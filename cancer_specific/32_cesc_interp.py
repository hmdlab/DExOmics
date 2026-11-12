#!/usr/bin/env python3
"""
32_cesc_interp.py — CESC の解釈を新モデルで再計算（旧 Supp S4 の置換。30 の CESC 版）

3 プロセス構成: --part xgb / cnn / analysis / all
- xgb: CESC XGBoost（fold 毎確定パラメータ、5 seed）再学習 + TreeSHAP（OOF）
- cnn: CESC pretrained CNN（28 の凍結 config・pre_ep 再現、3 seed）+ Expected Gradients（OOF）
- analysis: 因子レベル mean|attr|・SMARCB1/POLR2A の順位検証・モデル間一致
出力: results/exp5_interp/cesc_* / cesc_interp_summary.json
"""
import os, sys, json, time, subprocess, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
PART = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "all"
QUICK = "--quick" in sys.argv
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; F = f"{ROOT}/results/features"
E4 = f"{ROOT}/results/exp4_dl"; CACHE = f"{E4}/cesc_tensors"; OUT = f"{ROOT}/results/exp5_interp"
BATCH = 512
XPAR = {0: dict(max_depth=3, min_child_weight=1), 1: dict(max_depth=3, min_child_weight=1),
        2: dict(max_depth=3, min_child_weight=1), 3: dict(max_depth=3, min_child_weight=1),
        4: dict(max_depth=3, min_child_weight=20)}
CFG = {0: dict(C=64, drop=0.1, lr=1e-3), 1: dict(C=96, drop=0.2, lr=3e-4), 2: dict(C=96, drop=0.2, lr=1e-3),
       3: dict(C=128, drop=0.3, lr=1e-3), 4: dict(C=64, drop=0.1, lr=1e-3)}
PRE_EP = {0: 13, 1: 19, 2: 21, 3: 14, 4: 24}

if PART == "all":
    for part in ["xgb", "cnn", "analysis"]:
        print(f"=== part {part}", flush=True)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--part", part] + (["--quick"] if QUICK else []))
        if r.returncode != 0: sys.exit(f"part {part} failed")
    sys.exit(0)

# ---- CESC 遺伝子集合・fold（23/28 と同一） ----
de = pd.read_csv(f"{ROOT}/results/de_cesc/de_results.csv").set_index("gene_id")
de = de[de.gene_type.isin(["protein_coding", "lncRNA"]) & de.log2FC.notna()]
ann = pd.read_parquet(f"{F}/gene_annotation_gencode26.parquet").set_index("gene_id")
bind = pd.read_parquet(f"{F}/binding_gene_level_hela.parquet")
genes_idx = de.index.intersection(bind.index).intersection(ann.index)
df = de.loc[genes_idx, ["log2FC", "label"]].join(ann.loc[genes_idx, ["chrom"]])
df = df[df.chrom.isin([f"chr{i}" for i in range(1, 23)] + ["chrX"])]
folds = pd.read_csv(f"{ROOT}/results/exp1_nested_cv_cesc/folds.csv").set_index("chrom")["fold"]
df["fold"] = df.chrom.map(folds)
y = df.log2FC.values.astype(np.float32); fold = df.fold.values
omics = pd.concat([pd.read_parquet(f"{F}/{n}.parquet") for n in
                   ["cnv_gene_level_cesc", "mutation_gene_level_cesc", "methylation_gene_level_cesc"]], axis=1)
print(f"[{PART}] CESC genes {len(df)}", flush=True)
RAW_COLS = ("methy_", "log_length")

if PART == "xgb":
    import xgboost as xgb, shap
    from sklearn.preprocessing import StandardScaler
    X = pd.concat([bind.reindex(df.index).drop(columns=["n_bins_prom", "n_bins_rna"]),
                   omics.reindex(df.index).fillna(0.0)], axis=1)
    cols = list(X.columns)
    SEEDS = [1] if QUICK else [1, 2, 3, 4, 5]
    SH = np.zeros((len(df), len(cols)), dtype=np.float32)
    for k in range(5):
        tr = fold != k; te = ~tr
        keep = [c for c in cols if (X[tr][c] != 0).sum() >= 30]
        lg = [c for c in keep if not c.startswith(RAW_COLS)]
        def tf(Z):
            Z = Z.copy(); Z[lg] = np.log1p(Z[lg].clip(lower=0)); return Z
        sc = StandardScaler().fit(tf(X[tr][keep]))
        Xa, Xt = sc.transform(tf(X[tr][keep])), sc.transform(tf(X[te][keep]))
        kidx = [cols.index(c) for c in keep]
        fs = np.zeros((te.sum(), len(keep)), dtype=np.float32)
        for s in SEEDS:
            mdl = xgb.XGBRegressor(n_estimators=(50 if QUICK else 600), learning_rate=0.03, subsample=0.8,
                                   colsample_bytree=0.5, n_jobs=10, random_state=s, **XPAR[k]).fit(Xa, y[tr])
            fs += shap.TreeExplainer(mdl).shap_values(Xt) / len(SEEDS)
        SH[np.where(te)[0][:, None], np.array(kidx)[None, :]] = fs
        print(f"xgb fold {k} done", flush=True)
    np.save(f"{OUT}/cesc_shap_xgb.npy", SH)
    json.dump(cols, open(f"{OUT}/cesc_shap_cols.json", "w"))
    print("saved cesc_shap_xgb")

if PART == "cnn":
    import torch, torch.nn as nn
    DEV = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    P = np.log1p(np.load(f"{CACHE}/prom.npy")); R = np.log1p(np.load(f"{CACHE}/rna.npy"))
    nbins = np.load(f"{CACHE}/nbins.npy")
    NF_P, LP = P.shape[1], P.shape[2]; NF_R, LR = R.shape[1], R.shape[2]
    O = omics.reindex(df.index).fillna(0.0).values.astype(np.float32)
    O = np.hstack([O, np.log1p(nbins[:, None]).astype(np.float32)])
    pc = pd.read_csv(f"{ROOT}/data/pancan_data/transcriptome/pancanatlas_expid.txt", sep="\t", index_col=0)
    pc.index = [g.split(".")[0] for g in pc.index]
    pc = pc.drop(columns=["CESC", "MedianExp"], errors="ignore")
    pc = pc[~pc.index.duplicated(keep="first")]
    common = df.index.intersection(pc.index)
    pre_pos = pd.Series(range(len(df)), index=df.index).loc[common].values
    Ypre = pc.loc[common].values.astype(np.float32)
    Ypre = Ypre - Ypre.mean(axis=1, keepdims=True)

    class DWBranch(nn.Module):
        def __init__(self, nf, L, C, drop):
            super().__init__()
            self.a = nn.Sequential(nn.Conv1d(nf, nf, 5, padding=2, groups=nf), nn.GELU(),
                                   nn.Conv1d(nf, C, 1), nn.GELU(), nn.Dropout(drop))
            self.pe = nn.Parameter(torch.zeros(C, L))
            self.b = nn.Sequential(nn.Conv1d(C, C, 3, padding=1), nn.GELU())
        def forward(self, x):
            return self.b(self.a(x) + self.pe)

    class Model(nn.Module):
        def __init__(self, C, drop, n_pre):
            super().__init__()
            self.pb = DWBranch(NF_P, LP, C, drop); self.rb = DWBranch(NF_R, LR, C, drop)
            self.head = nn.Sequential(nn.LayerNorm(4 * C + O.shape[1]), nn.Dropout(drop),
                                      nn.Linear(4 * C + O.shape[1], 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))
            self.pre_head = nn.Linear(4 * C, n_pre)
        def trunk(self, p, r):
            hp = self.pb(p); hr = self.rb(r)
            return torch.cat([hp.mean(-1), hp.amax(-1), hr.mean(-1), hr.amax(-1)], dim=1)
        def forward(self, p, r, o):
            return self.head(torch.cat([self.trunk(p, r), o], dim=1)).squeeze(-1)
        def forward_pre(self, p, r):
            return self.pre_head(self.trunk(p, r))

    def train_fs(k, s):
        tr_idx = np.where(fold != k)[0]
        mu, sd_ = O[tr_idx].mean(0), O[tr_idx].std(0) + 1e-8; Ostd = (O - mu) / sd_
        tr_set = set(tr_idx); idx_pre = np.array([i for i in pre_pos if i in tr_set])
        sp = np.isin(pre_pos, idx_pre)
        Ys = ((Ypre - Ypre[sp].mean(0)) / (Ypre[sp].std(0) + 1e-8))[sp]
        torch.manual_seed(s); np.random.seed(s)
        cfg = CFG[k]
        m = Model(cfg["C"], cfg["drop"], Ypre.shape[1]).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        hub = nn.HuberLoss(delta=2.0); Yt = torch.from_numpy(Ys)
        for ep in range(2 if QUICK else PRE_EP[k]):
            m.train(); perm = np.random.permutation(len(idx_pre))
            for b in range(0, len(perm), BATCH):
                j = perm[b:b + BATCH]; sel = idx_pre[j]
                opt.zero_grad()
                hub(m.forward_pre(torch.from_numpy(P[sel]).to(DEV), torch.from_numpy(R[sel]).to(DEV)), Yt[j].to(DEV)).backward()
                opt.step()
        opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        for ep in range(3 if QUICK else 10):     # 28 の fine-tune 中央値相当（epoch は fold 毎ログが無いため固定 10、注記）
            m.train(); perm = np.random.permutation(len(tr_idx))
            for b in range(0, len(perm), BATCH):
                sel = tr_idx[perm[b:b + BATCH]]
                o = torch.from_numpy(Ostd[sel].astype(np.float32)).to(DEV)
                opt.zero_grad()
                hub(m(torch.from_numpy(P[sel]).to(DEV), torch.from_numpy(R[sel]).to(DEV), o),
                    torch.from_numpy(y[sel]).to(DEV)).backward()
                opt.step()
        return m, Ostd, tr_idx

    def expected_gradients(m, Ostd, te_idx, bg_idx, nsamples, rng):
        m.eval(); n = len(te_idx)
        ep_ = np.zeros((n, NF_P, LP), dtype=np.float32)
        er_ = np.zeros((n, NF_R, LR), dtype=np.float32)
        eo_ = np.zeros((n, O.shape[1]), dtype=np.float32)
        Ost = Ostd.astype(np.float32)
        for b0 in range(0, n, 256):
            sel = te_idx[b0:b0 + 256]; nb = len(sel)
            xp = torch.from_numpy(P[sel]).to(DEV); xr = torch.from_numpy(R[sel]).to(DEV)
            xo = torch.from_numpy(Ost[sel]).to(DEV)
            ap = torch.zeros_like(xp); ar = torch.zeros_like(xr); ao = torch.zeros_like(xo)
            for _ in range(nsamples):
                bg = bg_idx[rng.integers(0, len(bg_idx), nb)]
                bp = torch.from_numpy(P[bg]).to(DEV); br = torch.from_numpy(R[bg]).to(DEV)
                bo = torch.from_numpy(Ost[bg]).to(DEV)
                al = torch.from_numpy(rng.random(nb).astype(np.float32)).to(DEV)[:, None, None]
                ip = (bp + al * (xp - bp)).requires_grad_(True)
                ir = (br + al * (xr - br)).requires_grad_(True)
                io = (bo + al[:, :, 0] * (xo - bo)).requires_grad_(True)
                gp, gr, go = torch.autograd.grad(m(ip, ir, io).sum(), [ip, ir, io])
                ap += (xp - bp) * gp; ar += (xr - br) * gr; ao += (xo - bo) * go
            ep_[b0:b0 + nb] = (ap / nsamples).detach().cpu().numpy()
            er_[b0:b0 + nb] = (ar / nsamples).detach().cpu().numpy()
            eo_[b0:b0 + nb] = (ao / nsamples).detach().cpu().numpy()
        return ep_, er_, eo_

    rng = np.random.default_rng(0)
    SEEDS = [1] if QUICK else [1, 2, 3]
    NS = 4 if QUICK else 32
    EGP = np.zeros((len(df), NF_P, LP), dtype=np.float32)
    EGR = np.zeros((len(df), NF_R, LR), dtype=np.float32)
    EGO = np.zeros((len(df), O.shape[1]), dtype=np.float32)
    for k in range(5):
        te_idx = np.where(fold == k)[0]
        for s in SEEDS:
            t0 = time.time()
            m, Ostd, tr_idx = train_fs(k, s)
            bg = tr_idx[rng.choice(len(tr_idx), 50, replace=False)]
            a, b, c = expected_gradients(m, Ostd, te_idx, bg, NS, rng)
            EGP[te_idx] += a / len(SEEDS); EGR[te_idx] += b / len(SEEDS); EGO[te_idx] += c / len(SEEDS)
            print(f"cnn fold {k} seed {s} ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{OUT}/cesc_eg_prom.npy", EGP); np.save(f"{OUT}/cesc_eg_rna.npy", EGR); np.save(f"{OUT}/cesc_eg_omics.npy", EGO)
    json.dump({"omics_cols": list(omics.columns) + ["log_n_bins"]}, open(f"{OUT}/cesc_eg_meta.json", "w"))
    print("saved cesc EG")

if PART == "analysis":
    from scipy.stats import spearmanr
    SH = np.load(f"{OUT}/cesc_shap_xgb.npy"); xcols = json.load(open(f"{OUT}/cesc_shap_cols.json"))
    EGP = np.load(f"{OUT}/cesc_eg_prom.npy"); EGR = np.load(f"{OUT}/cesc_eg_rna.npy"); EGO = np.load(f"{OUT}/cesc_eg_omics.npy")
    ocols = json.load(open(f"{OUT}/cesc_eg_meta.json"))["omics_cols"]
    prom_names = [c for c in bind.columns if c.startswith("prom__")]
    rna_names = [c for c in bind.columns if c.startswith("rna__")]
    cnn_fac = np.concatenate([EGP.sum(2), EGR.sum(2), EGO], axis=1)
    cnn_names = prom_names + rna_names + ocols
    xmap = {c: i for i, c in enumerate(xcols)}
    shared = [c for c in cnn_names if c in xmap and c != "log_n_bins"]
    XS = SH[:, [xmap[c] for c in shared]]; CS = cnn_fac[:, [cnn_names.index(c) for c in shared]]
    mx = np.abs(XS).mean(0); mx = mx / mx.sum(); mc = np.abs(CS).mean(0); mc = mc / mc.sum()
    rank_x = pd.Series(mx, index=shared).rank(ascending=False)
    rank_c = pd.Series(mc, index=shared).rank(ascending=False)
    res = {"n_shared": len(shared),
           "top10_xgb": [shared[i] for i in np.argsort(-mx)[:10]],
           "top10_cnn": [shared[i] for i in np.argsort(-mc)[:10]],
           "crossmodel_spearman": round(float(spearmanr(mx, mc)[0]), 3)}
    for f_ in ["prom__SMARCB1", "prom__POLR2A"]:
        if f_ in shared:
            res[f_] = {"rank_xgb": int(rank_x[f_]), "rank_cnn": int(rank_c[f_]),
                       "of": len(shared),
                       "signed_mean_by_class_xgb": {g: round(float(XS[(df.label == g).values, shared.index(f_)].mean()), 5)
                                                    for g in ["UP", "NonDEG", "DOWN"]}}
    np.save(f"{OUT}/cesc_factor_share_xgb.npy", mx); np.save(f"{OUT}/cesc_factor_share_cnn.npy", mc)
    json.dump({"shared": shared}, open(f"{OUT}/cesc_shared_factors.json", "w"))
    json.dump(res, open(f"{OUT}/cesc_interp_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))

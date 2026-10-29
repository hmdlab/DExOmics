#!/usr/bin/env python3
"""
30_interpretability.py — 新モデル（XGB + 事前学習 CNN）の解釈ブロック（notes/05 の事前指定に準拠）

3 プロセス構成（libomp 衝突回避）:
  --part xgb      : fold 毎再学習(5 seed) + TreeSHAP → OOF SHAP 行列（factor 678 列）
  --part xgb-nocnv: cnv 4 列除外の XGB 再学習（増幅の必要性検証、1 seed）
  --part cnn      : 27 プロトコル再現(3 seed) + 手実装 Expected Gradients → OOF EG（因子別 + 位置別）
  --part analysis : 安定性・モデル間一致・増幅再現・位置プロファイルの集計
  --part all      : xgb → xgb-nocnv → cnn → analysis を順次実行

出力: results/exp5_interp/
実行: venv/bin/python analysis/30_interpretability.py [--quick]
"""
import os, sys, json, time, subprocess, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
PART = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "all"
QUICK = "--quick" in sys.argv
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; T = f"{ROOT}/results/tensors"; F = f"{ROOT}/results/features"
E4 = f"{ROOT}/results/exp4_dl"; OUT = f"{ROOT}/results/exp5_interp"; os.makedirs(OUT, exist_ok=True)
BATCH = 512
FOLD_CFG = {0: dict(C=96, drop=0.2, lr=3e-4), 1: dict(C=128, drop=0.2, lr=1e-3), 2: dict(C=96, drop=0.2, lr=1e-3),
            3: dict(C=96, drop=0.2, lr=1e-3), 4: dict(C=128, drop=0.3, lr=1e-3)}
XPARAMS = {0: dict(max_depth=5, min_child_weight=5), 1: dict(max_depth=7, min_child_weight=1),
           2: dict(max_depth=5, min_child_weight=1), 3: dict(max_depth=5, min_child_weight=1),
           4: dict(max_depth=3, min_child_weight=1)}   # 06 all|xgboost の fold 毎確定値

if PART == "all":
    for part in ["xgb", "xgb-nocnv", "cnn", "analysis"]:
        cmd = [sys.executable, os.path.abspath(__file__), "--part", part] + (["--quick"] if QUICK else [])
        print(f"=== part {part}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0: sys.exit(f"part {part} failed")
    sys.exit(0)

genes = pd.read_csv(f"{T}/genes.csv", index_col=0)
y = genes.log2FC.values.astype(np.float32); fold = genes.fold.values
RAW_COLS = ("methy_", "log_length")

def build_xagg():
    bind = pd.read_parquet(f"{F}/binding_gene_level.parquet")
    omics = pd.concat([pd.read_parquet(f"{F}/{n}.parquet") for n in ["cnv_gene_level", "mutation_gene_level", "methylation_gene_level"]], axis=1)
    return pd.concat([bind.reindex(genes.index).drop(columns=["n_bins_prom", "n_bins_rna"]),
                      omics.reindex(genes.index).fillna(0.0)], axis=1)

# ============================ XGB parts ============================
if PART in ("xgb", "xgb-nocnv"):
    import xgboost as xgb, shap
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    X = build_xagg()
    if PART == "xgb-nocnv":
        X = X.drop(columns=[c for c in X.columns if c.startswith("cnv__")])
    cols = list(X.columns)
    SEEDS = [1] if PART == "xgb-nocnv" else ([1] if QUICK else [1, 2, 3, 4, 5])
    shap_sum = np.zeros((len(genes), len(cols)), dtype=np.float32)
    shap_absmean_per_seed = []
    pred = np.full(len(genes), np.nan)
    for k in range(5):
        tr = fold != k; te = ~tr
        keep = [c for c in cols if (X[tr][c] != 0).sum() >= 30]
        lg = [c for c in keep if not c.startswith(RAW_COLS)]
        def tf(Z):
            Z = Z.copy(); Z[lg] = np.log1p(Z[lg].clip(lower=0)); return Z
        sc = StandardScaler().fit(tf(X[tr][keep]))
        Xa = sc.transform(tf(X[tr][keep])); Xt = sc.transform(tf(X[te][keep]))
        kidx = [cols.index(c) for c in keep]
        fold_shap = np.zeros((te.sum(), len(keep)), dtype=np.float32)
        fold_pred = np.zeros(te.sum())
        for s in SEEDS:
            mdl = xgb.XGBRegressor(n_estimators=(50 if QUICK else 600), learning_rate=0.03, subsample=0.8,
                                   colsample_bytree=0.5, n_jobs=10, random_state=s, **XPARAMS[k]).fit(Xa, y[tr])
            fold_pred += mdl.predict(Xt) / len(SEEDS)
            ex = shap.TreeExplainer(mdl)              # tree_path_dependent（高速・決定的）
            sv = ex.shap_values(Xt)
            fold_shap += sv / len(SEEDS)
            if PART == "xgb":
                m = np.zeros(len(cols)); m[kidx] = np.abs(sv).mean(0)
                shap_absmean_per_seed.append(dict(fold=k, seed=s, **{c: float(v) for c, v in zip(cols, m) if v > 0}))
        shap_sum[np.where(te)[0][:, None], np.array(kidx)[None, :]] = fold_shap
        pred[te] = fold_pred
        print(f"[{PART}] fold {k} done ({len(keep)} feats)", flush=True)
    if PART == "xgb":
        np.save(f"{OUT}/shap_xgb_oof.npy", shap_sum)
        json.dump(cols, open(f"{OUT}/shap_xgb_cols.json", "w"))
        pd.DataFrame(shap_absmean_per_seed).to_csv(f"{OUT}/shap_xgb_absmean_per_seed.csv", index=False)
        print("saved shap_xgb_oof.npy")
    else:
        rho = float(spearmanr(y, pred)[0])
        json.dump({"oof_spearman_nocnv": rho, "note": "cnv 4 列除外・seed1。比較対象は all|xgboost seed 平均 0.357 / 単一 seed ~0.346"},
                  open(f"{OUT}/xgb_nocnv.json", "w"), indent=1)
        print("no-cnv OOF spearman:", round(rho, 4))

# ============================ CNN part ============================
if PART == "cnn":
    import torch, torch.nn as nn
    DEV = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    P = np.log1p(np.load(f"{T}/prom.npy")); R = np.log1p(np.load(f"{T}/rna.npy"))
    omics = pd.concat([pd.read_parquet(f"{F}/{n}.parquet") for n in ["cnv_gene_level", "mutation_gene_level", "methylation_gene_level"]], axis=1)
    OM = omics.reindex(genes.index).fillna(0.0).values.astype(np.float32)
    OM = np.hstack([OM, np.log1p(genes.n_bins_rna.values[:, None]).astype(np.float32)])
    OCOLS = list(omics.columns) + ["log_n_bins"]
    pc = pd.read_csv(f"{ROOT}/data/pancan_data/transcriptome/pancanatlas_expid.txt", sep="\t", index_col=0)
    pc.index = [g.split(".")[0] for g in pc.index]
    pc = pc.drop(columns=["LIHC", "MedianExp"], errors="ignore")
    pc = pc[~pc.index.duplicated(keep="first")]
    common = genes.index.intersection(pc.index)
    pre_pos = pd.Series(range(len(genes)), index=genes.index).loc[common].values
    Ypre_raw = pc.loc[common].values.astype(np.float32)
    Ypre_raw = Ypre_raw - Ypre_raw.mean(axis=1, keepdims=True)
    plog = json.load(open(f"{E4}/pretrain_log.json"))["runs"]
    PRE_EP = {r["fold"]: r["pre_ep"] for r in plog}
    MED_EP = {(r["fold"], r["seed"]): r["med_ep"] for r in plog}
    SEEDS = [1] if QUICK else [1, 2, 3]
    NS = 4 if QUICK else 32     # EG サンプル対 / 遺伝子

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
            self.pb = DWBranch(550, 30, C, drop); self.rb = DWBranch(86, 32, C, drop)
            self.head = nn.Sequential(nn.LayerNorm(4 * C + OM.shape[1]), nn.Dropout(drop),
                                      nn.Linear(4 * C + OM.shape[1], 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))
            self.pre_head = nn.Linear(4 * C, n_pre)
        def trunk(self, p, r):
            hp = self.pb(p); hr = self.rb(r)
            return torch.cat([hp.mean(-1), hp.amax(-1), hr.mean(-1), hr.amax(-1)], dim=1)
        def forward(self, p, r, o):
            return self.head(torch.cat([self.trunk(p, r), o], dim=1)).squeeze(-1)
        def forward_pre(self, p, r):
            return self.pre_head(self.trunk(p, r))

    def train_fold_seed(k, s, cfg):
        tr_idx = np.where(fold != k)[0]
        mu, sd_ = OM[tr_idx].mean(0), OM[tr_idx].std(0) + 1e-8; Ostd = (OM - mu) / sd_
        tr_set = set(tr_idx); idx_pre = np.array([i for i in pre_pos if i in tr_set])
        sel_pre = np.isin(pre_pos, idx_pre)
        Ypre_std = ((Ypre_raw - Ypre_raw[sel_pre].mean(0)) / (Ypre_raw[sel_pre].std(0) + 1e-8))[sel_pre]
        torch.manual_seed(s); np.random.seed(s)
        m = Model(cfg["C"], cfg["drop"], Ypre_raw.shape[1]).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        hub = nn.HuberLoss(delta=2.0); Yt = torch.from_numpy(Ypre_std)
        for ep in range(2 if QUICK else PRE_EP[k]):
            m.train(); perm = np.random.permutation(len(idx_pre))
            for b in range(0, len(perm), BATCH):
                j = perm[b:b + BATCH]; sel = idx_pre[j]
                opt.zero_grad()
                hub(m.forward_pre(torch.from_numpy(P[sel]).to(DEV), torch.from_numpy(R[sel]).to(DEV)), Yt[j].to(DEV)).backward()
                opt.step()
        opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        for ep in range(3 if QUICK else MED_EP[(k, s)]):
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
        """EG を test 遺伝子ごとに算出。返り値: (eg_p [n,550,30], eg_r [n,86,32], eg_o [n,43])"""
        m.eval()
        n = len(te_idx)
        eg_p = np.zeros((n, 550, 30), dtype=np.float32)
        eg_r = np.zeros((n, 86, 32), dtype=np.float32)
        eg_o = np.zeros((n, OM.shape[1]), dtype=np.float32)
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
                out = m(ip, ir, io).sum()
                gp, gr, go = torch.autograd.grad(out, [ip, ir, io])
                ap += (xp - bp) * gp; ar += (xr - br) * gr; ao += (xo - bo) * go
            eg_p[b0:b0 + nb] = (ap / nsamples).detach().cpu().numpy()
            eg_r[b0:b0 + nb] = (ar / nsamples).detach().cpu().numpy()
            eg_o[b0:b0 + nb] = (ao / nsamples).detach().cpu().numpy()
        return eg_p, eg_r, eg_o

    rng = np.random.default_rng(0)
    EGP = np.zeros((len(genes), 550, 30), dtype=np.float32)
    EGR = np.zeros((len(genes), 86, 32), dtype=np.float32)
    EGO = np.zeros((len(genes), OM.shape[1]), dtype=np.float32)
    conv_report = {}
    for k in range(5):
        te_idx = np.where(fold == k)[0]
        for s in SEEDS:
            t0 = time.time()
            m, Ostd, tr_idx = train_fold_seed(k, s, FOLD_CFG[k])
            bg_idx = tr_idx[rng.choice(len(tr_idx), 50, replace=False)]
            ep_, er_, eo_ = expected_gradients(m, Ostd, te_idx, bg_idx, NS, rng)
            EGP[te_idx] += ep_ / len(SEEDS); EGR[te_idx] += er_ / len(SEEDS); EGO[te_idx] += eo_ / len(SEEDS)
            if k == 0 and s == 1 and not QUICK:
                sub = te_idx[:400]
                r1 = expected_gradients(m, Ostd, sub, bg_idx, 64, np.random.default_rng(1))
                a32 = np.abs(ep_[:400]).sum((0, 2)); a64 = np.abs(r1[0]).sum((0, 2))
                from scipy.stats import spearmanr as _sp
                conv_report["prom_factor_rank_spearman_ns32_vs_64"] = round(float(_sp(a32, a64)[0]), 4)
                bg2 = tr_idx[np.argsort(np.abs(genes.log2FC.values[tr_idx]))[:50]]   # 低|logFC| マッチ背景（感度）
                r2 = expected_gradients(m, Ostd, sub, bg2, 32, np.random.default_rng(2))
                a_alt = np.abs(r2[0]).sum((0, 2))
                conv_report["prom_factor_rank_spearman_randombg_vs_lowlfcbg"] = round(float(_sp(a32, a_alt)[0]), 4)
            print(f"cnn fold {k} seed {s} EG done ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{OUT}/eg_prom_oof.npy", EGP); np.save(f"{OUT}/eg_rna_oof.npy", EGR); np.save(f"{OUT}/eg_omics_oof.npy", EGO)
    json.dump({"omics_cols": OCOLS, "nsamples": NS, "seeds": SEEDS, "convergence": conv_report},
              open(f"{OUT}/eg_meta.json", "w"), indent=1)
    print("saved EG matrices |", conv_report)

# ============================ analysis part ============================
if PART == "analysis":
    from scipy.stats import spearmanr
    bind = pd.read_parquet(f"{F}/binding_gene_level.parquet")
    prom_names = [c for c in bind.columns if c.startswith("prom__")]
    rna_names = [c for c in bind.columns if c.startswith("rna__")]
    SX = np.load(f"{OUT}/shap_xgb_oof.npy"); xcols = json.load(open(f"{OUT}/shap_xgb_cols.json"))
    EGP = np.load(f"{OUT}/eg_prom_oof.npy"); EGR = np.load(f"{OUT}/eg_rna_oof.npy"); EGO = np.load(f"{OUT}/eg_omics_oof.npy")
    meta = json.load(open(f"{OUT}/eg_meta.json")); ocols = meta["omics_cols"]
    # 因子レベル: CNN = 位置合計（signed）
    cnn_fac = np.concatenate([EGP.sum(2), EGR.sum(2), EGO], axis=1)
    cnn_names = prom_names + rna_names + ocols
    xmap = {c: i for i, c in enumerate(xcols)}
    shared = [c for c in cnn_names if c in xmap and c != "log_n_bins"]
    XS = SX[:, [xmap[c] for c in shared]]
    CS = cnn_fac[:, [cnn_names.index(c) for c in shared]]
    res = {"n_shared_factors": len(shared)}
    # (i) モデル間一致（fold 別 Spearman of mean|attr|、top-20 Jaccard、符号一致）
    per_fold = []
    for k in range(5):
        te = fold == k
        mx = np.abs(XS[te]).mean(0); mc = np.abs(CS[te]).mean(0)
        rho = float(spearmanr(mx, mc)[0])
        tx = set(np.argsort(-mx)[:20]); tc = set(np.argsort(-mc)[:20])
        jac = len(tx & tc) / len(tx | tc)
        sgn = float(np.mean(np.sign(XS[te].mean(0)) == np.sign(CS[te].mean(0))))
        per_fold.append(dict(fold=k, spearman=rho, top20_jaccard=jac, sign_concord=sgn))
    pf = pd.DataFrame(per_fold); pf.to_csv(f"{OUT}/crossmodel_agreement.csv", index=False)
    res["crossmodel"] = {"spearman_mean": round(float(pf.spearman.mean()), 3),
                         "spearman_range": [round(float(pf.spearman.min()), 3), round(float(pf.spearman.max()), 3)],
                         "top20_jaccard_mean": round(float(pf.top20_jaccard.mean()), 3),
                         "sign_concord_mean": round(float(pf.sign_concord.mean()), 3)}
    # (ii) 上位因子（両モデル mean|attr|、参考: 安定コアとの重なり）
    mx = np.abs(XS).mean(0); mc = np.abs(CS).mean(0)
    res["top10_xgb"] = [shared[i] for i in np.argsort(-mx)[:10]]
    res["top10_cnn"] = [shared[i] for i in np.argsort(-mc)[:10]]
    # (iii) 増幅の再現: signed attr(cnv) と logFC の関係（ブロック bootstrap CI）
    yv = genes.log2FC.values; chrom = genes.chrom.values; chroms = np.unique(chrom)
    rng = np.random.default_rng(0)
    for label, A, names in [("xgb", SX, shared), ("cnn", CS, shared)]:
        for feat in ["cnv__Amp_rate", "cnv__mean_cn"]:
            if feat not in names: continue
            a = A[:, names.index(feat)]
            rho = float(spearmanr(a, yv)[0])
            boot = []
            for b in range(1000):
                samp = rng.choice(chroms, len(chroms), replace=True)
                idx = np.concatenate([np.where(chrom == c)[0] for c in samp])
                boot.append(spearmanr(a[idx], yv[idx])[0])
            ci = np.quantile(boot, [0.025, 0.975]).round(4).tolist()
            grp = {g: round(float(a[genes.label.values == g].mean()), 4) for g in ["UP", "NonDEG", "DOWN"]}
            res[f"amp_{label}_{feat}"] = {"spearman_vs_logFC": round(rho, 4), "ci": ci, "group_means": grp}
    # 必要性（xgb-nocnv の結果を転記）
    try:
        res["cnv_necessity"] = json.load(open(f"{OUT}/xgb_nocnv.json"))
    except Exception:
        pass
    # (iv) 位置プロファイル（探索的）
    np.save(f"{OUT}/pos_profile_prom.npy", np.abs(EGP).mean((0, 1)) if False else np.abs(EGP).sum(1).mean(0))
    np.save(f"{OUT}/pos_profile_rna.npy", np.abs(EGR).sum(1).mean(0))
    res["convergence"] = meta.get("convergence", {})
    json.dump(res, open(f"{OUT}/interp_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))

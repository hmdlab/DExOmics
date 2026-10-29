#!/usr/bin/env python3
"""
29_direct_classification.py — (b) 直接 3 クラス分類学習 vs 回帰→閾値（PI 指示、codex 設計チェック反映）

macOS の libomp 二重ロード（xgboost と torch）によるデッドロック回避のため 3 プロセス構成:
  --part xgb : XGB-clf（multi:softprob、balanced weight、grid depth{3,5,7}、5-seed 確率平均）
  --part cnn : CNN-clf（事前学習 trunk 転用 + 3 logits、weighted CE、監視 = weighted val CE、3-seed）
  --part eval: 両者読込 → argmax（primary）+ prior 補正（感度）で評価
  --part all（既定）: 上記 3 つを subprocess で順次実行

出力: results/exp4_dl/cls_prob_{xgb,cnn}.npy, classification_direct.csv, oof_classification.parquet
実行: venv/bin/python analysis/29_direct_classification.py [--quick]
"""
import os, sys, json, time, subprocess, warnings, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")

PART = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "all"
QUICK = "--quick" in sys.argv
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; T = f"{ROOT}/results/tensors"; F = f"{ROOT}/results/features"
OUT = f"{ROOT}/results/exp4_dl"
BATCH = 512; MAX_EPOCH = 3 if QUICK else 150; PATIENCE = 10
XSEEDS = [1, 2, 3, 4, 5]; CSEEDS = [1, 2, 3]
FOLD_CFG = {0: dict(C=96, drop=0.2, lr=3e-4), 1: dict(C=128, drop=0.2, lr=1e-3), 2: dict(C=96, drop=0.2, lr=1e-3),
            3: dict(C=96, drop=0.2, lr=1e-3), 4: dict(C=128, drop=0.3, lr=1e-3)}

if PART == "all":
    for part in ["xgb", "cnn", "eval"]:
        cmd = [sys.executable, os.path.abspath(__file__), "--part", part] + (["--quick"] if QUICK else [])
        print(f"=== running part {part}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0: sys.exit(f"part {part} failed ({r.returncode})")
    sys.exit(0)

genes = pd.read_csv(f"{T}/genes.csv", index_col=0)
ycls = genes.label.map({"DOWN": 0, "NonDEG": 1, "UP": 2}).values
fold = genes.fold.values
CW = len(ycls) / (3 * np.bincount(ycls))
print(f"[{PART}] class counts {np.bincount(ycls)} weights {CW.round(2)}", flush=True)

def macro_prauc(y, prob):
    return float(np.mean([average_precision_score(y == c, prob[:, c]) for c in range(3)]))

# ============================== XGB part ==============================
if PART == "xgb":
    import xgboost as xgb
    omics = pd.concat([pd.read_parquet(f"{F}/{n}.parquet") for n in ["cnv_gene_level", "mutation_gene_level", "methylation_gene_level"]], axis=1)
    bind = pd.read_parquet(f"{F}/binding_gene_level.parquet")
    Xagg = pd.concat([bind.reindex(genes.index).drop(columns=["n_bins_prom", "n_bins_rna"]),
                      omics.reindex(genes.index).fillna(0.0)], axis=1)
    sw = CW[ycls]
    GRID = [3, 5, 7] if not QUICK else [5]
    probs = {s: np.full((len(genes), 3), np.nan) for s in XSEEDS}
    for k in range(5):
        tr = fold != k; te = ~tr
        keep = [c for c in Xagg.columns if (Xagg[tr][c] != 0).sum() >= 30]
        Xa, Xt = Xagg[tr][keep].values, Xagg[te][keep].values
        ftr = fold[tr]; best, best_s = None, -np.inf
        for d in GRID:
            scores = []
            for hold in np.unique(ftr):
                m = ftr != hold; v = ftr == hold
                mdl = xgb.XGBClassifier(n_estimators=(50 if QUICK else 600), learning_rate=0.03, subsample=0.8,
                                        colsample_bytree=0.5, max_depth=d, min_child_weight=1, tree_method="hist",
                                        objective="multi:softprob", n_jobs=10, random_state=0)
                mdl.fit(Xa[m], ycls[tr][m], sample_weight=sw[tr][m])
                scores.append(macro_prauc(ycls[tr][v], mdl.predict_proba(Xa[v])))
            s = float(np.nanmean(scores))
            print(f"xgb-clf fold {k} depth {d} inner {s:.3f}", flush=True)
            if s > best_s: best, best_s = d, s
        for sd in XSEEDS:
            mdl = xgb.XGBClassifier(n_estimators=(50 if QUICK else 600), learning_rate=0.03, subsample=0.8,
                                    colsample_bytree=0.5, max_depth=best, min_child_weight=1, tree_method="hist",
                                    objective="multi:softprob", n_jobs=10, random_state=sd)
            mdl.fit(Xa, ycls[tr], sample_weight=sw[tr])
            probs[sd][te] = mdl.predict_proba(Xt)
        print(f"xgb-clf fold {k} FINAL depth {best}", flush=True)
    np.save(f"{OUT}/cls_prob_xgb.npy", np.mean([probs[s] for s in XSEEDS], axis=0))
    print("saved cls_prob_xgb.npy")

# ============================== CNN part ==============================
if PART == "cnn":
    import torch, torch.nn as nn
    DEV = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    P = np.log1p(np.load(f"{T}/prom.npy")); R = np.log1p(np.load(f"{T}/rna.npy"))
    omics = pd.concat([pd.read_parquet(f"{F}/{n}.parquet") for n in ["cnv_gene_level", "mutation_gene_level", "methylation_gene_level"]], axis=1)
    O = omics.reindex(genes.index).fillna(0.0).values.astype(np.float32)
    O = np.hstack([O, np.log1p(genes.n_bins_rna.values[:, None]).astype(np.float32)])
    pc = pd.read_csv(f"{ROOT}/data/pancan_data/transcriptome/pancanatlas_expid.txt", sep="\t", index_col=0)
    pc.index = [g.split(".")[0] for g in pc.index]
    pc = pc.drop(columns=["LIHC", "MedianExp"], errors="ignore")
    pc = pc[~pc.index.duplicated(keep="first")]
    common = genes.index.intersection(pc.index)
    pre_pos = pd.Series(range(len(genes)), index=genes.index).loc[common].values
    Ypre_raw = pc.loc[common].values.astype(np.float32)
    Ypre_raw = Ypre_raw - Ypre_raw.mean(axis=1, keepdims=True)
    try:
        PRE_EP = {r["fold"]: r.get("pre_ep", 10) for r in json.load(open(f"{OUT}/pretrain_log.json"))["runs"]}
    except Exception:
        PRE_EP = {k: 10 for k in range(5)}

    class DWBranch(nn.Module):
        def __init__(self, nf, L, C, drop):
            super().__init__()
            self.a = nn.Sequential(nn.Conv1d(nf, nf, 5, padding=2, groups=nf), nn.GELU(),
                                   nn.Conv1d(nf, C, 1), nn.GELU(), nn.Dropout(drop))
            self.pe = nn.Parameter(torch.zeros(C, L))
            self.b = nn.Sequential(nn.Conv1d(C, C, 3, padding=1), nn.GELU())
        def forward(self, x):
            return self.b(self.a(x) + self.pe)

    class ModelCls(nn.Module):
        def __init__(self, C, drop, n_pre):
            super().__init__()
            self.pb = DWBranch(550, 30, C, drop); self.rb = DWBranch(86, 32, C, drop)
            self.head = nn.Sequential(nn.LayerNorm(4 * C + O.shape[1]), nn.Dropout(drop),
                                      nn.Linear(4 * C + O.shape[1], 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 3))
            self.pre_head = nn.Linear(4 * C, n_pre)
        def trunk(self, p, r):
            hp = self.pb(p); hr = self.rb(r)
            return torch.cat([hp.mean(-1), hp.amax(-1), hr.mean(-1), hr.amax(-1)], dim=1)
        def forward(self, p, r, o):
            return self.head(torch.cat([self.trunk(p, r), o], dim=1))
        def forward_pre(self, p, r):
            return self.pre_head(self.trunk(p, r))

    @torch.no_grad()
    def predict_prob(mdl, idx, Ostd):
        mdl.eval(); out = []
        for b in range(0, len(idx), 2048):
            sel = idx[b:b + 2048]
            lg = mdl(torch.from_numpy(P[sel]).to(DEV), torch.from_numpy(R[sel]).to(DEV),
                     torch.from_numpy(Ostd[sel].astype(np.float32)).to(DEV))
            out.append(torch.softmax(lg, dim=1).cpu().numpy())
        return np.concatenate(out)

    def pretrain(mdl, idx_pre, Ypre_std, max_epoch):
        opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.HuberLoss(delta=2.0); Yt = torch.from_numpy(Ypre_std)
        for ep in range(max_epoch):
            mdl.train(); perm = np.random.permutation(len(idx_pre))
            for b in range(0, len(perm), BATCH):
                j = perm[b:b + BATCH]; sel = idx_pre[j]
                opt.zero_grad()
                lossf(mdl.forward_pre(torch.from_numpy(P[sel]).to(DEV), torch.from_numpy(R[sel]).to(DEV)),
                      Yt[j].to(DEV)).backward()
                opt.step()
        return mdl

    def finetune_cls(mdl, cfg, idx_tr, Ostd, idx_val=None, fixed_epochs=None):
        opt = torch.optim.AdamW(mdl.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss(weight=torch.tensor(CW, dtype=torch.float32).to(DEV))
        best_s, best_ep, bad = -np.inf, 0, 0
        yt = torch.from_numpy(ycls)
        for ep in range(1, (fixed_epochs or MAX_EPOCH) + 1):
            mdl.train(); perm = np.random.permutation(len(idx_tr))
            for b in range(0, len(perm), BATCH):
                sel = idx_tr[perm[b:b + BATCH]]
                o = torch.from_numpy(Ostd[sel].astype(np.float32)).to(DEV)
                opt.zero_grad()
                lossf(mdl(torch.from_numpy(P[sel]).to(DEV), torch.from_numpy(R[sel]).to(DEV), o),
                      yt[sel].to(DEV)).backward()
                opt.step()
            if idx_val is not None and fixed_epochs is None:
                pv = predict_prob(mdl, idx_val, Ostd)
                w = CW[ycls[idx_val]]
                ce = float(-(w * np.log(np.clip(pv[np.arange(len(idx_val)), ycls[idx_val]], 1e-8, 1))).mean())
                if -ce > best_s: best_s, best_ep, bad = -ce, ep, 0
                else:
                    bad += 1
                    if bad >= PATIENCE: break
        return mdl, best_s, best_ep

    probs = {s: np.full((len(genes), 3), np.nan) for s in CSEEDS}
    for k in range(5):
        cfg = FOLD_CFG[k]
        tr_idx = np.where(fold != k)[0]; te_idx = np.where(fold == k)[0]
        mu, sd_ = O[tr_idx].mean(0), O[tr_idx].std(0) + 1e-8; Ostd = (O - mu) / sd_
        tr_set = set(tr_idx); idx_pre = np.array([i for i in pre_pos if i in tr_set])
        sel_pre = np.isin(pre_pos, idx_pre)
        Ypre_std = ((Ypre_raw - Ypre_raw[sel_pre].mean(0)) / (Ypre_raw[sel_pre].std(0) + 1e-8))[sel_pre]
        pre_ep = 2 if QUICK else PRE_EP.get(k, 10)
        for s in CSEEDS:
            torch.manual_seed(s); np.random.seed(s)
            t0 = time.time(); eps = []
            for hold in sorted(set(fold[tr_idx]))[:2]:
                ia = tr_idx[fold[tr_idx] != hold]; iv = tr_idx[fold[tr_idx] == hold]
                ia_set = set(ia); ip = np.array([i for i in pre_pos if i in ia_set])
                sp = np.isin(pre_pos, ip)
                Ys = ((Ypre_raw - Ypre_raw[sp].mean(0)) / (Ypre_raw[sp].std(0) + 1e-8))[sp]
                m = ModelCls(cfg["C"], cfg["drop"], Ypre_raw.shape[1]).to(DEV)
                pretrain(m, ip, Ys, pre_ep)
                _, _, ep = finetune_cls(m, cfg, ia, Ostd, idx_val=iv)
                eps.append(ep)
            med_ep = max(int(np.median(eps)), 5)
            m = ModelCls(cfg["C"], cfg["drop"], Ypre_raw.shape[1]).to(DEV)
            pretrain(m, idx_pre, Ypre_std, pre_ep)
            m, _, _ = finetune_cls(m, cfg, tr_idx, Ostd, fixed_epochs=med_ep)
            probs[s][te_idx] = predict_prob(m, te_idx, Ostd)
            print(f"cnn-clf fold {k} seed {s} ft {med_ep} ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{OUT}/cls_prob_cnn.npy", np.mean([probs[s] for s in CSEEDS], axis=0))
    print("saved cls_prob_cnn.npy")

# ============================== eval part ==============================
if PART == "eval":
    PX = np.load(f"{OUT}/cls_prob_xgb.npy"); PC = np.load(f"{OUT}/cls_prob_cnn.npy")
    PE_ = (PX + PC) / 2

    def evaluate(name, prob, prior_correct=False):
        if prior_correct:
            prob = prob / CW[None, :]; prob = prob / prob.sum(1, keepdims=True)
        pred = prob.argmax(1)
        def mf1(y, p):
            f = 0.0
            for c in range(3):
                tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
                f += 0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
            return f / 3
        cm = np.zeros((3, 3), int)
        for a in range(3):
            for b in range(3): cm[a, b] = np.sum((ycls == a) & (pred == b))
        r = dict(model=name, pred_ratio=json.dumps((np.bincount(pred, minlength=3) / len(pred)).round(3).tolist()),
                 confusion=json.dumps(cm.tolist()), macro_prauc=macro_prauc(ycls, prob),
                 prauc_down=float(average_precision_score(ycls == 0, prob[:, 0])),
                 prauc_up=float(average_precision_score(ycls == 2, prob[:, 2])),
                 macro_f1=float(mf1(ycls, pred)),
                 recall_down=float(np.sum((pred == 0) & (ycls == 0)) / np.sum(ycls == 0)),
                 recall_up=float(np.sum((pred == 2) & (ycls == 2)) / np.sum(ycls == 2)),
                 recall_non=float(np.sum((pred == 1) & (ycls == 1)) / np.sum(ycls == 1)))
        print(f"{name:16s} macroPR {r['macro_prauc']:.3f} | PR down {r['prauc_down']:.3f} up {r['prauc_up']:.3f} | "
              f"macroF1 {r['macro_f1']:.3f} | recall D {r['recall_down']:.3f} U {r['recall_up']:.3f} N {r['recall_non']:.3f} | pred {r['pred_ratio']}", flush=True)
        return r
    rows = [evaluate("xgb-clf", PX), evaluate("cnn-clf", PC), evaluate("ens-clf", PE_),
            evaluate("xgb-clf-prior", PX, True), evaluate("cnn-clf-prior", PC, True), evaluate("ens-clf-prior", PE_, True)]
    pd.DataFrame(rows).to_csv(f"{OUT}/classification_direct.csv", index=False)
    oof = genes[["log2FC", "label", "chrom", "fold"]].copy()
    for i, c in enumerate(["DOWN", "NonDEG", "UP"]):
        oof[f"xgbclf_p{c}"] = PX[:, i]; oof[f"cnnclf_p{c}"] = PC[:, i]
    oof.to_parquet(f"{OUT}/oof_classification.parquet")
    print("saved classification_direct.csv / oof_classification.parquet")

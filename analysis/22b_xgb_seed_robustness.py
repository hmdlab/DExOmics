#!/usr/bin/env python3
"""
22b_xgb_seed_robustness.py — XGBoost 5-seed average OOF (the symmetric counterpart of 22 for the DL arm)

Re-implements the inline analysis that produced results/exp4_dl/oof_xgb_seedavg.parquet on 2026-09-21:
the "all" feature set (aggregate binding + omics, as in 06) is refit on each outer-train fold with the
fold-wise hyper-parameters that 06 selected for all|xgboost (XPARAMS below; no re-selection), for
random_state = 1..5. Pre-processing is identical to 06/30 (non-zero in >= 30 training genes, log1p for
count-like columns, standardisation fitted on the training fold).

Input : results/tensors/genes.csv (gene order / fold / log2FC), results/features/*.parquet
Output: results/exp4_dl/oof_xgb_seedavg.parquet  (columns xgb_seed1..5, all|xgboost_seedavg)
        results/exp4_dl/xgb_seedavg_out.txt        (per-seed and seed-averaged Spearman)
Run   : python analysis/22b_xgb_seed_robustness.py [--quick]
Reference values (Mac Studio, 2026-09-21): per-seed 0.3469/0.3542/0.3523/0.3464/0.3488, seed-avg 0.357.
"""
import os, sys, warnings, numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv
ROOT = DATA_ROOT; T = f"{ROOT}/results/tensors"; F = f"{ROOT}/results/features"
OUT = f"{ROOT}/results/exp4_dl"; os.makedirs(OUT, exist_ok=True)
SEEDS = [1] if QUICK else [1, 2, 3, 4, 5]
XPARAMS = {0: dict(max_depth=5, min_child_weight=5), 1: dict(max_depth=7, min_child_weight=1),
           2: dict(max_depth=5, min_child_weight=1), 3: dict(max_depth=5, min_child_weight=1),
           4: dict(max_depth=3, min_child_weight=1)}   # fold-wise choices of 06 all|xgboost (frozen)
RAW_COLS = ("methy_", "log_length")

genes = pd.read_csv(f"{T}/genes.csv", index_col=0)
y = genes.log2FC.values.astype(np.float32); fold = genes.fold.values
bind = pd.read_parquet(f"{F}/binding_gene_level.parquet")
omics = pd.concat([pd.read_parquet(f"{F}/{n}.parquet") for n in ["cnv_gene_level", "mutation_gene_level", "methylation_gene_level"]], axis=1)
X = pd.concat([bind.reindex(genes.index).drop(columns=["n_bins_prom", "n_bins_rna"]), omics.reindex(genes.index).fillna(0.0)], axis=1)
print("genes:", len(genes), "| columns:", X.shape[1], flush=True)

preds = {s: np.full(len(genes), np.nan) for s in SEEDS}
for k in range(5):
    tr = fold != k; te = ~tr
    keep = [c for c in X.columns if (X[tr][c] != 0).sum() >= 30]
    lg = [c for c in keep if not c.startswith(RAW_COLS)]
    def tf(Z):
        Z = Z.copy(); Z[lg] = np.log1p(Z[lg].clip(lower=0)); return Z
    sc = StandardScaler().fit(tf(X[tr][keep]))
    Xa = sc.transform(tf(X[tr][keep])); Xt = sc.transform(tf(X[te][keep]))
    for s in SEEDS:
        mdl = xgb.XGBRegressor(n_estimators=(50 if QUICK else 600), learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
                               n_jobs=8, random_state=s, **XPARAMS[k]).fit(Xa, y[tr])
        preds[s][te] = mdl.predict(Xt)
    print(f"fold {k} done", flush=True)

oof = genes[["log2FC", "label", "chrom", "fold"]].copy()
for s in SEEDS: oof[f"xgb_seed{s}"] = preds[s]
oof["all|xgboost_seedavg"] = np.mean([preds[s] for s in SEEDS], axis=0)
oof.to_parquet(f"{OUT}/oof_xgb_seedavg.parquet")
per_seed = [round(float(spearmanr(y, preds[s])[0]), 4) for s in SEEDS]
avg = round(float(spearmanr(y, oof["all|xgboost_seedavg"])[0]), 4)
with open(f"{OUT}/xgb_seedavg_out.txt", "w") as fh:
    fh.write("\n".join([f"fold {k} done" for k in range(5)] + [f"per-seed: {per_seed}", f"seed-avg: {avg}", ""]))
print("per-seed:", per_seed); print("seed-avg:", avg)

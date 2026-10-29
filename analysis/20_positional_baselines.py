#!/usr/bin/env python3
"""
20_positional_baselines.py — 2×2 比較の ③「位置分解入力 × 非 DL」（表現効果とモデル効果の分離用）

入力表現: flatten した位置特徴 prom [550×30=16,500] + rna pooled [86×32=2,752] + n_bins_rna + omics 42 列。
（① 集約+XGB=0.346 / ② 集約+MLP=0.252 は既存。④ 位置+DL は 21 で。）
プロトコルは 06 と同一: folds.csv の外側 5 fold、内側 4 fold（外側訓練群）で選択、
染色体 bootstrap は 09/06 と同じ後段で統合する。

モデル:
- xgboost(hist): 木は単調変換不変なので生値のまま。grid は depth {3,5,7}×mcw {1}（計算量を抑えた縮小 grid。
  06 の「all」で mcw=1 が全 fold 選択されたため）
- ridge: log1p+標準化、alpha {1,10,100,1000}

出力: results/exp4_dl/oof_positional.parquet, metrics_positional.csv
実行: venv/bin/python analysis/20_positional_baselines.py [--quick]
"""
import os, sys, json, warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
import xgboost as xgb
warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; T = f"{ROOT}/results/tensors"; F = f"{ROOT}/results/features"
OUT = f"{ROOT}/results/exp4_dl"; os.makedirs(OUT, exist_ok=True)

genes = pd.read_csv(f"{T}/genes.csv", index_col=0)
P = np.load(f"{T}/prom.npy").reshape(len(genes), -1)
R = np.load(f"{T}/rna.npy").reshape(len(genes), -1)
omics = pd.concat([pd.read_parquet(f"{F}/{n}.parquet") for n in ["cnv_gene_level", "mutation_gene_level", "methylation_gene_level"]], axis=1)
O = omics.reindex(genes.index).fillna(0.0).values.astype(np.float32)
X = np.hstack([P, R, genes.n_bins_rna.values[:, None].astype(np.float32), O])
if "--with-agg" in sys.argv:   # ③': 集約 636 列も併給（ハイブリッド DL との対称 arm）
    bind = pd.read_parquet(f"{F}/binding_gene_level.parquet")
    A = bind.reindex(genes.index).drop(columns=["n_bins_prom", "n_bins_rna"]).values.astype(np.float32)
    X = np.hstack([X, A])
y = genes.log2FC.values.astype(np.float32)
fold = genes.fold.values
print("X:", X.shape, "| y:", y.shape, flush=True)
del P, R

GRIDS = {
    "xgboost": [dict(max_depth=d, min_child_weight=1) for d in ([5] if QUICK else [3, 5, 7])],
    "ridge": [dict(alpha=a) for a in ([100.0] if QUICK else [1.0, 10.0, 100.0, 1000.0])],
}
def make(name, **p):
    if name == "xgboost":
        return xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
                                tree_method="hist", n_jobs=10, random_state=0, **p)
    return Ridge(**p)

def prep_ridge(Xtr, Xte):
    mu = np.log1p(Xtr).mean(axis=0); sd = np.log1p(Xtr).std(axis=0) + 1e-8
    return (np.log1p(Xtr) - mu) / sd, (np.log1p(Xte) - mu) / sd

oof = genes[["log2FC", "label", "chrom", "fold"]].copy(); records = []
for name, grid in GRIDS.items():
    col = f"{'posagg' if '--with-agg' in sys.argv else 'positional'}|{name}"; pred = np.full(len(genes), np.nan); chosen = {}
    for k in range(5):
        tr = fold != k; te = ~tr
        ftr = fold[tr]; best, best_s = None, -np.inf
        for params in grid:
            scores = []
            for hold in np.unique(ftr):
                m = ftr != hold; v = ftr == hold
                if name == "ridge":
                    Xa, Xv = prep_ridge(X[tr][m], X[tr][v])
                else:
                    Xa, Xv = X[tr][m], X[tr][v]
                mdl = make(name, **params).fit(Xa, y[tr][m])
                scores.append(spearmanr(y[tr][v], mdl.predict(Xv))[0])
            s = np.nanmean(scores)
            print(f"  {col} fold {k} {params} inner {s:.3f}", flush=True)
            if s > best_s: best, best_s = params, s
        if name == "ridge":
            Xa, Xt = prep_ridge(X[tr], X[te])
        else:
            Xa, Xt = X[tr], X[te]
        mdl = make(name, **best).fit(Xa, y[tr]); pred[te] = mdl.predict(Xt)
        chosen[k] = (best, round(float(best_s), 3))
    oof[col] = pred
    r = dict(model=col, rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
             r2=float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
             spearman=float(spearmanr(y, pred)[0]),
             prauc_up=float(average_precision_score(genes.label == "UP", pred)),
             prauc_down=float(average_precision_score(genes.label == "DOWN", -pred)),
             params=json.dumps({k: v[0] for k, v in chosen.items()}))
    records.append(r)
    print(f"{col:22s} spearman {r['spearman']:.3f} rmse {r['rmse']:.3f} PR-AUC up {r['prauc_up']:.3f} down {r['prauc_down']:.3f}", flush=True)
sfx = "_agg" if "--with-agg" in sys.argv else ""
oof.to_parquet(f"{OUT}/oof_positional{sfx}.parquet"); pd.DataFrame(records).to_csv(f"{OUT}/metrics_positional{sfx}.csv", index=False)
print("saved")

#!/usr/bin/env python3
"""
16_loco_sensitivity.py — leave-one-chromosome-out（LOCO）感度解析（codex R5-1 防御③）

5 fold 染色体ブロック CV の「推論単位が 22-23 クラスタ」という批判に対し、
染色体 1 個抜き（23 fold）でも主結果（all|xgboost の Spearman）が保存されるかを確認。
ハイパラは 06 の最頻値（max_depth=5, min_child_weight=1）に固定（チューニング分散を除いた感度解析）。

出力: results/exp1_nested_cv/loco_sensitivity.csv（染色体ごとの n・Spearman）＋ pooled OOF Spearman
実行: venv/bin/python analysis/16_loco_sensitivity.py
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; F = f"{ROOT}/results/features"
OUT = f"{ROOT}/results/exp1_nested_cv"

de = pd.read_csv(f"{ROOT}/results/de_lihc/de_results.csv").set_index("gene_id")
de = de[de.gene_type.isin(["protein_coding", "lncRNA"]) & de.log2FC.notna()]
ann = pd.read_parquet(f"{F}/gene_annotation_gencode26.parquet").set_index("gene_id")
bind = pd.read_parquet(f"{F}/binding_gene_level.parquet")
omics = pd.concat([pd.read_parquet(f"{F}/{n}.parquet") for n in ["cnv_gene_level", "mutation_gene_level", "methylation_gene_level"]], axis=1)
genes = de.index.intersection(bind.index).intersection(ann.index)
df = de.loc[genes, ["log2FC", "label"]].join(ann.loc[genes, ["chrom"]])
df = df[df.chrom.isin([f"chr{i}" for i in range(1, 23)] + ["chrX"])]
X = pd.concat([bind.loc[df.index].drop(columns=["n_bins_prom", "n_bins_rna"]), omics.reindex(df.index).fillna(0.0)], axis=1)
y = df.log2FC.values
print("genes:", len(df))

RAW_COLS = ("methy_", "log_length")
def prep(Xtr, Xte):
    keep = [c for c in Xtr.columns if (Xtr[c] != 0).sum() >= 30]
    Xtr, Xte = Xtr[keep], Xte[keep]
    lg = [c for c in keep if not c.startswith(RAW_COLS)]
    def tf(Z):
        Z = Z.copy(); Z[lg] = np.log1p(Z[lg].clip(lower=0)); return Z
    sc = StandardScaler().fit(tf(Xtr))
    return sc.transform(tf(Xtr)), sc.transform(tf(Xte))

pred = np.full(len(df), np.nan); rows = []
for c in sorted(df.chrom.unique(), key=lambda s: (len(s), s)):
    te = (df.chrom == c).values; tr = ~te
    Xa, Xt = prep(X[tr], X[te])
    mdl = xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
                           max_depth=5, min_child_weight=1, n_jobs=8, random_state=0).fit(Xa, y[tr])
    pred[te] = mdl.predict(Xt)
    rho = float(spearmanr(y[te], pred[te])[0])
    rows.append(dict(chrom=c, n=int(te.sum()), spearman=rho))
    print(f"{c:6s} n {te.sum():5d} spearman {rho:.3f}", flush=True)
res = pd.DataFrame(rows)
pooled = float(spearmanr(y, pred)[0])
res.to_csv(f"{OUT}/loco_sensitivity.csv", index=False)
print(f"\npooled LOCO spearman {pooled:.3f} | per-chrom median {res.spearman.median():.3f} "
      f"min {res.spearman.min():.3f} ({res.loc[res.spearman.idxmin(), 'chrom']}) max {res.spearman.max():.3f}")

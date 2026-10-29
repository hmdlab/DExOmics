#!/usr/bin/env python3
"""
17_importance_stability.py — 特徴重要度の fold 間安定性（解釈セクションの防御、codex R5-1）

06 と同じ 5 fold で all featset の XGBoost（最頻ハイパラ固定）を外側訓練集合ごとに再学習し、
gain importance の fold 間 Spearman 相関と top-20 の共通率を出す。旧稿の attribution 批判
（因果的語法・不安定性）への防御として「何位までが安定か」を定量化する。

出力: results/exp1_nested_cv/importance_by_fold.parquet, importance_stability.json
実行: venv/bin/python analysis/17_importance_stability.py
"""
import os, json, itertools, numpy as np, pandas as pd
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
df = de.loc[genes, ["log2FC"]].join(ann.loc[genes, ["chrom"]])
df = df[df.chrom.isin([f"chr{i}" for i in range(1, 23)] + ["chrX"])]
X = pd.concat([bind.loc[df.index].drop(columns=["n_bins_prom", "n_bins_rna"]), omics.reindex(df.index).fillna(0.0)], axis=1)
y = df.log2FC.values
folds = pd.read_csv(f"{OUT}/folds.csv").set_index("chrom")["fold"]
df["fold"] = df.chrom.map(folds)

RAW_COLS = ("methy_", "log_length")
imp = {}
for k in range(5):
    tr = (df.fold != k).values
    Xtr = X[tr]
    keep = [c for c in Xtr.columns if (Xtr[c] != 0).sum() >= 30]
    Xtr = Xtr[keep]
    lg = [c for c in keep if not c.startswith(RAW_COLS)]
    Z = Xtr.copy(); Z[lg] = np.log1p(Z[lg].clip(lower=0))
    Z = pd.DataFrame(StandardScaler().fit_transform(Z), columns=keep)
    mdl = xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
                           max_depth=5, min_child_weight=1, n_jobs=8, random_state=0).fit(Z, y[tr])
    imp[k] = pd.Series(mdl.feature_importances_, index=keep)
    print(f"fold {k} top5:", imp[k].nlargest(5).round(4).to_dict(), flush=True)

M = pd.DataFrame(imp).fillna(0.0)
M.to_parquet(f"{OUT}/importance_by_fold.parquet")
pairs = list(itertools.combinations(range(5), 2))
rhos = [float(spearmanr(M[a], M[b])[0]) for a, b in pairs]
top20 = {k: set(M[k].nlargest(20).index) for k in range(5)}
overlaps = [len(top20[a] & top20[b]) / 20 for a, b in pairs]
core20 = set.intersection(*top20.values())
summary = {
    "spearman_between_folds": {"mean": round(float(np.mean(rhos)), 3), "min": round(min(rhos), 3), "max": round(max(rhos), 3)},
    "top20_overlap": {"mean": round(float(np.mean(overlaps)), 3), "min": round(min(overlaps), 3)},
    "top20_in_all_5_folds": sorted(core20),
    "mean_importance_top15": M.mean(axis=1).nlargest(15).round(4).to_dict(),
}
json.dump(summary, open(f"{OUT}/importance_stability.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))

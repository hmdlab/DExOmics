#!/usr/bin/env python3
"""
06b_mlp.py — 06 と同一 fold・同一前処理での MLP（簡易 DL 系）比較

目的: 「強い非 DL ベースラインとの公平比較」に DL 側代表を足す。06 の染色体ブロック fold
（results/exp1_nested_cv/folds.csv）を読み込んで完全に同じ分割・prep を用いる。
featset は binding / all の 2 つ。inner CV も 06 と同じ 4 fold（外側訓練群）。

出力: results/exp1_nested_cv/oof_mlp.parquet, metrics_mlp.csv
実行: venv/bin/python analysis/06b_mlp.py [--quick]
"""
import os, sys, json, warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; F = f"{ROOT}/results/features"
OUT = f"{ROOT}/results/exp1_nested_cv"

# ---------- assemble（06 と同一） ----------
de = pd.read_csv(f"{ROOT}/results/de_lihc/de_results.csv").set_index("gene_id")
de = de[de.gene_type.isin(["protein_coding", "lncRNA"]) & de.log2FC.notna()]
ann = pd.read_parquet(f"{F}/gene_annotation_gencode26.parquet").set_index("gene_id")
bind = pd.read_parquet(f"{F}/binding_gene_level.parquet")
omics = [pd.read_parquet(f"{F}/{n}.parquet") for n in ["cnv_gene_level", "mutation_gene_level", "methylation_gene_level"]]
X_omics = pd.concat(omics, axis=1)
genes = de.index.intersection(bind.index).intersection(ann.index)
df = de.loc[genes, ["log2FC", "label", "gene_type"]].join(ann.loc[genes, ["chrom", "length"]])
df = df[df.chrom.isin([f"chr{i}" for i in range(1, 23)] + ["chrX"])]
Xb = bind.loc[df.index].drop(columns=["n_bins_prom", "n_bins_rna"])
Xo = X_omics.reindex(df.index).fillna(0.0)
assign = pd.read_csv(f"{OUT}/folds.csv").set_index("chrom")["fold"]   # 06 の fold を再利用
df["fold"] = df.chrom.map(assign)
assert df.fold.notna().all()
FEATSETS = {"binding": Xb, "all": pd.concat([Xb, Xo], axis=1)}
y = df.log2FC.values
print("genes:", len(df), "| folds:", df.fold.value_counts().sort_index().to_dict())

RAW_COLS = ("methy_", "log_length")
def prep(Xtr, Xte):
    keep = [c for c in Xtr.columns if (Xtr[c] != 0).sum() >= 30]
    Xtr, Xte = Xtr[keep], Xte[keep]
    lg = [c for c in keep if not c.startswith(RAW_COLS)]
    def tf(Z):
        Z = Z.copy(); Z[lg] = np.log1p(Z[lg].clip(lower=0)); return Z
    sc = StandardScaler().fit(tf(Xtr))
    return sc.transform(tf(Xtr)), sc.transform(tf(Xte))

GRID = ([dict(hidden_layer_sizes=(256, 64), alpha=1e-3)] if QUICK else
        [dict(hidden_layer_sizes=h, alpha=a) for h in [(256, 64), (512, 128)] for a in [1e-3, 1e-4]])
def make(**p):
    return MLPRegressor(max_iter=200, early_stopping=True, n_iter_no_change=10,
                        learning_rate_init=1e-3, batch_size=256, random_state=0, **p)

oof = df[["log2FC", "label", "chrom", "fold"]].copy(); records = []
for fs_name, X in FEATSETS.items():
    col = f"{fs_name}|mlp"; pred = np.full(len(df), np.nan); chosen = {}
    for k in range(5):
        tr = (df.fold != k).values; te = ~tr
        ftr = df.fold.values[tr]; best, best_s = None, -np.inf
        for params in GRID:
            scores = []
            for hold in pd.Series(ftr).unique():
                m = ftr != hold; v = ftr == hold
                Xa, Xv = prep(X[tr][m], X[tr][v])
                mdl = make(**params).fit(Xa, y[tr][m])
                scores.append(spearmanr(y[tr][v], mdl.predict(Xv))[0])
            s = np.nanmean(scores)
            if s > best_s: best, best_s = params, s
        Xa, Xt = prep(X[tr], X[te])
        mdl = make(**best).fit(Xa, y[tr]); pred[te] = mdl.predict(Xt)
        chosen[k] = (best, round(float(best_s), 3)); print(f"{col} fold {k} best {best} inner {best_s:.3f}", flush=True)
    oof[col] = pred
    r = dict(featset=fs_name, model="mlp", rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
             r2=float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
             spearman=float(spearmanr(y, pred)[0]),
             prauc_up=float(average_precision_score(df.label == "UP", pred)),
             prauc_down=float(average_precision_score(df.label == "DOWN", -pred)),
             params=json.dumps({k: v[0] for k, v in chosen.items()}, default=str))
    records.append(r)
    print(f"{col:22s} rmse {r['rmse']:.3f} r2 {r['r2']:.3f} spearman {r['spearman']:.3f} PR-AUC up {r['prauc_up']:.3f} down {r['prauc_down']:.3f}", flush=True)
oof.to_parquet(f"{OUT}/oof_mlp.parquet"); pd.DataFrame(records).to_csv(f"{OUT}/metrics_mlp.csv", index=False)
print("saved oof_mlp.parquet / metrics_mlp.csv")

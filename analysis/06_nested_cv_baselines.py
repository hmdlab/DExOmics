#!/usr/bin/env python3
"""
06_nested_cv_baselines.py — TCGA 単独 log2FC を目的変数にした染色体ブロック nested CV（実験 1 本体）

codex Round 2 の設計:
  - 目的変数: TCGA-LIHC の log2FC（連続、pydeseq2 MLE）。副次的に UP/DOWN ラベルの PR-AUC を予測値から評価。
  - 外側 5 fold × 内側 4 fold。fold は染色体を分断せず、遺伝子数が均衡するよう染色体をグループ化して固定。
  - 前処理（標準化・因子フィルタ ≥30 遺伝子）は外側訓練集合内で fit。
  - 比較群: mean / 長さ+結合総数のみ / ElasticNet / XGBoost。特徴セット: binding / omics / all（ablation）。
  - 指標: 外側 OOF の RMSE・R²・Spearman、UP/DOWN の PR-AUC（予測 log2FC を score に）。
    染色体単位の paired bootstrap で 95% 区間（固定 OOF 条件付き）。

入力: results/de_lihc/de_results.csv、results/features/*.parquet
出力: results/exp1_nested_cv/ (oof_predictions.parquet, metrics.csv, metrics_bootstrap.csv, folds.csv)
実行: venv/bin/python analysis/06_nested_cv_baselines.py [--quick]
"""
import os, sys, json, itertools, warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import xgboost as xgb
warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; F = f"{ROOT}/results/features"
CANCER = os.environ.get("CANCER", "LIHC")   # CANCER=CESC で CESC 版（HeLa 結合 + *_cesc omics、探索的）
SFX_BIND = "" if CANCER == "LIHC" else "_hela"
SFX_OM = "" if CANCER == "LIHC" else f"_{CANCER.lower()}"
OUT = f"{ROOT}/results/exp1_nested_cv" + ("" if CANCER == "LIHC" else f"_{CANCER.lower()}")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

# ---------- assemble ----------
de = pd.read_csv(f"{ROOT}/results/de_{CANCER.lower()}/de_results.csv").set_index("gene_id")
de = de[de.gene_type.isin(["protein_coding", "lncRNA"]) & de.log2FC.notna()]
ann = pd.read_parquet(f"{F}/gene_annotation_gencode26.parquet").set_index("gene_id")
bind = pd.read_parquet(f"{F}/binding_gene_level{SFX_BIND}.parquet")
omics = []; used_modalities = []
MODALITIES = [f"cnv_gene_level{SFX_OM}", f"mutation_gene_level{SFX_OM}", f"methylation_gene_level{SFX_OM}"]
for name in MODALITIES:
    p = f"{F}/{name}.parquet"
    if os.path.exists(p): omics.append(pd.read_parquet(p)); used_modalities.append(name)
missing_mod = [n for n in MODALITIES if n not in used_modalities]
if missing_mod and "--allow-partial" not in sys.argv:
    sys.exit(f"missing omics modalities: {missing_mod} (use --allow-partial to run anyway)")
X_omics = pd.concat(omics, axis=1) if omics else pd.DataFrame(index=bind.index)
genes = de.index.intersection(bind.index).intersection(ann.index)
df = de.loc[genes, ["log2FC", "label", "gene_type"]].join(ann.loc[genes, ["chrom", "length"]])
df = df[df.chrom.isin([f"chr{i}" for i in range(1, 23)] + ["chrX"])]
Xb = bind.loc[df.index].drop(columns=["n_bins_prom", "n_bins_rna"])
Xo = X_omics.reindex(df.index).fillna(0.0)          # 変異/CNV/メチル化が無い遺伝子は 0（原著と同じ扱い）
ctrl = pd.DataFrame({"log_length": np.log10(df.length), "prom_total": Xb.filter(like="prom__").drop(columns="prom__mask").sum(axis=1),
                     "rna_total": Xb.filter(like="rna__").drop(columns="rna__exon").sum(axis=1), "n_bins_rna": bind.loc[df.index, "n_bins_rna"]}, index=df.index)
print("genes:", len(df), "| binding feats", Xb.shape[1], "| omics feats", Xo.shape[1]); print(df.label.value_counts().to_dict())

# ---------- chromosome-block folds (5 groups balanced by gene count) ----------
chr_counts = df.chrom.value_counts().sort_values(ascending=False)
groups = {c: None for c in chr_counts.index}; loads = [0] * 5; assign = {}
for c, n in chr_counts.items():                       # greedy LPT
    k = int(np.argmin(loads)); assign[c] = k; loads[k] += n
df["fold"] = df.chrom.map(assign)
pd.Series(assign, name="fold").rename_axis("chrom").to_csv(f"{OUT}/folds.csv")
print("fold sizes:", df.fold.value_counts().sort_index().to_dict())

FEATSETS = {"binding": Xb, "omics": Xo, "all": pd.concat([Xb, Xo], axis=1), "control": ctrl}
y = df.log2FC.values

RAW_COLS = ("methy_", "log_length")   # 符号付き Δβ と log 済み列は log1p しない。列種で固定し featset 間で一貫させる

def prep(Xtr, Xte):
    keep = [c for c in Xtr.columns if (Xtr[c] != 0).sum() >= 30 or c in ctrl.columns]   # ≥30 遺伝子で非ゼロ
    Xtr, Xte = Xtr[keep], Xte[keep]
    lg = [c for c in keep if not c.startswith(RAW_COLS)]
    def tf(Z):
        Z = Z.copy()
        Z[lg] = np.log1p(Z[lg].clip(lower=0))
        return Z
    sc = StandardScaler().fit(tf(Xtr))
    return sc.transform(tf(Xtr)), sc.transform(tf(Xte))

def inner_cv(model_fn, grid, Xtr, ytr, ftr):
    """4-fold inner CV over chromosome groups inside the outer-train set; returns best params."""
    inner_chr = pd.Series(ftr).unique(); best, best_s = None, -np.inf
    for params in grid:
        scores = []
        for hold in inner_chr:
            m = ftr != hold; v = ftr == hold
            Xa, Xv = prep(Xtr[m], Xtr[v])
            mdl = model_fn(**params).fit(Xa, ytr[m]); scores.append(spearmanr(ytr[v], mdl.predict(Xv))[0])
        s = np.nanmean(scores)
        if s > best_s: best, best_s = params, s
    return best, best_s

MODELS = {
    "ridge": (lambda **p: Ridge(**p), [dict(alpha=a) for a in ([1.0] if QUICK else [0.1, 1, 10, 100, 1000])]),
    "elasticnet": (lambda **p: ElasticNet(max_iter=5000, **p), [dict(alpha=a, l1_ratio=l) for a in ([0.01] if QUICK else [0.001, 0.01, 0.1]) for l in ([0.5] if QUICK else [0.2, 0.5, 0.9])]),
    "xgboost": (lambda **p: xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, n_jobs=8, random_state=0, **p),
                [dict(max_depth=d, min_child_weight=w) for d in ([4] if QUICK else [3, 5, 7]) for w in ([5] if QUICK else [1, 5, 20])]),
}
oof = df[["log2FC", "label", "chrom", "fold"]].copy(); records = []
for fs_name, X in FEATSETS.items():
    for m_name, (fn, grid) in MODELS.items():
        if fs_name == "control" and m_name != "ridge": continue
        col = f"{fs_name}|{m_name}"; pred = np.full(len(df), np.nan); chosen = {}
        for k in range(5):
            tr = (df.fold != k).values; te = ~tr
            # inner folds: the 4 remaining outer groups
            best, s = inner_cv(fn, grid, X[tr], y[tr], df.fold.values[tr])
            Xa, Xt = prep(X[tr], X[te]); mdl = fn(**best).fit(Xa, y[tr]); pred[te] = mdl.predict(Xt); chosen[k] = (best, round(float(s), 3))
        oof[col] = pred
        r = dict(featset=fs_name, model=m_name, rmse=float(np.sqrt(np.mean((y - pred) ** 2))), r2=float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
                 spearman=float(spearmanr(y, pred)[0]), prauc_up=float(average_precision_score(df.label == "UP", pred)), prauc_down=float(average_precision_score(df.label == "DOWN", -pred)),
                 params=json.dumps({k: v[0] for k, v in chosen.items()}))
        records.append(r); print(f"{col:22s} rmse {r['rmse']:.3f} r2 {r['r2']:.3f} spearman {r['spearman']:.3f} PR-AUC up {r['prauc_up']:.3f} down {r['prauc_down']:.3f}", flush=True)
oof["mean|baseline"] = np.nan
for k in range(5): oof.loc[df.fold == k, "mean|baseline"] = y[(df.fold != k).values].mean()
pb = oof["mean|baseline"].values   # 指標は他モデルと同じく OOF 列から計算（bootstrap との整合）
records.append(dict(featset="none", model="mean", rmse=float(np.sqrt(np.mean((y - pb) ** 2))), r2=float(1 - np.sum((y - pb) ** 2) / np.sum((y - y.mean()) ** 2)),
                    spearman=float(spearmanr(y, pb)[0]), prauc_up=float(average_precision_score(df.label == "UP", pb)), prauc_down=float(average_precision_score(df.label == "DOWN", -pb)), params=""))
oof.to_parquet(f"{OUT}/oof_predictions.parquet"); pd.DataFrame(records).to_csv(f"{OUT}/metrics.csv", index=False)
json.dump({"used_modalities": used_modalities, "featset_ncols": {k: int(v.shape[1]) for k, v in FEATSETS.items()}, "n_genes": int(len(df))},
          open(f"{OUT}/run_meta.json", "w"), indent=1)

# ---------- paired bootstrap over chromosomes ----------
chroms = df.chrom.unique(); cols = [c for c in oof.columns if "|" in c]; boot = []
for b in range(200 if QUICK else 1000):
    samp = rng.choice(chroms, size=len(chroms), replace=True)
    idx = np.concatenate([np.where(df.chrom.values == c)[0] for c in samp])
    yy = y[idx]; row = {}
    for c in cols:
        p = oof[c].values[idx]; row[c] = spearmanr(yy, p)[0] if np.nanstd(p) > 0 else 0.0
    boot.append(row)
boot = pd.DataFrame(boot); ci = boot.quantile([0.025, 0.5, 0.975]).T; ci.columns = ["lo", "med", "hi"]
diffs = {f"{a} - {b}": (boot[a] - boot[b]).quantile([0.025, 0.5, 0.975]).tolist() for a, b in [("all|xgboost", "binding|xgboost"), ("all|xgboost", "all|ridge"), ("all|xgboost", "control|ridge"), ("binding|xgboost", "omics|xgboost"), ("all|xgboost", "omics|xgboost")] if a in boot and b in boot}
ci.to_csv(f"{OUT}/metrics_bootstrap_spearman.csv"); json.dump(diffs, open(f"{OUT}/bootstrap_differences.json", "w"), indent=1)
print("\nSpearman 95% CI (chromosome bootstrap):\n", ci.round(3)); print("\nDifferences:", json.dumps({k: [round(x, 3) for x in v] for k, v in diffs.items()}, indent=1))

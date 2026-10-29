#!/usr/bin/env python3
"""
09_liri_external_validation.py — 実験 2 後半: 06 の OOF 予測 log2FC を LIRI-JP 実測 log2FC と照合

設計（codex Round 3 の成功パターン判定に対応）:
- 各モデルの OOF 予測（未見染色体上の予測値）と LIRI-JP 実測 log2FC の遺伝子間 Spearman・符号一致率。
- 参照上限: 同じ遺伝子集合での TCGA 実測 vs LIRI 実測の Spearman（(ii)。予測はこの一部を保持できれば成功）。
- 染色体単位 paired bootstrap で 95% CI（固定 OOF 条件付き）。「保持率」= rho(pred,LIRI)/rho(TCGA,LIRI) も出す。
- 符号一致は LIRI の DE 遺伝子（|log2FC|>1 & padj<0.05）上でも算出（ノイズ遺伝子で薄まるのを避ける）。

入力: results/exp1_nested_cv/oof_predictions.parquet、results/de_liri/de_results.csv
出力: results/exp2_liri/metrics.csv, bootstrap_ci.csv
実行: venv/bin/python analysis/09_liri_external_validation.py
"""
import os, json, numpy as np, pandas as pd
from scipy.stats import spearmanr

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT
OUT = f"{ROOT}/results/exp2_liri"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

oof = pd.read_parquet(f"{ROOT}/results/exp1_nested_cv/oof_predictions.parquet")
mlp_p = f"{ROOT}/results/exp1_nested_cv/oof_mlp.parquet"
if os.path.exists(mlp_p):
    mlp = pd.read_parquet(mlp_p)
    oof = oof.join(mlp[[c for c in mlp.columns if c.endswith("|mlp")]], how="left")
liri = pd.read_csv(f"{ROOT}/results/de_liri/de_results.csv").dropna(subset=["gene_id"]).set_index("gene_id")
m = oof.join(liri[["log2FC", "label"]].rename(columns={"log2FC": "liri_logfc", "label": "liri_label"}), how="inner").dropna(subset=["liri_logfc"])
print("shared genes:", len(m), "| liri labels:", m.liri_label.value_counts().to_dict())

PRIMARY = "all|xgboost"   # 事前指定の主モデル（06 の主解析）。他モデルの並びは exploratory。
model_cols = [c for c in oof.columns if "|" in c]
ref_rho = float(spearmanr(m.log2FC, m.liri_logfc)[0])   # (ii) TCGA 実測 vs LIRI 実測
de = m[m.liri_label != "NonDEG"]
rows = []
for c in model_cols + ["log2FC"]:
    name = "TCGA_actual" if c == "log2FC" else c
    p = m[c].values
    rho = float(spearmanr(p, m.liri_logfc)[0])
    rows.append(dict(model=name, spearman_liri=rho,
                     normalized_rho=(rho / ref_rho if abs(ref_rho) >= 0.1 else np.nan),
                     sign_all=float((np.sign(p) == np.sign(m.liri_logfc)).mean()),
                     sign_liri_de=float((np.sign(m.loc[de.index, c]) == np.sign(de.liri_logfc)).mean())))
res = pd.DataFrame(rows).sort_values("spearman_liri", ascending=False)
res.to_csv(f"{OUT}/metrics.csv", index=False)
print("(model ranking is exploratory; primary model is pre-specified:", PRIMARY, ")")
print(res.round(3).to_string(index=False))

# bootstrap: 比（normalized rho）も各 resample 内で計算（相関した比の CI）
chroms = m.chrom.unique(); boot = []
for b in range(1000):
    samp = rng.choice(chroms, size=len(chroms), replace=True)
    idx = np.concatenate([np.where(m.chrom.values == c)[0] for c in samp])
    yy = m.liri_logfc.values[idx]
    rr = spearmanr(m.log2FC.values[idx], yy)[0]
    row = {}
    for c in model_cols + ["log2FC"]:
        rho_b = spearmanr(m[c].values[idx], yy)[0] if np.nanstd(m[c].values[idx]) > 0 else 0.0
        row[c] = rho_b
        if c != "log2FC":
            row[f"ratio:{c}"] = rho_b / rr if abs(rr) >= 0.1 else np.nan
    boot.append(row)
boot = pd.DataFrame(boot)
ci = boot.quantile([0.025, 0.5, 0.975]).T; ci.columns = ["lo", "med", "hi"]
ci.to_csv(f"{OUT}/bootstrap_ci.csv")
print("\n95% CI (chromosome bootstrap; ratio:* = normalized rho):\n", ci.round(3))

# paired 差（同一 resample 内、R5-1 の防御: 統合の追加価値は外部でも残るか）
PAIRS = [("all|xgboost", "omics|xgboost"), ("all|xgboost", "binding|xgboost"),
         ("all|xgboost", "control|ridge"), ("binding|xgboost", "omics|xgboost")]
diffs = {f"{a} - {b}": (boot[a] - boot[b]).quantile([0.025, 0.5, 0.975]).tolist()
         for a, b in PAIRS if a in boot and b in boot}
json.dump(diffs, open(f"{OUT}/bootstrap_differences.json", "w"), indent=1)
print("\npaired differences on LIRI (lo/med/hi):")
for k, v in diffs.items(): print(f"  {k}: [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]")
pr = res[res.model == PRIMARY].iloc[0]
print(f"\nprimary model {PRIMARY}: rho={pr.spearman_liri:.3f} normalized_rho={pr.normalized_rho:.2f} ref(TCGA vs LIRI)={ref_rho:.3f}")

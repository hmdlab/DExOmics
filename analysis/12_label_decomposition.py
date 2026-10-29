#!/usr/bin/env python3
"""
12_label_decomposition.py — downDEG 過少の逐次分解: 方法差（DESeq2 vs voom 近似）と選択効果（HepG2 一致フィルタ）

codex Round 4-1 の設計。ランナーとして 11 の入力書き出し → Rscript 11 → 分解集計まで一括実行。

手順:
1. 01 の counts_matrix.parquet から voom 用 TSV（counts / coldata / gene_length）を書き出し
2. Rscript analysis/11_voom_labels.R（voom 近似ラベル）
3. 分解集計:
   - 方法差: 共通遺伝子上で DESeq2 ラベル (01) vs voom 近似ラベル (11) の crosstab
   - 選択効果: voom-DOWN（= 原著 TCGA 側パイプラインの近似）の各遺伝子が
     配布データ（HepG2 一致フィルタ後）で DOWN として残った / NonDEG 化 / データセットから消えた の内訳
   - 注意: voom は cor.cut 前処理と GC 正規化を省いた近似（完全再現ではない）
出力: results/de_lihc/label_decomposition.json / decomposition_*.csv
実行: venv/bin/python analysis/12_label_decomposition.py
"""
import os, json, subprocess, sys
import numpy as np, pandas as pd

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT
DE = f"{ROOT}/results/de_lihc"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- 1. voom 入力の書き出し ----
if not os.path.exists(f"{DE}/voom_labels.csv") or "--force" in sys.argv:
    counts = pd.read_parquet(f"{DE}/counts_matrix.parquet")
    man = pd.read_csv(f"{ROOT}/gdc/LIHC/manifest_star_counts.csv")
    man = man[man.sample_type.isin(["Primary Tumor", "Solid Tissue Normal"])].drop_duplicates("sample")
    cond = man.set_index("sample").sample_type.map({"Primary Tumor": "Tumor", "Solid Tissue Normal": "Normal"})
    cond = cond.loc[counts.columns]
    ann = pd.read_parquet(f"{ROOT}/results/features/gene_annotation_gencode26.parquet").set_index("gene_id")
    counts.to_csv(f"{DE}/counts_for_voom.tsv.gz", sep="\t", compression="gzip")
    pd.DataFrame({"sample": counts.columns, "condition": cond.values}).to_csv(f"{DE}/coldata_for_voom.tsv", sep="\t", index=False)
    ann[["length"]].to_csv(f"{DE}/gene_length.tsv", sep="\t")
    # ---- 2. Rscript ----
    r = subprocess.run(["Rscript", f"{SCRIPT_DIR}/11_voom_labels.R"], capture_output=True, text=True)
    print(r.stdout[-2000:]); print(r.stderr[-1500:], file=sys.stderr)
    r.check_returncode()

# ---- 3. 分解集計 ----
deseq = pd.read_csv(f"{DE}/de_results.csv").set_index("gene_id")
voom = pd.read_csv(f"{DE}/voom_labels.csv").set_index("gene_id")
cur = pd.read_csv(f"{ROOT}/data/TCGAprocessed/LIHC/DEG_info.csv").set_index("Gene")

both = deseq[["label"]].join(voom[["label"]], how="inner", lsuffix="_deseq2", rsuffix="_voom")
ct_method = pd.crosstab(both.label_deseq2, both.label_voom)
ct_method.to_csv(f"{DE}/decomposition_method_crosstab.csv")
print("方法差 (rows=DESeq2, cols=voom近似):\n", ct_method, "\n")

vd = voom[voom.label == "DOWN"]
status = pd.Series("dropped_from_dataset", index=vd.index)
inboth = vd.index.intersection(cur.index)
status.loc[inboth] = cur.loc[inboth, "DElabel"].map({"DOWN": "kept_as_DOWN", "NonDEG": "kept_as_NonDEG", "UP": "kept_as_UP"})
flow_down = status.value_counts().to_dict()
print("選択効果 (voom近似-DOWN の配布データでの行方):", flow_down)

vu = voom[voom.label == "UP"]
status_u = pd.Series("dropped_from_dataset", index=vu.index)
inboth_u = vu.index.intersection(cur.index)
status_u.loc[inboth_u] = cur.loc[inboth_u, "DElabel"].map({"DOWN": "kept_as_DOWN", "NonDEG": "kept_as_NonDEG", "UP": "kept_as_UP"})
flow_up = status_u.value_counts().to_dict()

summary = {
    "voom_labels": voom.label.value_counts().to_dict(),
    "deseq2_labels": deseq.label.value_counts().to_dict(),
    "method_crosstab_note": "voom は cor.cut/GC 正規化を省略した近似。共通遺伝子 n=%d" % len(both),
    "flow_voom_down_in_distributed": flow_down,
    "flow_voom_up_in_distributed": flow_up,
    "n_current_dataset_genes": int(len(cur)),
}
json.dump(summary, open(f"{DE}/label_decomposition.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))

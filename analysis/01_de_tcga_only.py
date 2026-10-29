#!/usr/bin/env python3
"""
01_de_tcga_only.py — TCGA-LIHC 単独の腫瘍 vs 正常 DE 解析（pydeseq2）

目的: 現行ラベル（TCGA と HepG2 の DE 判定が一致する遺伝子だけを残す）を捨て、
TCGA だけから連続の log2FC と DE ラベルを作る。改善計画の最初の実験。

入力: ~/Data/DExOmics/gdc/LIHC/star_counts/<file_id>__<name>.tsv (GDC STAR - Counts, open access)
      ~/Data/DExOmics/gdc/LIHC/manifest_star_counts.csv
出力: ~/Data/DExOmics/results/de_lihc/
      counts_matrix.parquet   遺伝子 × 検体の unstranded カウント
      de_results.csv          gene_id, gene_name, gene_type, baseMean, log2FC, lfcSE, pvalue, padj, label
      summary.json            件数など

設計:
- 検体: Primary Tumor (01) と Solid Tissue Normal (11) のみ。Recurrent は除外。
  同一検体に複数ファイルがある場合は 1 つ目を採用（GDC の重複 aliquot）。
- 遺伝子: ENSG のみ、PAR_Y を除く。バージョン番号を落とす。
- フィルタ: 全検体の 25% 以上で count >= 10（edgeR の filterByExpr に近い緩い基準）。
  ※ 原著は TCGAbiolinks の gcContent 正規化 + 25% 分位フィルタ + limma-voom を TPM に適用していた。
    ここでは生カウントに DESeq2 系（pydeseq2）を使う。方法差は結果ファイルに明記。
- ラベル: padj < 0.05 かつ |log2FC| > 1 で UP / DOWN、それ以外 NonDEG（原著の閾値と同じ）。
- lfcShrink は使わない（連続目的変数には MLE の log2FC を使い、shrink 版は別列で保存可）。

実行: ~/Data/DExOmics/venv/bin/python analysis/01_de_tcga_only.py
"""
import os, sys, json, glob, gzip
import numpy as np, pandas as pd

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT
CANCER = os.environ.get("CANCER", "LIHC")   # 環境変数で癌種を切替（例: CANCER=CESC。CESC は正常 3 例のため探索的）
SRC = f"{ROOT}/gdc/{CANCER}/star_counts"
MAN = f"{ROOT}/gdc/{CANCER}/manifest_star_counts.csv"
OUT = f"{ROOT}/results/de_{CANCER.lower()}"
os.makedirs(OUT, exist_ok=True)

man = pd.read_csv(MAN)
man = man[man.sample_type.isin(["Primary Tumor", "Solid Tissue Normal"])].copy()
man["sample_code"] = man["sample"].str[13:15]
man = man.drop_duplicates("sample", keep="first")
print("samples:", man.sample_type.value_counts().to_dict())

def read_counts(fid, fname):
    p = f"{SRC}/{fid}__{fname}"
    df = pd.read_csv(p, sep="\t", comment="#")
    df = df[df.gene_id.str.startswith("ENSG") & ~df.gene_id.str.contains("_PAR_Y")]
    df["gid"] = df.gene_id.str.split(".").str[0]
    return df.set_index("gid")[["gene_name", "gene_type", "unstranded"]]

mats, meta = [], None
for r in man.itertuples():
    d = read_counts(r.file_id, r.file_name)
    if meta is None:
        meta = d[["gene_name", "gene_type"]]
    mats.append(d["unstranded"].rename(r.sample))
counts = pd.concat(mats, axis=1)
counts = counts.loc[meta.index]
counts.to_parquet(f"{OUT}/counts_matrix.parquet")
print("counts matrix:", counts.shape)

# filter low expression: count>=10 in >=25% samples
keep = (counts >= 10).mean(axis=1) >= 0.25
counts_f = counts[keep]
print("genes after filter:", counts_f.shape[0], "of", counts.shape[0])

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
coldata = man.set_index("sample")[["sample_type"]].loc[counts_f.columns]
coldata["condition"] = np.where(coldata.sample_type == "Primary Tumor", "Tumor", "Normal")
dds = DeseqDataSet(counts=counts_f.T, metadata=coldata[["condition"]], design="~condition", quiet=True, n_cpus=8)
dds.deseq2()
st = DeseqStats(dds, contrast=["condition", "Tumor", "Normal"], quiet=True, n_cpus=8)
st.summary()
res = st.results_df.copy()
res.index.name = "gene_id"
res = res.join(meta, how="left")
res["label"] = "NonDEG"
sig = (res.padj < 0.05) & (res.log2FoldChange.abs() > 1)
res.loc[sig & (res.log2FoldChange > 0), "label"] = "UP"
res.loc[sig & (res.log2FoldChange < 0), "label"] = "DOWN"
res = res.rename(columns={"log2FoldChange": "log2FC"})
res.to_csv(f"{OUT}/de_results.csv")

summary = {
    "n_tumor": int((coldata.condition == "Tumor").sum()),
    "n_normal": int((coldata.condition == "Normal").sum()),
    "n_genes_total": int(counts.shape[0]),
    "n_genes_tested": int(counts_f.shape[0]),
    "labels_all": res.label.value_counts().to_dict(),
    "labels_pc_lnc": res[res.gene_type.isin(["protein_coding", "lncRNA"])].label.value_counts().to_dict(),
    "method": "pydeseq2 Wald test, design ~condition (Tumor vs Normal), filter count>=10 in >=25% samples, padj<0.05 & |log2FC|>1",
}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))

# compare with the distributed (agreement-filtered) labels
cur = pd.read_csv(f"{ROOT}/data/TCGAprocessed/{CANCER}/DEG_info.csv").set_index("Gene")
cmp = res[["label"]].join(cur[["DElabel"]], how="inner")
ct = pd.crosstab(cmp.label, cmp.DElabel)
ct.to_csv(f"{OUT}/crosstab_vs_current_labels.csv")
print("\ncrosstab (rows=TCGA-only new, cols=current agreement labels):\n", ct)

#!/usr/bin/env python3
"""
08_liri_logfc.py — ICGC LIRI-JP の raw_read_count から腫瘍 vs 非腫瘍肝の log2FC を計算（実験 2 の前半）

入力: ~/Data/DExOmics/icgc/LIRI-JP/{exp_seq,specimen}/  (07 で取得。gene_model=RefSeq symbol, GRCh37)
設計:
- 検体: specimen_type が "Primary tumour - solid tissue" (腫瘍) と "Normal - solid tissue" (隣接非腫瘍肝) の
  RNA-seq を持つ sample。血液由来 Normal は発現に無いはずだが除外条件に明記。
- 同一 sample_id に複数行（重複 part）は raw_read_count を最初の値で採用。
- カウント: raw_read_count を四捨五入して整数化（RSEM 系の非整数があり得るため。方法差は summary に明記）。
- フィルタ・検定は TCGA 側 (01) と同一: count>=10 in >=25% samples、pydeseq2 Wald ~condition、
  padj<0.05 & |log2FC|>1 で UP/DOWN。
- シンボル→ENSG は GENCODE v26 の gene_name で対応（重複シンボルは除外）。symbol レベルの結果も保存。
- 最後に TCGA (01) の log2FC との遺伝子間 Spearman・符号一致率を出す（これが実測 vs 実測の参照上限）。

出力: ~/Data/DExOmics/results/de_liri/ (counts_matrix.parquet, de_results.csv, summary.json)
実行: venv/bin/python analysis/08_liri_logfc.py
"""
import os, glob, gzip, json
import numpy as np, pandas as pd

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT
SRC = f"{ROOT}/icgc/LIRI-JP"
OUT = f"{ROOT}/results/de_liri"; os.makedirs(OUT, exist_ok=True)

# ---- specimen: type ごとの specimen id ----
spec_cols = gzip.open(f"{SRC}/headers/specimen.tsv.gz", "rt").readline().rstrip("\n").split("\t")
spec = pd.concat([pd.read_csv(p, sep="\t", header=None, names=spec_cols, low_memory=False)
                  for p in sorted(glob.glob(f"{SRC}/specimen/*.gz"))]).drop_duplicates("icgc_specimen_id")
print("specimen types:", spec.specimen_type.value_counts().to_dict())

# ---- exp_seq: sample × gene の raw_read_count ----
exp_cols = gzip.open(f"{SRC}/headers/exp_seq.tsv.gz", "rt").readline().rstrip("\n").split("\t")
use = ["icgc_donor_id", "icgc_specimen_id", "icgc_sample_id", "gene_id", "raw_read_count"]
parts = []
for p in sorted(glob.glob(f"{SRC}/exp_seq/*.gz")):
    df = pd.read_csv(p, sep="\t", header=None, names=exp_cols, usecols=use, low_memory=False)
    parts.append(df)
exp = pd.concat(parts)
exp = exp.merge(spec[["icgc_specimen_id", "specimen_type"]], on="icgc_specimen_id", how="left")
print("exp_seq rows:", len(exp), "| samples:", exp.icgc_sample_id.nunique(),
      "| specimen types in exp:", exp.drop_duplicates("icgc_sample_id").specimen_type.value_counts().to_dict())

keep_types = {"Primary tumour - solid tissue": "Tumor", "Normal - solid tissue": "Normal"}
exp = exp[exp.specimen_type.isin(keep_types)]
exp["condition"] = exp.specimen_type.map(keep_types)
exp = exp.drop_duplicates(["icgc_sample_id", "gene_id"], keep="first")

counts = exp.pivot_table(index="gene_id", columns="icgc_sample_id", values="raw_read_count", aggfunc="first")
meta = exp.drop_duplicates("icgc_sample_id").set_index("icgc_sample_id")[["icgc_donor_id", "condition"]].loc[counts.columns]
print("counts matrix:", counts.shape, "| conditions:", meta.condition.value_counts().to_dict(),
      "| donors with both:", (meta.groupby("icgc_donor_id").condition.nunique() == 2).sum())
counts = counts.round().astype("Int64").astype(float).fillna(np.nan)
frac_na = counts.isna().mean().mean()
counts = counts.dropna(axis=0, how="any")   # 全 sample で測定された遺伝子のみ
print("genes measured in all samples:", counts.shape[0], f"(mean NA frac {frac_na:.3f})")
counts.to_parquet(f"{OUT}/counts_matrix.parquet")

keep = (counts >= 10).mean(axis=1) >= 0.25
cf_ = counts[keep].astype(int)
print("genes after filter:", cf_.shape[0])

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
dds = DeseqDataSet(counts=cf_.T, metadata=meta[["condition"]], design="~condition", quiet=True, n_cpus=8)
dds.deseq2()
st = DeseqStats(dds, contrast=["condition", "Tumor", "Normal"], quiet=True, n_cpus=8)
st.summary()
res = st.results_df.copy(); res.index.name = "gene_symbol"
res = res.rename(columns={"log2FoldChange": "log2FC"})
res["label"] = "NonDEG"
sig = (res.padj < 0.05) & (res.log2FC.abs() > 1)
res.loc[sig & (res.log2FC > 0), "label"] = "UP"
res.loc[sig & (res.log2FC < 0), "label"] = "DOWN"

# symbol -> ENSG (GENCODE v26, 重複シンボル除外)
ann = pd.read_parquet(f"{ROOT}/results/features/gene_annotation_gencode26.parquet")
sym = ann.drop_duplicates("gene_name", keep=False).set_index("gene_name")["gene_id"]
res["gene_id"] = res.index.map(sym)
res.to_csv(f"{OUT}/de_results.csv")

summary = {
    "n_tumor": int((meta.condition == "Tumor").sum()), "n_normal": int((meta.condition == "Normal").sum()),
    "n_donors_both": int((meta.groupby("icgc_donor_id").condition.nunique() == 2).sum()),
    "n_genes_tested": int(cf_.shape[0]), "labels": res.label.value_counts().to_dict(),
    "n_mapped_ensg": int(res.gene_id.notna().sum()),
    "method": "raw_read_count (RefSeq symbol, GRCh37) rounded to int, filter count>=10 in >=25%, pydeseq2 Wald ~condition Tumor vs Normal, padj<0.05 & |log2FC|>1",
}

# ---- TCGA 実測との比較（実測 vs 実測の参照上限）----
tcga = pd.read_csv(f"{ROOT}/results/de_lihc/de_results.csv").set_index("gene_id")
m = res.dropna(subset=["gene_id"]).set_index("gene_id")[["log2FC", "label"]].join(
    tcga[["log2FC", "label"]], how="inner", lsuffix="_liri", rsuffix="_tcga").dropna()
from scipy.stats import spearmanr
rho = float(spearmanr(m.log2FC_liri, m.log2FC_tcga)[0])
sign_all = float((np.sign(m.log2FC_liri) == np.sign(m.log2FC_tcga)).mean())
de_tcga = m[m.label_tcga != "NonDEG"]
sign_de = float((np.sign(de_tcga.log2FC_liri) == np.sign(de_tcga.log2FC_tcga)).mean())
summary.update({"n_shared_genes": int(len(m)), "spearman_liri_vs_tcga": round(rho, 4),
                "sign_concordance_all": round(sign_all, 4), "sign_concordance_tcga_de": round(sign_de, 4)})
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))

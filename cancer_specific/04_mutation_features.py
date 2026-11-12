#!/usr/bin/env python3
"""
04_mutation_features.py — GDC masked MAF から遺伝子レベルの変異頻度特徴（原著 Mu_wrangle の再現）

原著: 各患者 × 遺伝子 × Variant_Classification 9 種（5'Flank, 5'UTR, Nonsense_Mutation, Splice_Site,
Splice_Region, Intron, 3'UTR, 3'Flank, RNA）を 0/1 にし、common_patients で平均した頻度。
Primary tumor（sample code 01）のみ。cscape=FALSE（ドライバー判定なし）。

注: 原著の 9 種は Missense / Frame_Shift 等を含まない（意図は不明。原著の選択をそのまま再現し、
拡張版 mut_all_* も別途保存する）。

入力: ~/Data/DExOmics/gdc/LIHC/maf/*.maf.gz
出力: ~/Data/DExOmics/results/features/mutation_gene_level.parquet
      ~/Data/DExOmics/results/features/mutation_patients.csv（変異データを持つ患者の一覧）
"""
import os, glob, gzip, numpy as np, pandas as pd
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT
OUT = f"{ROOT}/results/features"; os.makedirs(OUT, exist_ok=True)
MUT9 = ["5'Flank", "5'UTR", "Nonsense_Mutation", "Splice_Site", "Splice_Region", "Intron", "3'UTR", "3'Flank", "RNA"]

rows = []
for p in sorted(glob.glob(f"{ROOT}/gdc/LIHC/maf/*.maf.gz")):
    df = pd.read_csv(p, sep="\t", comment="#", low_memory=False, usecols=["Gene", "Variant_Classification", "Tumor_Sample_Barcode"])
    df = df[df.Tumor_Sample_Barcode.str[13:15] == "01"]
    df["patient"] = df.Tumor_Sample_Barcode.str[:12]
    rows.append(df[["patient", "Gene", "Variant_Classification"]])
maf = pd.concat(rows).drop_duplicates()
maf["Gene"] = maf["Gene"].str.split(".").str[0]
patients = sorted(maf.patient.unique())
print("patients with mutation data:", len(patients), "records:", len(maf))
pd.Series(patients, name="patient").to_csv(f"{OUT}/mutation_patients.csv", index=False)

# per patient×gene×class 0/1, then frequency over patients
pres = maf.assign(v=1).pivot_table(index="Gene", columns="Variant_Classification", values="v", aggfunc=lambda s: 0, fill_value=0)  # placeholder shape
cnt = maf.groupby(["Gene", "Variant_Classification"]).patient.nunique().unstack(fill_value=0)
freq = cnt / len(patients)
feat9 = pd.DataFrame(index=freq.index)
for c in MUT9:
    feat9[f"mut__{c}"] = freq[c] if c in freq.columns else 0.0
feat_all = freq.add_prefix("mutall__")
out = pd.concat([feat9, feat_all], axis=1)
out.index.name = "gene_id"
out.to_parquet(f"{OUT}/mutation_gene_level.parquet")
print("genes with any mutation:", out.shape[0]); print(feat9.describe().T[["mean", "max"]])

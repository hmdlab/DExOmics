#!/usr/bin/env python3
"""
03_cnv_features.py — GDC ASCAT3 gene-level copy number から Amp_rate / Del_rate（原著再現）

原著: 患者ごとの gene-level copy_number（ASCAT3）で、>2 を増幅、<2 を欠失として
common_patients に占める割合。Primary tumor 検体のみ（ファイル 1 つ = 腫瘍/正常ペア 1 患者）。

入力: ~/Data/DExOmics/gdc/LIHC/cnv/*.tsv  （列: gene_id gene_name chromosome start end copy_number min_copy_number max_copy_number）
出力: results/features/cnv_gene_level.parquet（gene_id × Amp_rate, Del_rate, mean_cn, frac_na）
      results/features/cnv_patients.csv
"""
import os, glob, numpy as np, pandas as pd
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT
OUT = f"{ROOT}/results/features"; os.makedirs(OUT, exist_ok=True)
man = pd.read_csv(f"{ROOT}/gdc/LIHC/manifest_cnv.csv").set_index("file_id")

cols = {}
for p in sorted(glob.glob(f"{ROOT}/gdc/LIHC/cnv/*")):
    if p.endswith(".part"): continue
    fid = os.path.basename(p).split("__")[0]
    patient = man.loc[fid, "case"]
    df = pd.read_csv(p, sep="\t", usecols=["gene_id", "copy_number"])
    df["gid"] = df.gene_id.str.split(".").str[0]
    s = df.drop_duplicates("gid").set_index("gid")["copy_number"]
    if patient in cols:   # a patient may have >1 file (e.g. two normals); keep first
        continue
    cols[patient] = s
cn = pd.DataFrame(cols)
patients = list(cn.columns)
print("patients with CNV:", len(patients), "genes:", cn.shape[0])
pd.Series(patients, name="patient").to_csv(f"{OUT}/cnv_patients.csv", index=False)
out = pd.DataFrame({
    "cnv__Amp_rate": (cn > 2).sum(axis=1) / len(patients),
    "cnv__Del_rate": (cn < 2).sum(axis=1) / len(patients),
    "cnv__mean_cn": cn.mean(axis=1),
    "cnv__frac_na": cn.isna().mean(axis=1),
})
out.index.name = "gene_id"
out.to_parquet(f"{OUT}/cnv_gene_level.parquet")
print(out.describe().T[["mean", "max"]])

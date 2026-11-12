#!/usr/bin/env python3
"""
05_methylation_features.py — 450k β 値から遺伝子×CGI 位置別の平均 Δβ（原著 TCGAanalyze_DMC 再現＋ラベル非依存版）

原著: TCGAanalyze_DMC(TP vs NT, p.cut=0.01, diffmean.cut=0.25) で有意プローブのみ残し、
probe→gene（genesUniq が単一遺伝子のもの）に付け、CGIposition 別に mean(Δβ) を遺伝子集約。
非有意は 0 埋め。列: Island, N_Shore, N_Shelf, S_Shore, S_Shelf, restPositions(=NA)。

査読（R3-4, B1-5）: 有意プローブだけを平均するとバイアス。ここでは
  (A) 原著再現版  methy_sig__<pos>   有意プローブのみ（Wilcoxon rank-sum BH p<0.01 & |Δmean|>0.25）
  (B) 全プローブ版 methy_all__<pos>   有意判定に依らず全プローブの Δmean を遺伝子集約
の両方を出す。学習時にどちらを使うかを ablation で比べる。

入力: gdc/LIHC/methy/*.txt（2 列: probe, beta。GDC SeSAMe 形式）、manifest_methy.csv、gdc/annot/methy_probe_annot.tsv.gz
出力: results/features/methylation_gene_level.parquet、methylation_patients.csv、methylation_probe_stats.parquet
"""
import os, glob, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT
OUT = f"{ROOT}/results/features"; os.makedirs(OUT, exist_ok=True)
man = pd.read_csv(f"{ROOT}/gdc/LIHC/manifest_methy.csv").set_index("file_id")

series = {}; types = {}
for p in sorted(glob.glob(f"{ROOT}/gdc/LIHC/methy/*")):
    if p.endswith(".part"): continue
    fid = os.path.basename(p).split("__")[0]
    st = man.loc[fid, "sample_type"]
    if st not in ("Primary Tumor", "Solid Tissue Normal"): continue
    sample = man.loc[fid, "sample"]; patient = sample[:12]
    df = pd.read_csv(p, sep="\t", header=None, names=["probe", "beta"], index_col=0)
    key = (patient, "TP" if st == "Primary Tumor" else "NT")
    if key in series: continue          # replicates: keep first (原著は平均だが件数は僅少)
    series[key] = df["beta"]; types[key] = key[1]
beta = pd.DataFrame(series)
print("samples:", pd.Series(types).value_counts().to_dict(), "probes:", beta.shape[0])
beta = beta.dropna(axis=0, how="any")   # 原著: NA を含むプローブは除外
print("probes without NA:", beta.shape[0])
tp = beta.loc[:, [k for k in beta.columns if k[1] == "TP"]]
nt = beta.loc[:, [k for k in beta.columns if k[1] == "NT"]]
pd.Series(sorted({k[0] for k in tp.columns}), name="patient").to_csv(f"{OUT}/methylation_patients.csv", index=False)

# Wilcoxon rank-sum (Mann-Whitney) per probe, vectorized in chunks
pvals = np.empty(beta.shape[0]);
X = tp.values; Y = nt.values
for i in range(0, beta.shape[0], 20000):
    pvals[i:i+20000] = stats.mannwhitneyu(X[i:i+20000], Y[i:i+20000], axis=1, alternative="two-sided").pvalue
padj = multipletests(pvals, method="fdr_bh")[1]
delta = tp.mean(axis=1) - nt.mean(axis=1)      # mean.TP.minus.mean.NT
ps = pd.DataFrame({"delta": delta, "p": pvals, "padj": padj}, index=beta.index)
ps["sig"] = (ps.padj < 0.01) & (ps.delta.abs() > 0.25)
ps.to_parquet(f"{OUT}/methylation_probe_stats.parquet")
print("significant probes:", int(ps.sig.sum()))

annot = pd.read_csv(f"{ROOT}/gdc/annot/methy_probe_annot.tsv.gz", sep="\t", usecols=["probeID", "genesUniq", "CGIposition"], low_memory=False).set_index("probeID")
annot = annot[annot.genesUniq.notna() & ~annot.genesUniq.str.contains(";")]
annot["pos"] = annot.CGIposition.fillna("restPositions")
ps = ps.join(annot, how="inner")
def agg(sub, prefix):
    g = sub.groupby(["genesUniq", "pos"]).delta.mean().unstack(fill_value=0.0)
    g.columns = [f"{prefix}__{c}" for c in g.columns]
    return g
A = agg(ps[ps.sig], "methy_sig"); B = agg(ps, "methy_all")
out = A.join(B, how="outer").fillna(0.0)
# map gene symbol -> ensembl via distributed mapped_geneID (GENCODE v26 symbols) ※ シンボル重複は先勝ち
m = pd.read_csv(f"{ROOT}/data/TCGAprocessed/LIHC/mapped_geneID.csv").drop_duplicates("gene_name").set_index("gene_name")["Gene"]
out = out.join(m.rename("gene_id"), how="inner").set_index("gene_id")
out.to_parquet(f"{OUT}/methylation_gene_level.parquet")
print("genes with methylation features:", out.shape, "\n", out.filter(like="methy_sig").describe().T[["mean", "max"]])

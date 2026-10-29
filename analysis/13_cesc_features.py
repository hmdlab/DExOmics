#!/usr/bin/env python3
"""
13_cesc_features.py — CESC 用の特徴抽出一式（HeLa 結合 + CESC の CNV/変異/メチル化）

02/03/04/05 の CESC 版（探索的解析。CESC は Solid Tissue Normal 3 例しか無く、
メチル化の有意プローブ判定はほぼ無力 → methy_all（ラベル非依存 Δmean）を主とし
methy_sig は参考出力に留める）。

入力: data/{promoter,rna}_features/encode_hela_*、gdc/CESC/{cnv,maf,methy}、gdc/annot/
出力: results/features/binding_gene_level_hela.parquet、{cnv,mutation,methylation}_gene_level_cesc.parquet
実行: venv/bin/python analysis/13_cesc_features.py
"""
import os, sys, glob, gzip, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT
OUT = f"{ROOT}/results/features"; os.makedirs(OUT, exist_ok=True)
ONLY = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else {"binding", "cnv", "mutation", "methylation"}

# ---------- HeLa binding (02 相当) ----------
if "binding" in ONLY and not os.path.exists(f"{OUT}/binding_gene_level_hela.parquet"):
    def load(kind):
        d = pd.read_pickle(f"{ROOT}/data/{kind}_features/encode_hela_{kind}.pkl")
        names = pd.read_csv(f"{ROOT}/data/{kind}_features/encode_hela_{kind}_feature_name.txt.gz", header=None)[0].tolist()
        gid = d["Name"].str.split(".").str[0].tolist()
        sums = np.stack([np.asarray(m.sum(axis=1)).ravel() for m in d["feature_vec"]]).astype(np.float32)
        nbins = np.array([m.shape[1] for m in d["feature_vec"]])
        assert sums.shape[1] == len(names)
        return gid, names, sums, nbins
    gp, pn, ps, pb = load("promoter"); gr, rn, rs, rb = load("rna")
    assert gp == gr
    gl = pd.concat([pd.DataFrame(ps, index=gp, columns=[f"prom__{n}" for n in pn]),
                    pd.DataFrame(rs, index=gr, columns=[f"rna__{n}" for n in rn])], axis=1)
    gl["n_bins_prom"] = pb; gl["n_bins_rna"] = rb
    gl.index.name = "gene_id"; gl = gl[~gl.index.duplicated(keep="first")]
    gl.to_parquet(f"{OUT}/binding_gene_level_hela.parquet")
    print("HeLa binding:", gl.shape, "| prom", len(pn), "rna", len(rn))

# ---------- CNV (03 相当) ----------
if "cnv" in ONLY and not os.path.exists(f"{OUT}/cnv_gene_level_cesc.parquet"):
    man = pd.read_csv(f"{ROOT}/gdc/CESC/manifest_cnv.csv").set_index("file_id")
    cols = {}
    for p in sorted(glob.glob(f"{ROOT}/gdc/CESC/cnv/*")):
        if p.endswith(".part"): continue
        fid = os.path.basename(p).split("__")[0]
        patient = man.loc[fid, "case"]
        if patient in cols: continue
        df = pd.read_csv(p, sep="\t", usecols=["gene_id", "copy_number"])
        df["gid"] = df.gene_id.str.split(".").str[0]
        cols[patient] = df.drop_duplicates("gid").set_index("gid")["copy_number"]
    cn = pd.DataFrame(cols)
    out = pd.DataFrame({"cnv__Amp_rate": (cn > 2).sum(axis=1) / cn.shape[1], "cnv__Del_rate": (cn < 2).sum(axis=1) / cn.shape[1],
                        "cnv__mean_cn": cn.mean(axis=1), "cnv__frac_na": cn.isna().mean(axis=1)})
    out.index.name = "gene_id"; out.to_parquet(f"{OUT}/cnv_gene_level_cesc.parquet")
    print("CESC CNV:", cn.shape[1], "patients", out.shape)

# ---------- mutation (04 相当) ----------
if "mutation" in ONLY and not os.path.exists(f"{OUT}/mutation_gene_level_cesc.parquet"):
    MUT9 = ["5'Flank", "5'UTR", "Nonsense_Mutation", "Splice_Site", "Splice_Region", "Intron", "3'UTR", "3'Flank", "RNA"]
    rows = []
    for p in sorted(glob.glob(f"{ROOT}/gdc/CESC/maf/*.maf.gz")):
        df = pd.read_csv(p, sep="\t", comment="#", low_memory=False, usecols=["Gene", "Variant_Classification", "Tumor_Sample_Barcode"])
        df = df[df.Tumor_Sample_Barcode.str[13:15] == "01"]
        df["patient"] = df.Tumor_Sample_Barcode.str[:12]
        rows.append(df[["patient", "Gene", "Variant_Classification"]])
    maf = pd.concat(rows).drop_duplicates()
    maf["Gene"] = maf["Gene"].str.split(".").str[0]
    n_pat = maf.patient.nunique()
    cnt = maf.groupby(["Gene", "Variant_Classification"]).patient.nunique().unstack(fill_value=0)
    freq = cnt / n_pat
    feat9 = pd.DataFrame(index=freq.index)
    for c in MUT9: feat9[f"mut__{c}"] = freq[c] if c in freq.columns else 0.0
    out = pd.concat([feat9, freq.add_prefix("mutall__")], axis=1); out.index.name = "gene_id"
    out.to_parquet(f"{OUT}/mutation_gene_level_cesc.parquet")
    print("CESC mutation:", n_pat, "patients", out.shape)

# ---------- methylation (05 相当、NT=3 の制約下) ----------
if "methylation" in ONLY and not os.path.exists(f"{OUT}/methylation_gene_level_cesc.parquet"):
    man = pd.read_csv(f"{ROOT}/gdc/CESC/manifest_methy.csv").set_index("file_id")
    series = {}
    for p in sorted(glob.glob(f"{ROOT}/gdc/CESC/methy/*")):
        if p.endswith(".part"): continue
        fid = os.path.basename(p).split("__")[0]
        st = man.loc[fid, "sample_type"]
        if st not in ("Primary Tumor", "Solid Tissue Normal"): continue
        key = (man.loc[fid, "sample"][:12], "TP" if st == "Primary Tumor" else "NT")
        if key in series: continue
        series[key] = pd.read_csv(p, sep="\t", header=None, names=["probe", "beta"], index_col=0)["beta"]
    beta = pd.DataFrame(series).dropna(axis=0, how="any")
    tp = beta.loc[:, [k for k in beta.columns if k[1] == "TP"]]
    nt = beta.loc[:, [k for k in beta.columns if k[1] == "NT"]]
    print("CESC methy: TP", tp.shape[1], "NT", nt.shape[1], "probes", beta.shape[0])
    pvals = np.empty(beta.shape[0])
    for i in range(0, beta.shape[0], 20000):
        pvals[i:i+20000] = stats.mannwhitneyu(tp.values[i:i+20000], nt.values[i:i+20000], axis=1, alternative="two-sided").pvalue
    padj = multipletests(pvals, method="fdr_bh")[1]
    ps = pd.DataFrame({"delta": tp.mean(axis=1) - nt.mean(axis=1), "p": pvals, "padj": padj}, index=beta.index)
    ps["sig"] = (ps.padj < 0.01) & (ps.delta.abs() > 0.25)
    print("significant probes (NT=3, 参考値):", int(ps.sig.sum()))
    annot = pd.read_csv(f"{ROOT}/gdc/annot/methy_probe_annot.tsv.gz", sep="\t", usecols=["probeID", "genesUniq", "CGIposition"], low_memory=False).set_index("probeID")
    annot = annot[annot.genesUniq.notna() & ~annot.genesUniq.str.contains(";")]
    annot["pos"] = annot.CGIposition.fillna("restPositions")
    ps = ps.join(annot, how="inner")
    def agg(sub, prefix):
        g = sub.groupby(["genesUniq", "pos"]).delta.mean().unstack(fill_value=0.0)
        g.columns = [f"{prefix}__{c}" for c in g.columns]; return g
    out = agg(ps[ps.sig], "methy_sig").join(agg(ps, "methy_all"), how="outer").fillna(0.0)
    m = pd.read_csv(f"{ROOT}/data/TCGAprocessed/CESC/mapped_geneID.csv").drop_duplicates("gene_name").set_index("gene_name")["Gene"]
    out = out.join(m.rename("gene_id"), how="inner").set_index("gene_id")
    out.to_parquet(f"{OUT}/methylation_gene_level_cesc.parquet")
    print("CESC methylation:", out.shape)
print("done")

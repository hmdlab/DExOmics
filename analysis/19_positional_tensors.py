#!/usr/bin/env python3
"""
19_positional_tensors.py — 配布 pkl の位置分解特徴を学習用テンソルに変換（DL 実験の前処理）

- promoter: 遺伝子ごと 550因子×30bin → [N, 550, 30] float32（そのまま）
- rna: 86因子×可変bin（1〜1,968）→ 因子軸を保ち、位置軸を adaptive average pooling で 32bin に固定
  → [N, 86, 32]。実 bin 数 n_bins_rna は別ベクトルで保存（スケール情報が消えるため）。
  ※ pooling は「平均密度」なので合計量は n_bins を掛ければ復元可能。
- 遺伝子順序は 06 の学習対象（de_results ∩ binding ∩ annotation、chr1-22+X、17,807 遺伝子）に一致させ、
  fold 割当も保存。

出力: ~/Data/DExOmics/results/tensors/{prom.npy, rna.npy, genes.csv, meta.json}
実行: venv/bin/python analysis/19_positional_tensors.py
"""
import os, json, numpy as np, pandas as pd

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; F = f"{ROOT}/results/features"
OUT = f"{ROOT}/results/tensors"; os.makedirs(OUT, exist_ok=True)
RNA_BINS = 32

# 06 と同じ遺伝子集合・fold
de = pd.read_csv(f"{ROOT}/results/de_lihc/de_results.csv").set_index("gene_id")
de = de[de.gene_type.isin(["protein_coding", "lncRNA"]) & de.log2FC.notna()]
ann = pd.read_parquet(f"{F}/gene_annotation_gencode26.parquet").set_index("gene_id")
bind = pd.read_parquet(f"{F}/binding_gene_level.parquet")
genes = de.index.intersection(bind.index).intersection(ann.index)
df = de.loc[genes, ["log2FC", "label"]].join(ann.loc[genes, ["chrom"]])
df = df[df.chrom.isin([f"chr{i}" for i in range(1, 23)] + ["chrX"])]
folds = pd.read_csv(f"{ROOT}/results/exp1_nested_cv/folds.csv").set_index("chrom")["fold"]
df["fold"] = df.chrom.map(folds)
print("genes:", len(df))

def load(kind):
    d = pd.read_pickle(f"{ROOT}/data/{kind}_features/encode_hepg2_{kind}.pkl")
    gid = d["Name"].str.split(".").str[0].values
    first = {}
    for i, g in enumerate(gid):
        if g not in first: first[g] = i     # 重複 gene_id は先勝ち（02 と同じ）
    return d, first

target = df.index.tolist()

dp, fp = load("promoter")
P = np.zeros((len(target), 550, 30), dtype=np.float32)
for j, g in enumerate(target):
    m = dp["feature_vec"][fp[g]]
    P[j] = np.asarray(m.todense(), dtype=np.float32)
np.save(f"{OUT}/prom.npy", P)
print("prom tensor:", P.shape, f"{P.nbytes/1e9:.2f} GB | nonzero frac {(P>0).mean():.4f}")
del P, dp

dr, fr = load("rna")
R = np.zeros((len(target), 86, RNA_BINS), dtype=np.float32)
nbins = np.zeros(len(target), dtype=np.int32)
for j, g in enumerate(target):
    m = np.asarray(dr["feature_vec"][fr[g]].todense(), dtype=np.float32)   # [86, L]
    L = m.shape[1]; nbins[j] = L
    if L == RNA_BINS:
        R[j] = m
    else:
        # adaptive average pooling: 各出力 bin は入力区間の平均（密度を保存）
        edges = np.linspace(0, L, RNA_BINS + 1)
        for b in range(RNA_BINS):
            lo, hi = int(np.floor(edges[b])), max(int(np.ceil(edges[b + 1])), int(np.floor(edges[b])) + 1)
            R[j, :, b] = m[:, lo:hi].mean(axis=1)
np.save(f"{OUT}/rna.npy", R)
df.assign(n_bins_rna=nbins).to_csv(f"{OUT}/genes.csv")
json.dump({"n_genes": len(target), "rna_bins": RNA_BINS, "prom_shape": [550, 30], "rna_shape": [86, RNA_BINS],
           "note": "rna is mean density per pooled bin; multiply by n_bins_rna/RNA_BINS for totals"},
          open(f"{OUT}/meta.json", "w"), indent=1)
print("rna tensor:", R.shape, f"{R.nbytes/1e9:.2f} GB | nonzero frac {(R>0).mean():.4f}")
print("done")

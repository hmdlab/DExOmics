#!/usr/bin/env python3
"""00_gene_annotation.py — GENCODE v26 GTF（配布 pancan_data 内）から遺伝子注釈表を作る。
出力: ~/Data/DExOmics/results/features/gene_annotation_gencode26.parquet（gene_id, gene_name, gene_type, chrom, start, end, strand, length）"""
import os, re, pandas as pd
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; OUT = f"{ROOT}/results/features"; os.makedirs(OUT, exist_ok=True)
rows = []
with open(f"{ROOT}/data/pancan_data/references_v8_gencode.v26.GRCh38.genes.gtf") as f:
    for line in f:
        if line.startswith("#"): continue
        p = line.rstrip("\n").split("\t")
        if p[2] != "gene": continue
        a = dict(re.findall(r'(\S+) "([^"]*)"', p[8]))
        rows.append((a["gene_id"].split(".")[0], a.get("gene_name"), a.get("gene_type"), p[0], int(p[3]), int(p[4]), p[6]))
g = pd.DataFrame(rows, columns=["gene_id", "gene_name", "gene_type", "chrom", "start", "end", "strand"]).drop_duplicates("gene_id")
g["length"] = g.end - g.start + 1
g.to_parquet(f"{OUT}/gene_annotation_gencode26.parquet"); print(g.shape, g.chrom.value_counts().head(5).to_dict())

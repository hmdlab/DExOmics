#!/usr/bin/env python3
"""
02_binding_features.py — HepG2 の結合特徴（配布 pkl）を遺伝子レベルの表に展開する

配布データ（Yi の前処理済み）: 遺伝子ごとに (因子 × bin) の CSR 疎行列。**bin 数は遺伝子長に依存して可変**
（原著 CNN は length_align でパディング）。
  promoter: 550 TF（先頭 'mask' は位置マスク）、rna: 86 RBP（先頭 'exon' はエクソンマスク）

出力: ~/Data/DExOmics/results/features/
  binding_gene_level.parquet : 遺伝子 × (550 + 86) の bin 合計（結合カウント）＋ n_bins_prom / n_bins_rna。
  gene_index.csv : 遺伝子 ID（バージョン無し）
原著の「30 遺伝子以上に結合する因子」フィルタは学習時に fold 内で行う。
"""
import os, numpy as np, pandas as pd, scipy.sparse as sp
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT
OUT = f"{ROOT}/results/features"; os.makedirs(OUT, exist_ok=True)

def load(kind):
    d = pd.read_pickle(f"{ROOT}/data/{kind}_features/encode_hepg2_{kind}.pkl")
    names = pd.read_csv(f"{ROOT}/data/{kind}_features/encode_hepg2_{kind}_feature_name.txt.gz", header=None)[0].tolist()
    gid = d["Name"].str.split(".").str[0].tolist()
    sums = np.stack([np.asarray(m.sum(axis=1)).ravel() for m in d["feature_vec"]]).astype(np.float32)
    nbins = np.array([m.shape[1] for m in d["feature_vec"]])
    assert sums.shape[1] == len(names)
    return gid, names, sums, nbins

gp, pn, ps, pb = load("promoter"); gr, rn, rs, rb = load("rna")
assert gp == gr
prom = pd.DataFrame(ps, index=gp, columns=[f"prom__{n}" for n in pn])
rna = pd.DataFrame(rs, index=gr, columns=[f"rna__{n}" for n in rn])
gl = pd.concat([prom, rna], axis=1); gl["n_bins_prom"] = pb; gl["n_bins_rna"] = rb
gl.index.name = "gene_id"; gl = gl[~gl.index.duplicated(keep="first")]
gl.to_parquet(f"{OUT}/binding_gene_level.parquet"); pd.Series(gl.index).to_csv(f"{OUT}/gene_index.csv", index=False)
print("genes:", len(gl), "| promoter factors", len(pn), "| rna factors", len(rn))
print("rna bins: min/median/max", rb.min(), int(np.median(rb)), rb.max(), "| promoter bins", pb.min(), int(np.median(pb)), pb.max())
print("genes with any promoter binding:", int((prom.drop(columns='prom__mask').sum(axis=1) > 0).sum()), "| any RNA binding:", int((rna.drop(columns='rna__exon').sum(axis=1) > 0).sum()))
print("factors binding >=30 genes: promoter", int(((prom > 0).sum(axis=0) >= 30).sum()), "rna", int(((rna > 0).sum(axis=0) >= 30).sum()))

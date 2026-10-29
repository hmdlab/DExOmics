#!/usr/bin/env python3
"""
22_dl_seed_robustness.py — stage-2 (depthwise_pe) の seed 頑健性: fold 毎最良 config を seed 1–5 で再学習し
seed 平均 OOF を作る（codex R2/R4: ensemble 主張には学習不確実性を含めた評価が必要）。

fold 毎 config/epochs は run_dl_stage2.log の FINAL 行を転記（内側 fold で決定済みの値。再選択なし）。
出力: results/exp4_dl/oof_dl_seedavg.parquet（列 positional|depthwise_pe_seedavg と各 seed 列）
実行: venv/bin/python analysis/22_dl_seed_robustness.py
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
import torch, torch.nn as nn

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; T = f"{ROOT}/results/tensors"; F = f"{ROOT}/results/features"
OUT = f"{ROOT}/results/exp4_dl"
DEV = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
BATCH = 512; SEEDS = [1, 2, 3, 4, 5]
# run_dl_stage2.log の FINAL（fold 毎最良 config・median epoch、内側 fold で確定済み）
FOLD_CFG = {0: (dict(C=96, drop=0.2, lr=3e-4), 13), 1: (dict(C=128, drop=0.2, lr=1e-3), 7),
            2: (dict(C=96, drop=0.2, lr=1e-3), 8), 3: (dict(C=96, drop=0.2, lr=1e-3), 7),
            4: (dict(C=128, drop=0.3, lr=1e-3), 10)}

genes = pd.read_csv(f"{T}/genes.csv", index_col=0)
P = np.log1p(np.load(f"{T}/prom.npy")); R = np.log1p(np.load(f"{T}/rna.npy"))
omics = pd.concat([pd.read_parquet(f"{F}/{n}.parquet") for n in ["cnv_gene_level", "mutation_gene_level", "methylation_gene_level"]], axis=1)
O = omics.reindex(genes.index).fillna(0.0).values.astype(np.float32)
O = np.hstack([O, np.log1p(genes.n_bins_rna.values[:, None]).astype(np.float32)])
y = genes.log2FC.values.astype(np.float32); fold = genes.fold.values

class DWBranch(nn.Module):
    def __init__(self, nf, L, C, drop):
        super().__init__()
        self.a = nn.Sequential(nn.Conv1d(nf, nf, 5, padding=2, groups=nf), nn.GELU(),
                               nn.Conv1d(nf, C, 1), nn.GELU(), nn.Dropout(drop))
        self.pe = nn.Parameter(torch.zeros(C, L))
        self.b = nn.Sequential(nn.Conv1d(C, C, 3, padding=1), nn.GELU())
    def forward(self, x):
        return self.b(self.a(x) + self.pe)

class Depthwise(nn.Module):
    def __init__(self, C, drop):
        super().__init__()
        self.pb = DWBranch(550, 30, C, drop); self.rb = DWBranch(86, 32, C, drop)
        self.head = nn.Sequential(nn.LayerNorm(4 * C + O.shape[1]), nn.Dropout(drop),
                                  nn.Linear(4 * C + O.shape[1], 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))
    def forward(self, p, r, o):
        hp = self.pb(p); hr = self.rb(r)
        return self.head(torch.cat([hp.mean(-1), hp.amax(-1), hr.mean(-1), hr.amax(-1), o], dim=1)).squeeze(-1)

@torch.no_grad()
def predict(mdl, idx, Ostd):
    mdl.eval(); out = []
    for b in range(0, len(idx), 2048):
        sel = idx[b:b + 2048]
        out.append(mdl(torch.from_numpy(P[sel]).to(DEV), torch.from_numpy(R[sel]).to(DEV),
                       torch.from_numpy(Ostd[sel].astype(np.float32)).to(DEV)).cpu().numpy())
    return np.concatenate(out)

oof = genes[["log2FC", "label", "chrom", "fold"]].copy()
preds = {s: np.full(len(genes), np.nan) for s in SEEDS}
for k in range(5):
    cfg, n_ep = FOLD_CFG[k]
    tr_idx = np.where(fold != k)[0]; te_idx = np.where(fold == k)[0]
    mu, sd = O[tr_idx].mean(0), O[tr_idx].std(0) + 1e-8
    Ostd = (O - mu) / sd
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        mdl = Depthwise(cfg["C"], cfg["drop"]).to(DEV)
        opt = torch.optim.AdamW(mdl.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        lossf = nn.HuberLoss(delta=2.0)
        for ep in range(n_ep):
            mdl.train(); perm = np.random.permutation(len(tr_idx))
            for b in range(0, len(perm), BATCH):
                sel = tr_idx[perm[b:b + BATCH]]
                p = torch.from_numpy(P[sel]).to(DEV); r = torch.from_numpy(R[sel]).to(DEV)
                o = torch.from_numpy(Ostd[sel].astype(np.float32)).to(DEV)
                opt.zero_grad(); lossf(mdl(p, r, o), torch.from_numpy(y[sel]).to(DEV)).backward(); opt.step()
        preds[s][te_idx] = predict(mdl, te_idx, Ostd)
        print(f"fold {k} seed {s} done", flush=True)
for s in SEEDS: oof[f"pe_seed{s}"] = preds[s]
oof["positional|depthwise_pe_seedavg"] = np.mean([preds[s] for s in SEEDS], axis=0)
oof.to_parquet(f"{OUT}/oof_dl_seedavg.parquet")
sa = oof["positional|depthwise_pe_seedavg"]
per_seed = [float(spearmanr(y, preds[s])[0]) for s in SEEDS]
print("per-seed spearman:", [round(v, 4) for v in per_seed])
print("seed-avg spearman:", round(float(spearmanr(y, sa)[0]), 4))

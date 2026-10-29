#!/usr/bin/env python3
"""
25_original_arch_pe.py — 原著 ConcatedNet の回帰版（PE あり/なし）を同一 nested CV で評価

原著 (code/scripts/cancer_specific/evaluation_lihc/network.py + pretrained/LIHC_params.json):
- 入力 [B, 1, L, F]。conv1 kernel=(1, F) で因子軸を一括線形混合 → ReLU → (tuned では位置 conv なし)
  → 位置方向 sum → RNA/DNA 結合 → FC(64) → TCGA MLP(96×3) と concat → 出力。dropout 0.5、Xavier 初期化。
- tuned: C=32、n_ConvLayer=1（**位置畳み込みゼロ = 位置情報をほぼ使わない構造**）。

本改造（PI 指示: 位置 embedding + log2FC 回帰）:
- n_out=1、Huber(δ=2)。学習プロトコルは 21/22/23 と同一（AdamW、内側 fold で config/epoch、median epoch）。
- PE: conv1 出力 [B, C, L, 1] に学習可能 PE[C, L] を加算（ReLU 前）。arm = orignet（PE なし）/ orignet_pe。
- 入力は他 DL arm と同一の 32-bin pooled テンソル（表現を揃えアーキテクチャ差のみを比較）。
- config 5 点/arm: tuned(C32,L1) / C32,L2 / C64,L1 / C64,L2 / tuned lr3e-4（探索予算は stage-2 と同数、別掲透明化）。

出力: results/exp4_dl/oof_orignet.parquet, metrics_orignet.csv
実行: venv/bin/python analysis/25_original_arch_pe.py [--quick]
"""
import os, sys, json, time, warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
import torch, torch.nn as nn, torch.nn.functional as F_
warnings.filterwarnings("ignore")
torch.manual_seed(0); np.random.seed(0)

QUICK = "--quick" in sys.argv
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; T = f"{ROOT}/results/tensors"; F = f"{ROOT}/results/features"
OUT = f"{ROOT}/results/exp4_dl"
DEV = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
BATCH = 512; MAX_EPOCH = 3 if QUICK else 150; PATIENCE = 10

genes = pd.read_csv(f"{T}/genes.csv", index_col=0)
P = np.log1p(np.load(f"{T}/prom.npy"))                     # [N, 550, 30]
R = np.log1p(np.load(f"{T}/rna.npy"))                      # [N, 86, 32]
omics = pd.concat([pd.read_parquet(f"{F}/{n}.parquet") for n in ["cnv_gene_level", "mutation_gene_level", "methylation_gene_level"]], axis=1)
O = omics.reindex(genes.index).fillna(0.0).values.astype(np.float32)
O = np.hstack([O, np.log1p(genes.n_bins_rna.values[:, None]).astype(np.float32)])
y = genes.log2FC.values.astype(np.float32); fold = genes.fold.values
# 原著の入力形式 [B, 1, L, F] へ: transpose して factor を最後の軸に
NP_, LP = P.shape[1], P.shape[2]; NR_, LR = R.shape[1], R.shape[2]
print(f"prom F={NP_} L={LP} | rna F={NR_} L={LR} | omics {O.shape[1]} | dev {DEV}", flush=True)

class OrigBranch(nn.Module):
    """conv1(1,F) → [+PE] → ReLU → (n_layer-1 個の位置 conv(k,1)) → 位置 sum"""
    def __init__(self, F_dim, L, C, n_layer, use_pe, k=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, C, kernel_size=(1, F_dim))
        self.pe = nn.Parameter(torch.zeros(C, L)) if use_pe else None
        self.extra = nn.ModuleList([nn.Conv2d(C, C, kernel_size=(k, 1)) for _ in range(n_layer - 1)])
    def forward(self, x):                      # x: [B, 1, L, F]
        h = self.conv1(x)                      # [B, C, L, 1]
        if self.pe is not None:
            h = h + self.pe[None, :, :, None]
        h = F_.relu(h)
        for c in self.extra:
            h = F_.relu(c(h))                  # 位置次元が k-1 ずつ縮む（原著同様 padding なし）
        return torch.flatten(torch.sum(h, dim=2), 1)   # 位置 sum → [B, C]

class OrigNet(nn.Module):
    def __init__(self, C=32, n_layer=1, use_pe=True, fc=64, tcga_h=96, drop=0.5):
        super().__init__()
        self.rna = OrigBranch(NR_, LR, C, n_layer, use_pe)
        self.dna = OrigBranch(NP_, LP, C, n_layer, use_pe)
        self.fc_enc = nn.Linear(2 * C, fc)
        self.t1 = nn.Linear(O.shape[1], tcga_h); self.t2 = nn.Linear(tcga_h, tcga_h); self.t3 = nn.Linear(tcga_h, tcga_h)
        self.out = nn.Linear(fc + tcga_h, 1)
        self.drop = nn.Dropout(drop)
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.constant_(p, 0)
    def forward(self, p, r, o):
        xr = self.rna(r.unsqueeze(1).transpose(2, 3))      # [B, 86, 32] → [B,1,32,86]
        xd = self.dna(p.unsqueeze(1).transpose(2, 3))      # [B, 550, 30] → [B,1,30,550]
        xe = self.drop(F_.relu(self.fc_enc(torch.cat([xr, xd], 1))))
        xt = self.drop(F_.relu(self.t3(F_.relu(self.t2(F_.relu(self.t1(o)))))))
        return self.out(torch.cat([xt, xe], 1)).squeeze(-1)

GRID = [dict(C=32, n_layer=1, lr=1e-3), dict(C=32, n_layer=2, lr=1e-3), dict(C=64, n_layer=1, lr=1e-3),
        dict(C=64, n_layer=2, lr=1e-3), dict(C=32, n_layer=1, lr=3e-4)]
if QUICK: GRID = GRID[:2]
ARMS = [("orignet_pe", True), ("orignet", False)]

@torch.no_grad()
def predict(mdl, idx, Ostd):
    mdl.eval(); out = []
    for b in range(0, len(idx), 2048):
        sel = idx[b:b + 2048]
        out.append(mdl(torch.from_numpy(P[sel]).to(DEV), torch.from_numpy(R[sel]).to(DEV),
                       torch.from_numpy(Ostd[sel].astype(np.float32)).to(DEV)).cpu().numpy())
    return np.concatenate(out)

def run_train(cfg, use_pe, idx_tr, Ostd, idx_val=None, fixed_epochs=None):
    mdl = OrigNet(cfg["C"], cfg["n_layer"], use_pe).to(DEV)
    opt = torch.optim.AdamW(mdl.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=2.0)
    best_s, best_ep, bad = -np.inf, 0, 0
    for ep in range(1, (fixed_epochs or MAX_EPOCH) + 1):
        mdl.train(); perm = np.random.permutation(len(idx_tr))
        for b in range(0, len(perm), BATCH):
            sel = idx_tr[perm[b:b + BATCH]]
            p = torch.from_numpy(P[sel]).to(DEV); r = torch.from_numpy(R[sel]).to(DEV)
            o = torch.from_numpy(Ostd[sel].astype(np.float32)).to(DEV)
            opt.zero_grad(); lossf(mdl(p, r, o), torch.from_numpy(y[sel]).to(DEV)).backward(); opt.step()
        if idx_val is not None and fixed_epochs is None:
            s = spearmanr(y[idx_val], predict(mdl, idx_val, Ostd))[0]
            if s > best_s: best_s, best_ep, bad = s, ep, 0
            else:
                bad += 1
                if bad >= PATIENCE: break
    return mdl, best_s, best_ep

oof = genes[["log2FC", "label", "chrom", "fold"]].copy(); records = []; log = {"grid": GRID, "runs": []}
for arm, use_pe in ARMS:
    pred = np.full(len(genes), np.nan)
    for k in range(5):
        tr_idx = np.where(fold != k)[0]; te_idx = np.where(fold == k)[0]
        mu, sd = O[tr_idx].mean(0), O[tr_idx].std(0) + 1e-8; Ostd = (O - mu) / sd
        best, best_s, eps = None, -np.inf, []
        for cfg in GRID:
            t0 = time.time(); scores, e = [], []
            for hold in sorted(set(fold[tr_idx])):
                ia = tr_idx[fold[tr_idx] != hold]; iv = tr_idx[fold[tr_idx] == hold]
                _, s, ep = run_train(cfg, use_pe, ia, Ostd, idx_val=iv)
                scores.append(s); e.append(ep)
            s = float(np.nanmean(scores))
            log["runs"].append(dict(arm=arm, fold=k, cfg=cfg, inner=round(s, 4), sec=round(time.time() - t0, 1)))
            print(f"{arm} fold {k} {cfg} inner {s:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if s > best_s: best, best_s, eps = cfg, s, e
        med_ep = max(int(np.median(eps)), 5)
        mdl, _, _ = run_train(best, use_pe, tr_idx, Ostd, fixed_epochs=med_ep)
        pred[te_idx] = predict(mdl, te_idx, Ostd)
        print(f"{arm} fold {k} FINAL {best} epochs {med_ep}", flush=True)
    col = f"positional|{arm}"; oof[col] = pred
    r = dict(model=col, spearman=float(spearmanr(y, pred)[0]), rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
             prauc_up=float(average_precision_score(genes.label == "UP", pred)),
             prauc_down=float(average_precision_score(genes.label == "DOWN", -pred)))
    records.append(r); print(f"{col:26s} spearman {r['spearman']:.3f}", flush=True)
oof.to_parquet(f"{OUT}/oof_orignet.parquet"); pd.DataFrame(records).to_csv(f"{OUT}/metrics_orignet.csv", index=False)
json.dump(log, open(f"{OUT}/orignet_training_log.json", "w"), indent=1)
print("saved")

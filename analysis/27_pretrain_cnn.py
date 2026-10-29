#!/usr/bin/env python3
"""
27_pretrain_cnn.py — pan-cancer 事前学習 → LIHC fine-tune（PI 指示・案3）

Pretext: 同一入力テンソルから TCGA pancanatlas の癌種別中央発現 31 次元（LIHC 列除外、列ごと訓練側標準化）
を multi-output 回帰。trunk = depthwise_pe の conv 枝のみ（omics head は pretext に含めない）。
Leakage 制御: 各外側 fold の訓練染色体の遺伝子のみで pretrain。LIHC 列除外。
Fine-tune: trunk を初期化に、全体を LIHC logFC で学習。config は stage-2 の fold 毎確定値（再選択なし）、
fine-tune epoch のみ内側 4 回転早期終了 → median。pretrain 40 epoch 固定・lr 1e-3。

出力: results/exp4_dl/oof_dl_pretrained.parquet, pretrain_log.json
実行: venv/bin/python analysis/27_pretrain_cnn.py [--quick] [--seeds 1]
"""
import os, sys, json, time, warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr
import torch, torch.nn as nn
warnings.filterwarnings("ignore")

QUICK = "--quick" in sys.argv
SEEDS = list(range(1, 1 + (int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 1)))
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; T = f"{ROOT}/results/tensors"; F = f"{ROOT}/results/features"
OUT = f"{ROOT}/results/exp4_dl"
DEV = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
BATCH = 512; MAX_EPOCH = 3 if QUICK else 150; PATIENCE = 10; PRE_EPOCH = 2 if QUICK else 40
FOLD_CFG = {0: dict(C=96, drop=0.2, lr=3e-4), 1: dict(C=128, drop=0.2, lr=1e-3), 2: dict(C=96, drop=0.2, lr=1e-3),
            3: dict(C=96, drop=0.2, lr=1e-3), 4: dict(C=128, drop=0.3, lr=1e-3)}   # stage-2 の fold 毎確定 config

genes = pd.read_csv(f"{T}/genes.csv", index_col=0)
P = np.log1p(np.load(f"{T}/prom.npy")); R = np.log1p(np.load(f"{T}/rna.npy"))
omics = pd.concat([pd.read_parquet(f"{F}/{n}.parquet") for n in ["cnv_gene_level", "mutation_gene_level", "methylation_gene_level"]], axis=1)
O = omics.reindex(genes.index).fillna(0.0).values.astype(np.float32)
O = np.hstack([O, np.log1p(genes.n_bins_rna.values[:, None]).astype(np.float32)])
y = genes.log2FC.values.astype(np.float32); fold = genes.fold.values

# pan-cancer target（LIHC 列除外）
pc = pd.read_csv(f"{ROOT}/data/pancan_data/transcriptome/pancanatlas_expid.txt", sep="\t", index_col=0)
pc.index = [g.split(".")[0] for g in pc.index]
pc = pc.drop(columns=["LIHC", "MedianExp"], errors="ignore")
pc = pc[~pc.index.duplicated(keep="first")]
common = genes.index.intersection(pc.index)
pos_in_genes = pd.Series(range(len(genes)), index=genes.index)
pre_pos = pos_in_genes.loc[common].values                       # tensor 行番号
Ypre_raw = pc.loc[common].values.astype(np.float32)             # [M, 31]
Ypre_raw = Ypre_raw - Ypre_raw.mean(axis=1, keepdims=True)      # 行中心化（癌種間の相対発現。絶対発現の shortcut 回避、事前指定）
print(f"pretrain genes: {len(common)} of {len(genes)} | targets {Ypre_raw.shape[1]} cancers | device {DEV}", flush=True)

class DWBranch(nn.Module):
    def __init__(self, nf, L, C, drop):
        super().__init__()
        self.a = nn.Sequential(nn.Conv1d(nf, nf, 5, padding=2, groups=nf), nn.GELU(),
                               nn.Conv1d(nf, C, 1), nn.GELU(), nn.Dropout(drop))
        self.pe = nn.Parameter(torch.zeros(C, L))
        self.b = nn.Sequential(nn.Conv1d(C, C, 3, padding=1), nn.GELU())
    def forward(self, x):
        return self.b(self.a(x) + self.pe)

class Model(nn.Module):
    def __init__(self, C, drop, n_out_pre=31):
        super().__init__()
        self.pb = DWBranch(550, 30, C, drop); self.rb = DWBranch(86, 32, C, drop)
        self.head = nn.Sequential(nn.LayerNorm(4 * C + O.shape[1]), nn.Dropout(drop),
                                  nn.Linear(4 * C + O.shape[1], 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))
        self.pre_head = nn.Linear(4 * C, n_out_pre)   # pretext 用（omics なし）
    def trunk(self, p, r):
        hp = self.pb(p); hr = self.rb(r)
        return torch.cat([hp.mean(-1), hp.amax(-1), hr.mean(-1), hr.amax(-1)], dim=1)
    def forward(self, p, r, o):
        return self.head(torch.cat([self.trunk(p, r), o], dim=1)).squeeze(-1)
    def forward_pre(self, p, r):
        return self.pre_head(self.trunk(p, r))

@torch.no_grad()
def predict(mdl, idx, Ostd):
    mdl.eval(); out = []
    for b in range(0, len(idx), 2048):
        sel = idx[b:b + 2048]
        out.append(mdl(torch.from_numpy(P[sel]).to(DEV), torch.from_numpy(R[sel]).to(DEV),
                       torch.from_numpy(Ostd[sel].astype(np.float32)).to(DEV)).cpu().numpy())
    return np.concatenate(out)

def pretrain(mdl, idx_pre, Ypre_std, val_mask=None, max_epoch=None, patience=5):
    """val_mask: idx_pre/Ypre_std の行に対する bool（染色体 val）。指定時は早期終了し best epoch を返す。"""
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=2.0)
    Yt = torch.from_numpy(Ypre_std)
    tr_rows = np.where(~val_mask)[0] if val_mask is not None else np.arange(len(idx_pre))
    best, best_ep, bad = np.inf, 0, 0
    n_ep = max_epoch or PRE_EPOCH
    for ep in range(1, n_ep + 1):
        mdl.train(); perm = np.random.permutation(len(tr_rows))
        for b in range(0, len(perm), BATCH):
            j = tr_rows[perm[b:b + BATCH]]; sel = idx_pre[j]
            p = torch.from_numpy(P[sel]).to(DEV); r = torch.from_numpy(R[sel]).to(DEV)
            opt.zero_grad(); lossf(mdl.forward_pre(p, r), Yt[j].to(DEV)).backward(); opt.step()
        if val_mask is not None:
            mdl.eval(); vr = np.where(val_mask)[0]
            with torch.no_grad():
                vl = 0.0
                for b in range(0, len(vr), 2048):
                    j = vr[b:b + 2048]; sel = idx_pre[j]
                    vl += float(lossf(mdl.forward_pre(torch.from_numpy(P[sel]).to(DEV), torch.from_numpy(R[sel]).to(DEV)), Yt[j].to(DEV))) * len(j)
                vl /= len(vr)
            if vl < best: best, best_ep, bad = vl, ep, 0
            else:
                bad += 1
                if bad >= patience: break
    return mdl, best_ep

def finetune(mdl, cfg, idx_tr, Ostd, idx_val=None, fixed_epochs=None):
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

oof = genes[["log2FC", "label", "chrom", "fold"]].copy(); log = {"pre_epoch": PRE_EPOCH, "runs": []}
preds = {s: np.full(len(genes), np.nan) for s in SEEDS}
for k in range(5):
    cfg = FOLD_CFG[k]
    tr_idx = np.where(fold != k)[0]; te_idx = np.where(fold == k)[0]
    mu, sd = O[tr_idx].mean(0), O[tr_idx].std(0) + 1e-8; Ostd = (O - mu) / sd
    tr_set = set(tr_idx); idx_pre = np.array([i for i in pre_pos if i in tr_set])   # 訓練染色体のみ
    sel_pre = np.isin(pre_pos, idx_pre)
    Ymu, Ysd = Ypre_raw[sel_pre].mean(0), Ypre_raw[sel_pre].std(0) + 1e-8
    Ypre_std = ((Ypre_raw - Ymu) / Ysd)[sel_pre]
    # --- pretext epoch を染色体 val（内側第1群）で決定（fold 毎 1 回、seed 0） ---
    torch.manual_seed(0); np.random.seed(0)
    g0 = sorted(set(fold[tr_idx]))[0]
    vmask = (fold[idx_pre] == g0)
    m0 = Model(cfg["C"], cfg["drop"]).to(DEV)
    _, pre_ep = pretrain(m0, idx_pre, Ypre_std, val_mask=vmask, max_epoch=(2 if QUICK else 60))
    pre_ep = max(pre_ep, 3)
    print(f"fold {k} pretext epochs (chrom-val) = {pre_ep}", flush=True)
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        t0 = time.time()
        # 内側回転で fine-tune epoch を決める（pretrain は回転の訓練側染色体のみ、epoch は fold 決定値で固定）
        eps = []
        for hold in sorted(set(fold[tr_idx]))[:2]:   # 回転 2 で epoch 推定（選択でなく epoch のみ）
            ia = tr_idx[fold[tr_idx] != hold]; iv = tr_idx[fold[tr_idx] == hold]
            ia_set = set(ia); ip = np.array([i for i in pre_pos if i in ia_set])
            sp = np.isin(pre_pos, ip)
            Ys = ((Ypre_raw - Ypre_raw[sp].mean(0)) / (Ypre_raw[sp].std(0) + 1e-8))[sp]
            m = Model(cfg["C"], cfg["drop"]).to(DEV)
            m, _ = pretrain(m, ip, Ys, max_epoch=pre_ep)
            _, s_in, ep = finetune(m, cfg, ia, Ostd, idx_val=iv)
            eps.append(ep)
        med_ep = max(int(np.median(eps)), 5)
        m = Model(cfg["C"], cfg["drop"]).to(DEV)
        m, _ = pretrain(m, idx_pre, Ypre_std, max_epoch=pre_ep)
        m, _, _ = finetune(m, cfg, tr_idx, Ostd, fixed_epochs=med_ep)
        preds[s][te_idx] = predict(m, te_idx, Ostd)
        log["runs"].append(dict(fold=k, seed=s, pre_ep=pre_ep, med_ep=med_ep, sec=round(time.time() - t0, 1)))
        print(f"fold {k} seed {s} pre_ep {pre_ep} ft_epochs {med_ep} ({time.time()-t0:.0f}s)", flush=True)
for s in SEEDS: oof[f"pretrained_seed{s}"] = preds[s]
oof["positional|depthwise_pe_pretrained"] = np.mean([preds[s] for s in SEEDS], axis=0)
oof.to_parquet(f"{OUT}/oof_dl_pretrained.parquet")
json.dump(log, open(f"{OUT}/pretrain_log.json", "w"), indent=1)
pa = oof["positional|depthwise_pe_pretrained"]
print("per-seed:", [round(float(spearmanr(y, preds[s])[0]), 4) for s in SEEDS])
print("pretrained (seed-avg if >1):", round(float(spearmanr(y, pa)[0]), 4))

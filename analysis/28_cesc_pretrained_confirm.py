#!/usr/bin/env python3
"""
28_cesc_pretrained_confirm.py — 事前学習版 ensemble の凍結 CESC 確証（notes/05 の凍結プロトコルに準拠）

規則（実行前に notes/05 に文書化済み）:
- CESC pretrained CNN（HeLa テンソル、pretext = pancanatlas 31 列（CESC 列除外・行中心化）、
  fold 毎 config は 23 の確定値、seed 1–3 平均）
- ens_pre = 0.5 × 既存 CESC all|xgboost OOF + 0.5 × pretrained CNN
- 成功 = 染色体 paired bootstrap で ens_pre − all|xgboost の 95% CI 下限 > 0

出力: results/exp4_dl/oof_dl_cesc_pretrained.parquet, ensemble_cesc_pretrained.json
実行: venv/bin/python analysis/28_cesc_pretrained_confirm.py [--quick]
"""
import os, sys, json, time, warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr
import torch, torch.nn as nn
warnings.filterwarnings("ignore")

QUICK = "--quick" in sys.argv
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; F = f"{ROOT}/results/features"
OUT = f"{ROOT}/results/exp4_dl"; CACHE = f"{OUT}/cesc_tensors"
DEV = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
BATCH = 512; MAX_EPOCH = 3 if QUICK else 150; PATIENCE = 10; SEEDS = [1, 2, 3]
FOLD_CFG = {0: dict(C=64, drop=0.1, lr=1e-3), 1: dict(C=96, drop=0.2, lr=3e-4), 2: dict(C=96, drop=0.2, lr=1e-3),
            3: dict(C=128, drop=0.3, lr=1e-3), 4: dict(C=64, drop=0.1, lr=1e-3)}   # 23 の確定値（凍結）

# ---- CESC 遺伝子集合・fold（23 と同一） ----
de = pd.read_csv(f"{ROOT}/results/de_cesc/de_results.csv").set_index("gene_id")
de = de[de.gene_type.isin(["protein_coding", "lncRNA"]) & de.log2FC.notna()]
ann = pd.read_parquet(f"{F}/gene_annotation_gencode26.parquet").set_index("gene_id")
bind = pd.read_parquet(f"{F}/binding_gene_level_hela.parquet")
genes_idx = de.index.intersection(bind.index).intersection(ann.index)
df = de.loc[genes_idx, ["log2FC", "label"]].join(ann.loc[genes_idx, ["chrom"]])
df = df[df.chrom.isin([f"chr{i}" for i in range(1, 23)] + ["chrX"])]
folds = pd.read_csv(f"{ROOT}/results/exp1_nested_cv_cesc/folds.csv").set_index("chrom")["fold"]
df["fold"] = df.chrom.map(folds)
P = np.log1p(np.load(f"{CACHE}/prom.npy")); R = np.log1p(np.load(f"{CACHE}/rna.npy"))
nbins = np.load(f"{CACHE}/nbins.npy")
NF_P, LP = P.shape[1], P.shape[2]; NF_R, LR = R.shape[1], R.shape[2]
omics = pd.concat([pd.read_parquet(f"{F}/{n}.parquet") for n in
                   ["cnv_gene_level_cesc", "mutation_gene_level_cesc", "methylation_gene_level_cesc"]], axis=1)
O = omics.reindex(df.index).fillna(0.0).values.astype(np.float32)
O = np.hstack([O, np.log1p(nbins[:, None]).astype(np.float32)])
y = df.log2FC.values.astype(np.float32); fold = df.fold.values
print(f"CESC genes {len(df)} | prom F={NF_P} rna F={NF_R} | dev {DEV}", flush=True)

# ---- pretext target（CESC 列除外・行中心化） ----
pc = pd.read_csv(f"{ROOT}/data/pancan_data/transcriptome/pancanatlas_expid.txt", sep="\t", index_col=0)
pc.index = [g.split(".")[0] for g in pc.index]
pc = pc.drop(columns=["CESC", "MedianExp"], errors="ignore")
pc = pc[~pc.index.duplicated(keep="first")]
common = df.index.intersection(pc.index)
pos_in = pd.Series(range(len(df)), index=df.index)
pre_pos = pos_in.loc[common].values
Ypre_raw = pc.loc[common].values.astype(np.float32)
Ypre_raw = Ypre_raw - Ypre_raw.mean(axis=1, keepdims=True)
print(f"pretrain genes {len(common)} | targets {Ypre_raw.shape[1]} cancers", flush=True)

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
    def __init__(self, C, drop, n_pre):
        super().__init__()
        self.pb = DWBranch(NF_P, LP, C, drop); self.rb = DWBranch(NF_R, LR, C, drop)
        self.head = nn.Sequential(nn.LayerNorm(4 * C + O.shape[1]), nn.Dropout(drop),
                                  nn.Linear(4 * C + O.shape[1], 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))
        self.pre_head = nn.Linear(4 * C, n_pre)
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

def pretrain(mdl, idx_pre, Ypre_std, val_mask=None, max_epoch=40, patience=5):
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=2.0); Yt = torch.from_numpy(Ypre_std)
    tr_rows = np.where(~val_mask)[0] if val_mask is not None else np.arange(len(idx_pre))
    best, best_ep, bad = np.inf, 0, 0
    for ep in range(1, max_epoch + 1):
        mdl.train(); perm = np.random.permutation(len(tr_rows))
        for b in range(0, len(perm), BATCH):
            j = tr_rows[perm[b:b + BATCH]]; sel = idx_pre[j]
            opt.zero_grad()
            lossf(mdl.forward_pre(torch.from_numpy(P[sel]).to(DEV), torch.from_numpy(R[sel]).to(DEV)),
                  Yt[j].to(DEV)).backward()
            opt.step()
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
            o = torch.from_numpy(Ostd[sel].astype(np.float32)).to(DEV)
            opt.zero_grad()
            lossf(mdl(torch.from_numpy(P[sel]).to(DEV), torch.from_numpy(R[sel]).to(DEV), o),
                  torch.from_numpy(y[sel]).to(DEV)).backward()
            opt.step()
        if idx_val is not None and fixed_epochs is None:
            s = spearmanr(y[idx_val], predict(mdl, idx_val, Ostd))[0]
            if s > best_s: best_s, best_ep, bad = s, ep, 0
            else:
                bad += 1
                if bad >= PATIENCE: break
    return mdl, best_s, best_ep

oof = df[["log2FC", "label", "chrom", "fold"]].copy()
preds = {s: np.full(len(df), np.nan) for s in SEEDS}
for k in range(5):
    cfg = FOLD_CFG[k]
    tr_idx = np.where(fold != k)[0]; te_idx = np.where(fold == k)[0]
    mu, sd = O[tr_idx].mean(0), O[tr_idx].std(0) + 1e-8; Ostd = (O - mu) / sd
    tr_set = set(tr_idx); idx_pre = np.array([i for i in pre_pos if i in tr_set])
    sel_pre = np.isin(pre_pos, idx_pre)
    Ypre_std = ((Ypre_raw - Ypre_raw[sel_pre].mean(0)) / (Ypre_raw[sel_pre].std(0) + 1e-8))[sel_pre]
    torch.manual_seed(0); np.random.seed(0)
    g0 = sorted(set(fold[tr_idx]))[0]
    _, pre_ep = pretrain(Model(cfg["C"], cfg["drop"], Ypre_raw.shape[1]).to(DEV), idx_pre, Ypre_std,
                         val_mask=(fold[idx_pre] == g0), max_epoch=(2 if QUICK else 60))
    pre_ep = max(pre_ep, 3)
    print(f"fold {k} pretext epochs = {pre_ep}", flush=True)
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        t0 = time.time(); eps = []
        for hold in sorted(set(fold[tr_idx]))[:2]:
            ia = tr_idx[fold[tr_idx] != hold]; iv = tr_idx[fold[tr_idx] == hold]
            ia_set = set(ia); ip = np.array([i for i in pre_pos if i in ia_set])
            sp = np.isin(pre_pos, ip)
            Ys = ((Ypre_raw - Ypre_raw[sp].mean(0)) / (Ypre_raw[sp].std(0) + 1e-8))[sp]
            m = Model(cfg["C"], cfg["drop"], Ypre_raw.shape[1]).to(DEV)
            m, _ = pretrain(m, ip, Ys, max_epoch=pre_ep)
            _, _, ep = finetune(m, cfg, ia, Ostd, idx_val=iv)
            eps.append(ep)
        med_ep = max(int(np.median(eps)), 5)
        m = Model(cfg["C"], cfg["drop"], Ypre_raw.shape[1]).to(DEV)
        m, _ = pretrain(m, idx_pre, Ypre_std, max_epoch=pre_ep)
        m, _, _ = finetune(m, cfg, tr_idx, Ostd, fixed_epochs=med_ep)
        preds[s][te_idx] = predict(m, te_idx, Ostd)
        print(f"fold {k} seed {s} pre_ep {pre_ep} ft {med_ep} ({time.time()-t0:.0f}s)", flush=True)
for s in SEEDS: oof[f"pre_seed{s}"] = preds[s]
oof["cesc_pretrained"] = np.mean([preds[s] for s in SEEDS], axis=0)
oof.to_parquet(f"{OUT}/oof_dl_cesc_pretrained.parquet")

# ---- 凍結判定 ----
o1 = pd.read_parquet(f"{ROOT}/results/exp1_nested_cv_cesc/oof_predictions.parquet")
prev = pd.read_parquet(f"{OUT}/oof_dl_cesc.parquet")   # 23 の非事前学習 CNN（副次比較用）
m = o1[["log2FC", "chrom", "all|xgboost"]].join(oof[["cesc_pretrained"]], how="inner").join(
    prev[["positional|depthwise_pe"]].rename(columns={"positional|depthwise_pe": "cesc_rand"}), how="left").dropna()
yv = m.log2FC.values; ch = m.chrom.unique()
m["ens_pre"] = (m["all|xgboost"] + m.cesc_pretrained) / 2
m["ens_rand"] = (m["all|xgboost"] + m.cesc_rand) / 2
rng = np.random.default_rng(0)
cols = ["all|xgboost", "cesc_pretrained", "cesc_rand", "ens_pre", "ens_rand"]
boot = {c: [] for c in cols}
for b in range(2000):
    samp = rng.choice(ch, len(ch), replace=True)
    idx = np.concatenate([np.where(m.chrom.values == c)[0] for c in samp])
    for c in cols: boot[c].append(spearmanr(yv[idx], m[c].values[idx])[0])
B = pd.DataFrame(boot)
res = {"point": {c: round(float(spearmanr(yv, m[c])[0]), 4) for c in cols}}
for a, bq in [("ens_pre", "all|xgboost"), ("ens_pre", "ens_rand"), ("cesc_pretrained", "cesc_rand"),
              ("cesc_pretrained", "all|xgboost")]:
    res[f"{a} - {bq}"] = (B[a] - B[bq]).quantile([0.025, 0.5, 0.975]).round(4).tolist()
res["frozen_success"] = bool(res["ens_pre - all|xgboost"][0] > 0)
json.dump(res, open(f"{OUT}/ensemble_cesc_pretrained.json", "w"), indent=1)
print(json.dumps(res, indent=1))

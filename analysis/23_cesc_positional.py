#!/usr/bin/env python3
"""
23_cesc_positional.py — 未使用 CESC での凍結 ensemble 規則の確証実験（codex R4-3 の最優先）

事前凍結した規則（LIHC の結果を見た後、CESC の DL 結果を見る前に固定）:
  ens = 0.5 * all|xgboost(CESC, 既存 exp1_nested_cv_cesc) + 0.5 * positional depthwise_pe CNN(CESC)
  成功 = CESC の染色体 paired bootstrap で ens − all|xgboost の 95% CI 下限 > 0

手順:
1. HeLa 位置 pkl（prom 62因子×30bin、rna 31因子×可変→32bin pool）を CESC の CV 遺伝子集合でテンソル化
2. LIHC stage-2 と同一プロトコル（5 config・内側4回転・median epoch）で pe CNN を nested CV
3. 凍結規則の ensemble を評価（±対照: DL 単体 vs ①）
出力: results/exp4_dl/cesc_{tensors cache}/, oof_dl_cesc.parquet, ensemble_cesc.json
実行: venv/bin/python analysis/23_cesc_positional.py
"""
import os, json, time, warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr
import torch, torch.nn as nn
warnings.filterwarnings("ignore")
torch.manual_seed(0); np.random.seed(0)

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; F = f"{ROOT}/results/features"
OUT = f"{ROOT}/results/exp4_dl"; CACHE = f"{OUT}/cesc_tensors"; os.makedirs(CACHE, exist_ok=True)
DEV = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
BATCH = 512; MAX_EPOCH = 150; PATIENCE = 10; RNA_BINS = 32

# ---- CESC の CV 遺伝子集合と fold（exp1_nested_cv_cesc と同一） ----
de = pd.read_csv(f"{ROOT}/results/de_cesc/de_results.csv").set_index("gene_id")
de = de[de.gene_type.isin(["protein_coding", "lncRNA"]) & de.log2FC.notna()]
ann = pd.read_parquet(f"{F}/gene_annotation_gencode26.parquet").set_index("gene_id")
bind = pd.read_parquet(f"{F}/binding_gene_level_hela.parquet")
genes = de.index.intersection(bind.index).intersection(ann.index)
df = de.loc[genes, ["log2FC", "label"]].join(ann.loc[genes, ["chrom"]])
df = df[df.chrom.isin([f"chr{i}" for i in range(1, 23)] + ["chrX"])]
folds = pd.read_csv(f"{ROOT}/results/exp1_nested_cv_cesc/folds.csv").set_index("chrom")["fold"]
df["fold"] = df.chrom.map(folds)
assert df.fold.notna().all()
print("CESC genes:", len(df), flush=True)

# ---- HeLa 位置テンソル（キャッシュ） ----
if not os.path.exists(f"{CACHE}/prom.npy"):
    def load(kind):
        d = pd.read_pickle(f"{ROOT}/data/{kind}_features/encode_hela_{kind}.pkl")
        gid = d["Name"].str.split(".").str[0].values
        first = {}
        for i, g in enumerate(gid):
            if g not in first: first[g] = i
        return d, first
    target = df.index.tolist()
    dp, fp = load("promoter")
    nf_p = dp["feature_vec"][0].shape[0]
    P = np.zeros((len(target), nf_p, 30), dtype=np.float32)
    for j, g in enumerate(target):
        P[j] = np.asarray(dp["feature_vec"][fp[g]].todense(), dtype=np.float32)
    np.save(f"{CACHE}/prom.npy", P); del dp
    dr, fr = load("rna")
    nf_r = dr["feature_vec"][0].shape[0]
    R = np.zeros((len(target), nf_r, RNA_BINS), dtype=np.float32)
    nbins = np.zeros(len(target), dtype=np.int32)
    for j, g in enumerate(target):
        m = np.asarray(dr["feature_vec"][fr[g]].todense(), dtype=np.float32); L = m.shape[1]; nbins[j] = L
        if L == RNA_BINS: R[j] = m
        else:
            edges = np.linspace(0, L, RNA_BINS + 1)
            for b in range(RNA_BINS):
                lo, hi = int(np.floor(edges[b])), max(int(np.ceil(edges[b + 1])), int(np.floor(edges[b])) + 1)
                R[j, :, b] = m[:, lo:hi].mean(axis=1)
    np.save(f"{CACHE}/rna.npy", R); np.save(f"{CACHE}/nbins.npy", nbins)
P = np.log1p(np.load(f"{CACHE}/prom.npy")); R = np.log1p(np.load(f"{CACHE}/rna.npy"))
nbins = np.load(f"{CACHE}/nbins.npy")
NF_P, NF_R = P.shape[1], R.shape[1]
print("tensors:", P.shape, R.shape, flush=True)

omics = pd.concat([pd.read_parquet(f"{F}/{n}.parquet") for n in
                   ["cnv_gene_level_cesc", "mutation_gene_level_cesc", "methylation_gene_level_cesc"]], axis=1)
O = omics.reindex(df.index).fillna(0.0).values.astype(np.float32)
O = np.hstack([O, np.log1p(nbins[:, None]).astype(np.float32)])
y = df.log2FC.values.astype(np.float32); fold = df.fold.values

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
        self.pb = DWBranch(NF_P, 30, C, drop); self.rb = DWBranch(NF_R, RNA_BINS, C, drop)
        self.head = nn.Sequential(nn.LayerNorm(4 * C + O.shape[1]), nn.Dropout(drop),
                                  nn.Linear(4 * C + O.shape[1], 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))
    def forward(self, p, r, o):
        hp = self.pb(p); hr = self.rb(r)
        return self.head(torch.cat([hp.mean(-1), hp.amax(-1), hr.mean(-1), hr.amax(-1), o], dim=1)).squeeze(-1)

CONFIGS = [dict(C=64, drop=0.1, lr=1e-3), dict(C=96, drop=0.2, lr=1e-3), dict(C=128, drop=0.3, lr=1e-3),
           dict(C=96, drop=0.2, lr=3e-4), dict(C=128, drop=0.2, lr=1e-3)]

@torch.no_grad()
def predict(mdl, idx, Ostd):
    mdl.eval(); out = []
    for b in range(0, len(idx), 2048):
        sel = idx[b:b + 2048]
        out.append(mdl(torch.from_numpy(P[sel]).to(DEV), torch.from_numpy(R[sel]).to(DEV),
                       torch.from_numpy(Ostd[sel].astype(np.float32)).to(DEV)).cpu().numpy())
    return np.concatenate(out)

def run_train(cfg, idx_tr, Ostd, idx_val=None, fixed_epochs=None):
    mdl = Depthwise(cfg["C"], cfg["drop"]).to(DEV)
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

pred = np.full(len(df), np.nan)
for k in range(5):
    tr_idx = np.where(fold != k)[0]; te_idx = np.where(fold == k)[0]
    mu, sd = O[tr_idx].mean(0), O[tr_idx].std(0) + 1e-8; Ostd = (O - mu) / sd
    best, best_s, eps = None, -np.inf, []
    for cfg in CONFIGS:
        scores, e = [], []
        for hold in sorted(set(fold[tr_idx])):
            ia = tr_idx[fold[tr_idx] != hold]; iv = tr_idx[fold[tr_idx] == hold]
            _, s, ep = run_train(cfg, ia, Ostd, idx_val=iv)
            scores.append(s); e.append(ep)
        s = float(np.nanmean(scores))
        print(f"fold {k} {cfg} inner {s:.3f}", flush=True)
        if s > best_s: best, best_s, eps = cfg, s, e
    med_ep = max(int(np.median(eps)), 5)
    mdl, _, _ = run_train(best, tr_idx, Ostd, fixed_epochs=med_ep)
    pred[te_idx] = predict(mdl, te_idx, Ostd)
    print(f"fold {k} FINAL {best} epochs {med_ep}", flush=True)

oof = df[["log2FC", "label", "chrom", "fold"]].copy(); oof["positional|depthwise_pe"] = pred
oof.to_parquet(f"{OUT}/oof_dl_cesc.parquet")

# ---- 凍結規則の ensemble 評価 ----
o1 = pd.read_parquet(f"{ROOT}/results/exp1_nested_cv_cesc/oof_predictions.parquet")
m = o1[["log2FC", "chrom", "all|xgboost", "all|elasticnet"]].join(oof[["positional|depthwise_pe"]], how="inner").dropna()
yv = m.log2FC.values; chroms = m.chrom.unique()
m["ens_dl"] = (m["all|xgboost"] + m["positional|depthwise_pe"]) / 2
m["ens_enet"] = (m["all|xgboost"] + m["all|elasticnet"]) / 2
rng = np.random.default_rng(0)
cols = ["all|xgboost", "positional|depthwise_pe", "ens_dl", "ens_enet"]
boot = {c: [] for c in cols}
for b in range(2000):
    samp = rng.choice(chroms, len(chroms), replace=True)
    idx = np.concatenate([np.where(m.chrom.values == c)[0] for c in samp])
    for c in cols: boot[c].append(spearmanr(yv[idx], m[c].values[idx])[0])
B = pd.DataFrame(boot)
res = {"point": {c: round(float(spearmanr(yv, m[c])[0]), 4) for c in cols}}
for a, bq in [("ens_dl", "all|xgboost"), ("ens_dl", "ens_enet"), ("positional|depthwise_pe", "all|xgboost")]:
    res[f"{a} - {bq}"] = (B[a] - B[bq]).quantile([0.025, 0.5, 0.975]).round(4).tolist()
json.dump(res, open(f"{OUT}/ensemble_cesc.json", "w"), indent=1)
print(json.dumps(res, indent=1))

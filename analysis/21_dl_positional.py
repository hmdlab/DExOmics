#!/usr/bin/env python3
"""
21_dl_positional.py — 2×2 比較の ④「位置分解入力 × DL」（PyTorch MPS）

候補 2 アーキテクチャ・計 9 config（codex R1 の予算合意: DL 側の総探索は XGBoost の 9 点と同数）:
- depthwise (5 config): 因子=channel の depthwise Conv1d(位置方向) → pointwise 混合 → GAP+GMP。
  prom [550,30] / rna [86,32] 別 encoder + omics head。~0.1-0.4M params。
- factorattn (4 config): 全因子共有の小型 track encoder + 因子 embedding + attention pooling。~0.2-0.7M。

プロトコル（06 と同一の公平性要件）:
- 外側 5 fold は folds.csv。config 選択と早期終了 epoch は内側 fold のみで決定
  （--inner-rot N で内側回転数を制御。最終モデルは外側訓練全体で median(best_epoch) だけ学習、test は不可視）。
- 入力 log1p、omics は訓練側統計で標準化。AdamW(wd=1e-4)、batch 512、patience 10。

出力: results/exp4_dl/oof_dl.parquet, metrics_dl.csv, dl_training_log.json
実行: venv/bin/python analysis/21_dl_positional.py [--quick] [--inner-rot 4] [--arch depthwise|factorattn|all]
"""
import os, sys, json, time, warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
import torch, torch.nn as nn
warnings.filterwarnings("ignore")
torch.manual_seed(0); np.random.seed(0)

QUICK = "--quick" in sys.argv
INNER_ROT = int(sys.argv[sys.argv.index("--inner-rot") + 1]) if "--inner-rot" in sys.argv else 4
ARCH = sys.argv[sys.argv.index("--arch") + 1] if "--arch" in sys.argv else "all"
DEV = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; T = f"{ROOT}/results/tensors"; F = f"{ROOT}/results/features"
OUT = f"{ROOT}/results/exp4_dl"; os.makedirs(OUT, exist_ok=True)
MAX_EPOCH = 3 if QUICK else 150; PATIENCE = 10; BATCH = 512

genes = pd.read_csv(f"{T}/genes.csv", index_col=0)
P = np.log1p(np.load(f"{T}/prom.npy"))
R = np.log1p(np.load(f"{T}/rna.npy"))
omics = pd.concat([pd.read_parquet(f"{F}/{n}.parquet") for n in ["cnv_gene_level", "mutation_gene_level", "methylation_gene_level"]], axis=1)
O = omics.reindex(genes.index).fillna(0.0).values.astype(np.float32)
O = np.hstack([O, np.log1p(genes.n_bins_rna.values[:, None]).astype(np.float32)])
if "--hybrid" in sys.argv:   # ④': 集約 636 列を head 側に併給（③' と対称）
    _bind = pd.read_parquet(f"{F}/binding_gene_level.parquet")
    A = np.log1p(_bind.reindex(genes.index).drop(columns=["n_bins_prom", "n_bins_rna"]).values.astype(np.float32))
    O = np.hstack([O, A])
y = genes.log2FC.values.astype(np.float32)
fold = genes.fold.values
print(f"prom {P.shape} rna {R.shape} omics {O.shape} device {DEV}", flush=True)

class DWBranch(nn.Module):
    """depthwise→pointwise→(+位置embedding)→conv。use_pe=True で絶対位置を学習可能にする（codex R2-1 対応）"""
    def __init__(self, nf, L, C, drop, use_pe):
        super().__init__()
        self.a = nn.Sequential(nn.Conv1d(nf, nf, 5, padding=2, groups=nf), nn.GELU(),
                               nn.Conv1d(nf, C, 1), nn.GELU(), nn.Dropout(drop))
        self.pe = nn.Parameter(torch.zeros(C, L)) if use_pe else None
        self.b = nn.Sequential(nn.Conv1d(C, C, 3, padding=1), nn.GELU())
    def forward(self, x):
        h = self.a(x)
        if self.pe is not None: h = h + self.pe
        return self.b(h)

class Depthwise(nn.Module):
    def __init__(self, C=96, drop=0.2, use_pe=False):
        super().__init__()
        self.pb = DWBranch(550, 30, C, drop, use_pe); self.rb = DWBranch(86, 32, C, drop, use_pe)
        self.head = nn.Sequential(nn.LayerNorm(4 * C + O.shape[1]), nn.Dropout(drop),
                                  nn.Linear(4 * C + O.shape[1], 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))
    def forward(self, p, r, o):
        hp = self.pb(p); hr = self.rb(r)
        z = torch.cat([hp.mean(-1), hp.amax(-1), hr.mean(-1), hr.amax(-1), o], dim=1)
        return self.head(z).squeeze(-1)

class FactorAttn(nn.Module):
    def __init__(self, E=48, drop=0.2):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(1, 8, 5, padding=2), nn.GELU(), nn.Conv1d(8, 16, 3, padding=1), nn.GELU())
        self.proj = nn.Linear(32, E)
        self.emb = nn.Embedding(550 + 86, E)
        self.q = nn.Parameter(torch.randn(E) * 0.1)
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.LayerNorm(E + O.shape[1]), nn.Dropout(drop),
                                  nn.Linear(E + O.shape[1], 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))
        self.register_buffer("fid", torch.arange(550 + 86))
    def encode(self, x):
        B, NF, L = x.shape
        h = self.enc(x.reshape(B * NF, 1, L))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)    # [B*NF, 32]
        return self.proj(h).reshape(B, NF, -1)
    def forward(self, p, r, o):
        tok = torch.cat([self.encode(p), self.encode(r)], dim=1) + self.emb(self.fid)[None]
        a = torch.softmax((tok * self.q).sum(-1) / np.sqrt(tok.shape[-1]), dim=1)
        z = self.drop((tok * a[..., None]).sum(1))
        return self.head(torch.cat([z, o], dim=1)).squeeze(-1)

CONFIGS = []
if "--hybrid" in sys.argv:
    for C, drop, lr in [(64, 0.1, 1e-3), (96, 0.2, 1e-3), (128, 0.3, 1e-3), (96, 0.2, 3e-4), (128, 0.2, 1e-3)]:
        CONFIGS.append(dict(arch="depthwise_pe", C=C, drop=drop, lr=lr))
elif "--stage2" in sys.argv:
    # stage-2（事後改善、budget は別掲で透明化）: 位置 embedding 版 depthwise 5 config
    for C, drop, lr in [(64, 0.1, 1e-3), (96, 0.2, 1e-3), (128, 0.3, 1e-3), (96, 0.2, 3e-4), (128, 0.2, 1e-3)]:
        CONFIGS.append(dict(arch="depthwise_pe", C=C, drop=drop, lr=lr))
else:
    for C, drop, lr in [(64, 0.1, 1e-3), (96, 0.2, 1e-3), (128, 0.3, 1e-3), (96, 0.2, 3e-4), (128, 0.1, 3e-4)]:
        CONFIGS.append(dict(arch="depthwise", C=C, drop=drop, lr=lr))
    for E, drop, lr in [(32, 0.1, 1e-3), (48, 0.2, 1e-3), (64, 0.3, 1e-3), (48, 0.2, 3e-4)]:
        CONFIGS.append(dict(arch="factorattn", E=E, drop=drop, lr=lr))
if ARCH != "all": CONFIGS = [c for c in CONFIGS if c["arch"] == ARCH]
if QUICK: CONFIGS = CONFIGS[:2]
SUFFIX = "_hybrid" if "--hybrid" in sys.argv else ("_stage2" if "--stage2" in sys.argv else "")

def build(cfg):
    if cfg["arch"].startswith("depthwise"):
        return Depthwise(cfg["C"], cfg["drop"], use_pe=cfg["arch"] == "depthwise_pe").to(DEV)
    return FactorAttn(cfg["E"], cfg["drop"]).to(DEV)

def std_omics(idx_tr):
    mu, sd = O[idx_tr].mean(0), O[idx_tr].std(0) + 1e-8
    return (O - mu) / sd

def run_train(cfg, idx_tr, idx_val=None, max_epoch=MAX_EPOCH, fixed_epochs=None):
    Ostd = std_omics(idx_tr)
    def tens(idx):
        return (torch.from_numpy(P[idx]), torch.from_numpy(R[idx]), torch.from_numpy(Ostd[idx].astype(np.float32)))
    mdl = build(cfg); opt = torch.optim.AdamW(mdl.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=2.0)
    ytr = torch.from_numpy(y[idx_tr])
    best_s, best_ep, bad = -np.inf, 0, 0
    n_ep = fixed_epochs or max_epoch
    for ep in range(1, n_ep + 1):
        mdl.train(); perm = np.random.permutation(len(idx_tr))
        for b in range(0, len(perm), BATCH):
            sel = idx_tr[perm[b:b + BATCH]]
            p, r, o = [t.to(DEV) for t in tens(sel)]
            opt.zero_grad(); loss = lossf(mdl(p, r, o), torch.from_numpy(y[sel]).to(DEV)); loss.backward(); opt.step()
        if idx_val is not None and fixed_epochs is None:
            s = spearmanr(y[idx_val], predict(mdl, idx_val, Ostd))[0]
            if s > best_s: best_s, best_ep, bad = s, ep, 0
            else:
                bad += 1
                if bad >= PATIENCE: break
    return mdl, Ostd, best_s, best_ep

@torch.no_grad()
def predict(mdl, idx, Ostd):
    mdl.eval(); out = []
    for b in range(0, len(idx), 2048):
        sel = idx[b:b + 2048]
        p = torch.from_numpy(P[sel]).to(DEV); r = torch.from_numpy(R[sel]).to(DEV)
        o = torch.from_numpy(Ostd[sel].astype(np.float32)).to(DEV)
        out.append(mdl(p, r, o).cpu().numpy())
    return np.concatenate(out)

oof = genes[["log2FC", "label", "chrom", "fold"]].copy()
log = {"configs": CONFIGS, "inner_rot": INNER_ROT, "runs": []}
pred_by_arch = {a: np.full(len(genes), np.nan) for a in {c["arch"] for c in CONFIGS}}
for k in range(5):
    tr_idx = np.where(fold != k)[0]; te_idx = np.where(fold == k)[0]
    inner_groups = sorted(set(fold[tr_idx]))[:INNER_ROT]
    per_arch = {}
    for ci, cfg in enumerate(CONFIGS):
        t0 = time.time(); scores, eps = [], []
        for hold in inner_groups:
            ia = tr_idx[fold[tr_idx] != hold]; iv = tr_idx[fold[tr_idx] == hold]
            _, _, s, ep = run_train(cfg, ia, iv)
            scores.append(s); eps.append(ep)
        s = float(np.nanmean(scores)); med_ep = int(np.median(eps))
        log["runs"].append(dict(fold=k, cfg=cfg, inner=round(s, 4), med_epoch=med_ep, sec=round(time.time() - t0, 1)))
        print(f"fold {k} {cfg} inner {s:.3f} med_ep {med_ep} ({time.time()-t0:.0f}s)", flush=True)
        a = cfg["arch"]
        if a not in per_arch or s > per_arch[a][1]: per_arch[a] = (cfg, s, med_ep)
    for a, (cfg, s, med_ep) in per_arch.items():
        mdl, Ostd, _, _ = run_train(cfg, tr_idx, None, fixed_epochs=max(med_ep, 5))
        pred_by_arch[a][te_idx] = predict(mdl, te_idx, Ostd)
        print(f"fold {k} FINAL {a} cfg {cfg} epochs {med_ep}", flush=True)

records = []
for a, pred in pred_by_arch.items():
    col = f"{'hybrid' if '--hybrid' in sys.argv else 'positional'}|{a}"; oof[col] = pred
    ok = ~np.isnan(pred)
    r = dict(model=col, spearman=float(spearmanr(y[ok], pred[ok])[0]),
             rmse=float(np.sqrt(np.mean((y[ok] - pred[ok]) ** 2))),
             prauc_up=float(average_precision_score(genes.label[ok] == "UP", pred[ok])),
             prauc_down=float(average_precision_score(genes.label[ok] == "DOWN", -pred[ok])))
    records.append(r); print(f"{col:24s} spearman {r['spearman']:.3f} rmse {r['rmse']:.3f}", flush=True)
oof.to_parquet(f"{OUT}/oof_dl{SUFFIX}.parquet"); pd.DataFrame(records).to_csv(f"{OUT}/metrics_dl{SUFFIX}.csv", index=False)
json.dump(log, open(f"{OUT}/dl_training_log{SUFFIX}.json", "w"), indent=1)
print("saved")

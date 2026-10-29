#!/usr/bin/env python3
"""
10_direction_diagnosis.py — 方向別（UP vs DOWN）性能診断（codex Round 4-3 設計、レビュー反映版）

問い: 「DOWN 方向は UP より説明が難しい」は、件数・発現量・効果量・特徴被覆を揃えても残るか。

設計（2026-09-20 codex レビュー反映）:
- 06 の OOF 予測を使用（追加学習なし）。
- 各 DOWN 遺伝子に UP 遺伝子を 1:1 マッチング。共変量 = log10(baseMean)、|log2FC|、結合特徴被覆率。
  標準化は UP∪DOWN のみで実施。**最適割当（linear_sum_assignment、最小総距離）**＋
  caliper（z 距離 > CALIPER の組を除外）。除外数と共変量ごとの SMD（マッチ前後）を出力。
- CI はマッチ済みペアの bootstrap（2,000 resample）。指標: MAE、符号正解率、
  balanced PR-AUC（DOWN: 正例=DOWN・score=-pred / UP: 正例=UP・score=pred）と方向差。

入力: results/exp1_nested_cv/oof_predictions.parquet、results/de_lihc/de_results.csv、
      results/features/binding_gene_level.parquet
出力: results/exp3_direction/diagnosis.csv, matched_pairs.csv, balance.csv
実行: venv/bin/python analysis/10_direction_diagnosis.py [--model "all|xgboost"]
"""
import os, sys, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from scipy.optimize import linear_sum_assignment

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT
OUT = f"{ROOT}/results/exp3_direction"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
CALIPER = float(sys.argv[sys.argv.index("--caliper") + 1]) if "--caliper" in sys.argv else 0.5   # z 距離
N_BOOT = 2000

oof = pd.read_parquet(f"{ROOT}/results/exp1_nested_cv/oof_predictions.parquet")
mlp_p = f"{ROOT}/results/exp1_nested_cv/oof_mlp.parquet"
if os.path.exists(mlp_p):
    mlp = pd.read_parquet(mlp_p)
    oof = oof.join(mlp[[c for c in mlp.columns if c.endswith("|mlp")]], how="left")
de = pd.read_csv(f"{ROOT}/results/de_lihc/de_results.csv").set_index("gene_id")
bind = pd.read_parquet(f"{ROOT}/results/features/binding_gene_level.parquet")
cov_bind = ((bind.filter(like="prom__") > 0).mean(axis=1) + (bind.filter(like="rna__") > 0).mean(axis=1)) / 2

df = oof.join(de[["baseMean"]], how="left")
df["coverage"] = cov_bind.reindex(df.index).fillna(0.0)
df = df.dropna(subset=["baseMean"])
model_cols = [c for c in oof.columns if "|" in c and c != "mean|baseline"]
if "--model" in sys.argv:
    model_cols = [sys.argv[sys.argv.index("--model") + 1]]

sub = df[df.label.isin(["UP", "DOWN"])]
C = pd.DataFrame({"e": np.log10(sub.baseMean + 1), "f": sub.log2FC.abs(), "c": sub.coverage}, index=sub.index)
Z = (C - C.mean()) / C.std()          # UP∪DOWN のみで標準化
down_idx = sub.index[sub.label == "DOWN"]; up_idx = sub.index[sub.label == "UP"]
print(f"DOWN {len(down_idx)} | UP {len(up_idx)} | models {model_cols}")

# 最適 1:1 割当（最小総距離）→ caliper 適用
D = np.linalg.norm(Z.loc[down_idx].values[:, None, :] - Z.loc[up_idx].values[None, :, :], axis=2)
ri, ci_ = linear_sum_assignment(D)
dist = D[ri, ci_]
keep = dist <= CALIPER
tag = "" if CALIPER == 0.5 else f"_caliper{CALIPER}"   # caliper 別に保存（上書き事故防止）
pairs = pd.DataFrame({"down": down_idx[ri[keep]], "up": up_idx[ci_[keep]], "distance": dist[keep]})
pairs.to_csv(f"{OUT}/matched_pairs{tag}.csv", index=False)
print(f"matched pairs {len(pairs)} | excluded by caliper {int((~keep).sum())} of {len(ri)}")

def smd(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
bal = []
for k in ["e", "f", "c"]:
    bal.append(dict(covariate=k,
                    smd_before=float(smd(C.loc[down_idx, k], C.loc[up_idx, k])),
                    smd_after=float(smd(C.loc[pairs.down, k].values, C.loc[pairs.up, k].values))))
bal = pd.DataFrame(bal); bal.to_csv(f"{OUT}/balance{tag}.csv", index=False)
print(bal.round(3).to_string(index=False))

rows = []
for mcol in model_cols:
    pd_, pu = df.loc[pairs.down, mcol].values, df.loc[pairs.up, mcol].values
    ad, au = df.loc[pairs.down, "log2FC"].values, df.loc[pairs.up, "log2FC"].values
    n = len(pairs)
    def metrics(idx):
        y = np.r_[np.ones(len(idx)), np.zeros(len(idx))]
        sc = np.r_[-pd_[idx], -pu[idx]]
        return dict(mae_down=float(np.abs(pd_[idx] - ad[idx]).mean()), mae_up=float(np.abs(pu[idx] - au[idx]).mean()),
                    sign_down=float((np.sign(pd_[idx]) == np.sign(ad[idx])).mean()), sign_up=float((np.sign(pu[idx]) == np.sign(au[idx])).mean()),
                    prauc_down=float(average_precision_score(y, sc)), prauc_up=float(average_precision_score(1 - y, -sc)))
    point = metrics(np.arange(n))
    boots = [metrics(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    bt = pd.DataFrame(boots)
    row = {"model": mcol, "n_pairs": n, **point}
    for k in ["mae", "sign", "prauc"]:
        gap = bt[f"{k}_down"] - bt[f"{k}_up"]
        row[f"{k}_gap"] = point[f"{k}_down"] - point[f"{k}_up"]
        row[f"{k}_gap_lo"], row[f"{k}_gap_hi"] = float(gap.quantile(0.025)), float(gap.quantile(0.975))
    rows.append(row)
    print(f"{mcol:22s} pairs {n} | MAE down {point['mae_down']:.3f} vs up {point['mae_up']:.3f} (gap CI [{row['mae_gap_lo']:.3f},{row['mae_gap_hi']:.3f}]) | "
          f"sign down {point['sign_down']:.3f} vs up {point['sign_up']:.3f} | PR-AUC down {point['prauc_down']:.3f} vs up {point['prauc_up']:.3f}", flush=True)
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/diagnosis{tag}.csv", index=False)
print("saved:", f"{OUT}/diagnosis{tag}.csv")

#!/usr/bin/env python3
"""
15_calibration_probe.py — 方向別バイアスは post-hoc 較正で救えるかの最小プローブ

実験 3 の所見「識別力は同等（balanced PR-AUC）だが予測が正方向にシフト」を受け、
外側 fold ごとに OOF 予測を quantile mapping（当該 fold の外側訓練集合の実測 logFC 分布へ写像）
した場合の符号正解率・MAE の変化を見る。写像は訓練側の分布のみで作るためリークなし。

出力: results/exp3_direction/calibration_probe.csv
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT
OUT = f"{ROOT}/results/exp3_direction"; os.makedirs(OUT, exist_ok=True)
oof = pd.read_parquet(f"{ROOT}/results/exp1_nested_cv/oof_predictions.parquet")
y = oof.log2FC.values
rows = []
for mcol in ["all|xgboost", "all|elasticnet", "binding|xgboost"]:
    raw = oof[mcol].values.copy()
    cal = np.full_like(raw, np.nan)
    for k in sorted(oof.fold.unique()):
        te = (oof.fold == k).values; tr = ~te
        ytr = np.sort(y[tr])
        r = pd.Series(raw[te]).rank(pct=True).values          # OOF 予測の fold 内順位
        cal[te] = np.quantile(ytr, np.clip(r, 0, 1))          # 訓練実測分布の同順位点へ写像
    # isotonic 較正（codex R5-2 レシピの近似: 各外側 fold の較正器は「他 4 fold の OOF (pred, actual)」で学習。
    # 厳密な cross-fitted 訓練内予測ではなく OOF ベースの近似である旨を明記）
    from sklearn.isotonic import IsotonicRegression
    iso = np.full_like(raw, np.nan)
    for k in sorted(oof.fold.unique()):
        te = (oof.fold == k).values; tr = ~te
        ir = IsotonicRegression(out_of_bounds="clip").fit(raw[tr], y[tr])
        iso[te] = ir.predict(raw[te])
    for name, p in [("raw", raw), ("quantile_mapped", cal), ("isotonic_crossfold", iso)]:
        m = {}
        for lab in ["DOWN", "UP", "NonDEG"]:
            s = (oof.label == lab).values
            m[f"sign_{lab.lower()}"] = float((np.sign(p[s]) == np.sign(y[s])).mean())
            m[f"mae_{lab.lower()}"] = float(np.abs(p[s] - y[s]).mean())
        rows.append(dict(model=mcol, variant=name, spearman=float(spearmanr(y, p)[0]),
                         frac_pred_negative=float((p < 0).mean()), **m))
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/calibration_probe.csv", index=False)
print(res.round(3).to_string(index=False))

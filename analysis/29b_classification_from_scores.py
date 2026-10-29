#!/usr/bin/env python3
"""
29b_classification_from_scores.py — 3-class (DOWN / NonDEG / UP) evaluation of the regression OOF scores
("(a) regression -> thresholds" in notes/05; counterpart of the direct classifiers in 29).

Re-implements the inline analysis that produced results/exp4_dl/classification_from_scores.csv:
  * models: XGBoost(5seed) = all|xgboost_seedavg, CNN pretrained(3seed) = positional|depthwise_pe_pretrained,
            Ensemble = their 0.5/0.5 average,
  * threshold-free metrics on the continuous score: PR-AUC / AUROC for UP (score = prediction) and for
    DOWN (score = -prediction),
  * class assignment with two thresholds (t_down < t_up: pred < t_down -> DOWN, pred > t_up -> UP),
    calibrated cross-fold: the thresholds applied to outer fold k maximise macro-F1 on the OOF predictions
    of the other four folds, so fold k's own predictions never inform its thresholds; grid of 0.02 steps on
    [-1.5, 1.5]. Caveat (same procedure as the 2026-09-21 analysis, kept for comparability): the OOF models
    of the other folds were trained on data that include fold k, so the threshold choice is not fully
    independent of fold k's labels. A strictly nested variant would need inner-fold OOF predictions.
  * macro-F1 and per-class recall of the resulting assignment.
If results/exp4_dl/oof_classification.parquet (from 29) exists, the direct ens-clf (argmax of averaged
probabilities) is compared with the score-based Ensemble by a paired chromosome bootstrap (2,000 resamples,
rng 0) on macro-F1 and PR-AUC(DOWN); the result is written to classification_paired_diff.json and is read by
31_paper_figures.py for the Fig 6 annotation.
Input : exp4_dl/oof_xgb_seedavg.parquet, oof_dl_pretrained.parquet [, oof_classification.parquet]
Output: exp4_dl/classification_from_scores.csv [, classification_paired_diff.json]
Run   : python analysis/29b_classification_from_scores.py
"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; E4 = f"{ROOT}/results/exp4_dl"
xg = pd.read_parquet(f"{E4}/oof_xgb_seedavg.parquet")
pr = pd.read_parquet(f"{E4}/oof_dl_pretrained.parquet")
m = xg[["log2FC", "label", "chrom", "fold", "all|xgboost_seedavg"]].join(pr[["positional|depthwise_pe_pretrained"]], how="inner")
m["Ensemble"] = (m["all|xgboost_seedavg"] + m["positional|depthwise_pe_pretrained"]) / 2
MODELS = {"XGBoost(5seed)": "all|xgboost_seedavg", "CNN pretrained(3seed)": "positional|depthwise_pe_pretrained", "Ensemble": "Ensemble"}
ycls = m.label.map({"DOWN": 0, "NonDEG": 1, "UP": 2}).values; fold = m.fold.values; chrom = m.chrom.values
GRID = np.round(np.arange(-1.5, 1.5001, 0.02), 2)

def macro_f1(y, p):
    f = 0.0
    for c in range(3):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f += 0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return f / 3

def assign(score, t_down, t_up):
    return np.where(score < t_down, 0, np.where(score > t_up, 2, 1))

def best_thresholds(score, y):
    best, bt = -1.0, (0.0, 0.0)
    for td in GRID:
        lo = assign(score, td, np.inf)                       # DOWN decided by td only
        for tu in GRID[GRID > td]:
            p = np.where(score > tu, 2, lo); f = macro_f1(y, p)
            if f > best: best, bt = f, (float(td), float(tu))
    return bt

rows, assigned = [], {}
for name, col in MODELS.items():
    s = m[col].values; pred = np.full(len(m), -1); thr = {}
    for k in sorted(np.unique(fold)):
        te = fold == k; td, tu = best_thresholds(s[~te], ycls[~te]); thr[int(k)] = (td, tu)
        pred[te] = assign(s[te], td, tu)
    assigned[name] = pred
    rows.append(dict(model=name,
                     prauc_up=float(average_precision_score(ycls == 2, s)), prauc_down=float(average_precision_score(ycls == 0, -s)),
                     auroc_up=float(roc_auc_score(ycls == 2, s)), auroc_down=float(roc_auc_score(ycls == 0, -s)),
                     macro_f1=float(macro_f1(ycls, pred)),
                     recall_up=float(np.mean(pred[ycls == 2] == 2)), recall_down=float(np.mean(pred[ycls == 0] == 0)),
                     recall_non=float(np.mean(pred[ycls == 1] == 1)), thresholds_by_fold=json.dumps(thr)))
    r = rows[-1]; print(f"{name:22s} PR up/down {r['prauc_up']:.3f}/{r['prauc_down']:.3f} | AUROC {r['auroc_up']:.3f}/{r['auroc_down']:.3f} | "
                        f"macro-F1 {r['macro_f1']:.3f} | recall U/D/N {r['recall_up']:.3f}/{r['recall_down']:.3f}/{r['recall_non']:.3f}", flush=True)
pd.DataFrame(rows).to_csv(f"{E4}/classification_from_scores.csv", index=False)

cls_p = f"{E4}/oof_classification.parquet"
if os.path.exists(cls_p):
    oc = pd.read_parquet(cls_p).reindex(m.index)
    P = np.stack([(oc[f"xgbclf_p{c}"].values + oc[f"cnnclf_p{c}"].values) / 2 for c in ["DOWN", "NonDEG", "UP"]], axis=1)
    pred_direct = P.argmax(1); pred_scores = assigned["Ensemble"]; s_ens = m["Ensemble"].values
    rng = np.random.default_rng(0); chroms = np.unique(chrom); d_f1, d_pr = [], []
    for _ in range(2000):
        samp = rng.choice(chroms, len(chroms), replace=True)
        idx = np.concatenate([np.where(chrom == c)[0] for c in samp]); yy = ycls[idx]
        d_f1.append(macro_f1(yy, pred_direct[idx]) - macro_f1(yy, pred_scores[idx]))
        d_pr.append(average_precision_score(yy == 0, P[idx, 0]) - average_precision_score(yy == 0, -s_ens[idx]))
    out = {"ens-clf (direct, 29) minus Ensemble (scores, this script)": {
        "macro_f1": [round(float(np.quantile(d_f1, q)), 4) for q in (0.025, 0.5, 0.975)],
        "prauc_down": [round(float(np.quantile(d_pr, q)), 4) for q in (0.025, 0.5, 0.975)]},
        "bootstrap": "chromosome-level paired bootstrap, 2000 resamples, rng seed 0 (lo, med, hi)"}
    json.dump(out, open(f"{E4}/classification_paired_diff.json", "w"), indent=1)
    print("paired diff direct - scores:", json.dumps(out[list(out)[0]]))
print("saved classification_from_scores.csv")

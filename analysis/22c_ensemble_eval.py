#!/usr/bin/env python3
"""
22c_ensemble_eval.py — ensemble evaluation on LIHC (internal OOF) and ICGC LIRI-JP (external), plus the
2x2 cell values used by Fig 5a.

Re-implements the inline analyses of 2026-09-21 (notes/05, Rounds 4-5 and the pre-training section) as a
script so that every number quoted in the manuscript's ensemble subsection can be regenerated:
  * point estimates (Spearman rho) of every single-model and ensemble OOF column,
  * paired chromosome-bootstrap 95% CIs of differences (2,000 resamples, rng seed 0, same resample for
    both members of each pair), internally and on LIRI-JP (genes shared with results/de_liri),
  * results/exp4_dl/fig5_cells.json: 2x2 cells (aggregate/positional x XGBoost/DL) with unpaired
    chromosome-bootstrap 95% CIs (1,000 resamples) for 24_fig5.py.
Ensembles are equal-weight (0.5/0.5) averages of OOF predictions; the manuscript's primary model is
ens_pre = 0.5 * all|xgboost_seedavg + 0.5 * positional|depthwise_pe_pretrained.

Input : exp1_nested_cv/oof_predictions.parquet, oof_mlp.parquet; exp4_dl/oof_positional.parquet,
        oof_dl_stage2.parquet, oof_dl_seedavg.parquet, oof_xgb_seedavg.parquet, oof_dl_pretrained.parquet;
        de_liri/de_results.csv (optional columns are skipped if the file is missing)
Output: results/exp4_dl/ensemble_lihc.json, results/exp4_dl/fig5_cells.json
Run   : python analysis/22c_ensemble_eval.py
Note  : point estimates are deterministic; bootstrap CIs regenerate from rng seed 0 and may differ from the
        values in the manuscript (computed interactively on 2026-09-21) in the third decimal.
"""
import os, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; R = f"{ROOT}/results"; E1 = f"{R}/exp1_nested_cv"; E4 = f"{R}/exp4_dl"
N_BOOT_PAIRED, N_BOOT_CELLS = 2000, 1000

def read(path, cols):
    if not os.path.exists(path):
        print("missing (skipped):", path); return None
    d = pd.read_parquet(path); return d[[c for c in cols if c in d.columns]]

base = pd.read_parquet(f"{E1}/oof_predictions.parquet")[["log2FC", "label", "chrom", "fold", "all|xgboost", "all|elasticnet", "omics|xgboost", "binding|xgboost"]]
parts = [read(f"{E1}/oof_mlp.parquet", ["all|mlp"]),
         read(f"{E4}/oof_positional.parquet", ["positional|xgboost"]),
         read(f"{E4}/oof_dl_stage2.parquet", ["positional|depthwise_pe"]),
         read(f"{E4}/oof_dl_seedavg.parquet", ["pe_seed1", "pe_seed2", "pe_seed3", "positional|depthwise_pe_seedavg"]),
         read(f"{E4}/oof_xgb_seedavg.parquet", ["all|xgboost_seedavg"]),
         read(f"{E4}/oof_dl_pretrained.parquet", ["positional|depthwise_pe_pretrained"])]
m = base.join([p for p in parts if p is not None], how="inner")
if {"pe_seed1", "pe_seed2", "pe_seed3"} <= set(m.columns):
    m["positional|depthwise_pe_rand3"] = m[["pe_seed1", "pe_seed2", "pe_seed3"]].mean(axis=1)   # matched random-init control for pretraining
ENS = {  # name: (member a, member b)
    "ens_single":  ("all|xgboost", "positional|depthwise_pe"),                    # stage-2, single seed each
    "ens_seedavg": ("all|xgboost_seedavg", "positional|depthwise_pe_seedavg"),    # both 5-seed averaged
    "ens_rand3":   ("all|xgboost_seedavg", "positional|depthwise_pe_rand3"),      # 3-seed random-init CNN
    "ens_pre":     ("all|xgboost_seedavg", "positional|depthwise_pe_pretrained"), # PRIMARY (pretrained CNN, 3 seeds)
    "ens_enet":    ("all|xgboost", "all|elasticnet"),                              # non-DL control
    "ens_mlp":     ("all|xgboost", "all|mlp"),                                     # aggregate-MLP control
}
for k, (a, b) in ENS.items():
    if a in m.columns and b in m.columns: m[k] = (m[a] + m[b]) / 2
PAIRS = [("ens_single", "all|xgboost"), ("ens_enet", "all|xgboost"), ("ens_mlp", "all|xgboost"),
         ("positional|depthwise_pe", "all|xgboost"),
         ("ens_seedavg", "all|xgboost_seedavg"), ("positional|depthwise_pe_seedavg", "all|xgboost_seedavg"),
         ("positional|depthwise_pe_pretrained", "positional|depthwise_pe_rand3"),
         ("positional|depthwise_pe_pretrained", "all|xgboost_seedavg"),
         ("ens_pre", "all|xgboost_seedavg"), ("ens_pre", "ens_rand3"), ("ens_pre", "ens_seedavg"),
         ("all|xgboost", "omics|xgboost")]
cols = [c for c in m.columns if c not in ("log2FC", "label", "chrom", "fold") and not c.startswith("pe_seed")]

def evaluate(df, yname, tag):
    yv = df[yname].values; chroms = df.chrom.unique(); rng = np.random.default_rng(0)
    point = {c: round(float(spearmanr(yv, df[c].values)[0]), 4) for c in cols}
    boot = {c: [] for c in cols}
    for _ in range(N_BOOT_PAIRED):
        samp = rng.choice(chroms, len(chroms), replace=True)
        idx = np.concatenate([np.where(df.chrom.values == c)[0] for c in samp])
        for c in cols: boot[c].append(spearmanr(yv[idx], df[c].values[idx])[0])
    B = pd.DataFrame(boot)
    diffs = {f"{a} - {b}": (B[a] - B[b]).quantile([0.025, 0.5, 0.975]).round(4).tolist() for a, b in PAIRS if a in B and b in B}
    print(f"\n[{tag}] n={len(df)}"); print(json.dumps(point, indent=1))
    for k, v in diffs.items(): print(f"  {k}: {v[1]:+.3f} [{v[0]:+.3f}, {v[2]:+.3f}]")
    return {"n_genes": int(len(df)), "point": point, "paired_diff_lo_med_hi": diffs}

res = {"internal_lihc": evaluate(m, "log2FC", "internal LIHC OOF")}
liri_p = f"{R}/de_liri/de_results.csv"
if os.path.exists(liri_p):
    liri = pd.read_csv(liri_p).dropna(subset=["gene_id"]).set_index("gene_id")
    ml = m.join(liri[["log2FC"]].rename(columns={"log2FC": "liri_logfc"}), how="inner").dropna(subset=["liri_logfc"])
    res["external_liri"] = evaluate(ml, "liri_logfc", "external LIRI-JP")
res["definitions"] = {"ensembles": ENS, "bootstrap": f"chromosome-level paired bootstrap, {N_BOOT_PAIRED} resamples, rng seed 0"}
json.dump(res, open(f"{E4}/ensemble_lihc.json", "w"), indent=1)

# ---- Fig 5a cells: unpaired chromosome bootstrap of the four 2x2 cells (same OOF columns as the figure) ----
cells = {}; yv = m.log2FC.values; chroms = m.chrom.unique(); rng = np.random.default_rng(0)
for c in ["all|xgboost", "all|mlp", "positional|xgboost", "positional|depthwise_pe"]:
    if c not in m.columns: continue
    bs = []
    for _ in range(N_BOOT_CELLS):
        samp = rng.choice(chroms, len(chroms), replace=True)
        idx = np.concatenate([np.where(m.chrom.values == cc)[0] for cc in samp])
        bs.append(spearmanr(yv[idx], m[c].values[idx])[0])
    cells[c] = {"point": round(float(spearmanr(yv, m[c].values)[0]), 4),
                "lo": round(float(np.quantile(bs, 0.025)), 4), "hi": round(float(np.quantile(bs, 0.975)), 4)}
json.dump(cells, open(f"{E4}/fig5_cells.json", "w"), indent=1)
print("\nfig5 cells:", json.dumps(cells)); print("saved ensemble_lihc.json / fig5_cells.json")

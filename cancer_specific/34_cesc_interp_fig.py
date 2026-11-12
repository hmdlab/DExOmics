#!/usr/bin/env python3
"""34_cesc_interp_fig.py — 新 Supp S4: CESC の解釈（再計算版、dumbbell + POLR2A/SMARCB1 検証）"""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
R = f"{DATA_ROOT}/results"
TEX = f"{PROJ_ROOT}/2024_Yi_Bioinformatics_TeX/supple_fig"
NOTES = f"{PROJ_ROOT}/notes/figures"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e8e8e5", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "white", "savefig.dpi": 300})
CB, CO = "#2a78d6", "#eb6834"

mx = np.load(f"{R}/exp5_interp/cesc_factor_share_xgb.npy")
mc = np.load(f"{R}/exp5_interp/cesc_factor_share_cnn.npy")
shared = json.load(open(f"{R}/exp5_interp/cesc_shared_factors.json"))["shared"]
summ = json.load(open(f"{R}/exp5_interp/cesc_interp_summary.json"))
combo = list(np.argsort(-(mx + mc))[:12])
for f_ in ["prom__POLR2A", "prom__SMARCB1"]:      # 検証対象 2 因子を明示的に併記
    i = shared.index(f_)
    if i not in combo: combo.append(i)
lab = lambda c: c.replace("prom__", "TF: ").replace("rna__", "RBP: ").replace("cnv__", "CNV: ").replace("methy_all__", "Me: ").replace("mutall__", "Mut: ").replace("mut__", "Mut: ")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.45})
ax = axes[0]
ys = np.arange(len(combo))[::-1]
for y0, i in zip(ys, combo):
    hl = shared[i] in ("prom__POLR2A", "prom__SMARCB1")
    ax.plot([mx[i], mc[i]], [y0, y0], color="#d9d8d3", lw=1.1, zorder=1)
    ax.plot(mx[i], y0, "o", color=CB, ms=4.5, zorder=2)
    ax.plot(mc[i], y0, "o", color=CO, ms=4.5, zorder=2)
    if hl: ax.axhspan(y0 - 0.42, y0 + 0.42, color="#fff3c9", zorder=0)
ax.set_yticks(ys); ax.set_yticklabels([lab(shared[i]) for i in combo], fontsize=6.8)
ax.set_xlabel("importance share (mean |attribution|)")
ax.legend(handles=[Line2D([], [], color=CB, marker="o", lw=0, label="XGBoost (TreeSHAP)"),
                   Line2D([], [], color=CO, marker="o", lw=0, label="CNN (Expected Grad.)")],
          frameon=False, fontsize=6.4, loc="lower right")
ax.set_title("a  Top factors, TCGA-CESC (revised models)", loc="left", fontsize=9)
ax.grid(axis="y", visible=False)
ax = axes[1]
grp = ["UP", "NonDEG", "DOWN"]; x = np.arange(3)
g = [summ["prom__POLR2A"]["signed_mean_by_class_xgb"][k] for k in grp]
ax.bar(x, g, 0.5, color=CB, edgecolor="white")
ax.axhline(0, color="#7a7a76", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(grp, fontsize=7.6)
ax.set_ylabel("mean signed SHAP (POLR2A)")
ax.text(0.03, 0.05, f"POLR2A rank: {summ['prom__POLR2A']['rank_xgb']}/135 (XGB), {summ['prom__POLR2A']['rank_cnn']}/135 (CNN)\n"
                    f"SMARCB1 rank: {summ['prom__SMARCB1']['rank_xgb']}/135 (XGB) — not a leading factor\n"
                    f"cross-model rank agreement: ρ = {summ['crossmodel_spearman']}",
        transform=ax.transAxes, fontsize=6.2, color="#52514e", va="bottom")
ax.set_title("b  POLR2A separates directions", loc="left", fontsize=9)
ax.grid(axis="x", visible=False)
for out in [f"{R}/figures/figS_cesc_interpret.pdf"] + ([f"{TEX}/figS_cesc_interpret.pdf"] if os.path.isdir(TEX) else []):   # manuscript copy only if the TeX tree exists
    fig.savefig(out, bbox_inches="tight")
fig.savefig(f"{NOTES}/figS_cesc_interpret.png", bbox_inches="tight")
print("saved figS_cesc_interpret")

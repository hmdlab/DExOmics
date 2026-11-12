#!/usr/bin/env python3
"""
24_fig5.py — Fig 5: DL 検討の決着図（codex R4-4 仕様）
左 a: 2×2 interaction plot（入力表現 × 学習器、絶対性能と染色体 bootstrap 95%CI）
右 b: XGBoost 基準の paired 差 forest（DL 単体と ensemble、LIHC 内部 / LIRI 外部 / CESC 凍結確証）
配色: XGBoost=#2a78d6 (blue) / DL=#eb6834 (orange) / ensemble=#1baf7a (aqua)。CI は未調整（脚注）。
出力: results/figures/fig5_dl_complementarity.{pdf,png} + notes/figures/
"""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; R = f"{ROOT}/results"
OUT = f"{R}/figures"; NOTES = f"{PROJ_ROOT}/notes/figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(NOTES, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e8e8e5", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "white", "savefig.dpi": 300})
CB, CO, CA = "#2a78d6", "#eb6834", "#1baf7a"

cells = json.load(open(f"{R}/exp4_dl/fig5_cells.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.42})

# ---- a: 2×2 interaction ----
ax = axes[0]
x = [0, 1]
xgb = [cells["all|xgboost"], cells["positional|xgboost"]]
dl = [cells["all|mlp"], cells["positional|depthwise_pe"]]
for pts, c, lab in [(xgb, CB, "non-DL (XGBoost)"), (dl, CO, "DL (MLP / positional CNN)")]:
    ax.plot(x, [p["point"] for p in pts], "-o", color=c, ms=5, lw=1.6, label=lab)
    for xi, p in zip(x, pts):
        ax.plot([xi, xi], [p["lo"], p["hi"]], color=c, lw=1.4, solid_capstyle="round")
for xi, p, dy in [(0, cells["all|xgboost"], 0.013), (1, cells["positional|xgboost"], -0.030),
                  (0, cells["all|mlp"], 0.013), (1, cells["positional|depthwise_pe"], 0.016)]:
    ax.text(xi, p["point"] + dy, f'{p["point"]:.3f}', ha="center", fontsize=7.6, color="#0b0b0b")
ax.set_xticks(x); ax.set_xticklabels(["aggregated", "position-resolved"]); ax.set_xlim(-0.35, 1.35)
ax.set_ylabel("Spearman ρ (out-of-fold)"); ax.set_ylim(0.20, 0.40)
ax.legend(frameon=False, fontsize=7.4, loc="lower right")
ax.set_title("a  Representation × learner (TCGA-LIHC)", loc="left", fontsize=9)
ax.grid(axis="x", visible=False)

# ---- b: forest of paired differences vs XGBoost ----
ax = axes[1]
rows = [  # (label, CNN(pretrained)−XGB, ens(pretrained)−XGB) — seed 平均・CESC は凍結規則（昇格後の主数値、2026-09-21 の値）
    ("LIHC internal",  (-0.004, 0.011, 0.026), (0.021, 0.029, 0.037)),
    ("LIRI-JP external", (0.001, 0.022, 0.043), (0.018, 0.030, 0.041)),
    ("CESC frozen",    (-0.006, 0.013, 0.038), (0.013, 0.023, 0.040)),
]
# regenerated values (22c → ensemble_lihc.json, 28 → ensemble_cesc_pretrained.json) replace the constants when present
_el = f"{R}/exp4_dl/ensemble_lihc.json"; _ec = f"{R}/exp4_dl/ensemble_cesc_pretrained.json"
if os.path.exists(_el):
    _e = json.load(open(_el)); _k = "paired_diff_lo_med_hi"
    for i, sect in [(0, "internal_lihc"), (1, "external_liri")]:
        if sect in _e:
            d = _e[sect][_k]; a, b = "positional|depthwise_pe_pretrained - all|xgboost_seedavg", "ens_pre - all|xgboost_seedavg"
            if a in d and b in d: rows[i] = (rows[i][0], tuple(d[a]), tuple(d[b])); print(f"panel b: {sect} from ensemble_lihc.json")
            else: print(f"panel b: {sect} keeps the 2026-09-21 constants (pairs missing in ensemble_lihc.json)")
if os.path.exists(_ec):
    _c = json.load(open(_ec))
    if "cesc_pretrained - all|xgboost" in _c and "ens_pre - all|xgboost" in _c:
        rows[2] = (rows[2][0], tuple(_c["cesc_pretrained - all|xgboost"]), tuple(_c["ens_pre - all|xgboost"])); print("panel b: CESC from ensemble_cesc_pretrained.json")
ys = np.arange(len(rows))[::-1]
for y0, (lab, ddl, dens) in zip(ys, rows):
    ax.plot([ddl[0], ddl[2]], [y0 + 0.16, y0 + 0.16], color=CO, lw=1.5, solid_capstyle="round")
    ax.plot(ddl[1], y0 + 0.16, "o", color="white", mec=CO, mew=1.4, ms=5, zorder=3)
    ax.plot([dens[0], dens[2]], [y0 - 0.16, y0 - 0.16], color=CA, lw=1.8, solid_capstyle="round")
    ax.plot(dens[1], y0 - 0.16, "o", color=CA, ms=5, zorder=3)
ax.axvline(0, color="#7a7a76", lw=0.9)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_ylim(-0.6, len(rows) - 0.4 + 0.55)
ax.set_xlim(-0.035, 0.055)
ax.set_xlabel("Δ Spearman ρ vs XGBoost alone (5-seed averaged)")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=CO, marker="o", mfc="white", mec=CO, lw=1.5, label="pretrained positional CNN alone"),
                   Line2D([], [], color=CA, marker="o", lw=1.8, label="ensemble (prespecified 0.5/0.5)")],
          frameon=False, fontsize=7.2, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02), handlelength=1.4, columnspacing=1.0)
ax.set_title("b  Complementary value across three settings", loc="left", fontsize=9)
ax.grid(axis="y", visible=False)
fig.text(0.99, -0.045, "95% CIs: chromosome-block bootstrap, unadjusted. CNN is pan-cancer pretrained; CESC row uses the frozen rule (the non-pretrained ensemble also passed: +0.014 [0.005, 0.028]).",
         ha="right", fontsize=6.2, color="#52514e")
for ext in ["pdf", "png"]:
    fig.savefig(f"{OUT}/fig5_dl_complementarity.{ext}", bbox_inches="tight")
fig.savefig(f"{NOTES}/fig5_dl_complementarity.png", bbox_inches="tight")
print("saved fig5")

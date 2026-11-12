#!/usr/bin/env python3
"""
26_fig_arch.py — Fig 5 で用いた DL モデル（depthwise_pe CNN）の構造図
左 (a): アーキテクチャ本体（promoter / RNA の 2 branch + omics head）
右 (b): 評価・ensemble の流れ（5-seed 平均、XGBoost との 0.5/0.5 等重み）
出力: results/figures/fig_s_architecture.{pdf,png} + notes/figures/
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
OUT = f"{DATA_ROOT}/results/figures"
NOTES = f"{PROJ_ROOT}/notes/figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(NOTES, exist_ok=True)
plt.rcParams.update({"font.size": 7.6, "figure.facecolor": "white", "savefig.dpi": 300})
CO, CB, CA, CG, CT = "#eb6834", "#2a78d6", "#1baf7a", "#f2f1ee", "#0b0b0b"
EDGE = "#8a8985"

fig = plt.figure(figsize=(8.6, 4.6))
axa = fig.add_axes([0.0, 0.0, 0.62, 1.0]); axb = fig.add_axes([0.64, 0.0, 0.36, 1.0])
for ax in (axa, axb):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def box(ax, x, y, w, h, text, fc=CG, ec=EDGE, fs=7.2, lw=1.0, tc=CT, style="round,pad=0.008,rounding_size=0.012"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)

def arrow(ax, x1, y1, x2, y2, c="#52514e"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=1.0, color=c, zorder=1))

# ---------- (a) architecture ----------
axa.text(0.02, 0.965, "a  Positional CNN (depthwise + positional embedding)", fontsize=9, fontweight="bold", ha="left")
xp, xr = 0.26, 0.62   # branch centers
bw, bh, dy = 0.30, 0.062, 0.088
y0 = 0.865
box(axa, xp, y0, bw, 0.075, "Promoter binding tensor\n550 TFs × 30 bins  (log1p)", fc="#fde8de", ec=CO)
box(axa, xr, y0, bw, 0.075, "RNA binding tensor\n86 RBPs × 32 pooled bins  (log1p)", fc="#fde8de", ec=CO)
steps = [
    ("Depthwise Conv1d  k=5\n(per-factor, groups=factors)", CG),
    ("Pointwise Conv1d 1×1\nfactors → C channels · GELU", CG),
    ("+ Positional embedding\nPE[C × L]  (learned)", "#fff3c9"),
    ("Conv1d  k=3 · GELU", CG),
    ("GAP ‖ GMP over positions\n→ 2C", CG),
]
for j, (txt, fc) in enumerate(steps):
    yy = y0 - 0.075 - (j + 0.5) * dy + 0.012
    for xc in (xp, xr):
        box(axa, xc, yy, bw, bh, txt, fc=fc, ec=("#d8b23a" if "embedding" in txt else EDGE))
        if j == 0: arrow(axa, xc, y0 - 0.040, xc, yy + bh / 2)
        else: arrow(axa, xc, yy + dy - bh / 2 + 0.001, xc, yy + bh / 2)
ylast = y0 - 0.075 - (len(steps) - 0.5) * dy + 0.012
# omics side input
xo = 0.90
box(axa, xo, 0.52, 0.17, 0.10, "Patient omics\naggregates (42)\n+ log n_bins\n(standardized)", fc="#e5eefb", ec=CB, fs=6.8)
# head
yc = 0.20
box(axa, 0.50, yc + 0.075, 0.62, 0.055, "concat  [2C (prom) + 2C (RNA) + omics 43]", fc=CG)
box(axa, 0.50, yc, 0.62, 0.055, "LayerNorm → Dropout → Linear 128 → GELU → Dropout → Linear 1", fc=CG)
box(axa, 0.50, yc - 0.085, 0.26, 0.055, "predicted log2FC", fc="#fde8de", ec=CO, fs=7.6)
arrow(axa, xp, ylast - bh / 2, xp, yc + 0.075 + 0.03); arrow(axa, xr, ylast - bh / 2, xr, yc + 0.075 + 0.03)
arrow(axa, xo, 0.52 - 0.052, xo - 0.06, yc + 0.075 + 0.03)
arrow(axa, 0.50, yc + 0.075 - 0.028, 0.50, yc + 0.028)
arrow(axa, 0.50, yc - 0.028, 0.50, yc - 0.085 + 0.028)
axa.text(0.02, 0.035, "Training: Huber (δ=2), AdamW (wd=1e-4), batch 512, early stopping;  config & epochs chosen in inner folds only.\n"
                      "C ∈ {64, 96, 128}, dropout ∈ {0.1–0.3}, lr ∈ {1e-3, 3e-4} — 5 configs (budget matched to XGBoost's 9-point grid stage).  ~0.1–0.4M params.",
         fontsize=6.8, color="#52514e", ha="left", va="bottom")

# ---------- (b) evaluation / ensemble ----------
axb.text(0.03, 0.965, "b  Evaluation & ensemble", fontsize=9, fontweight="bold", ha="left")
box(axb, 0.30, 0.80, 0.48, 0.10, "Positional CNN\n(architecture in a)\n× 5 seeds → averaged", fc="#fde8de", ec=CO)
box(axb, 0.30, 0.60, 0.48, 0.10, "XGBoost\naggregated 636 + omics\n× 5 seeds → averaged", fc="#e5eefb", ec=CB)
box(axb, 0.74, 0.70, 0.34, 0.10, "equal-weight mean\n(0.5 / 0.5, prespecified)", fc="#def3ea", ec=CA, fs=7.0)
arrow(axb, 0.54, 0.80, 0.57, 0.73); arrow(axb, 0.54, 0.60, 0.57, 0.67)
box(axb, 0.74, 0.50, 0.34, 0.075, "ensemble\nprediction", fc="#def3ea", ec=CA, fs=7.0)
arrow(axb, 0.74, 0.65, 0.74, 0.54)
box(axb, 0.50, 0.30, 0.90, 0.12,
    "Chromosome-blocked nested CV (5×4), stage-2 models\nOOF ρ:  CNN 0.364 · XGBoost 0.357 · ensemble 0.381\n(final pretrained ensemble: 0.386)", fc=CG, fs=7.0)
arrow(axb, 0.74, 0.462, 0.74, 0.365)
box(axb, 0.50, 0.13, 0.90, 0.13,
    "Confirmations (Δρ vs XGBoost, 95% CI > 0):\nLIRI-JP external +0.028 · frozen CESC rule +0.014", fc="#def3ea", ec=CA, fs=7.0)
arrow(axb, 0.50, 0.24, 0.50, 0.20)

for ext in ["pdf", "png"]:
    fig.savefig(f"{OUT}/fig_s_architecture.{ext}", bbox_inches="tight")
fig.savefig(f"{NOTES}/fig_s_architecture.png", bbox_inches="tight")
print("saved fig_s_architecture")

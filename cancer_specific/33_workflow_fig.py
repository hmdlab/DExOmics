#!/usr/bin/env python3
"""
33_workflow_fig.py — 改訂版 Fig 1（workflow 概観図）の再作図
(a) pan-cancer モデル（従来どおり）(b) 改訂 cancer-specific 解析 (c) 評価・解釈
出力: results/figures/fig0_workflow.{pdf,png} + notes/figures/ + TeX fig/
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
OUT = f"{DATA_ROOT}/results/figures"
NOTES = f"{PROJ_ROOT}/notes/figures"
TEX = f"{PROJ_ROOT}/2024_Yi_Bioinformatics_TeX/fig"
for d in (OUT, NOTES): os.makedirs(d, exist_ok=True)
plt.rcParams.update({"font.size": 7.2, "figure.facecolor": "white", "savefig.dpi": 300})
CB, CO, CA, CG, CY = "#2a78d6", "#eb6834", "#1baf7a", "#f2f1ee", "#fff3c9"
EDGE = "#8a8985"

fig = plt.figure(figsize=(10.5, 3.6))
axa = fig.add_axes([0.00, 0.0, 0.30, 1.0]); axb = fig.add_axes([0.31, 0.0, 0.40, 1.0]); axc = fig.add_axes([0.72, 0.0, 0.28, 1.0])
for ax in (axa, axb, axc):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def box(ax, x, y, w, h, text, fc=CG, ec=EDGE, fs=6.6, tc="#0b0b0b"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fc, ec=ec, lw=1.0, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)

def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.9, color="#52514e", zorder=1))

# ---------- (a) pan-cancer ----------
axa.text(0.03, 0.96, "a  Pan-cancer model (cross-cancer regularities)", fontsize=8.2, fontweight="bold")
box(axa, 0.5, 0.82, 0.86, 0.11, "Positional binding features\nGTRD TFs · POSTAR3 RBPs · TargetScan miRNAs\n(100-bp bins on promoters / RNAs)", fc="#e5eefb", ec=CB)
box(axa, 0.5, 0.60, 0.62, 0.09, "CNN (DEcode-style)", fc="#e5eefb", ec=CB)
box(axa, 0.5, 0.38, 0.86, 0.11, "Median expression + DE profiles\nacross 32 (PanCanAtlas) / 29 (PCAWG) cancers", fc=CG)
box(axa, 0.5, 0.14, 0.86, 0.11, "DeepLIFT attributions\n(e.g., AGO2 across cancers)", fc=CY, ec="#d8b23a")
arrow(axa, 0.5, 0.765, 0.5, 0.65); arrow(axa, 0.5, 0.555, 0.5, 0.435); arrow(axa, 0.5, 0.325, 0.5, 0.20)

# ---------- (b) cancer-specific (revised) ----------
axb.text(0.02, 0.96, "b  Cancer-specific analysis (revised)", fontsize=8.2, fontweight="bold")
box(axb, 0.26, 0.83, 0.44, 0.13, "TCGA (GDC open tier)\nSTAR counts · ASCAT3 CNV\n450K methylation · masked MAF", fc="#e5eefb", ec=CB)
box(axb, 0.74, 0.83, 0.44, 0.13, "ENCODE positional binding\n550 TFs × 30 promoter bins\n86 RBPs × transcript bins", fc="#fde8de", ec=CO)
box(axb, 0.26, 0.615, 0.44, 0.12, "Label audit → target =\ncontinuous log2FC\n(DESeq2, TCGA-only)", fc=CY, ec="#d8b23a")
box(axb, 0.74, 0.615, 0.44, 0.12, "Position-aware CNN\n(depthwise + positional emb.,\npan-cancer pretrained)", fc="#fde8de", ec=CO)
box(axb, 0.26, 0.40, 0.44, 0.12, "Strong baselines\nridge · elastic net · XGBoost\n(aggregated features)", fc="#e5eefb", ec=CB)
box(axb, 0.5, 0.175, 0.66, 0.11, "Prespecified equal-weight ensemble\n(XGBoost + pretrained CNN)", fc="#def3ea", ec=CA, fs=7.0)
arrow(axb, 0.26, 0.765, 0.26, 0.675); arrow(axb, 0.74, 0.765, 0.74, 0.675)
arrow(axb, 0.26, 0.555, 0.26, 0.46); arrow(axb, 0.26, 0.34, 0.38, 0.23); arrow(axb, 0.74, 0.555, 0.62, 0.23)

# ---------- (c) evaluation & interpretation ----------
axc.text(0.03, 0.96, "c  Evaluation & interpretation", fontsize=8.2, fontweight="bold")
box(axc, 0.5, 0.82, 0.9, 0.11, "Chromosome-blocked nested CV\n+ chromosome-block bootstrap CIs", fc=CG)
box(axc, 0.5, 0.615, 0.9, 0.11, "External replication\nICGC LIRI-JP (240 T / 197 N)", fc="#def3ea", ec=CA)
box(axc, 0.5, 0.41, 0.9, 0.11, "Frozen cross-cancer confirmation\nTCGA-CESC (rule fixed pre-training)", fc="#def3ea", ec=CA)
box(axc, 0.5, 0.175, 0.9, 0.13, "Interpretation (non-causal)\nTreeSHAP + expected gradients,\ncross-model agreement", fc=CY, ec="#d8b23a")
arrow(axc, 0.5, 0.765, 0.5, 0.67); arrow(axc, 0.5, 0.56, 0.5, 0.465); arrow(axc, 0.5, 0.355, 0.5, 0.24)

for ext in ["pdf", "png"]:
    fig.savefig(f"{OUT}/fig0_workflow.{ext}", bbox_inches="tight")
fig.savefig(f"{NOTES}/fig0_workflow.png", bbox_inches="tight")
if os.path.isdir(TEX): fig.savefig(f"{TEX}/fig0_workflow.pdf", bbox_inches="tight")   # manuscript copy, only if the TeX tree exists
print("saved fig0_workflow")

#!/usr/bin/env python3
"""
18_figures.py — 論文図版ドラフト 4 枚（PDF+PNG、results/figures/ と proj notes/figures/ に出力）

Fig1 ラベル分解フロー + crosstab、Fig2 nested CV フォレスト（LIHC/CESC）、
Fig3 LIRI 外部検証（hexbin + rho フォレスト）、Fig4 方向別診断（符号・PR-AUC・較正）。
配色: 検証済みパレット slot1-3 (#2a78d6 blue / #eb6834 orange / #1baf7a aqua) + gray。
"""
import os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
ROOT = DATA_ROOT; R = f"{ROOT}/results"
OUT = f"{R}/figures"; os.makedirs(OUT, exist_ok=True)
NOTES = f"{PROJ_ROOT}/notes/figures"; os.makedirs(NOTES, exist_ok=True)
C = {"all": "#2a78d6", "binding": "#eb6834", "omics": "#1baf7a", "control": "#7a7a76", "ref": "#0b0b0b"}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e8e8e5", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "white", "savefig.dpi": 300})

def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    fig.savefig(f"{NOTES}/{name}.png", bbox_inches="tight")
    plt.close(fig); print("saved", name)

# ---------- Fig 1: label decomposition ----------
dec = json.load(open(f"{R}/de_lihc/label_decomposition.json"))
ct = pd.read_csv(f"{R}/de_lihc/crosstab_vs_current_labels.csv", index_col=0)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.45})
ax = axes[0]
stages = ["DESeq2\n(TCGA only)", "voom-on-TPM\n(method approx.)", "+ HepG2 agreement\n(distributed)"]
vals = [1307, 516, 224]
bars = ax.barh(range(3)[::-1], vals, height=0.62, color=["#2a78d6", "#7aa9e0", "#c9d9ef"], edgecolor="white")
ax.set_xlim(0, 1450)
for i, (v, lab) in enumerate(zip(vals, ["", "−60.5% (method)", "−56.6% (selection)"])):
    inside = v > 900
    ax.text(v - 30 if inside else v + 25, 2 - i, f"{v:,}", va="center", ha="right" if inside else "left",
            fontsize=9, fontweight="bold", color="white" if inside else "#0b0b0b")
    if lab: ax.text(650, 2 - i + 0.46, lab, va="center", ha="left", fontsize=7.5, color="#52514e")
ax.set_yticks(range(3)[::-1]); ax.set_yticklabels(stages, fontsize=8)
ax.set_xlabel("DOWN-regulated genes (n)"); ax.set_title("a  Sequential shrinkage of DOWN labels", loc="left", fontsize=9)
ax.grid(axis="y", visible=False)
ax = axes[1]
ctn = ct.reindex(index=["UP", "NonDEG", "DOWN"], columns=["UP", "NonDEG", "DOWN"])
im = ax.imshow(np.log10(ctn.values + 1), cmap="Blues", aspect="auto")
for i in range(3):
    for j in range(3):
        v = ctn.values[i, j]
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8,
                color="white" if v > 2000 else "#0b0b0b")
ax.set_xticks(range(3)); ax.set_xticklabels(ctn.columns); ax.set_yticks(range(3)); ax.set_yticklabels(ctn.index)
ax.set_xlabel("Distributed labels (agreement-filtered)"); ax.set_ylabel("TCGA-only labels (DESeq2)")
ax.set_title("b  Label cross-tabulation (12,219 genes)", loc="left", fontsize=9); ax.grid(visible=False)
save(fig, "fig1_label_decomposition")

# ---------- Fig 2: forest (LIHC + CESC) ----------
def forest(ax, boot_csv, mlp_csv, title, show_mlp):
    ci = pd.read_csv(boot_csv, index_col=0)
    order = ["all|xgboost", "all|elasticnet", "all|ridge", "binding|xgboost", "binding|elasticnet",
             "binding|ridge", "omics|xgboost", "omics|elasticnet", "omics|ridge", "control|ridge", "mean|baseline"]
    rows = [(m, ci.loc[m]) for m in order if m in ci.index]
    if show_mlp and "all|mlp" in ci.index:
        rows.insert(3, ("all|mlp", ci.loc["all|mlp"])); rows.insert(7, ("binding|mlp", ci.loc["binding|mlp"]))
    ys = np.arange(len(rows))[::-1]
    for y, (m, r) in zip(ys, rows):
        g = m.split("|")[0] if m != "mean|baseline" else "control"
        ax.plot([r.lo, r.hi], [y, y], color=C.get(g, "#7a7a76"), lw=1.6, solid_capstyle="round")
        ax.plot(r.med, y, "o", color=C.get(g, "#7a7a76"), ms=4.5)
        ax.text(-0.24, y, m.replace("|", " · "), ha="left", va="center", fontsize=7.6)
    ax.axvline(0, color="#b9b8b3", lw=0.8)
    ax.set_yticks([]); ax.set_xlim(-0.25, 0.45); ax.set_xlabel("Spearman ρ (out-of-fold vs measured log2FC)")
    ax.set_title(title, loc="left", fontsize=9); ax.grid(axis="y", visible=False)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=False)
forest(axes[0], f"{R}/exp1_nested_cv/metrics_bootstrap_spearman.csv", None, "a  TCGA-LIHC (primary)", True)
forest(axes[1], f"{R}/exp1_nested_cv_cesc/metrics_bootstrap_spearman.csv", None, "b  TCGA-CESC (exploratory)", False)
save(fig, "fig2_nested_cv_forest")

# ---------- Fig 3: LIRI external ----------
oof = pd.read_parquet(f"{R}/exp1_nested_cv/oof_predictions.parquet")
liri = pd.read_csv(f"{R}/de_liri/de_results.csv").dropna(subset=["gene_id"]).set_index("gene_id")
m = oof.join(liri[["log2FC"]].rename(columns={"log2FC": "liri"}), how="inner").dropna(subset=["liri"])
ci2 = pd.read_csv(f"{R}/exp2_liri/bootstrap_ci.csv", index_col=0)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
hb = ax.hexbin(m["all|xgboost"], m.liri, gridsize=45, cmap="Blues", mincnt=1, linewidths=0)
ax.set_xlabel("Predicted log2FC (OOF, integrated XGBoost)"); ax.set_ylabel("LIRI-JP measured log2FC")
ax.axhline(0, color="#b9b8b3", lw=0.7); ax.axvline(0, color="#b9b8b3", lw=0.7)
r = ci2.loc["all|xgboost"]
ax.set_title(f"a  External cohort: ρ = {r.med:.2f} [{r.lo:.2f}, {r.hi:.2f}]", loc="left", fontsize=9)
ax.grid(visible=False)
ax = axes[1]
sel = ["log2FC", "all|xgboost", "all|elasticnet", "all|mlp", "omics|xgboost", "binding|xgboost", "control|ridge"]
labels = {"log2FC": "TCGA measured (reference)", "all|xgboost": "all · xgboost", "all|elasticnet": "all · elasticnet",
          "all|mlp": "all · mlp", "omics|xgboost": "omics · xgboost", "binding|xgboost": "binding · xgboost",
          "control|ridge": "control · ridge"}
ys = np.arange(len(sel))[::-1]
for y, k in zip(ys, sel):
    r = ci2.loc[k]; g = "ref" if k == "log2FC" else k.split("|")[0]
    ax.plot([r.lo, r.hi], [y, y], color=C[g], lw=1.6, solid_capstyle="round")
    ax.plot(r.med, y, "o", color=C[g], ms=4.5)
    ax.text(-0.62, y, labels[k], ha="left", va="center", fontsize=7.6)
ax.axvline(0, color="#b9b8b3", lw=0.8)
ax.set_yticks([]); ax.set_xlim(-0.63, 1.0); ax.set_xlabel("Spearman ρ vs LIRI-JP measured log2FC")
ax.set_title("b  Transfer of predictions to LIRI-JP", loc="left", fontsize=9); ax.grid(axis="y", visible=False)
save(fig, "fig3_liri_external")

# ---------- Fig 4: direction diagnosis ----------
diag = pd.read_csv(f"{R}/exp3_direction/diagnosis.csv").set_index("model")
cal = pd.read_csv(f"{R}/exp3_direction/calibration_probe.csv")
models = ["all|xgboost", "all|elasticnet", "binding|xgboost", "omics|xgboost", "control|ridge"]
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
ax = axes[0]
ys = np.arange(len(models))[::-1]
for y, mo in zip(ys, models):
    d = diag.loc[mo]
    ax.plot([d.sign_down, d.sign_up], [y, y], color="#d9d8d3", lw=1.2, zorder=1)
    ax.plot(d.sign_down, y, "o", color="#2a78d6", ms=5, zorder=2)
    ax.plot(d.sign_up, y, "o", color="#eb6834", ms=5, zorder=2)
    ax.text(-0.05, y, mo.replace("|", " · "), ha="right", va="center", fontsize=7.4)
ax.set_yticks([]); ax.set_xlim(-0.02, 1.02); ax.set_xlabel("Sign accuracy (matched pairs)")
ax.set_title("a  DOWN (blue) vs UP (orange), n = 840 matched pairs", loc="left", fontsize=9); ax.grid(axis="y", visible=False)
ax = axes[1]
for y, mo in zip(ys, models):
    d = diag.loc[mo]
    ax.plot([d.prauc_down, d.prauc_up], [y, y], color="#d9d8d3", lw=1.2, zorder=1)
    ax.plot(d.prauc_down, y, "o", color="#2a78d6", ms=5, zorder=2)
    ax.plot(d.prauc_up, y, "o", color="#eb6834", ms=5, zorder=2)
ax.axvline(0.5, color="#b9b8b3", lw=0.8, ls=":")
ax.set_yticks([]); ax.set_xlim(0.45, 0.75); ax.set_xlabel("Balanced PR-AUC (matched pairs)")
ax.set_title("b  Discrimination is preserved", loc="left", fontsize=9); ax.grid(axis="y", visible=False)
ax = axes[2]
sub = cal[cal.model == "all|xgboost"].set_index("variant")
variants = ["raw", "isotonic_crossfold", "quantile_mapped"]
vl = {"raw": "raw", "isotonic_crossfold": "isotonic", "quantile_mapped": "quantile-mapped"}
x = np.arange(len(variants)); w = 0.36
ax.bar(x - w / 2, [sub.loc[v, "sign_down"] for v in variants], w, color="#2a78d6", label="DOWN", edgecolor="white")
ax.bar(x + w / 2, [sub.loc[v, "sign_up"] for v in variants], w, color="#eb6834", label="UP", edgecolor="white")
for xi, v in zip(x, variants):
    ax.text(xi - w / 2, sub.loc[v, "sign_down"] + 0.02, f"{sub.loc[v, 'sign_down']:.2f}", ha="center", fontsize=7)
    ax.text(xi + w / 2, sub.loc[v, "sign_up"] + 0.02, f"{sub.loc[v, 'sign_up']:.2f}", ha="center", fontsize=7)
vl["quantile_mapped"] = "quantile-\nmapped"; vl["isotonic_crossfold"] = "isotonic"
ax.set_xticks(x); ax.set_xticklabels([vl[v] for v in variants], fontsize=7.6)
ax.set_ylim(0, 1.28); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("Sign accuracy")
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=0.9)
ax.set_title("c  Post-hoc calibration (all genes)", loc="left", fontsize=9); ax.grid(axis="x", visible=False)
save(fig, "fig4_direction_diagnosis")
print("done")

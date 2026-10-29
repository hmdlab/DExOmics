#!/usr/bin/env bash
# run_all.sh — ordered driver for the revision analysis (DExOmics, 2026-09 revision).
#
# Usage:  bash analysis/run_all.sh [--quick] [--data-root DIR] [--python BIN] STAGE [STAGE ...]
#   STAGE  data | features | de | exp1 | exp2 | exp4 | exp5 | figures | all
#          (all = features de exp1 exp2 exp4 exp5 figures; the `data` stage is never implied)
#   --quick     pass --quick to every script that supports it (smoke test: few epochs / small grids)
#   --data-root where raw data and results live (default $DEXOMICS_DATA or ~/Data/DExOmics)
#   --python    interpreter (default $DEXOMICS_DATA/venv/bin/python, else python3)
# Every step is logged to $DATA_ROOT/results/logs/<timestamp>_<step>.log; the driver stops at the first failure.
# Steps are listed in dependency order; see README.md for inputs/outputs and approximate run times.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
QUICK=0; STAGES=()
DATA_ROOT="${DEXOMICS_DATA:-$HOME/Data/DExOmics}"; PY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --data-root) DATA_ROOT="$2"; shift ;;
    --python) PY="$2"; shift ;;
    -h|--help) sed -n 2,11p "$0"; exit 0 ;;
    *) STAGES+=("$1") ;;
  esac; shift
done
[ ${#STAGES[@]} -gt 0 ] || { sed -n 2,11p "$0"; exit 1; }
mkdir -p "$DATA_ROOT"; DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
export DEXOMICS_DATA="$DATA_ROOT"
[ -n "$PY" ] || { PY="$DATA_ROOT/venv/bin/python"; [ -x "$PY" ] || PY=python3; }
LOG="$DATA_ROOT/results/logs"; mkdir -p "$LOG"
Q=(); [ $QUICK = 1 ] && Q=(--quick)
echo "data root : $DATA_ROOT"; echo "python    : $PY ($("$PY" --version 2>&1))"; echo "quick     : $QUICK"; echo "stages    : ${STAGES[*]}"

run() {   # run <label> [ENV=VAL ...] <script> [args...]
  local label="$1"; shift
  local envs=(); while [[ "$1" == *=* ]]; do envs+=("$1"); shift; done
  local ts; ts="$(date +%Y%m%d_%H%M%S)"; local logf="$LOG/${ts}_${label}.log"
  echo; echo "=== [$label] $(date '+%F %T')  ${envs[*]:-} $*" | tee "$logf"
  local t0=$SECONDS
  if [[ "$1" == *.R ]]; then env ${envs[@]+"${envs[@]}"} Rscript "$HERE/$@" 2>&1 | tee -a "$logf"
  elif [[ "$1" == *.sh ]]; then env ${envs[@]+"${envs[@]}"} bash "$HERE/$@" 2>&1 | tee -a "$logf"
  else env ${envs[@]+"${envs[@]}"} "$PY" "$HERE/$@" 2>&1 | tee -a "$logf"; fi
  echo "=== [$label] done in $((SECONDS - t0)) s" | tee -a "$logf"
}

stage_data() {
  mkdir -p "$DATA_ROOT/data" "$DATA_ROOT/data_drive" "$DATA_ROOT/gdc/annot"
  echo "--- distributed processed data (Google Drive folder 1etIOFisUnMDNoQ5UAiMHyz3Mo2n49dAk, link-shared; ~200 MB)"
  if [ ! -f "$DATA_ROOT/data/pancan_data/references_v8_gencode.v26.GRCh38.genes.gtf" ]; then
    if command -v gdown >/dev/null; then gdown --folder "https://drive.google.com/drive/folders/1etIOFisUnMDNoQ5UAiMHyz3Mo2n49dAk" -O "$DATA_ROOT/data_drive"
    elif command -v uvx >/dev/null; then uvx gdown --folder "https://drive.google.com/drive/folders/1etIOFisUnMDNoQ5UAiMHyz3Mo2n49dAk" -O "$DATA_ROOT/data_drive"
    else echo "install gdown (pip install gdown) or download the 4 tar.gz files manually into $DATA_ROOT/data_drive"; exit 1; fi
    for t in "$DATA_ROOT"/data_drive/*.tar.gz; do echo "extract $t"; tar -xzf "$t" -C "$DATA_ROOT/data"; done
  fi
  echo "--- GDC open-access files, TCGA-LIHC and TCGA-CESC (STAR counts, masked MAF, ASCAT3 CNV, 450k betas; ~14 GB)"
  run gdc_LIHC gdc/gdc_download_all.sh LIHC
  run gdc_CESC gdc/gdc_download_all.sh CESC
  echo "--- HM450 hg38 probe annotation (zhou-lab InfiniumAnnotationData, GENCODE v36; 17.7 MB)"
  [ -s "$DATA_ROOT/gdc/annot/methy_probe_annot.tsv.gz" ] || curl -L -o "$DATA_ROOT/gdc/annot/methy_probe_annot.tsv.gz" \
    "https://github.com/zhou-lab/InfiniumAnnotationData/raw/main/Anno/HM450/HM450.hg38.manifest.gencode.v36.tsv.gz"
  echo "--- ICGC LIRI-JP (release 28 open tier, ~170 MB)"
  run 07_liri 07_liri_download.py
}
stage_features() {
  run 00_annot 00_gene_annotation.py
  run 02_binding 02_binding_features.py
  run 03_cnv 03_cnv_features.py
  run 04_mutation 04_mutation_features.py
  run 05_methylation 05_methylation_features.py
  run 13_cesc_features 13_cesc_features.py
}
stage_de() {
  run 01_de_lihc 01_de_tcga_only.py
  run 01_de_cesc CANCER=CESC 01_de_tcga_only.py
  run 08_liri_logfc 08_liri_logfc.py
  command -v Rscript >/dev/null || echo "WARNING: Rscript not on PATH; 12 calls Rscript analysis/11_voom_labels.R (limma, edgeR)"
  run 12_label_decomposition 12_label_decomposition.py --force
}
stage_exp1() {
  run 06_nested_cv 06_nested_cv_baselines.py ${Q[@]+"${Q[@]}"}
  run 06b_mlp 06b_mlp.py ${Q[@]+"${Q[@]}"}
  run 06_nested_cv_cesc CANCER=CESC 06_nested_cv_baselines.py ${Q[@]+"${Q[@]}"}
  run 16_loco 16_loco_sensitivity.py
  run 17_importance 17_importance_stability.py
}
stage_exp2() {
  run 09_liri_validation 09_liri_external_validation.py
  run 10_direction 10_direction_diagnosis.py
  run 10_direction_c03 10_direction_diagnosis.py --caliper 0.3
  run 10_direction_c10 10_direction_diagnosis.py --caliper 1.0
  run 15_calibration 15_calibration_probe.py
}
stage_exp4() {
  run 19_tensors 19_positional_tensors.py
  run 20_positional 20_positional_baselines.py ${Q[@]+"${Q[@]}"}
  run 20_positional_agg 20_positional_baselines.py --with-agg ${Q[@]+"${Q[@]}"}
  run 21_dl 21_dl_positional.py ${Q[@]+"${Q[@]}"}
  run 21_dl_stage2 21_dl_positional.py --stage2 ${Q[@]+"${Q[@]}"}
  run 21_dl_hybrid 21_dl_positional.py --hybrid ${Q[@]+"${Q[@]}"}
  run 22_dl_seedavg 22_dl_seed_robustness.py
  run 22b_xgb_seedavg 22b_xgb_seed_robustness.py ${Q[@]+"${Q[@]}"}
  run 25_orignet 25_original_arch_pe.py ${Q[@]+"${Q[@]}"}
  run 27_pretrain 27_pretrain_cnn.py --seeds 3 ${Q[@]+"${Q[@]}"}
  run 23_cesc_dl 23_cesc_positional.py
  run 28_cesc_pretrained 28_cesc_pretrained_confirm.py ${Q[@]+"${Q[@]}"}
  run 22c_ensemble 22c_ensemble_eval.py
  run 29_classification 29_direct_classification.py ${Q[@]+"${Q[@]}"}
  run 29b_cls_from_scores 29b_classification_from_scores.py
}
stage_exp5() {
  run 30_interpretability 30_interpretability.py ${Q[@]+"${Q[@]}"}
  run 32_cesc_interp 32_cesc_interp.py ${Q[@]+"${Q[@]}"}
}
stage_figures() {
  run 18_figures 18_figures.py
  run 24_fig5 24_fig5.py
  run 26_fig_arch 26_fig_arch.py
  run 31_paper_figures 31_paper_figures.py
  run 33_workflow_fig 33_workflow_fig.py
  run 34_cesc_interp_fig 34_cesc_interp_fig.py
}
for s in "${STAGES[@]}"; do
  case "$s" in
    all) for t in features de exp1 exp2 exp4 exp5 figures; do "stage_$t"; done ;;
    data|features|de|exp1|exp2|exp4|exp5|figures) "stage_$s" ;;
    *) echo "unknown stage: $s"; exit 1 ;;
  esac
done
echo; echo "all requested stages finished $(date '+%F %T')"

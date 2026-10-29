#!/usr/bin/env bash
# smoke_test.sh — end-to-end check of the pipeline in a FRESH test root without touching the real data root.
#   bash analysis/smoke_test.sh            (real root = $DEXOMICS_DATA or ~/Data/DExOmics; must already hold data + results)
# Raw inputs (data/, gdc/, icgc/, venv/) and the large intermediates are symlinked into the test root, the small
# result folders are copied, then (1) the formerly-inline scripts (22b/22c/29b) and the figure scripts are run on
# the REAL intermediates so their outputs can be compared with the reference files, and (2) run_all.sh is executed
# with --quick for every stage except `data`. Logs: <test root>/verify/ and <test root>/results/logs/.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REAL="${DEXOMICS_DATA:-$HOME/Data/DExOmics}"
T="${1:-$HOME/Data/DExOmics_test_$(date +%Y%m%d_%H%M)}"
export DEXOMICS_DATA="$T" DEXOMICS_PROJ="$T"
PY="$REAL/venv/bin/python"; [ -x "$PY" ] || PY=python3
mkdir -p "$T/results/figures" "$T/verify/out"
for d in data gdc icgc venv; do [ -e "$REAL/$d" ] && ln -sfn "$REAL/$d" "$T/$d"; done
for d in features tensors de_lihc de_cesc de_liri exp5_interp; do [ -e "$REAL/results/$d" ] && ln -sfn "$REAL/results/$d" "$T/results/$d"; done
for d in exp1_nested_cv exp1_nested_cv_cesc exp2_liri exp3_direction exp4_dl; do [ -e "$T/results/$d" ] || cp -R "$REAL/results/$d" "$T/results/$d"; done
echo "### test root $T  (real root $REAL, python $PY)  $(date)"
echo "### PHASE 1: regenerate formerly-inline outputs from the real intermediates"
for s in 22b_xgb_seed_robustness.py 22c_ensemble_eval.py 29b_classification_from_scores.py 24_fig5.py 31_paper_figures.py 18_figures.py; do
  echo "--- $s $(date)"; "$PY" "$HERE/$s" > "$T/verify/${s%.py}.log" 2>&1; echo "exit=$?"; tail -2 "$T/verify/${s%.py}.log"
done
for f in xgb_seedavg_out.txt ensemble_lihc.json fig5_cells.json classification_from_scores.csv classification_paired_diff.json; do
  [ -f "$T/results/exp4_dl/$f" ] && cp "$T/results/exp4_dl/$f" "$T/verify/out/"
  [ -f "$REAL/results/exp4_dl/$f" ] && cp "$REAL/results/exp4_dl/$f" "$T/verify/ref_$f"
done
echo "### PHASE 2: staged driver with --quick  $(date)"
bash "$HERE/run_all.sh" --quick --data-root "$T" features de exp1 exp2 exp4 exp5 figures > "$T/verify/run_all_quick.log" 2>&1; echo "run_all exit=$?"
echo "### DONE $(date)   compare $T/verify/out/* with $T/verify/ref_*"

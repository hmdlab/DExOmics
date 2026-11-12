#!/usr/bin/env bash
# Prepare the working directories of the pan-cancer study.
#
# The 2024 pan-cancer scripts address their inputs and outputs with fixed relative paths two levels above
# their own directory (../../data, ../../results, ../../plots, ../../<project>_model, ../../test_metrics_<project>
# from pancancer/04_modeling/ and pancancer/05_interpret/; ../data and ../results from pancancer/ for 01_run.sh).
# With the code under pancancer/ these names resolve to the repository root. This script creates them there as
# symlinks into a workspace OUTSIDE the repository, so that no data, checkpoint or plot is ever tracked, and
# seeds <workspace>/data/modeling/ with the tracked parameter files and gene splits. The scripts themselves are
# not modified.
#
# Usage:  bash pancancer/setup_workspace.sh [WORKSPACE]
#         default WORKSPACE = $PANCANCER_WORKSPACE, else ~/Data/DExOmics/pancancer
#
# Re-running is safe: a symlink that already points elsewhere is reported and left alone (a dangling one is
# re-pointed); an existing data/modeling/<project>_params.json is never overwritten (train.py rewrites that file
# in place, and the rewritten values belong to the checkpoints trained with them).
# Several clones/worktrees that use the same WORKSPACE share (and overwrite) the same checkpoints and outputs:
# give each one its own WORKSPACE argument when runs must be kept apart. Symlinks are absolute: re-run this
# script after moving the clone or the workspace.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../pancancer
REPO="$(dirname "$HERE")"                                   # repository root
WS="${1:-${PANCANCER_WORKSPACE:-$HOME/Data/DExOmics/pancancer}}"
mkdir -p "$WS"
WS_ABS="$(cd "$WS" && pwd)"
case "$WS_ABS/" in
  "$REPO/"*) rmdir "$WS" 2>/dev/null || true
             echo "ERROR: the workspace ($WS_ABS) must be outside the repository ($REPO)" >&2; exit 1 ;;
esac
WS="$WS_ABS"

NAMES=(data results plots pancanatlas_model pcawg_model test_metrics_pancanatlas test_metrics_pcawg)
for n in "${NAMES[@]}"; do
  mkdir -p "$WS/$n"
  if [ -L "$REPO/$n" ]; then
    cur="$(readlink "$REPO/$n")"
    if [ "$cur" = "$WS/$n" ]; then echo "OK    $REPO/$n -> $cur"
    elif [ ! -e "$REPO/$n" ]; then
      rm "$REPO/$n"; ln -s "$WS/$n" "$REPO/$n"; echo "RELINK $REPO/$n -> $WS/$n   (was dangling: $cur)"
    else echo "KEEP  $REPO/$n -> $cur   (differs from $WS/$n; remove the link to re-point it)"; fi
  elif [ -e "$REPO/$n" ]; then
    echo "SKIP  $REPO/$n exists and is not a symlink   (move it out of the clone; nothing under it is tracked)"
  else
    ln -s "$WS/$n" "$REPO/$n"; echo "LINK  $REPO/$n -> $WS/$n"
  fi
done

# sub-directories the scripts write into without creating them (eval.py: plots/<p>_pretrained/; 01: results/*_features/)
mkdir -p "$WS/data/modeling" "$WS/results/rna_features" "$WS/results/promoter_features" "$WS/results/shap" \
         "$WS/plots/pancanatlas_pretrained" "$WS/plots/pcawg_pretrained" "$WS/pancanatlas_model/reg0.001" "$WS/pcawg_model/reg0.001"
for p in pancanatlas pcawg; do
  for s in train val test; do
    src="$HERE/config/gene_split/${p}_${s}.csv"; dst="$WS/data/modeling/${p}_${s}.csv"
    [ -e "$dst" ] || { cp "$src" "$dst"; echo "COPY  $dst"; }
  done
  src="$HERE/config/${p}_params.json"; dst="$WS/data/modeling/${p}_params.json"
  [ -e "$dst" ] || { cp "$src" "$dst"; echo "COPY  $dst   (train.py rewrites this file in place)"; }
done

cat <<EOF

Workspace: $WS
Still to be provided per project (pancanatlas, pcawg) -- see pancancer/README.md, section 4:
  $WS/data/<project>_expid.txt
  $WS/results/rna_features/output_postar_rna.pkl, output_postar_rna_feature_name.txt.gz
  $WS/results/rna_features/output_targetscan_rna.pkl, output_targetscan_rna_feature_name.txt.gz
  $WS/results/promoter_features/output_gtrd_promoter.pkl, output_gtrd_promoter_feature_name.txt.gz
  (to skip training) $WS/<project>_model/reg0.001/best_model.pth, epoch1..N.pth, feature_norm_stats.txt
EOF

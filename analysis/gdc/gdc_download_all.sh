#!/usr/bin/env bash
# gdc_download_all.sh — build GDC manifests and download all open-access files for one TCGA project.
# Usage:  bash analysis/gdc/gdc_download_all.sh LIHC        (or CESC)
# Env:    DEXOMICS_DATA (default ~/Data/DExOmics), PYTHON (default: $DEXOMICS_DATA/venv/bin/python or python3)
# Output: $DEXOMICS_DATA/gdc/<PROJ>/{manifest_*.csv, star/, maf/, cnv/, methy/} and star_counts -> star symlink.
set -euo pipefail
PROJ="${1:-LIHC}"
ROOT="${DEXOMICS_DATA:-$HOME/Data/DExOmics}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-$ROOT/venv/bin/python}"; [ -x "$PY" ] || PY=python3
export DEXOMICS_DATA="$ROOT"
mkdir -p "$ROOT/gdc/$PROJ"
echo "=== manifests for TCGA-$PROJ $(date)"
"$PY" "$HERE/gdc_manifest.py" "TCGA-$PROJ"
for k in cnv methy maf; do "$PY" "$HERE/gdc_manifest2.py" "TCGA-$PROJ" "$k"; done
echo "=== star counts $(date)"
"$PY" "$HERE/gdc_download.py" "$ROOT/gdc/$PROJ/manifest_star_counts.csv" "$ROOT/gdc/$PROJ/star"
ln -sfn star "$ROOT/gdc/$PROJ/star_counts"   # 01_de_tcga_only.py reads gdc/<PROJ>/star_counts/
for k in maf cnv methy; do
  echo "=== $k $(date)"; "$PY" "$HERE/gdc_download.py" "$ROOT/gdc/$PROJ/manifest_$k.csv" "$ROOT/gdc/$PROJ/$k"
done
echo "=== done $(date)"

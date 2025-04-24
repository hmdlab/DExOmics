#!/bin/bash

#SBATCH -t 0-72:00:00
#SBATCH -p medium
#SBATCH --mem-per-cpu 64g
#SBATCH -J CESC_methy

# Rscript load_expr.R LIHC
# Rscript load_cnv.R LIHC
# Rscript load_methy.R LIHC
# Rscript load_snv.R LIHC

# Rscript load_expr.R CESC
# Rscript load_cnv.R CESC
Rscript load_methy.R CESC
# Rscript load_snv.R CESC
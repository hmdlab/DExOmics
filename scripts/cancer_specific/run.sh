#!/bin/bash

#SBATCH -t 0-72:00:00
#SBATCH -p medium
#SBATCH --mem-per-cpu 64g
#SBATCH -J data_merge

# Rscript data_observe.R LIHC
# Rscript dea.R LIHC hepg2
Rscript data_merge.R LIHC hepg2 TRUE   
# python get_HepG2_genes.py LIHC hepg2

# Rscript data_observe.R CESC
# Rscript dea.R CESC hela
Rscript data_merge.R CESC hela FALSE
# python get_encode_genes.py CESC hela
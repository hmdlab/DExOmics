#!/bin/bash

#SBATCH -t 0-1:00:00
#SBATCH -p short
#SBATCH --mem-per-cpu 64g
#SBATCH -J LIHC_shap_plot

# Rscript data_observe.R LIHC
# Rscript dea.R LIHC hepg2
# Rscript data_merge.R LIHC hepg2 TRUE  
# python get_encode_genes.py LIHC hepg2

# Rscript data_observe.R CESC
# Rscript dea.R CESC hela
# Rscript data_merge.R CESC hela FALSE
# python get_encode_genes.py CESC hela

# python pretrain.py LIHC hepg2 ../../model_LIHC/concat/ -bs 50 -n 100 -lr 0.001 -step 30 -reg 0.0001
# python eval.py LIHC hepg2 ../../model_LIHC/concat/ -n 100 -reg 0.0001

# python compute_shap.py LIHC hepg2 ../../shap/ExpectedGrad_LIHC/
# Rscript summarize_SHAP.R LIHC ../../shap/ExpectedGrad_LIHC/
Rscript shap_plot.R ../../shap/ExpectedGrad_LIHC/ ../../plots_LIHC/global/
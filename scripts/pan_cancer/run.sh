#!/bin/bash

#SBATCH -t 0-1:00:00
#SBATCH -p short
#SBATCH --mem-per-cpu 64g
#SBATCH -J shap_plot


# python pretrain.py ../../pancanatlas_model/ -p pancanatlas -bs 50 -n 100 -lr 0.001 -step 30 -reg 0.001
# python eval.py ../../pancanatlas_model/ -p pancanatlas -n 100 -reg 0.001

# python compute_shap.py ../../shap/DeepLIFT_pancanatlas/ -p pancanatlas
# Rscript summarize_SHAP.R pancanatlas ../../shap/DeepLIFT_pancanatlas/
Rscript shap_plot.R pancanatlas ../../shap/DeepLIFT_pancanatlas/ ../../plots_pancanatlas/
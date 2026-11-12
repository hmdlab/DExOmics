#!/bin/bash

#SBATCH -p medium
#SBATCH -t 0-72:00:00
#SBATCH --mem-per-cpu 10g
#SBATCH -J shap_plot

# GLOBAL SHAP -------------------------------------------------------------------------
# PanCanAtlas
# python compute_shap.py ../../results/shap/DeepLIFT_pancanatlas/ -p pancanatlas
# Rscript summarize_SHAP.R 0.001 pancanatlas ../../results/shap/DeepLIFT_pancanatlas/
# python compute_shap.py ../../results/shap/ExpectedGrad_pancanatlas/ -p pancanatlas
# Rscript summarize_SHAP.R 0.001 pancanatlas ../../results/shap/ExpectedGrad_pancanatlas/
Rscript shap_plot.R pancanatlas ../../results/shap/DeepLIFT_pancanatlas/ ../../plots/


# PCAWG
# python compute_shap.py ../../results/shap/DeepLIFT_pcawg/ -p pcawg
# Rscript summarize_SHAP.R 0.001 pcawg ../../results/shap/DeepLIFT_pcawg/
# python compute_shap.py ../../results/shap/ExpectedGrad_pcawg/ -p pcawg
# Rscript summarize_SHAP.R 0.001 pcawg ../../results/shap/ExpectedGrad_pcawg/
Rscript shap_plot.R pcawg ../../results/shap/DeepLIFT_pcawg/ ../../plots/


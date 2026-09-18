#!/bin/bash

#SBATCH -p medium
#SBATCH -t 1-00:00:00
#SBATCH --mem=32G
#SBATCH -J split

# Rscript split_data.R LIHC FALSE
python full_train_auto.py LIHC hepg2 ../../model_LIHC/concat/ --n_trials 80 --final_epochs 100
python full_eval_auto.py LIHC hepg2 ../../model_LIHC/concat/
# python run_xgboost_auto.py LIHC hepg2 ../../model_LIHC/concat/

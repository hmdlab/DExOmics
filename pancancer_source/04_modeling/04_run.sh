#!/bin/bash

#SBATCH -p medium
#SBATCH -t 0-72:00:00
#SBATCH --mem-per-cpu 10g
#SBATCH -J pancan_plot

# reg_values=(0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5)

# for reg_value in "${reg_valuessdf[@]}"
# do
#     echo "Training with reg = $reg_value"
#     # python train.py ../../results/full_model/reg$reg_value/ -bs 50 -hs 64 -n 100 -lr 0.001 -step 30 -reg $reg_value
#     # python baseline_train.py ../../results/baseline_model/reg$reg_value/ -bs 50 -hs 64 -n 100 -lr 0.001 -step 30 -reg $reg_value
#     python encode_train.py ../../results/encode_model/reg$reg_value/ -bs 50 -hs 32 -n 100 -lr 0.001 -step 30 -reg $reg_value
# done
# 
# python train.py ../../pancanatlas_model/reg0.001/ -p pancanatlas -bs 50 -n 100 -lr 0.001 -step 30 -reg 0.001
# python eval.py ../../pancanatlas_model/reg0.001/ -p pancanatlas -n 100 -reg 0.001


# python train.py ../../pcawg_model/reg0.001/ -p pcawg -bs 50 -n 100 -lr 0.001 -step 30 -reg 0.001
# python eval.py ../../pcawg_model/reg0.001/ -p pcawg -n 100 -reg 0.001

Rscript pancanatlas_visual.R pancanatlas
Rscript pcawg_visual.R pcawg

#!/bin/bash

#SBATCH -t 0-1:00:00
#SBATCH -p short
#SBATCH --mem-per-cpu 64g
#SBATCH -J pancanatlas_train


python pretrain.py ../../pancanatlas_model/ -p pancanatlas -bs 50 -n 100 -lr 0.001 -step 30 -reg 0.001
# python eval.py ../../pancanatlas_model/ -p pancanatlas -n 100 -reg 0.001


# python train.py ../../pcawg_model/reg0.001/ -p pcawg -bs 50 -n 100 -lr 0.001 -step 30 -reg 0.001
# python eval.py ../../pcawg_model/reg0.001/ -p pcawg -n 100 -reg 0.001
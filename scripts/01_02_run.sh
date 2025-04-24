#!/bin/bash

#SBATCH -t 0-72:00:00
#SBATCH -p medium
#SBATCH --mem-per-cpu 64g
#SBATCH -J 02_to_sparse

# Rscript 01_bed_to_RNA_coord.R -b "../data/hepg2_bed_promoter" -n 100 -g "../data/pancan_data/references_v8_gencode.v26.GRCh38.genes.gtf" -t "promoter" -o "../data/promoter_features/encode_hepg2_promoter" -s "ENCODE"
# Rscript 01_bed_to_RNA_coord.R -b "../data/hepg2_bed_rna" -n 100 -g "../data/pancan_data/references_v8_gencode.v26.GRCh38.genes.gtf" -t "rna" -o "../data/rna_features/encode_hepg2_rna" -s "ENCODE"
# Rscript 01_bed_to_RNA_coord.R -b "../data/hela_bed_promoter" -n 100 -g "../data/pancan_data/references_v8_gencode.v26.GRCh38.genes.gtf" -t "promoter" -o "../data/promoter_features/encode_hela_promoter" -s "ENCODE"
# Rscript 01_bed_to_RNA_coord.R -b "../data/hela_bed_rna" -n 100 -g "../data/pancan_data/references_v8_gencode.v26.GRCh38.genes.gtf" -t "rna" -o "../data/rna_features/encode_hela_rna" -s "POSTAR3"


python 02_to_sparse.py ../data/promoter_features/encode_hepg2_promoter.txt
python 02_to_sparse.py ../data/rna_features/encode_hepg2_rna.txt
python 02_to_sparse.py ../data/promoter_features/encode_hela_promoter.txt
python 02_to_sparse.py ../data/rna_features/encode_hela_rna.txt
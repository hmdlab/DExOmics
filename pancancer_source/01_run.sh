#!/bin/bash

#$ -cwd 
#$ -V 
#$ -l gpu
#$ -l d_rt=48:00:00
#$ -l s_rt=48:00:00
#$ -l s_vmem=64G 
#$ -l mem_req=64G
#$ -N RNAtrans
#$ -S /bin/bash

export PATH=$HOME/local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/lib64/R/lib:$HOME/local/lib64:$LD_LIBRARY_PATH

# Rscript 01_bed_to_RNA_coord.R -b "../../data/gtrd/Meta-clusters_by_TF/bed" -n 100 -g "../data/references_v8_gencode.v26.GRCh38.genes.gtf" -t "promoter" -o "../results/promoter_features/output_gtrd_promoter" -s "GTRD"
Rscript 01_bed_to_RNA_coord.R -b "../../data/postar3/bed_rna" -n 100 -g "../data/references_v8_gencode.v26.GRCh38.genes.gtf" -t "rna" -o "../results/rna_features/output_postar_rna" -s "POSTAR3"
# Rscript 01_bed_to_RNA_coord.R -b "../../data/targetscan/hg38_beds" -n 100 -g "../data/references_v8_gencode.v26.GRCh38.genes.gtf" -t "rna" -o "../results/rna_features/output_targetscan_rna" -s "TARGETSCAN"

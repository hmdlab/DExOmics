### This script visualizes the raw data and finds common samples and 
### samples having replicates in omics data types

library(tidyverse)
library(stringr)
library(ComplexHeatmap)
library(TCGAbiolinks)
library(SummarizedExperiment)


# Download omics data ----------------------------------------------------------
# Get cancer from the terminal command line
cancer <- as.character(commandArgs(trailingOnly = TRUE))

# Expression
# dataDEGs <- get(load("../../data/TCGAdata/dataDEGs.rda"))
dataFilt <- get(load(paste0("../../../DEomics/data/TCGAdata/", cancer, "/", cancer, ".expr.prep.rda")))

# CNV
CNV <- get(load(paste0("../../../DEomics/data/TCGAdata/", cancer, "/", cancer, ".cnv.rda")))
CNV <- assay(CNV)
rm(data)

# Mutation
Mut <- get(load(paste0("../../../DEomics/data/TCGAdata/", cancer, "/", cancer, ".snv.rda")))

# Methylation
Met <- get(load(paste0("../../../DEomics/data/TCGAdata/", cancer, "/", cancer, ".methy.rda")))
Met <- assay(Met)
rm(data)



# Observe data -----------------------------------------------------------------
# Find overlapping samples in all data types
# Build up lists
sample_list <- list("Expression" = unique(str_sub(colnames(dataFilt), 1, 12)),
                     "CNV" = unique(str_sub(colnames(CNV), 1, 12)),
                     "Mutation" = unique(str_sub(Mut$Tumor_Sample_Barcode, 1, 12)),
                     "Methylation" = unique(str_sub(colnames(Met), 1, 12)))


# Input matrix of UpSet function
sample_matrix <- make_comb_mat(sample_list)

# Extract overlapping patients
common <- extract_comb(sample_matrix, "1111")
# (All of them are tumor samples!!)

write.csv(common,
          file=paste0("../../data/TCGAdata/", cancer, "/common_patients.csv"),
          row.names = F)

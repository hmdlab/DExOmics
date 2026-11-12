### This script processed PCAWG gene expression data.
library(tidyverse)
library(purrr)
library(dplyr)
library(stringr)
library(readr)
library(data.table)
library(SummarizedExperiment)
library(TCGAbiolinks)

# Load data
expr_data <- read.table("../../../data/GDCdata/PCAWG/tophat_star_fpkm.v2_aliquot_gl.tsv", sep="\t", header=TRUE)
patient_info <- read.delim("../../../data/GDCdata/PCAWG/rnaseq.extended.metadata.aliquot_id.V4.tsv", header=TRUE, stringsAsFactors=FALSE)

# Correct gene IDs
expr_df <- expr_data %>%
  mutate(gene_id = feature) %>% 
  select(-feature) %>% 
  filter(gene_id != "?") %>% 
  # Delete duplicated gene_ids
  group_by(gene_id) %>%
  filter(n() == 1) %>%
  ungroup() %>% 
  column_to_rownames(var = "gene_id")
colnames(expr_df) <- gsub("\\.", "-", colnames(expr_df))
colnames(expr_df) <- str_sub(colnames(expr_df), 2)
rownames(expr_df) <- sapply(strsplit(rownames(expr_df), "\\."), `[`, 1)

# Delete normal samples
normal_samples <- patient_info %>% 
  filter(specimen_type == "Normal - tissue adjacent to primary" | specimen_type == "Normal - solid tissue") %>% 
  pull(aliquot_id)
expr_df <- expr_df %>% select(-one_of(normal_samples))

# Replace Aliquot_id with project code
# Gather same cancer types
cancer_types <- unique(patient_info$project_code)
new_df <- data.frame(matrix(ncol = length(cancer_types), nrow = nrow(expr_df)))
colnames(new_df) <- cancer_types

# Iterate every cancer type
for (cancer in cancer_types) {
  
  # Map cancer type to patient barcode
  patients_in_cancer <- patient_info %>%
    filter(project_code == cancer) %>%
    pull(aliquot_id)
  
  # 确保选择的列在原数据框中存在
  patients_in_cancer <- intersect(patients_in_cancer, colnames(expr_df))
  
  # 计算每一行相同癌症类型列的中位数，并创建新列
  new_df[[cancer]] <- apply(expr_df[, patients_in_cancer], 1, median)
}
rownames(new_df) <- rownames(expr_df)
cancer_df <- na.omit(new_df)

# Keep genes with expressed greater than two TPM in at least one cancer
cancer_df <- cancer_df[rowSums(cancer_df > 2) > 0, ]

# Log2 transformation
cancer_df <- cancer_df %>% mutate(across(everything(), ~ ifelse(. >= 0, log2(. + 0.25), NA)))
cancer_df <- na.omit(cancer_df)

# Calculate median expression
cancer_df$MedianExp <- apply(cancer_df[, -1], 1, function(x) median(x, na.rm = FALSE))

# Rownames to colum
cancer_df <- cancer_df %>% rownames_to_column("Name")
write.table(cancer_df, file = "../../data/pcawg_exp.txt", sep = "\t", row.names = FALSE, col.names = TRUE)
### This script processed PanCanAtlas gene expression data.
library(tidyverse)
library(purrr)
library(dplyr)
library(stringr)
library(readr)
library(data.table)
library(SummarizedExperiment)
library(TCGAbiolinks)


find_replicates <- function(omics_df){
  # Delete normal samples and non-primary tumor samples
  normal_index <- which(as.numeric(str_sub(colnames(omics_df), 14, 15)) > 9)
  omics_df <- omics_df[, -normal_index]
  nonTP_index <- which(as.numeric(str_sub(colnames(omics_df), 14, 15)) > 1)
  omics_df <- omics_df[, -nonTP_index]
  
  # Find total patients
  total_patients <- str_sub(colnames(omics_df), 1, 12)
  
  # Find replicated sample codes
  rep_patients <- unique(total_patients[duplicated(total_patients)])
  if (length(rep_patients) != 0){
    for (p in rep_patients){
      tmp <- omics_df %>% select(starts_with(p))
      
      # Calculate mean of the replicated samples
      row_means <- apply(tmp, 1, mean, na.rm=TRUE)
      row_means <- round(row_means, 3)
      omics_df <- cbind(omics_df, row_means)
      
      # Rename last column
      colnames(omics_df)[length(colnames(omics_df))] <- p
      # Remain integers
      omics_df[,length(colnames(omics_df))] <- round(omics_df[,length(colnames(omics_df))])
      # Delete replicates
      omics_df <- omics_df[, !colnames(omics_df) %in% colnames(tmp)] 
    }
  }
  # Change all column names to sample ID
  colnames(omics_df) <- str_sub(colnames(omics_df), 1, 12)
  return(omics_df)
}



# Load data
expr_data <- read.table("../../../data/GDCdata/PanCanAtlas/EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv", sep="\t", header=TRUE)
patient_info <- read.delim("../../../data/GDCdata/PanCanAtlas/clinical_PANCAN_patient_with_followup.tsv", header=TRUE, stringsAsFactors=FALSE)

# Correct gene IDs
expr_df <- expr_data %>%
  mutate(gene_id = sub("\\|.*", "", gene_id)) %>%
  filter(gene_id != "?") %>% 
  # Delete duplicated gene_ids
  group_by(gene_id) %>%
  filter(n() == 1) %>%
  ungroup() %>% 
  column_to_rownames(var = "gene_id")
colnames(expr_df) <- gsub("\\.", "-", colnames(expr_df))

# Deal with replicates
expr_df <- find_replicates(expr_df)

# Gather same cancer types
cancer_types <- unique(patient_info$acronym)
new_df <- data.frame(matrix(ncol = length(cancer_types), nrow = nrow(expr_df)))
colnames(new_df) <- cancer_types

# Iterate every cancer type
for (cancer in cancer_types) {
  
  # Map cancer type to patient barcode
  patients_in_cancer <- patient_info %>%
    filter(acronym == cancer) %>%
    pull(bcr_patient_barcode)
  
  # Confirm the existence of patient barcode in expr_df
  patients_in_cancer <- intersect(patients_in_cancer, colnames(expr_df))
  
  # Take the median for each cancer type
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
write.table(cancer_df, file = "../../data/pancanatlas_exp.txt", sep = "\t", row.names = FALSE, col.names = TRUE)

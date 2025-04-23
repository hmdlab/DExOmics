#### This script is to separately apply DEA on TCGA gene expression data and ENCODE gene expresson data, and find out the overlap.
library(SummarizedExperiment)
library(TCGAbiolinks)
library(tidyverse)
library(readr)
library(stringr)
library(ggrepel)

# Get cancer from the terminal command line
cancer <- as.character(commandArgs(trailingOnly = TRUE))[1]
cell_line <- as.character(commandArgs(trailingOnly = TRUE))[2]

# TCGA DEA --------------------------------------------------------
# *** Import data ***
expr_data <- get(load(paste0("../../data/TCGAdata/", cancer, "/", cancer, ".exp.rda")))
met_data <- get(load(paste0("../../data/TCGAdata/", cancer, "/", cancer, ".methy.rda")))

# *** DEA using both TPM counts from TCGA and ENCODE***
# Get TCGA TPM-normalized counts
tcga_expr <- as.data.frame(assay(expr_data))
tcga_tpm <- as.data.frame(expr_data@assays@data@listData[["tpm_unstrand"]])
colnames(tcga_tpm) <- expr_data$barcode
rownames(tcga_tpm) <- rownames(tcga_expr)


# Delete uncertain genes
tcga_tpm <- tcga_tpm %>% 
  rownames_to_column("gene_id") %>%
  filter(str_starts(gene_id, "ENSG")) %>% 
  filter(!str_detect(gene_id, "_PAR_Y"))
tcga_tpm$gene_id <- sapply(strsplit(tcga_tpm$gene_id, "\\."), `[`, 1)
rownames(tcga_tpm) <- tcga_tpm$gene_id

# selection of normal samples "NT"
samplesNT_tcga <- TCGAquery_SampleTypes(barcode = colnames(tcga_tpm), typesample = c("NT"))

# selection of tumor samples "TP"
samplesTP_tcga <- TCGAquery_SampleTypes(barcode = colnames(tcga_tpm), typesample = c("TP"))

# TCGA DEA
dataDEG <- TCGAanalyze_DEA(
  mat1 = tcga_tpm[,samplesNT_tcga],
  mat2 = tcga_tpm[,samplesTP_tcga],
  Cond1type = "Normal",
  Cond2type = "Tumor",
  pipeline = "limma",
  voom = TRUE,
  fdr.cut = 0.05 ,
  logFC.cut = 1,
  method = "glmLRT"
)
print("done")

# Add DE label
dataDEG <- dataDEG %>% 
  mutate(tcga_DElabel = case_when(logFC > 1 ~ "UP",
                             logFC < 1 ~ "DOWN"))
dataDEG <- rownames_to_column(dataDEG, var = "geneID")

# Merge with gene information table
genes <- as.data.frame(rowRanges(expr_data)) %>% 
  filter(str_starts(gene_id, "ENSG")) %>% 
  filter(!str_detect(gene_id, "_PAR_Y"))
genes <- rownames_to_column(genes, var = "geneID")
genes$geneID <- sapply(strsplit(genes$geneID, "\\."), `[`, 1)
dataDEG <- inner_join(genes, dataDEG, by = "geneID") %>% 
  select(geneID, gene_type, gene_name, logFC, P.Value, tcga_DElabel)


# Save data 
# Check directory 
folder_path <- paste0("../../data/TCGAprocessed/", cancer, "/")
if (!dir.exists(folder_path)) {
  dir.create(folder_path, recursive = TRUE)
} 
save(dataDEG, file = paste0(folder_path, "dataDEG.rda"))



# *** DMA ***
# Delete NAs
met <- subset(met_data, subset = (rowSums(is.na(assay(met_data))) == 0))

# *** DMA ***
dataDMC <- TCGAanalyze_DMC(
  data = met, 
  groupCol = "shortLetterCode",
  group1 = "TP",
  group2 = "NT",
  p.cut = 0.01,
  diffmean.cut = 0.25,
  legend = "State",
  plot.filename = paste0("../../plots_", cancer, "/DMC_volcano.pdf")
)
save(dataDMC, file = paste0(folder_path, "dataDMC.rda"))



# ENCODE DEA --------------------------------------------------------
# *** Data preparation ***
# Read files
file_list1 <- list.files(path = paste0("../../data/", cell_line, "_expr_tumor"),
                        pattern = "^ENCFF.*\\.tsv$")
file_list2 <- list.files(path = paste0("../../data/", cell_line, "_expr_normal"),
                        pattern = "^ENCFF.*\\.tsv$")

if (length(file_list1) != 0 && length(file_list2) != 0){
  # Load HepG2 data (disease group)
  tumor_data <- list.files(path = paste0("../../data/", cell_line, "_expr_tumor"), 
                        pattern = "^ENCFF.*\\.tsv$", 
                        full.names = TRUE) %>%
    map(read_tsv) %>%
    reduce(bind_cols) %>% 
    select(all_of("gene_id...1"), starts_with("TPM..."))
  n_sample = length(file_list1)
  colnames(tumor_data) <- c("gene_id", paste0("TP", seq(1, n_sample)))

  # Delete gene_ids that not starts with ENSG and uncertained genes
  tumor_data <- tumor_data %>% 
    filter(str_starts(gene_id, "ENSG")) %>% 
    filter(!str_detect(gene_id, "_PAR_Y"))
  tumor_data$gene_id <- sapply(strsplit(tumor_data$gene_id, "\\."), `[`, 1)


  # Load liver tissue data (control group)
  normal_data <- list.files(path = paste0("../../data/", cell_line, "_expr_normal"), 
                          pattern = "^ENCFF.*\\.tsv$", 
                          full.names = TRUE) %>%
    map(read_tsv) %>%
    reduce(bind_cols) %>% 
    select(all_of("gene_id...1"), starts_with("TPM..."))
  n_sample = length(file_list2)
  colnames(normal_data) <- c("gene_id", paste0("NT", seq(1, n_sample)))

  # Delete gene_ids that not starts with ENSG and uncertained genes
  normal_data <- normal_data %>% 
    filter(str_starts(gene_id, "ENSG")) %>% 
    filter(!str_detect(gene_id, "_PAR_Y"))
  normal_data$gene_id <- sapply(strsplit(normal_data$gene_id, "\\."), `[`, 1)


  # Merge data frames
  encode_expr_df <- tumor_data %>% 
    inner_join(normal_data, by = "gene_id")
  encode_expr_df <- as.data.frame(encode_expr_df)
  row.names(encode_expr_df) <- encode_expr_df$gene_id
  encode_expr_df <- encode_expr_df[-1]
  save(encode_expr_df, file = paste0("../../data/", cell_line, "_expr_data.rda"))


  # *** DEA ***
  samplesTP_encode <- colnames(tumor_data)[-1]
  samplesNT_encode <- colnames(normal_data)[-1]

  encodeDEG <- TCGAanalyze_DEA(
    mat1 = encode_expr_df[,samplesNT_encode],
    mat2 = encode_expr_df[,samplesTP_encode],
    Cond1type = "Normal",
    Cond2type = "Tumor",
    pipeline = "limma",
    voom = TRUE,
    fdr.cut = 0.05 ,
    logFC.cut = 1,
    method = "glmLRT")

  # Add DE label
  encodeDEG <- encodeDEG %>% 
    mutate(encode_DElabel = case_when(logFC > 1 ~ "UP",
                              logFC < 1 ~ "DOWN"))
  encodeDEG <- rownames_to_column(encodeDEG, var = "geneID")
  save(encodeDEG, file = paste0("../../data/", cell_line, "DEG.rda"))
}else{
  print("Counld find both transcriptome data sources.")
}


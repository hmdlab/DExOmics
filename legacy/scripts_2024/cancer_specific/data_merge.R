### This script wrangles different omics data to a gene-level base
library(tidyverse)
library(purrr)
library(dplyr)
library(stringr)
library(readr)
library(data.table)
library(SummarizedExperiment)
library(TCGAbiolinks)
library(ComplexHeatmap)
source("data_func.R")

# Get cancer from the terminal command line
cancer <- as.character(commandArgs(trailingOnly = TRUE))[1]
cell_line <- as.character(commandArgs(trailingOnly = TRUE))[2]
encode_expr_on <- as.character(commandArgs(trailingOnly = TRUE))[3]

# Load preprocessed data -------------------------------------------------------
# Expression
tcgaDEG <- get(load(paste0("../../data/TCGAprocessed/", cancer, "/dataDEG.rda")))
dataExpr <- get(load(paste0("../../data/TCGAdata/", cancer, "/", cancer, ".exp.rda")))
if (encode_expr_on){
  encodeDEG <- get(load(paste0("../../data/", cell_line, "DEG.rda")))
  encode_expr_df <- get(load(paste0("../../data/", cell_line, "_expr_data.rda")))
}

# CNV
CNV <- get(load(paste0("../../data/TCGAdata/", cancer, "/", cancer, ".cnv.rda")))

# Mutation
Mut <- get(load(paste0("../../data/TCGAdata/", cancer, "/", cancer, ".snv.rda")))

# Methylation
probes <- read_tsv("../../data/TCGAdata/HM450.hg38.manifest.gencode.v36.tsv")
Met <- get(load(paste0("../../data/TCGAdata/", cancer, "/", cancer, ".methy.rda")))
dataDMC <- get(load(paste0("../../data/TCGAprocessed/", cancer, "/dataDMC.rda")))

# Common patients
common_patients <- read.csv(paste0("../../data/TCGAdata/", cancer, "/common_patients.csv"))$x




# Wrangle data -----------------------------------------------------------------
# *** Expression ***
gene_info <- as.data.frame(rowRanges(dataExpr)) %>%
  filter(str_starts(gene_id, "ENSG")) %>% 
  filter(!str_detect(gene_id, "_PAR_Y"))
gene_info$gene_id <- sapply(strsplit(gene_info$gene_id, "\\."), `[`, 1)       # modify gene ID

# Make mapping data frame
map_info <- gene_info %>% 
  # Keep only lncRNA and protein-coding genes in differential expression data
  filter(gene_type %in% c("protein_coding", "lncRNA")) %>% 
  select(gene_id, gene_name, gene_type) %>% 
  rownames_to_column("name") %>% 
  select(-name) %>% 
  mutate(Gene = gene_id) %>% 
  select(-gene_id) %>% 
  distinct(Gene, .keep_all = TRUE) #%>% 
  # inner_join(as.data.frame(dataFilt) %>% 
  #              rownames_to_column("Gene") %>% 
  #              select(Gene),
  #            by = "Gene")
write_csv(map_info, file = paste0("../../data/TCGAprocessed/", cancer, "/mapped_geneID.csv"))

# Prepare tcga DEGs
tcgaDEG_prepare <- tcgaDEG %>%
  filter(gene_type %in% c("protein_coding", "lncRNA")) %>%
  distinct(geneID, .keep_all = TRUE) %>% 
  mutate(Gene = geneID) %>%
  select(-geneID) %>%
  full_join(map_info, 
            by = c("Gene", "gene_name", "gene_type")) %>% 
  mutate(tcga_DElabel = ifelse(is.na(tcga_DElabel), "NonDEG", tcga_DElabel)) %>% 
  select(-c(P.Value))

# Prepare encode DEGs
if (encode_expr_on){
  encodeDEG_prepare <- encodeDEG %>% 
  select(geneID, logFC, encode_DElabel) %>% 
  full_join(encode_expr_df %>% 
              rownames_to_column("geneID") %>% 
              select("geneID"),
            by = "geneID") %>% 
  mutate(encode_DElabel = ifelse(is.na(encode_DElabel), "NonDEG", encode_DElabel)) %>%
  mutate(Gene = geneID) %>%
  # Select only protein-coding genes and lncRNAs
  inner_join(map_info, by = c("Gene")) %>%
  select(Gene, gene_name, gene_type, logFC, encode_DElabel)

  # Merge DEGs
  dataDEG_prepare <- inner_join(tcgaDEG_prepare, encodeDEG_prepare, by = c("Gene", "gene_name", "gene_type")) %>% 
  # Selet out only the overlapping genes
  filter(tcga_DElabel == encode_DElabel) %>% 
  mutate(DElabel = tcga_DElabel) %>% 
  select(Gene, gene_name, gene_type, DElabel)
  write_csv(dataDEG_prepare, file = paste0("../../data/TCGAprocessed/", cancer, "/DEG_info.csv"))
  write_csv(tcgaDEG_prepare %>% mutate(DElabel = tcga_DElabel) %>% select(-tcga_DElabel), file = paste0("../../data/TCGAprocessed/", cancer, "/tcgaDEG.csv"))
  write_csv(encodeDEG_prepare %>% mutate(DElabel = encode_DElabel) %>% select(-encode_DElabel), file = paste0("../../data/", cell_line, "DEG.csv"))

}else {
  dataDEG_prepare <- tcgaDEG_prepare %>% mutate(DElabel = tcga_DElabel) %>% select(-tcga_DElabel) %>% select(Gene, gene_name, gene_type, DElabel)
  write_csv(dataDEG_prepare, file = paste0("../../data/TCGAprocessed/", cancer, "/DEG_info.csv"))
  write_csv(tcgaDEG_prepare %>% mutate(DElabel = tcga_DElabel) %>% select(-tcga_DElabel), file = paste0("../../data/TCGAprocessed/", cancer, "/tcgaDEG.csv"))
}


# *** CNV ***
CNV <- assay(CNV)
colnames(CNV) <- str_sub(colnames(CNV), 1, 12)      # change column names
rownames(CNV) <- sapply(strsplit(rownames(CNV), "\\."), `[`, 1)     # change row names
cnv <- CNV[!apply(is.na(CNV), 1, all), ]       # delete NA rows
cnv <- as.data.frame(cnv)
cnv <- cnv[, which(colnames(cnv) %in% common_patients)]      # keep common patients

# Average the values of the patients
# Average patients' values
genewise_cnv <- cnv %>% 
  rownames_to_column("Gene") %>% 
  rowwise() %>%
  mutate(Amp_rate = round(sum(c_across(starts_with("TCGA")) > 2) / length(common_patients), 3)) %>% 
  mutate(Del_rate = round(sum(c_across(starts_with("TCGA")) < 2) / length(common_patients), 3)) %>% 
  select(Gene, Amp_rate, Del_rate)
write_csv(genewise_cnv, file = paste0("../../data/TCGAprocessed/", cancer, "/genewise_cnv.csv"))



# *** Methylation ***
met <- as.data.frame(assay(Met)) 
met <- find_replicates(met)     # averge replicates
met <- met[, which(colnames(met) %in% common_patients)]      # keep common patients
met_wrangeled <- dataDMC %>% 
  # delete non-differentially methylated probes
  filter(!status == "Not Significant") %>% 
  rownames_to_column("probeID") %>% 
  mutate(delta = mean.TP.minus.mean.NT) %>% 
  select(probeID, delta) %>% 
  inner_join(probes %>% 
               select(probeID, genesUniq, CGIposition) %>% 
               # Keep only genesUniq with one gene/transcript
               filter(!grepl(';', genesUniq)),
             by = "probeID") %>% 
  group_by(genesUniq, CGIposition) %>% 
  # calculate mean delta across probes for each gene
  summarise_at(vars(-probeID), mean) %>% 
  pivot_wider(names_from = CGIposition,
              values_from = -c(genesUniq, CGIposition),
              values_fill = 0) %>% 
  ungroup()
# change column name
colnames(met_wrangeled)[2] <- "restPositions"
# met_wrangeled <- met_wrangeled %>% select(-Other)

# map gene symbol to ensembl ID
genewise_met <- met_wrangeled %>% 
  inner_join(map_info %>% select(-gene_type), by = join_by("genesUniq" == "gene_name")) %>%
  select(-genesUniq)
write_csv(genewise_met, file = paste0("../../data/TCGAprocessed/", cancer, "/genewise_met.csv"))



# *** Mutation ***
# Make gene-wise mutation table
genewise_mut <- Mu_wrangle(mu_df=Mut, patientIDs=common_patients, cscape=FALSE)
write_csv(genewise_mut, file = paste0("../../data/TCGAprocessed/", cancer, "/genewise_mut.csv"))



# *** Merge ***
merged_tcga <- full_join(genewise_cnv, genewise_met, by = "Gene") %>% 
  mutate_all(~ replace(., is.na(.), 0)) %>%
  full_join(genewise_mut, by = "Gene") %>% 
  mutate_all(~ replace(., is.na(.), 0)) %>%
  inner_join(tcgaDEG_prepare %>% 
               select(Gene, tcga_DElabel), 
             by = "Gene") %>%
  mutate(DElabel = tcga_DElabel) %>%
  select(-tcga_DElabel)
write_csv(merged_tcga, file = paste0("../../data/TCGAprocessed/", cancer, "/merged_tcga.csv"))


# Full join
merged_tcga_encode <- full_join(genewise_cnv, genewise_met, by = "Gene") %>% 
  mutate_all(~ replace(., is.na(.), 0)) %>%
  full_join(genewise_mut, by = "Gene") %>% 
  mutate_all(~ replace(., is.na(.), 0)) %>%
  inner_join(dataDEG_prepare %>% 
               select(Gene, DElabel), 
             by = "Gene")
write_csv(merged_tcga_encode, file = paste0("../../data/TCGAprocessed/", cancer, "/merged_tcga_encode.csv"))
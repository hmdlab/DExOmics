library(tidyverse)
library(readr)
library(SummarizedExperiment)

# Load data
deg_data <- read.table("../../data/pcawg_exp.txt", header = TRUE, sep = "\t")
rbp_genes <- read_csv("../../data/modeling/rbp_genes.csv")$Name
mirna_genes <- read_csv("../../data/modeling/miRNA_genes.csv")$Name
tf_genes <- read_csv("../../data/modeling/tf_genes.csv")$Name
dataExpr <- get(load("../../../DEomics/data/TCGAdata/LIHC/LIHC.exp.rda"))
gene_info <- as.data.frame(rowRanges(dataExpr)) %>%
  filter(str_starts(gene_id, "ENSG")) %>% 
  filter(!str_detect(gene_id, "_PAR_Y"))
gene_info$gene_id <- sapply(strsplit(gene_info$gene_id, "\\."), `[`, 1)       # modify gene ID


# Merge with gene_info to get chromosome information
deg_data <- deg_data %>% inner_join(gene_info %>% select(seqnames, gene_name, gene_id), join_by("Name" == "gene_id"))

# Set seed
set.seed(1)

# trian:val = 4:1
# Only interaction data
# Merge with interaction data
merged_data <- deg_data %>% distinct(Name, .keep_all=TRUE) %>% filter(Name %in% rbp_genes) %>% filter(Name %in% mirna_genes) %>% filter(Name %in% tf_genes)
test_data <- merged_data %>% filter(seqnames == "chr1") %>% select(Name)
write_csv(test_data, file = "../../data/modeling/pcawg_test.csv")
train_val_data <- merged_data[!merged_data$Name %in% test_data$Name,]
train_data <- as.data.frame(sample(train_val_data$Name, size = length(train_val_data$Name) * 0.8))
colnames(train_data) <- c("Name")
val_data <- as.data.frame(setdiff(train_val_data$Name, train_data$Name))
colnames(val_data) <- c("Name")
write_csv(train_data, file = "../../data/modeling/pcawg_train.csv")
write_csv(val_data, file = "../../data/modeling/pcawg_val.csv")

# Save file
deg_data <- deg_data %>% select(-c(seqnames, gene_name)) %>% select(Name, everything())
write.table(deg_data, file = "../../data/pcawg_expid.txt", sep = "\t", row.names = FALSE, col.names = TRUE)
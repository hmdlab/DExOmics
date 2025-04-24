### This script extracts the RBP data of HeLa cell line from POSTAR3, and splits them into .bed files.
library(tidyverse)
library(dplyr)
library(stringr)
library(readr)
library(data.table)

data <- read_delim("../../data/postar3/human.txt", col_names = FALSE)
cell_lines <- split(data, data$X8)
df <- data.frame(cell_line = names(cell_lines), rbp_num = NA)
for (i in 1:length(cell_lines)){
  num <- length(unique(cell_lines[[i]]$X6))
  df$rbp_num[i] <- num
}
HeLa_bed_rna <- cell_lines[["HeLa"]] %>% 
  filter(X6 != "RBP_occupancy")

grouped_data <- split(HeLa_bed_rna, HeLa_bed_rna$X6)

folder_path <- "../data/HeLa_bed_rna/"
if (!dir.exists(folder_path)) {
  dir.create(folder_path, recursive = TRUE)
} 

for (name in names(grouped_data)) {
  group_df <- grouped_data[[name]]
  bed_data <- group_df[, c("X1", "X2", "X3")]
  bed_filename <- paste0(folder_path, name, ".bed")
  write.table(bed_data, bed_filename, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
}
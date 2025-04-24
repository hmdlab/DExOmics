# ------------------ TCGA data for mRNA expression ----------------------------
library(SummarizedExperiment)
library(TCGAbiolinks)
library(tidyverse)
library(stringr)

# Cancer type from terminal argument
cancer <- as.character(commandArgs(trailingOnly = TRUE))
project <- paste0("TCGA-", cancer)

# Folder to save data
folder_path <- file.path("../data/TCGAdata", cancer)
if (!dir.exists(folder_path)) dir.create(folder_path, recursive = TRUE)

# ------------------ Query setup ----------------------------
message("Creating query...")
if (cancer == "BRCA") {
  dataClin <- GDCquery_clinic(project = project, type = "clinical")
  female_barcodes <- dataClin$submitter_id[dataClin$gender == "female"]

  query.exp <- GDCquery(
    project = project,
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification",
    workflow.type = "STAR - Counts",
    barcode = female_barcodes
  )
} else {
  query.exp <- GDCquery(
    project = project,
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification",
    workflow.type = "STAR - Counts"
  )
}

# ------------------ Download with error handling ----------------------------
message("Downloading expression data...")
download_success <- tryCatch({
  GDCdownload(query = query.exp, method = "api", directory = "./GDCdata", files.per.chunk = 10)
  TRUE
}, error = function(e) {
  message("Download failed: ", e$message)
  FALSE
})

if (!download_success) quit(save = "no", status = 1)

# ------------------ Prepare and save expression data ----------------------------
exp_file <- file.path(folder_path, paste0(cancer, ".exp.rda"))

if (!file.exists(exp_file)) {
  message("Preparing expression data...")
  tumor_exp <- tryCatch({
    GDCprepare(query = query.exp,
               save = TRUE,
               save.filename = exp_file,
               directory = "./GDCdata")
  }, error = function(e) {
    message("GDCprepare failed: ", e$message)
    quit(save = "no", status = 1)
  })
  save(tumor_exp, file = exp_file)
} else {
  message("Loading existing expression data...")
  tumor_exp <- get(load(exp_file))
}

# ------------------ Preprocessing ----------------------------
message("Preprocessing expression data...")
dataPrep <- TCGAanalyze_Preprocessing(object = tumor_exp, cor.cut = 0.6)

# GC-normalization
message("Normalizing expression data...")
data("geneInfoHT", package = "TCGAbiolinks")
dataNorm <- TCGAanalyze_Normalization(tabDF = dataPrep, geneInfo = geneInfoHT, method = "gcContent")

# Filtering
message("Filtering low expression genes...")
dataFilt <- TCGAanalyze_Filtering(tabDF = dataNorm, method = "quantile", qnt.cut = 0.25)

# Save processed data
prep_file <- file.path(folder_path, paste0(cancer, ".expr.prep.rda"))
save(dataFilt, file = prep_file)

write.table(rownames(dataFilt), file = file.path(folder_path, "expr_rownames.txt"), sep = "\t", col.names = FALSE, row.names = FALSE)
write.table(colnames(dataFilt), file = file.path(folder_path, "expr_colnames.txt"), sep = "\t", col.names = FALSE, row.names = FALSE)

# ------------------ Clean up ----------------------------
message("Cleaning up GDCdata folder...")
if (exists("tumor_exp")) {
  unlink("./GDCdata", recursive = TRUE)
  if (file.exists("MANIFEST.txt")) file.remove("MANIFEST.txt")
}

# ------------------ Output gene types summary ----------------------------
message("Counting gene types...")
transcript <- as.data.frame(rowData(tumor_exp))
transcript_summary <- transcript %>%
  group_by(gene_type) %>%
  summarise(n = n(), .groups = "drop")
write_csv(transcript_summary, file = file.path(folder_path, paste0(cancer, "_gene_types.csv")))

message("All done.")

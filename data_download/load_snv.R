###------------- TCGA SNV -------------------
library(TCGAbiolinks)
library(SummarizedExperiment)

# Get cancer from the terminal command line
cancer <- as.character(commandArgs(trailingOnly = TRUE))
project <- paste0("TCGA-", cancer)

# Check directory 
folder_path <- paste0("../data/TCGAdata/", cancer, "/")
if (!dir.exists(folder_path)) {
  dir.create(folder_path, recursive = TRUE)
}

# Data query
if(cancer == "BRCA"){
  dataClin <- GDCquery_clinic(project = project, type = "clinical")
  female_barcodes <- dataClin$submitter_id[which(dataClin$gender == "female")]
  query.SNV <- GDCquery(project = project,
                        data.category = "Simple Nucleotide Variation",
                        data.type = "Masked Somatic Mutation",
                        # legacy = FALSE,
                        barcode = female_barcodes)
}else{
  query.SNV <- GDCquery(project = project,
                        data.category = "Simple Nucleotide Variation",
                        data.type = "Masked Somatic Mutation")
                        # legacy = FALSE)
}

# Data download
GDCdownload(query = query.SNV, method = "api", directory = "./GDCdata", files.per.chunk = 10)

# Data prepare
if (file.exists(paste0(folder_path, cancer, ".snv.rda")) == FALSE){
  tumor_snv <- GDCprepare(query = query.SNV,
                          save = TRUE,
                          save.filename = paste0(folder_path, cancer, ".snv.rda"),
                          directory = "./GDCdata")}


# Delete GDCdata
unlink("./GDCdata", recursive = TRUE)
file.remove("MANIFEST.txt")

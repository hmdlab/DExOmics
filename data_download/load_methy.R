###------------- TCGA Methylation-------------------
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
query.met <- GDCquery(project = project,
                        # legacy =FALSE,
                        data.category = "DNA Methylation",
                        data.type = "Methylation Beta Value",
                        platform = "Illumina Human Methylation 450",
                        # sample.type = c("Primary Tumor", "Solid Tissue Normal"),
                        barcode = female_barcodes)


}else{
query.met <- GDCquery(project = project,
                        # legacy =FALSE,
                        data.category = "DNA Methylation",
                        data.type = "Methylation Beta Value",
                        platform = "Illumina Human Methylation 450")
                        # sample.type = c("Primary Tumor", "Solid Tissue Normal")

}

# Data download
GDCdownload(query = query.met, method = "api", directory = "./GDCdata", files.per.chunk = 10)

# Data prepare
if (file.exists(paste0(folder_path, cancer, ".methy.rda")) == FALSE){
  tumor_met <- GDCprepare(query = query.met,
                          save = TRUE,
                          save.filename = paste0(folder_path, cancer, ".methy.rda"),
                          directory = "./GDCdata")}


# Delete GDCdata
unlink("./GDCdata", recursive = TRUE)
file.remove("MANIFEST.txt")

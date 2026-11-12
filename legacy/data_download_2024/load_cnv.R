###------------- TCGA-LIHC CNV ----------------###
library(TCGAbiolinks)
library(SummarizedExperiment)
library(GenomicRanges)
library(reshape2)
library(readr)

# Get cancer from the terminal command line
cancer <- as.character(commandArgs(trailingOnly = TRUE))
project <- paste0("TCGA-", cancer)

# Check directory 
folder_path <- paste0("../data/TCGAdata/", cancer, "/")
if (!dir.exists(folder_path)) {
  dir.create(folder_path, recursive = TRUE)
} 

# For BRCA, only include female samples
if(cancer == "BRCA"){
  dataClin <- GDCquery_clinic(project = project, type = "clinical")
  female_barcodes <- dataClin$submitter_id[which(dataClin$gender == "female")]
  query.cnv <- GDCquery(project = project,
                        data.category = "Copy Number Variation",
                        # legacy = FALSE,
                        data.type = "Gene Level Copy Number",
                        workflow.type = "ASCAT3",
                        barcode = female_barcodes)
  
}else{
  query.cnv <- GDCquery(project = project,
                        data.category = "Copy Number Variation",
                        # legacy = FALSE,
                        data.type = "Gene Level Copy Number",
                        workflow.type = "ASCAT3")
  
}

#Download data to working directory
GDCdownload(query = query.cnv, method = "api", directory = "./GDCdata", files.per.chunk = 10)

#Read, prepare, and save the downloaded data 
if (file.exists(paste0(folder_path, cancer, ".cnv.rda")) == FALSE){
  tumor_cnv <- GDCprepare(query = query.cnv,
                          save = TRUE,
                          save.filename = paste0(folder_path, cancer, ".cnv.rda"),
                          directory = "./GDCdata")}

# Delete GDCdata
unlink("./GDCdata", recursive = TRUE)
file.remove("MANIFEST.txt")

### This script map the genomic locations in the bed narrowPeak files from ENCODE to local transcript locations, 
### and generates the RNA and promoter features for each transcript.

options(stringsAsFactors = FALSE)
library(optparse)
library(tidyverse)
library(data.table)
library(rtracklayer)
library(plyranges)
library(GenomicFeatures)


option_list = list(
  make_option(c("-b", "--bed_directory"), type="character", default=NULL,
              help="bed directory", metavar="character"),
  make_option(c("-n", "--bin"), type="character", default=100,
              help="bin", metavar="character"),
  make_option(c("-g", "--gtf_file"), type="character", default=NULL,
              help="gtf file", metavar="character"),
  make_option(c("-t", "--input_type"), type="character", default=NULL,
              help="input type", metavar="character"),
  make_option(c("-o", "--output"), type="character", default=NULL,
              help="output", metavar="character"),
  make_option(c("-s", "--source"), type="character", default=NULL,
              help="data source", metavar="character")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

bed_directory = opt$bed_directory
bin = as.numeric(opt$bin)
output = opt$output
gtf_file = opt$gtf_file
input_type = opt$input_type
database = opt$source


# read gene coordinate
gtf = import(gtf_file)
if(input_type=="promoter"){
  txdb <- GenomicFeatures::makeTxDbFromGRanges(gtf)
  genome_coord<-promoters(txdb, upstream = 2000, downstream = 1000)
  genome_coord$transcript_id=genome_coord$tx_name
  genome_coord$tx_id=NULL
  genome_coord$tx_name=NULL
}else if(input_type=="rna"){
  gtf%>%
    as_tibble() %>%
    filter(type=="exon") %>%
    dplyr::select(seqnames,start,end,width,strand,transcript_id) %>%
    GRanges() -> genome_coord
}

seqlevelsStyle(genome_coord)="UCSC"
genome_coord = split(genome_coord,as.factor(genome_coord$transcript_id))


list.files(path = bed_directory,pattern = "*.bed",full.names = T) %>%
  lapply(., function(bed_file){
    print(bed_file)
    # read bed file
    bed = data.table::fread(bed_file)%>%
      dplyr::select(seqname=V1,start=V2,end=V3)%>%#,strand=V6)%>%
      mutate(start=start+1)%>%
      # *** delete uncertain peaks***
      filter(!str_detect(seqname, "_")) %>%
      GRanges()

    # convert genome coordinate to RNA coordinate
    bed_coord <- mapToTranscripts(bed, genome_coord,ignore.strand = TRUE)
    
    # split into bins (only if peaks can be found on RNA)
    if (nrow(bed_coord %>% as_tibble()) != 0){
      bed_coord %>%
        as_tibble() %>%
        # *** delete genes with negative indices ***
        filter(start >= 0) %>% 
        rowwise() %>%
        mutate(pos = list(start:end)) %>%
        dplyr::select(transcript_id=seqnames,pos) %>%
        unnest(cols = c(pos)) %>%
        unique() %>%
        group_by(transcript_id) %>%
        mutate(indx=cut(pos,
                        breaks =seq(from=0,
                                    to = ceiling((max(pos))/bin)*bin,
                                    by = bin),
                        include.lowest = T) %>%
                 as.numeric()) %>%
        group_by(transcript_id,indx) %>%
        summarise(N=dplyr::n()) %>%
        mutate(indx=indx-1) -> bed_coord_bined
      
      # *** map feature names ***
      if (database == "ENCODE"){
        meta <- read.table(file = paste0(bed_directory, "/metadata.tsv"), header = TRUE, sep = "\t")
        target <- meta %>% 
          filter(File.accession == unlist(strsplit(basename(bed_file), split = "\\."))[1]) %>% 
          pull(Experiment.target)
        target <- unlist(strsplit(target, split = "\\-"))[1]
      } else{
        target <- unlist(strsplit(basename(bed_file), split = "\\."))[1]
      }
      
      bed_coord_bined %>%
        mutate(feature_name = target) %>%
        return()
      }
  }) %>%
  bind_rows() -> ds

# exon index
genome_coord %>%
  unlist() %>%
  group_by(transcript_id) %>%
  summarise(width=sum(width)) %>%
  as_tibble() %>%
  mutate(max_bin = floor(width/bin),
         last_bin_val= width%%bin) %>%
  rowwise() %>%
  mutate(indx = list(0:max_bin)) %>%
  dplyr::select(-max_bin,-width) %>%
  unnest(cols = c(indx)) %>%
  mutate(N=100) %>%
  group_by(transcript_id) %>%
  mutate(N=ifelse(indx%in%max(indx), last_bin_val, N)) %>%
  dplyr::select(-last_bin_val) %>%
  mutate(feature_name = ifelse(input_type == "promoter", "mask", "exon")) %>%
  bind_rows(.,ds) %>%
  filter(N>0) -> ds

ds$feature_name %>% 
  unique() %>%
  factor() %>%
  relevel(.,ref = ifelse(input_type=="promoter", "mask", "exon")) -> feature_names

ds%>%
  ungroup()%>%
  mutate(feature_name = factor(feature_name,levels = feature_names) %>% 
           as.numeric()-1) %>%
  dplyr::rename(col_indx = indx,
                row_indx = feature_name,
                val = N) %>%
  group_by(Name=transcript_id) %>%
  summarise(feature_dim=paste0(length(feature_names),",",max(col_indx)+1),
            row_indx = paste0(row_indx,collapse = ","),
            col_indx = paste0(col_indx,collapse = ","),
            val = paste0(val,collapse = ",")) -> ds

ds%>%
  data.table::fwrite(.,file=paste0(output,".txt"),
                     append = F,quote = F,sep = "\t",row.names = F,col.names = T)

ds$Name%>%
  write.table(.,file=paste0(output,"_gene_name.txt"),
              append = F,quote = F,sep = "\t",row.names = F,col.names = F)
system(paste0("gzip ",output,"_gene_name.txt"))

feature_names%>%
  write.table(.,file=paste0(output,"_feature_name.txt"),
              append = F,quote = F,sep = "\t",row.names = F,col.names = F)
system(paste0("gzip ",output,"_feature_name.txt"))

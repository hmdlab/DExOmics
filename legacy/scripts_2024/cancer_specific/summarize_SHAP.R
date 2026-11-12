# Descriptions of outputs
# RNA_importance_mean.txt: Average GradSHAP score for each RNA regulator in each gene class.
# promoter_importance_mean.txt: Average GradSHAP score for each promoter regulator in each gene class.
# omics_importance_mean.txt: Average GradSHAP score for each omics factor from TCGA in each gene class.

library(tidyverse)
library(ggpubr)
library(broom)

cancer <- as.character(commandArgs(trailingOnly = TRUE))[1]
outloc <- as.character(commandArgs(trailingOnly = TRUE))[2]

feature_names = read_delim(paste0("../../model_", cancer, "/concat/feature_norm_stats.txt"), delim = ",",col_names = T)

# TF importance ####
feature_names%>%
  filter(feature_type=="promoter_range")%>%
  .$feature_name%>%
  gsub("promoter_annot_|_GTRD.rds|.rds","",.)-> col_name
feature_names%>%
  filter(feature_type=="deg_stat")%>%
  arrange(row_indx) -> deg_stat

lapply(0:(nrow(deg_stat)-1), function(sample_indx){
  read_delim(paste0(outloc, "/DNA_",sample_indx,".txt.gz"),
             delim = "\t",col_names = F) -> EG_ds
  
  EG_mat=matrix(0,nrow=nrow(EG_ds),ncol = length(col_name))
  rownames(EG_mat)=EG_ds$X2
  colnames(EG_mat)=col_name
  
  for (i in 1:nrow(EG_ds)){
    EG_mat[i,]=EG_ds%>%.$X3%>%.[i]%>%
      str_split(.,pattern = ",")%>%.[[1]]%>%as.numeric()
  }
  
  EG_mat%>%as.data.frame()%>%
    mutate(Gene=rownames(EG_mat))%>%
    gather(feat_name,EG,-Gene)%>%
    mutate(sample_name=deg_stat$feature_name[sample_indx+1])%>%
    ungroup%>%
    group_by(feat_name,sample_name)%>%
    summarise(EG_mean=mean(EG))%>%
    dplyr::select(sample_name,feat_name,EG_mean)%>%
    return()
}) %>%
  bind_rows() %>%
  spread(feat_name,EG_mean)->promoter_importance_mean
write.table(promoter_importance_mean, file = paste0(outloc, "/promoter_importance_mean.txt"),
            append = F,quote = F,sep = "\t",row.names = F,col.names = T)



# RNA importance ####
feature_names%>%
  filter(feature_type=="mRNA_range")%>%
  .$feature_name-> col_name

lapply(0:(nrow(deg_stat)-1), function(sample_indx){
  read_delim(paste0(outloc, "/RNA_",sample_indx,".txt.gz"),
             delim = "\t",col_names = F) -> EG_ds
  
  EG_mat=matrix(0,nrow=nrow(EG_ds),ncol = length(col_name))
  rownames(EG_mat)=EG_ds$X2
  colnames(EG_mat)=col_name
  
  for (i in 1:nrow(EG_ds)){
    EG_mat[i,]=EG_ds%>%.$X3%>%.[i]%>%
      str_split(.,pattern = ",")%>%.[[1]]%>%as.numeric()
  }
  
  EG_mat%>%as.data.frame()%>%
    mutate(Gene=rownames(EG_mat))%>%
    gather(feat_name,EG,-Gene)%>%
    mutate(sample_name=deg_stat$feature_name[sample_indx+1])%>%
    ungroup%>%
    group_by(feat_name,sample_name)%>%
    summarise(EG_mean=mean(EG))%>%
    dplyr::select(sample_name,feat_name,EG_mean)%>%
    return()
})%>%bind_rows()%>%
  spread(feat_name,EG_mean)->RNA_importance_mean
write.table(RNA_importance_mean,file = paste0(outloc, "/RNA_importance_mean.txt"),
            append = F,quote = F,sep = "\t",row.names = F,col.names = T)




# Omics importance ####
feature_names%>%
  filter(feature_type=="omics_range")%>%
  .$feature_name-> col_name

lapply(0:(nrow(deg_stat)-1), function(sample_indx){
  read_delim(paste0(outloc, "/omics_",sample_indx,".txt.gz"),
             delim = "\t",col_names = F) -> EG_ds
  
  EG_mat=matrix(0,nrow=nrow(EG_ds),ncol=length(col_name))
  rownames(EG_mat)=EG_ds$X2
  colnames(EG_mat)=col_name
  
  for (i in 1:nrow(EG_ds)){
    EG_mat[i,]=EG_ds%>%.$X3%>%.[i]%>%
      str_split(.,pattern = ",")%>%.[[1]]%>%as.numeric()
  }
  
  EG_mat%>%as.data.frame()%>%
    mutate(Gene=rownames(EG_mat))%>%
    gather(feat_name,EG,-Gene)%>%
    mutate(sample_name=deg_stat$feature_name[sample_indx+1])%>%
    ungroup%>%
    group_by(feat_name,sample_name)%>%
    summarise(EG_mean=mean(EG))%>%
    dplyr::select(sample_name,feat_name,EG_mean)%>%
    return()
})%>%bind_rows()%>%
  spread(feat_name,EG_mean)->omics_importance_mean
write.table(omics_importance_mean,file = paste0(outloc, "/omics_importance_mean.txt"),
            append = F,quote = F,sep = "\t",row.names = F,col.names = T)
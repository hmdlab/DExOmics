library(tidyverse)
library(ggpubr)
library(broom)

project <- as.character(commandArgs(trailingOnly = TRUE))[1]
sample_names = read_delim(paste0("../../", project, "_model/feature_norm_stats.txt"), delim = ",",col_names = T) %>%
  filter(feature_type=="deg_stat")%>%.$feature_name
geneids = read_delim(paste0("../../metrics/", project, "_geneid.txt.gz"), delim = "\t",col_names = F)%>%.$X1

actual = read_delim(paste0("../../metrics/", project, "_actual.txt.gz"), delim = "\t",col_names = F)
colnames(actual) = sample_names

prediction = read_delim(paste0("../../metrics/", project, "_prediction.txt.gz"), delim = "\t",col_names = F)
colnames(prediction) = sample_names

actual%>%
  as.data.frame()%>%
  mutate(geneid=geneids)%>%
  gather(sample,actual,-geneid)%>%
  inner_join(.,
             prediction%>%
               as.data.frame()%>%
               mutate(geneid=geneids)%>%
               gather(sample,prediction,-geneid),
             by=c("geneid","sample"))%>%
  group_by(sample)%>%
  do(cor.test(.$actual,.$prediction,method = "spearman")%>%broom::tidy()) -> cor_tbl

write.table(cor_tbl,file=paste0("../../metrics/", project, "_cor_tbl.txt"), append = F,quote = F,sep = "\t",row.names = F,col.names = T)  

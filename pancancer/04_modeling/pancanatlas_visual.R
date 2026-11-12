# This script analyzes and visualizes model performance on the test set.
library(tidyverse)
library(purrr)
library(dplyr)
library(stringr)
library(readr)
library(data.table)
library(ggplot2)
library(reshape2)
library(ggpubr)
library(patchwork)
library(pheatmap)


# Data loading
project <- as.character(commandArgs(trailingOnly = TRUE))[1]
cor_tbl <- read.delim(paste0("../../test_metrics_", project, "/cor_tbl.txt"))
sample_names = read_delim(paste0("../../", project, "_model/reg0.001/feature_norm_stats.txt"), delim = ",",col_names = T) %>%
  filter(feature_type=="deg_stat")%>%.$feature_name
actual = read_delim(paste0("../../test_metrics_", project, "/actual.txt.gz"), delim = "\t",col_names = F)
colnames(actual) <- sample_names
pred = read_delim(paste0("../../test_metrics_", project, "/prediction.txt.gz"), delim = "\t",col_names = F)
colnames(pred) <- sample_names
print(colnames(pred))

# Spearman correlation barplot
barplot1 <- ggplot(cor_tbl, aes(x = reorder(sample, estimate), y = estimate)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "", x = "Cancer Type", y = "Spearman's Rho") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 10),  
    axis.text.y = element_text(size = 6),
    axis.title.x = element_text(size = 10), 
    axis.title.y = element_text(size = 10) 
  )
ggsave(paste0("../../plots/", project, "_spearman.pdf"), plot = barplot1, width = 8, height = 4, dpi = 300)

cor_tbl2 <- cor_tbl %>% 
  filter(sample != "MedianExp") %>% 
  mutate(system = case_when(
    sample %in% c("KIRC", "KIRP", "KICH", "BLCA", "PRAD", "TGCT") ~ "Urinary",
    sample %in% c("BRCA", "CESC", "OV", "UCEC", "UCS") ~ "Gynecological",
    sample %in% c("CHOL", "COAD", "ESCA", "PAAD", "READ", "STAD") ~ "Digestive",
    sample %in% c("LUAD", "LUSC") ~ "Respiratory",
    sample %in% c("DLBC") ~ "Hematologic and Lymphatic",
    sample %in% c("GBM", "LGG") ~ "Nervous",
    sample %in% c("ACC", "PCPG", "THCA") ~ "Endocrine",
    sample %in% c("HNSC", "LIHC", "MESO", "SARC", "SKCM", "THYM", "UVM") ~ "Others",
    TRUE ~ "Miscellaneous"
  )) %>% 
  arrange(system, sample) %>% 
  mutate(sample = fct_inorder(sample))
print("Mean Spearman correlation coefficient:")
print(mean(cor_tbl2$estimate))
barplot2 <- ggplot(cor_tbl2, aes(x = sample, y = estimate, fill = system)) +
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "Set3") + 
  coord_flip() +
  labs(title = "", x = "Cancer Type", y = "Spearman's Rho") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 20),  
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 24), 
    axis.title.y = element_text(size = 24)
  )
ggsave(paste0("../../plots/", project, "_system_spearman.pdf"), plot = barplot2, width = 12, height = 6, dpi = 300)

# Regression plots
plot_list <- list()
for (i in 1:(length(sample_names)-1)){
  col_data <- as.data.frame(cbind(pred[,i], actual[,i]))
  colnames(col_data) <- c("pred_exp", "actual_exp")

  reg_plot <- ggplot(col_data, aes(x = pred_exp, y = actual_exp)) +
    geom_hex() +
    scale_fill_gradient(low = "skyblue", high = "salmon") +
    stat_cor(method = "spearman", size = 2) +
    labs(title = sample_names[i], x = "", y = "") +  
    theme_minimal() +
    theme(text = element_text(size = 8),  
          panel.grid.major = element_blank())

  # Combine the plots
  plot_list[[i]] <- reg_plot
}
combined_plot <- wrap_plots(plot_list, ncol = 5) + 
  plot_layout(guides = "collect") &  
  theme(legend.position = "none") 
ggsave(paste0("../../plots/", project, "_combined_actual_pred.pdf"), plot = combined_plot, width = 12, height = 12, dpi = 300)

MedianExp_data <- as.data.frame(cbind(pred$MedianExp, actual$MedianExp))
colnames(MedianExp_data) <- c("pred_exp", "actual_exp")
reg_plot1 <- ggplot(MedianExp_data, aes(x = pred_exp, y = actual_exp)) +
  geom_hex() +  
  scale_fill_gradient(low = "skyblue", high = "salmon") + 
  stat_cor(method = "spearman", label.x = 2, label.y = 6, size = 8) +
  labs(title = "", x = "Predicted expression", y = "Actual expression", fill = "Count") +  
  theme_minimal() + 
  theme(legend.position = "right",
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),  
        panel.background = element_blank(), 
        legend.text = element_text(size = 18),
        legend.title = element_text(size = 18),
        axis.text = element_text(size = 20),
        axis.title = element_text(size = 24))
ggsave(paste0("../../plots/", project, "_MedianExp_actual_pred.pdf"), plot = reg_plot1, width = 8, height = 6, dpi = 300)
  


# Cancer-specific pairewise correlation heatmap
pred2 <- pred %>% select(c(BLCA, KICH, KIRC, KIRP, PRAD, TGCT, # 泌尿生殖系统癌症
                           BRCA, CESC, OV, UCEC, UCS, # 妇科癌症
                           CHOL, COAD, ESCA, PAAD, READ, STAD, # 消化系统癌症
                           LUAD, LUSC, # 呼吸系统癌症
                           DLBC, # 血液和淋巴系统癌症
                           GBM, LGG, # 神经系统癌症
                           ACC, PCPG, THCA, # 内分泌系统癌症
                           HNSC, LIHC, MESO, SARC, SKCM, THYM, UVM)) # 其他特殊类型癌症

actual2 <- actual %>% select(c(BLCA, KICH, KIRC, KIRP, PRAD, TGCT, # 泌尿生殖系统癌症
                           BRCA, CESC, OV, UCEC, UCS, # 妇科癌症
                           CHOL, COAD, ESCA, PAAD, READ, STAD, # 消化系统癌症
                           LUAD, LUSC, # 呼吸系统癌症
                           DLBC, # 血液和淋巴系统癌症
                           GBM, LGG, # 神经系统癌症
                           ACC, PCPG, THCA, # 内分泌系统癌症
                           HNSC, LIHC, MESO, SARC, SKCM, THYM, UVM)) # 其他特殊类型癌症
correlations <- cor(pred2, actual2)
correlations_melted <- melt(cor(correlations), varnames = c("Variable1", "Variable2"))
heatmap1 <- ggplot(correlations_melted, aes(Variable1, Variable2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, 
                       limit = c(-1, 1),
                       space = "Lab",
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  coord_fixed() +
  labs(x = "Predicted expression", y = "Actual expression")
ggsave(paste0("../../plots/", project, "_pairwise_heatmap.pdf"), plot = heatmap1, width = 8, height = 6, dpi = 300)

# Heatmap after clustering
correlations <- cor(pred[,-ncol(pred)], actual[,-ncol(actual)]) # not include medianExp
output_file <- paste0("../../plots/", project, "_pairwise_heatmap_clustering.png")
png(filename = output_file, width = 10, height = 10, units = "in", res = 300)
pheatmap(
  correlations,
  clustering_distance_rows = "euclidean",
  clustering_distance_cols = "euclidean",
  clustering_method = "average",
  display_numbers = FALSE,
  fontsize = 18,
  fontsize_row = 20,  
  fontsize_col = 20,
  angle_col = 270
)
dev.off()
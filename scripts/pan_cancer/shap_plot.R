### This script visulizes the results of the shap values
library(tidyverse)
library(purrr)
library(dplyr)
library(stringr)
library(readr)
library(data.table)
library(ggplot2)
library(reshape2)
library(ggpubr)
library(broom)
library(patchwork)
library(ComplexHeatmap)

cancer <- as.character(commandArgs(trailingOnly = TRUE))[1]
inloc <- as.character(commandArgs(trailingOnly = TRUE))[2]
outloc <- as.character(commandArgs(trailingOnly = TRUE))[3]
if (!dir.exists(outloc)) {
  dir.create(outloc, recursive = TRUE)
} 

# Read files
promoter_importance_mean <- read_delim(paste0(inloc, "/promoter_importance_mean.txt"))
RNA_importance_mean <- read_delim(paste0(inloc, "/RNA_importance_mean.txt"))


# Split TFs, RBPs and miRNAs
TF_df <- promoter_importance_mean %>% 
  filter(sample_name != "MedianExp")
RBP_df <- RNA_importance_mean %>% 
  filter(sample_name != "MedianExp") %>% 
  select(!starts_with("miR"))
miRNA_df <- RNA_importance_mean %>% 
  filter(sample_name != "MedianExp") %>% 
  select(sample_name, starts_with("miR"))

# SHAP results analysis
tf_long <- reshape2::melt(TF_df, id.vars = "sample_name") %>% 
  mutate(factor = "tf")
rbp_long <- reshape2::melt(RBP_df, id.vars = "sample_name") %>% 
  mutate(factor = "rbp")
mirna_long <- reshape2::melt(miRNA_df, id.vars = "sample_name") %>% 
  mutate(factor = "mirna")

# Generate the violin plot with an embedded boxplot ---------------------------------------------------------
tf_average <- TF_df %>% select(-sample_name) %>% mutate(TF = rowMeans(.)) %>% select(TF)
tf_average <- apply(TF_df %>% select(-sample_name), 1, mean)

rbp_average <- RBP_df %>% select(-sample_name) %>% mutate(RBP = rowMeans(.)) %>% select(RBP)
rbp_average <- apply(RBP_df %>% select(-sample_name), 1, mean)

mirna_average <- miRNA_df %>% select(-sample_name) %>% mutate(miRNA = rowMeans(.)) %>% select(miRNA)
mirna_average <- apply(miRNA_df %>% select(-sample_name), 1, mean)

average_df <- cbind(cbind(tf_average, rbp_average), mirna_average)
average_df <- as.data.frame(average_df)
colnames(average_df) <- c("TF", "RBP", "miRNA")

# Comapre the effect of miRNA, RBP and TF
df_long <- average_df %>%
  pivot_longer(cols = everything(), names_to = "Category", values_to = "Value")

# Two-sample Wilcoxon test
p_values <- list()
categories <- unique(df_long$Category)

for (i in 1:(length(categories) - 1)) {
  for (j in (i + 1):length(categories)) {
    cat1 <- categories[i]
    cat2 <- categories[j]
    p_value <- wilcox.test(average_df[[cat1]], average_df[[cat2]])$p.value
    p_values <- append(p_values, list(data.frame(group1 = cat1, group2 = cat2, p_value = p_value)))
  }
}

# Combine p-values
p_values_df <- bind_rows(p_values)
# Annotate significance
p_values_df$significance <- ifelse(p_values_df$p_value < 0.001, "***",
                                   ifelse(p_values_df$p_value < 0.01, "**",
                                          ifelse(p_values_df$p_value < 0.05, "*",
                                                 sprintf("p = %.3g", p_values_df$p_value))))  # p ≥ 0.05 直接显示数值

# Plotting
contribution_boxplot <- ggplot(df_long, aes(x = Category, y = Value, fill = Category)) +
  geom_violin(alpha = 0.7) +  
  geom_boxplot(width = 0.2, outlier.shape = NA, color = "black") + 
  geom_jitter(width = 0.1, alpha = 0.4) +  
  scale_y_log10() + 
  geom_signif(comparisons = list(c("TF", "RBP"), c("TF", "miRNA"), c("RBP", "miRNA")),
              annotations = p_values_df$significance,
              y_position = c(0, -0.5, -1),
              tip_length = 0.02, textsize = 5) +
  theme_classic() + 
  labs(title = "",
       x = "", 
       y = "Contribution Score") +
  theme(legend.position = "none",
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 18))
ggsave(paste0(outloc, cancer, "_DeepLIFT_contribution_boxplot.pdf"), plot = contribution_boxplot, width = 10, height = 8, dpi = 300)




# Top 5 shap values of the factors for each cancer type ---------------------------------------------------------
data_long <- rbind(rbind(mirna_long, tf_long), rbp_long)
selected_factor <- c()
for (s in unique(data_long$sample_name)){
  sub_df <- data_long %>% 
    filter(sample_name == s) %>% 
    filter(variable != "mask" & variable != "exon") %>% 
    mutate(abs_value = abs(value)) %>% 
    arrange(desc(abs_value)) %>% 
    head(5)
  selected_factor <- c(selected_factor, as.character(sub_df$variable))
}
data_short <- data_long %>% filter(variable %in% selected_factor)
df_wide <- dcast(data_short, sample_name ~ variable, value.var = "value")

# Create matrix
df_wide_mat <- df_wide %>% 
  column_to_rownames("sample_name") %>% 
  as.matrix()
factors <- data_short$factor[seq(from = 1, to = nrow(data_short), by = length(unique(data_short$sample_name)))]

# ComplexHeatmap
pdf(paste0(outloc, cancer, "_DeepLIFT_heatmap.pdf"), width = 12, height = 8)
Heatmap(
  df_wide_mat, 
  name = "DeepLIFT score",
  top_annotation = HeatmapAnnotation(
    df = data.frame(Factor = factors),
    col = list(Factor = c("tf" = "#B3CDE3", "rbp" = "#CCEBC5", "mirna" = "#FBB4AE"))),
  cluster_rows = TRUE,
  cluster_columns = TRUE,
  show_row_names = TRUE,
  show_column_names = TRUE,
  heatmap_legend_param = list(
    legend_gp = gpar(fontsize = 14),  
    title_gp = gpar(fontsize = 14)   
  ),
  row_names_gp = gpar(fontsize = 14),  
  column_names_gp = gpar(fontsize = 14), 
  column_title_gp = gpar(fontsize = 18),  
  row_title_gp = gpar(fontsize = 18)) 
dev.off()




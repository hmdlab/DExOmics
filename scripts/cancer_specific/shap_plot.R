### This script visulizes the results of the shap values
library(tidyverse)
library(purrr)
library(dplyr)
library(stringr)
library(readr)
library(data.table)
library(ggplot2)
library(RColorBrewer)
library(reshape2)
library(ggpubr)
library(broom)
library(patchwork)
library(ComplexHeatmap)

inloc <- as.character(commandArgs(trailingOnly = TRUE))[1]
outloc <- as.character(commandArgs(trailingOnly = TRUE))[2]
if (!dir.exists(outloc)) {
  dir.create(outloc, recursive = TRUE)
} 

# Read files
omics_importance_mean <- read_delim(paste0(inloc, "/omics_importance_mean.txt"))
promoter_importance_mean <- read_delim(paste0(inloc, "/promoter_importance_mean.txt"))
RNA_importance_mean <- read_delim(paste0(inloc, "/RNA_importance_mean.txt"))


# SHAP results analysis
omics_long <- reshape2::melt(omics_importance_mean, id.vars = "sample_name") %>% 
  mutate(factor = "omics")
tf_long <- reshape2::melt(promoter_importance_mean, id.vars = "sample_name") %>% 
  mutate(factor = "tf")
rbp_long <- reshape2::melt(RNA_importance_mean, id.vars = "sample_name") %>% 
  mutate(factor = "rbp")

# Generate the violin plot with an embedded boxplot ---------------------------------------------------------
box1 <- ggplot(omics_long, aes(x = sample_name, y = value, fill = sample_name)) +
  geom_violin(trim = FALSE) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +  # Adding a white boxplot inside the violin
  labs(x = "",
       y = "Predictive contribution") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3") + # Adjust colors if desired
  theme(legend.position = "none") +  # Removes the legend
  theme(
    legend.position = "none",  # Removes the legend
    panel.grid.major = element_blank(),  # Removes major grid lines
    panel.grid.minor = element_blank(),  # Removes minor grid lines
    panel.background = element_blank(),  # Removes background
    axis.line = element_line(colour = "black"),  # Adds axis lines
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.text = element_text(size = 18),
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 24)
  )

box2 <- ggplot(tf_long, aes(x = sample_name, y = value, fill = sample_name)) +
  geom_violin(trim = FALSE) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +  # Adding a white boxplot inside the violin
  labs(x = "",
       y = "") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3") + # Adjust colors if desired
  theme(legend.position = "none") +  # Removes the legend
  theme(
    legend.position = "none",  # Removes the legend
    panel.grid.major = element_blank(),  # Removes major grid lines
    panel.grid.minor = element_blank(),  # Removes minor grid lines
    panel.background = element_blank(),  # Removes background
    axis.line = element_line(colour = "black"),  # Adds axis lines
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.text = element_text(size = 18),
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 24)
  )


box3 <- ggplot(rbp_long, aes(x = sample_name, y = value, fill = sample_name)) +
  geom_violin(trim = FALSE) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +  # Adding a white boxplot inside the violin
  labs(x = "", 
       y = "") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3") + # Adjust colors if desired
  theme(legend.position = "none") + # Removes the legend
  theme(
    legend.position = "none",  # Removes the legend
    panel.grid.major = element_blank(),  # Removes major grid lines
    panel.grid.minor = element_blank(),  # Removes minor grid lines
    panel.background = element_blank(),  # Removes background
    axis.line = element_line(colour = "black"),  # Adds axis lines
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.text = element_text(size = 18),
    axis.text = element_text(size = 24),
    axis.title = element_text(size = 24)
  )
box_plot <- box1 + box2 + box3
ggsave(paste0(outloc, "omics_promoter_rna_EG.pdf"), plot = box_plot, width = 10, height = 6, dpi = 300)





# Top 20 shap values of the factors for upDEGs and downDEGs ---------------------------------------------------------
data_long <- rbind(rbind(omics_long, tf_long), rbp_long)
upDEG_top10 <- bind_rows(data_long %>% 
                           filter(value < 0) %>% 
                           filter(sample_name == "upDEG") %>% 
                           filter(variable != "mask" & variable != "exon") %>% 
                           arrange(value) %>% 
                           head(10),
                         data_long %>% 
                           filter(value > 0) %>% 
                           filter(sample_name == "upDEG") %>% 
                           filter(variable != "mask" & variable != "exon") %>% 
                           arrange(desc(value)) %>% 
                           head(10)) 
top_factors <- upDEG_top10 %>% select(variable) %>% rename(upDEG = variable)
pastel_colors <- brewer.pal(n = 3, name = "Pastel1")
names(pastel_colors) <- c("omics", "tf", "rbp")                                         
upDEG_top10_plot <- ggplot(upDEG_top10, aes(x = reorder(variable, -value), y = value, fill = factor)) +
  geom_bar(stat = "identity", position = "dodge") +
  # scale_fill_brewer(palette = "Pastel1") +  # Using a pastel color palette
  scale_fill_manual(values = pastel_colors) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        legend.text = element_text(size = 18),
        legend.title = element_text(size = 18),
        axis.text = element_text(size = 20),
        axis.title = element_text(size = 24)) +
  labs(x = "", y = "Expected Grad.", fill = "Factor")
ggsave(paste0(outloc, "upDEG_top10EG.pdf"), plot = upDEG_top10_plot, width = 10, height = 6, dpi = 300)


downDEG_top10 <- bind_rows(data_long %>% 
                           filter(value < 0) %>% 
                           filter(sample_name == "downDEG") %>% 
                           filter(variable != "mask" & variable != "exon") %>% 
                           arrange(value) %>% 
                           head(10),
                         data_long %>% 
                           filter(value > 0) %>% 
                           filter(sample_name == "downDEG") %>% 
                           filter(variable != "mask" & variable != "exon") %>% 
                           arrange(desc(value)) %>% 
                           head(10))
top_factors <- top_factors %>% mutate(downDEG = downDEG_top10$variable)                           
downDEG_top10_plot <- ggplot(downDEG_top10, aes(x = reorder(variable, -value), y = value, fill = factor)) +
  geom_bar(stat = "identity", position = "dodge") +
  # scale_fill_brewer(palette = "Pastel1") +  # Using a pastel color palette
  scale_fill_manual(values = pastel_colors) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        legend.text = element_text(size = 18),
        legend.title = element_text(size = 18),
        axis.text = element_text(size = 20),
        axis.title = element_text(size = 24)) +
  labs(x = "", y = "Expected Grad.", fill = "Factor")
ggsave(paste0(outloc, "downDEG_top10EG.pdf"), plot = downDEG_top10_plot, width = 10, height = 6, dpi = 300)
gene_name <- basename(normalizePath(outloc, mustWork = FALSE))
write.csv(top_factors, file = paste0(inloc, gene_name, "_top_factors.csv"), row.names = FALSE)





# Expected gradient heatmap ---------------------------------------------------------
nonDEG_df <- data_long %>% 
  filter(sample_name == "nonDEG") %>% 
  filter(variable != "mask" & variable != "exon") %>% 
  mutate(abs_value = abs(value)) %>% 
  arrange(desc(abs_value)) %>% 
  head(30)

upDEG_df <- data_long %>% 
  filter(sample_name == "upDEG") %>% 
  filter(variable != "mask" & variable != "exon") %>% 
  mutate(abs_value = abs(value)) %>% 
  arrange(desc(abs_value)) %>% 
  head(30)

downDEG_df <- data_long %>% 
  filter(sample_name == "downDEG") %>%
  filter(variable != "mask" & variable != "exon") %>% 
  mutate(abs_value = abs(value)) %>% 
  arrange(desc(abs_value)) %>% 
  head(30)

selected_factor <- c(nonDEG_df$variable, upDEG_df$variable, downDEG_df$variable)
data_short <- data_long %>% 
  filter(variable %in% selected_factor)

df_wide <- dcast(data_short, sample_name ~ variable, value.var = "value")


# ComplexHeatmap
df_wide_mat <- df_wide %>% 
  column_to_rownames("sample_name") %>% 
  as.matrix()

factors <- data_short$factor[seq(from = 1, to = nrow(data_short), by = length(unique(data_short$sample_name)))]

# Fix row order
row_order <- c("upDEG", "downDEG", "nonDEG")
df_wide_mat <- df_wide_mat[row_order, , drop = FALSE]

pdf(paste0(outloc, "EG_heatmap.pdf"), width = 14, height = 6)
# Create heatmap
Heatmap(
  df_wide_mat, 
  name = "EG",
  top_annotation = HeatmapAnnotation(
    df = data.frame(Factor = factors), 
    col = list(Factor = pastel_colors),
    annotation_legend_param = list(
      Factor = list(
        title_gp  = gpar(fontsize = 18),
        labels_gp = gpar(fontsize = 18) 
      )
    )
  ),
  cluster_rows = FALSE,
  cluster_columns = TRUE,
  show_row_names = TRUE,
  show_column_names = TRUE,
  row_order = row_order,
  heatmap_legend_param = list(
    title_gp  = gpar(fontsize = 18), 
    labels_gp = gpar(fontsize = 18)
  ),
  row_names_gp = gpar(fontsize = 20),
  column_names_gp = gpar(fontsize = 20))


dev.off()


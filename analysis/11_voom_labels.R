#!/usr/bin/env Rscript
# 11_voom_labels.R — 原著近似の limma-voom ラベル（TCGA-LIHC 単独、HepG2 一致フィルタなし）
#
# 原著: STAR counts → TCGAanalyze_Preprocessing(cor.cut=0.6) → gcContent 正規化 → 25% 分位フィルタ
#       → TPM に limma-voom（FDR 0.05・|logFC|>1）
# 本近似: TPM（GENCODE v26 遺伝子長）→ 平均 TPM が 25% 分位超の遺伝子 → voom(TPM)+lmFit+eBayes。
#         cor.cut 前処理と GC 正規化は省略（方法差分解の近似である旨を 12 の出力に明記）。
#
# 入力: results/de_lihc/counts_for_voom.tsv.gz（gene_id × sample、01 の counts を 12 のランナーが書き出す）
#       results/de_lihc/coldata_for_voom.tsv（sample, condition）
#       results/features/gene_annotation_gencode26 の長さ → gene_length.tsv（ランナーが書き出す）
# 出力: results/de_lihc/voom_labels.csv（gene_id, logFC, adj_p, label）
suppressMessages({ library(limma); library(edgeR) })
root <- file.path(Sys.getenv("DEXOMICS_DATA", unset = path.expand("~/Data/DExOmics")), "results/de_lihc")   # set by analysis/dexo_paths.py when called from 12
cnt <- read.delim(gzfile(file.path(root, "counts_for_voom.tsv.gz")), row.names = 1, check.names = FALSE)
col <- read.delim(file.path(root, "coldata_for_voom.tsv"))
len <- read.delim(file.path(root, "gene_length.tsv"), row.names = 1)
stopifnot(all(colnames(cnt) == col$sample))
common <- intersect(rownames(cnt), rownames(len))
cnt <- cnt[common, ]; L <- len[common, "length"]

# TPM
rpk <- sweep(as.matrix(cnt), 1, L / 1e3, "/")
tpm <- sweep(rpk, 2, colSums(rpk) / 1e6, "/")

# 25% 分位フィルタ（原著 TCGAbiolinks quantile filter の近似: 平均 TPM）
mu <- rowMeans(tpm)
keep <- mu > quantile(mu, 0.25)
tpm <- tpm[keep, ]
cat("genes after quantile filter:", nrow(tpm), "\n")

design <- model.matrix(~0 + factor(col$condition, levels = c("Normal", "Tumor")))
colnames(design) <- c("Normal", "Tumor")
v <- voom(tpm, design)          # 原著同様 TPM を voom に投入
fit <- eBayes(contrasts.fit(lmFit(v, design), makeContrasts(Tumor - Normal, levels = design)))
tt <- topTable(fit, number = Inf, sort.by = "none")
lab <- ifelse(tt$adj.P.Val < 0.05 & abs(tt$logFC) > 1, ifelse(tt$logFC > 0, "UP", "DOWN"), "NonDEG")
out <- data.frame(gene_id = rownames(tpm), logFC = tt$logFC, adj_p = tt$adj.P.Val, label = lab)
write.csv(out, file.path(root, "voom_labels.csv"), row.names = FALSE)
print(table(lab))

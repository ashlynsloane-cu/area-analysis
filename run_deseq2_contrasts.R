#!/usr/bin/env Rscript
# ==============================================================================
# run_deseq2_contrasts.R
#
# This script runs differential expression analysis (DESeq2) comparing:
#   1. Resilience Program: Resilient vs. Canonical Disease (High-Burden background)
#   2. Alternative Etiology Program: Alternative Etiology vs. Typical Healthy (Low-Burden)
#
# Usage:
#   Rscript run_deseq2_contrasts.R <path_to_raw_counts.csv>
#
# If no raw counts file is specified, it will look for 'data/raw_counts_matrix.csv'
# or run in a robust Mock Mode with simulated counts to verify the installation
# and pipeline execution.
# ==============================================================================

# Suppress warnings for clean output
options(warn = -1)

# Check and load packages
required_packages <- c("BiocManager", "ggplot2", "pheatmap")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(paste("Installing package:", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

# Install Bioconductor packages if missing
bioc_packages <- c("DESeq2")
for (pkg in bioc_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(paste("Installing Bioconductor package:", pkg))
    BiocManager::install(pkg, update = FALSE, ask = FALSE)
  }
}

library(DESeq2)
library(ggplot2)

# ==============================================================================
# 1. SETUP & PATHS
# ==============================================================================
args <- commandArgs(trailingOnly = TRUE)
raw_counts_file <- if(length(args) >= 1) args[1] else "data/raw_counts_matrix.csv"

resilience_sheet_file <- "results/deseq2_resilience_samplesheet.csv"
alt_etiology_sheet_file <- "results/deseq2_alternative_etiology_samplesheet.csv"

# Function to normalize IDs to ensure robust matching across packages
normalize_id <- function(id) {
  s <- gsub("[_.-]", "", trimws(tolower(id)))
  if (substring(s, 1, 1) == "x" && nchar(s) > 1) {
    s <- substring(s, 2)
  }
  return(s)
}

# Check if sheets exist
if (!file.exists(resilience_sheet_file) || !file.exists(alt_etiology_sheet_file)) {
  stop("Error: Could not find DESeq2 sample sheets. Please run 'cluster_rae_subtypes.R' first!")
}

cat("\n=== STARTING DESEQ2 COHORT CONTRASTS ===\n")

# ==============================================================================
# 2. RUN PIPELINE
# ==============================================================================
run_deseq_analysis <- function(counts_path, samplesheet_path, contrast_name, ref_level, treat_level, output_prefix) {
  cat(sprintf("\n--- Analyzing Contrast: %s (%s vs. %s) ---\n", contrast_name, treat_level, ref_level))
  
  # Load samplesheet
  metadata <- read.csv(samplesheet_path, check.names = FALSE)
  metadata$Normalized_ID <- sapply(metadata$Participant, normalize_id)
  
  # Load or mock raw counts
  if (!file.exists(counts_path)) {
    cat("[Note] Raw counts file not found. Generating simulated counts for testing...\n")
    set.seed(123)
    num_mock_genes <- 1000
    mock_matrix <- matrix(
      rnbinom(num_mock_genes * nrow(metadata), mu = 200, size = 1/0.2),
      nrow = num_mock_genes,
      ncol = nrow(metadata)
    )
    
    # Introduce some mock biology (enrich some genes in treatment)
    treat_indices <- which(metadata$Cohort == treat_level)
    # Make some genes upregulated in treatment
    mock_matrix[1:50, treat_indices] <- mock_matrix[1:50, treat_indices] * runif(50, 2.5, 6.0)
    # Make some genes downregulated in treatment
    mock_matrix[51:100, treat_indices] <- round(mock_matrix[51:100, treat_indices] / runif(50, 2.5, 6.0))
    
    rownames(mock_matrix) <- paste0("ENSG", sprintf("%011d", 1:num_mock_genes))
    colnames(mock_matrix) <- metadata$Participant
    counts_df <- as.data.frame(mock_matrix)
  } else {
    counts_df <- read.csv(counts_path, row.names = 1, check.names = FALSE)
  }
  
  # Normalize count columns and align with metadata
  colnames_norm <- sapply(colnames(counts_df), normalize_id)
  matched_cols <- intersect(colnames_norm, metadata$Normalized_ID)
  
  if (length(matched_cols) == 0) {
    stop("Error: No patient columns in count matrix matched the sample sheet. Check your column headers!")
  }
  
  # Subset counts & metadata to matched participants
  expr_aligned <- counts_df[, match(matched_cols, colnames_norm), drop = FALSE]
  metadata_aligned <- metadata[match(matched_cols, metadata$Normalized_ID), ]
  
  # Ensure cohort is a factor with reference level correctly set
  metadata_aligned$Cohort <- factor(metadata_aligned$Cohort)
  metadata_aligned$Cohort <- relevel(metadata_aligned$Cohort, ref = ref_level)
  
  # Build DESeq2 DataSet
  dds <- DESeqDataSetFromMatrix(
    countData = round(as.matrix(expr_aligned)),
    colData = metadata_aligned,
    design = ~ Cohort
  )
  
  # Filter lowly expressed genes (mean count >= 1, consistent with AREA rank stability QC)
  keep <- rowMeans(counts(dds)) >= 1
  dds <- dds[keep, ]
  cat(sprintf(" -> Filtered matrix to %d genes with mean count >= 1\n", nrow(dds)))
  
  # Run DESeq2 differential analysis
  cat(" -> Executing Wald significance tests...\n")
  dds <- DESeq(dds)
  
  # Extract results
  res <- results(dds, contrast = c("Cohort", treat_level, ref_level))
  res_df <- as.data.frame(res)
  res_df$Gene <- rownames(res_df)
  res_df <- res_df[, c("Gene", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj")]
  
  # Sort by p-value significance
  res_df <- res_df[order(res_df$padj, na.last = TRUE), ]
  
  # Filter significant DEGs (padj < 0.05 and abs(LFC) > 1)
  sig_degs <- subset(res_df, padj < 0.05 & abs(log2FoldChange) >= 1)
  up_degs <- subset(sig_degs, log2FoldChange > 0)
  down_degs <- subset(sig_degs, log2FoldChange < 0)
  
  cat(sprintf(" -> Contrast Successful! Total Significant DEGs (adj. pval < 0.05, |LFC| >= 1.0): %d\n", nrow(sig_degs)))
  cat(sprintf("    - Upregulated in %s: %d genes\n", treat_level, nrow(up_degs)))
  cat(sprintf("    - Downregulated in %s: %d genes\n", treat_level, nrow(down_degs)))
  
  # Save full result table
  dir.create("results", showWarnings = FALSE)
  out_csv <- sprintf("results/deseq2_%s_results.csv", output_prefix)
  write.csv(res_df, out_csv, row.names = FALSE)
  cat(sprintf(" -> Saved full DEG table to: '%s'\n", out_csv))
  
  # Save significant DEG table
  sig_out_csv <- sprintf("results/deseq2_%s_sig_DEGs.csv", output_prefix)
  write.csv(sig_degs, sig_out_csv, row.names = FALSE)
  
  # --------------------------------------------------------------------------
  # GENERATE VISUALIZATIONS
  # --------------------------------------------------------------------------
  dir.create("figures", showWarnings = FALSE)
  
  # A. Volcano Plot
  res_df$Threshold <- "Not Significant"
  res_df$Threshold[res_df$padj < 0.05 & res_df$log2FoldChange >= 1] <- sprintf("Upregulated in %s", treat_level)
  res_df$Threshold[res_df$padj < 0.05 & res_df$log2FoldChange <= -1] <- sprintf("Downregulated in %s", treat_level)
  
  p_volcano <- ggplot(res_df, aes(x = log2FoldChange, y = -log10(padj), color = Threshold)) +
    geom_point(alpha = 0.6, size = 1.2) +
    scale_color_manual(values = c("Not Significant" = "grey70", 
                                  "Upregulated in Resilient" = "royalblue", 
                                  "Downregulated in Resilient" = "indianred",
                                  "Upregulated in Alternative_Etiology" = "darkorchid",
                                  "Downregulated in Alternative_Etiology" = "gold3")) +
    geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "grey40") +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey40") +
    labs(
      title = sprintf("Volcano Plot: %s", contrast_name),
      subtitle = sprintf("%s (n=%d) vs. %s (n=%d)", treat_level, nrow(metadata_aligned[metadata_aligned$Cohort == treat_level,]), ref_level, nrow(metadata_aligned[metadata_aligned$Cohort == ref_level,])),
      x = "Log2 Fold Change",
      y = "-Log10 Adjusted P-Value"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  volcano_fig <- sprintf("figures/deseq2_%s_volcano.png", output_prefix)
  ggsave(volcano_fig, plot = p_volcano, width = 7, height = 5, dpi = 150)
  cat(sprintf(" -> Saved Volcano Plot to: '%s'\n", volcano_fig))
  
  # B. Principal Component Analysis (PCA)
  vsd <- vst(dds, blind = FALSE)
  p_pca <- plotPCA(vsd, intgroup = "Cohort") +
    labs(
      title = sprintf("PCA Plot: %s Cohort Segregation", contrast_name),
      x = "PC1: Variance Explained",
      y = "PC2: Variance Explained"
    ) +
    theme_minimal()
  
  pca_fig <- sprintf("figures/deseq2_%s_pca.png", output_prefix)
  ggsave(pca_fig, plot = p_pca, width = 6, height = 4, dpi = 150)
  cat(sprintf(" -> Saved PCA Plot to: '%s'\n", pca_fig))
}

# Run Resilience Contrast: Resilient (Treat) vs. Canonical_Disease (Reference)
run_deseq_analysis(
  counts_path = raw_counts_file,
  samplesheet_path = resilience_sheet_file,
  contrast_name = "Resilience Program",
  ref_level = "Canonical_Disease",
  treat_level = "Resilient",
  output_prefix = "resilience"
)

# Run Alternative Etiology Contrast: Alternative_Etiology (Treat) vs. Typical_Healthy (Reference)
run_deseq_analysis(
  counts_path = raw_counts_file,
  samplesheet_path = alt_etiology_sheet_file,
  contrast_name = "Alternative Etiology Program",
  ref_level = "Typical_Healthy",
  treat_level = "Alternative_Etiology",
  output_prefix = "alternative_etiology"
)

cat("\n=== DESEQ2 CONTRASTS COMPLETED SUCCESSFULLY ===\n")


#!/usr/bin/env Rscript
# ==============================================================================
# cluster_rae_subtypes.R (v2)
#
# This R script performs unsupervised K-means clustering on your binarized 
# Patient-by-Gene Risk-Associated Expression (RAE) matrix. It calculates 
# cluster diagnostic metrics (Elbow and Silhouette), performs the clustering, 
# and automatically stratifies your cohort into four distinct clinical groups 
# (Resilient, Canonical Diseased, Alternative Etiology, Typical Healthy) 
# to generate your downstream DESeq2 design sample sheets.
#
# Usage:
#   Rscript cluster_rae_subtypes.R <comorbidity_name> <number_of_clusters_k>
#
# Example (Obstructive Sleep Apnea with k=3):
#   Rscript cluster_rae_subtypes.R MONDO_obstructive_sleep_apnea_syndrome 3
# ==============================================================================

# Suppress annoying loading warnings
options(warn = -1)

# Check and install/load standard packages if needed
required_packages <- c("cluster", "ggplot2")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(paste("Installing package:", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(cluster)
library(ggplot2)

# ==============================================================================
# 1. SETUP & COMMAND LINE ARGUMENTS
# ==============================================================================
args <- commandArgs(trailingOnly = TRUE)

# Default values if no arguments are passed
selected_disease <- if(length(args) >= 1) args[1] else "MONDO_sleep_apnea_syndrome"
k_clusters <- if(length(args) >= 2) as.integer(args[2]) else 3

# Define input paths
rae_file <- "RAE_matrix_for_clustering.csv"
comorb_file <- "data/filtered_binary_attributes_dataframe.csv"

# Check if input files exist
if (!file.exists(rae_file)) {
  # Fallback to look inside data/
  if (file.exists(file.path("data", rae_file))) {
    rae_file <- file.path("data", rae_file)
  } else {
    stop(paste("Error: Could not find", rae_file, "in current directory or 'data/' folder."))
  }
}

if (!file.exists(comorb_file)) {
  stop(paste("Error: Could not find", comorb_file, ". Please make sure it is in your data/ folder."))
}

# ==============================================================================
# 2. LOAD & SUBSET THE DATA
# ==============================================================================
cat("\n=== STARTING RAE MOLECULAR SUBTYPE CLUSTERING ===\n")
cat("Loading datasets...\n")

# Load binary RAE matrix and comorbidities
rae_df <- read.csv(rae_file, row.names = 1, check.names = FALSE)
comorb_df <- read.csv(comorb_file, check.names = FALSE)

# Normalize patient IDs in comorbidities to match RAE row names
normalize_id <- function(id) {
  s <- gsub("[_.-]", "", trimws(tolower(id)))
  # Strip R-style prepended X if present
  if (substring(s, 1, 1) == "x" && nchar(s) > 1) {
    s <- substring(s, 2)
  }
  return(s)
}
comorb_df$Normalized_ID <- sapply(comorb_df$Participant, normalize_id)
rownames(rae_df) <- sapply(rownames(rae_df), normalize_id)

# Match overlapping patients
common_ids <- intersect(rownames(rae_df), comorb_df$Normalized_ID)
cat(paste("Successfully aligned", length(common_ids), "T21 patients across expression and clinical sheets.\n"))

# Subset both tables to matched patients
rae_df <- rae_df[common_ids, ]
comorb_df <- comorb_df[match(common_ids, comorb_df$Normalized_ID), ]

# ==============================================================================
# 3. FILTER TO DISEASE-SPECIFIC GENES
# ==============================================================================
# The RAE columns are formatted as "GENE_SYMBOL" or "GENE_SYMBOL_DISEASE"
# Let's subset our clustering strictly to the genes associated with the selected comorbidity
cat(paste("\nFiltering RAE matrix for comorbidity:", selected_disease, "...\n"))

# If the columns contain comorbidity suffixes from multi-disease runs, subset them
# Otherwise, we cluster based on the whole matrix
disease_cols <- colnames(rae_df)
if (any(grepl(selected_disease, disease_cols))) {
  disease_cols <- disease_cols[grepl(selected_disease, disease_cols)]
  cluster_matrix <- rae_df[, disease_cols]
  cat(paste(" -> Found", length(disease_cols), "significant RAE genes specifically for", selected_disease, "\n"))
} else {
  cluster_matrix <- rae_df
  cat(paste(" -> No suffix matched. Clustering on all", ncol(rae_df), "available columns.\n"))
}

# Double check that we have columns to cluster
if (ncol(cluster_matrix) == 0) {
  stop("Error: No genes remain in the matrix for clustering. Make sure the comorbidity name is correct.")
}

# ==============================================================================
# 4. CLUSTERING DIAGNOSTICS (ELBOW & SILHOUETTE)
# ==============================================================================
cat("\n[Step 1/3] Calculating clustering diagnostics (k = 2 to 10)...\n")

max_k <- min(10, nrow(cluster_matrix) - 1)
wss <- numeric(max_k)
sil_widths <- numeric(max_k)

# Calculate Within-Cluster Sum of Squares (WSS) and Average Silhouette Widths
for (k in 2:max_k) {
  km <- kmeans(cluster_matrix, centers = k, nstart = 25, iter.max = 50)
  wss[k] <- km$tot.withinss
  
  # Calculate silhouette widths (using binary distance metric for binary vectors)
  sil <- silhouette(km$cluster, dist(cluster_matrix, method = "binary"))
  sil_widths[k] <- mean(sil[, 3])
}
wss[1] <- kmeans(cluster_matrix, centers = 1, nstart = 25)$tot.withinss # WSS for k=1

# Plot Elbow and Silhouette curves side-by-side
diag_df <- data.frame(
  k = 1:max_k,
  WSS = wss,
  Silhouette = c(NA, sil_widths[2:max_k])
)

# Create figures directory if it doesn't exist
dir.create("figures", showWarnings = FALSE)

# Plot Elbow
p1 <- ggplot(diag_df, aes(x = k, y = WSS)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "darkblue", size = 2) +
  labs(title = "Elbow Method (WSS)", x = "Number of Clusters (k)", y = "Total Within-Cluster SS") +
  theme_minimal()

# Plot Silhouette Widths
p2 <- ggplot(diag_df[-1, ], aes(x = k, y = Silhouette)) +
  geom_line(color = "red", size = 1) +
  geom_point(color = "darkred", size = 2) +
  labs(title = "Silhouette Analysis", x = "Number of Clusters (k)", y = "Average Silhouette Width") +
  theme_minimal()

# Save diagnostic plots
ggsave("figures/clustering_elbow_plot.png", plot = p1, width = 6, height = 4, dpi = 150)
ggsave("figures/clustering_silhouette_plot.png", plot = p2, width = 6, height = 4, dpi = 150)

cat(" -> Diagnostic plots saved to 'figures/clustering_elbow_plot.png' and 'figures/clustering_silhouette_plot.png'\n")

# ==============================================================================
# 5. EXECUTE FINAL K-MEANS & BURDEN PROFILING
# ==============================================================================
cat(paste("\n[Step 2/3] Running final K-Means with k =", k_clusters, "...\n"))

# Set random seed for reproducibility
set.seed(42)
final_km <- kmeans(cluster_matrix, centers = k_clusters, nstart = 50, iter.max = 100)

# Add cluster assignments to our patients
patient_clusters <- data.frame(
  Normalized_ID = common_ids,
  Participant = comorb_df$Participant,
  Cluster = final_km$cluster,
  Disease_Status = as.integer(as.logical(comorb_df[[selected_disease]]))
)

# Calculate the RAE burden (cumulative sum of 1s) per patient
patient_clusters$RAE_Burden <- rowSums(cluster_matrix)

# Classify each cluster's average burden to identify the "High Burden" background
cluster_profiles <- aggregate(RAE_Burden ~ Cluster, data = patient_clusters, FUN = mean)
colnames(cluster_profiles)[2] <- "Mean_RAE_Burden"

# FIX: In R's order() function, the argument is 'decreasing', not 'descending'!
cluster_profiles <- cluster_profiles[order(cluster_profiles$Mean_RAE_Burden, decreasing = TRUE), ]

cat("\nSummary of Patient Clusters:\n")
for (i in 1:nrow(cluster_profiles)) {
  c_id <- cluster_profiles$Cluster[i]
  c_mean <- cluster_profiles$Mean_RAE_Burden[i]
  c_size <- sum(patient_clusters$Cluster == c_id)
  c_cases <- sum(patient_clusters$Cluster == c_id & patient_clusters$Disease_Status == 1)
  prev_rate <- (c_cases / c_size) * 100
  cat(sprintf(" -> Cluster %d: %d patients | Mean RAE Burden: %.2f genes | Disease Prevalence: %.1f%%\n", 
              c_id, c_size, c_mean, prev_rate))
}

# Identify the highest-burden cluster (the one with the largest mean RAE burden)
high_burden_cluster <- cluster_profiles$Cluster[1]
cat(sprintf("\nIdentified Cluster %d as the primary High Cumulative Risk Burden cohort (Mean Burden: %.2f genes).\n", 
            high_burden_cluster, cluster_profiles$Mean_RAE_Burden[1]))

# ==============================================================================
# 6. COHORT STRATIFICATION & SAMPLESHEET GENERATION
# ==============================================================================
cat("\n[Step 3/3] Stratifying cohorts and writing DESeq2 samplesheets...\n")

# Apply the 4-way stratification rules
patient_clusters$Cohort <- "Typical_Healthy" # Default baseline

# Group 1: Resilient (High burden cluster, but healthy)
patient_clusters$Cohort[patient_clusters$Cluster == high_burden_cluster & patient_clusters$Disease_Status == 0] <- "Resilient"

# Group 2: Canonical Disease (High burden cluster, and diseased)
patient_clusters$Cohort[patient_clusters$Cluster == high_burden_cluster & patient_clusters$Disease_Status == 1] <- "Canonical_Disease"

# Group 3: Alternative Etiology (Low burden cluster, and diseased)
patient_clusters$Cohort[patient_clusters$Cluster != high_burden_cluster & patient_clusters$Disease_Status == 1] <- "Alternative_Etiology"

# Group 4: Typical Healthy (Low burden cluster, and healthy)
patient_clusters$Cohort[patient_clusters$Cluster != high_burden_cluster & patient_clusters$Disease_Status == 0] <- "Typical_Healthy"

# Count sample sizes per stratified cohort
cohort_counts <- table(patient_clusters$Cohort)
cat("\nStratified Cohort Sizes:\n")
for (cohort_name in names(cohort_counts)) {
  cat(sprintf(" -> %s: %d patients\n", cohort_name, cohort_counts[cohort_name]))
}

# Create output directories if they don't exist
dir.create("results", showWarnings = FALSE)

# Generate DESeq2 sample sheets
# Samplesheet A: Resilience Contrast (Resilient vs. Canonical Disease)
resilience_sheet <- patient_clusters[patient_clusters$Cohort %in% c("Resilient", "Canonical_Disease"), ]
write.csv(resilience_sheet, "results/deseq2_resilience_samplesheet.csv", row.names = FALSE)
cat(" -> Saved resilience sample sheet: 'results/deseq2_resilience_samplesheet.csv'\n")

# Samplesheet B: Alternative Etiology Contrast (Alternative vs. Typical Healthy)
alt_etiology_sheet <- patient_clusters[patient_clusters$Cohort %in% c("Alternative_Etiology", "Typical_Healthy"), ]
write.csv(alt_etiology_sheet, "results/deseq2_alternative_etiology_samplesheet.csv", row.names = FALSE)
cat(" -> Saved alternative etiology sample sheet: 'results/deseq2_alternative_etiology_samplesheet.csv'\n")

cat("\n=== CLUSTERING & COHORT STRATIFICATION SUCCESSFUL ===\n")


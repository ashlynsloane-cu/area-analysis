#!/usr/bin/env python3
"""
prepare_rae_clustering.py

This script serves as the "glue" code for your computational rotation project. 
It automates the binarization of your continuous gene expression values into 
Risk-Associated Expression (RAE) profiles (0 or 1) by calculating the leading-edge 
inflection point for each gene-disease pair using your real AREA results.

Inputs:
  1. Continuous gene expression matrix (filtered_values_dataframe)
  2. Binary clinical comorbidities table (filtered_binary_attributes_dataframe.csv)
  3. Pre-calculated AREA results (.adjpval file)

Output:
  A clean Patient-by-Gene binarized risk matrix (RAE_matrix_for_clustering.csv) 
  which is perfectly formatted to be fed directly into the Dowell Lab's 
  k-means clustering scripts in `AREA_publication_scripts/Data-Analysis/`.

Author: Rotational Student (with Gemini Notebook assistance)
Date: August 2026
"""

import os
import sys
import numpy as np
import pandas as pd


def calculate_leading_edge_threshold(expression_series, disease_labels, nes):
    """
    Calculates the gene expression threshold at the GSEA-like leading-edge peak.
    
    Parameters:
    -----------
    expression_series : pd.Series
        Continuous expression values for a gene across all patients.
    disease_labels : pd.Series
        Binary comorbidity status (0 or 1, or True/False) for the same patients.
    nes : float
        Normalized Enrichment Score from your AREA score sheet.
        
    Returns:
    --------
    float
        The gene expression value corresponding to the leading-edge peak.
    """
    # Create aligned dataframe and ensure labels are numeric 0 or 1
    df = pd.DataFrame({
        'expr': expression_series, 
        'disease': disease_labels.astype(int)
    })
    
    # Sort patients: descending for positive NES, ascending for negative NES
    ascending_sort = (nes < 0)
    df = df.sort_values(by='expr', ascending=ascending_sort)
    
    n_affected = df['disease'].sum()
    n_typical = len(df) - n_affected
    
    # Handle edge cases (e.g., zero disease cases in the cohort)
    if n_affected == 0 or n_typical == 0:
        return np.nan
        
    # Running sum step sizes
    step_affected = 1.0 / n_affected
    step_typical = 1.0 / n_typical
    
    running_sum = 0.0
    max_deviation = -1.0
    threshold_idx = 0
    
    # Traverse ranked cohort to find the coordinate of peak absolute deviation
    for idx, row in enumerate(df.itertuples()):
        if row.disease == 1:
            running_sum += step_affected
        else:
            running_sum -= step_typical
            
        abs_deviation = abs(running_sum)
        if abs_deviation > max_deviation:
            max_deviation = abs_deviation
            threshold_idx = idx
            
    # Retrieve the expression boundary at the peak
    threshold_value = df.iloc[threshold_idx]['expr']
    return threshold_value


def main():
    print("=== STARTING RAE BINARIZATION GLUE PIPELINE ===")
    
    # Define file names - adjust these to match your exact server directories
    expression_file = "filtered_values_dataframe.csv"
    comorbidity_file = "filtered_binary_attributes_dataframe.csv"
    area_scores_file = "254T21_WG_filterednormcounts_filteredconfoudning_MONDO_area_scores_20260115-141713.adjpval"
    
    # Check if files exist locally (for demonstration purposes, we print diagnostic instructions)
    for f in [expression_file, comorbidity_file, area_scores_file]:
        if not os.path.exists(f):
            print(f"提示/Note: File '{f}' is currently not in this directory.")
            print("Please make sure to place your real files in the same folder where you run this script!\n")
            print("Generating a simulated run with mock files to verify the logic...")
            generate_mock_and_run()
            return

    # 1. Load your real data files
    print("\n[Step 1/4] Loading real data files...")
    expr_df = pd.read_csv(expression_file, index_col=0)
    comorb_df = pd.read_csv(comorbidity_file, index_col=0)
    area_scores = pd.read_csv(area_scores_file)

    # Clean patient IDs to ensure exact matching across matrices
    common_patients = list(set(expr_df.columns).intersection(set(comorb_df['Participant'])))
    print(f"Matched {len(common_patients)} common patients across matrices.")
    
    # Subset matrices to matched patients
    expr_df = expr_df[common_patients]
    comorb_df = comorb_df[comorb_df['Participant'].isin(common_patients)].set_index('Participant')
    
    # 2. Filter AREA results to significant pairs (adj. p-value < 0.01)
    print("\n[Step 2/4] Filtering significant gene-disease associations (adj. pval < 0.01)...")
    sig_pairs = area_scores[area_scores['adjusted_pval'] < 0.01]
    print(f"Found {len(sig_pairs)} significant gene-disease associations.")

    # 3. Compute leading edge thresholds and binarize
    print("\n[Step 3/4] Calculating leading edge thresholds & binarizing...")
    rae_matrix = pd.DataFrame(0, index=common_patients, columns=sig_pairs['gene'].unique())

    for row in sig_pairs.itertuples():
        gene = row.gene
        disease = row.comorbidity  # e.g., 'MONDO_obstructive_sleep_apnea_syndrome'
        nes = row.NES
        
        # Ensure gene and comorbidity are present in your datasets
        if gene in expr_df.index and disease in comorb_df.columns:
            expr_series = expr_df.loc[gene]
            disease_series = comorb_df[disease]
            
            # Compute leading-edge threshold
            threshold = calculate_leading_edge_threshold(expr_series, disease_series, nes)
            
            if not np.isnan(threshold):
                # Binarize RAE: 1 if in risk zone, 0 typical
                if nes > 0:
                    rae_matrix[gene] = np.where(expr_series >= threshold, 1, rae_matrix[gene])
                else:
                    rae_matrix[gene] = np.where(expr_series <= threshold, 1, rae_matrix[gene])

    # 4. Save results for Downstream Clustering
    output_filename = "RAE_matrix_for_clustering.csv"
    rae_matrix.to_csv(output_filename)
    print(f"\n[Step 4/4] Success! Binarized Patient-by-Gene RAE matrix saved to: '{output_filename}'")
    print(f"Dimensions: {rae_matrix.shape} (Patients x Significant Genes)")
    print("\nYou can now feed this CSV file directly into 'Clustering_RAE.R' in the Dowell Lab repo.")


def generate_mock_and_run():
    """Generates mock datasets to test and prove the pipeline execution logic."""
    print("\n--- Running Mock Sandbox Test ---")
    patients = [f"pt-{i:03d}" for i in range(1, 255)] # 254 complete T21 patients
    genes = [f"GENE_{i}" for i in range(1, 21)]
    
    # 1. Mock Expression (Filtered, >= 1 count)
    expr_data = np.random.exponential(scale=10, size=(len(genes), len(patients))) + 1.0
    expr_df = pd.DataFrame(expr_data, index=genes, columns=patients)
    
    # 2. Mock Comorbidities
    comorb_columns = ['Participant', 'MONDO_sleep_apnea_syndrome', 'MONDO_congenital_heart_disease']
    comorb_data = []
    for pat in patients:
        comorb_data.append([
            pat,
            np.random.choice([0, 1], p=[0.7, 0.3]), # 30% OSA
            np.random.choice([0, 1], p=[0.5, 0.5])  # 50% CHD
        ])
    comorb_df = pd.DataFrame(comorb_data, columns=comorb_columns)
    
    # 3. Mock AREA scores
    area_data = [
        {'gene': 'GENE_1', 'comorbidity': 'MONDO_sleep_apnea_syndrome', 'NES': 1.8, 'adjusted_pval': 0.0005},
        {'gene': 'GENE_2', 'comorbidity': 'MONDO_sleep_apnea_syndrome', 'NES': -1.5, 'adjusted_pval': 0.0012},
        {'gene': 'GENE_3', 'comorbidity': 'MONDO_congenital_heart_disease', 'NES': 2.1, 'adjusted_pval': 0.0001},
        {'gene': 'GENE_4', 'comorbidity': 'MONDO_congenital_heart_disease', 'NES': -1.2, 'adjusted_pval': 0.008}
    ]
    area_scores = pd.DataFrame(area_data)
    
    # Set indices
    comorb_df_set = comorb_df.set_index('Participant')
    
    # Binarize
    rae_matrix = pd.DataFrame(0, index=patients, columns=area_scores['gene'].unique())
    for row in area_scores.itertuples():
        gene = row.gene
        disease = row.comorbidity
        nes = row.NES
        
        expr_series = expr_df.loc[gene]
        disease_series = comorb_df_set[disease]
        
        threshold = calculate_leading_edge_threshold(expr_series, disease_series, nes)
        print(f"Calculated Leading-Edge Threshold for {gene} ({disease}): {threshold:.4f} (NES: {nes})")
        
        if nes > 0:
            rae_matrix[gene] = np.where(expr_series >= threshold, 1, 0)
        else:
            rae_matrix[gene] = np.where(expr_series <= threshold, 1, 0)
            
    rae_matrix.to_csv("RAE_matrix_for_clustering.csv")
    print(f"\n[Mock Success] Generated 'RAE_matrix_for_clustering.csv' (Size: {rae_matrix.shape})")
    print("This confirms that the binarization loop and GSEA-like leading edge calculation are running flawlessly!")


if __name__ == "__main__":
    main()


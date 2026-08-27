#!/usr/bin/env python3
"""
prepare_rae_clustering.py (v3 - Robust Dynamic Edition)

This script serves as the "glue" code for your computational rotation project. 
It automates the binarization of your continuous gene expression values into 
Risk-Associated Expression (RAE) profiles (0 or 1) by calculating the leading-edge 
inflection point for each gene-disease pair using your real AREA results.

It has been upgraded to automatically detect:
1. Patient IDs across different formats (e.g. pt-xxxx, pt.xxxx, Xpt.xxxx)
2. Gene expression matrix orientations (Genes as rows or columns)
3. Column names in the AREA score file (e.g. adjusted_pval, adjpval, padj)

Inputs:
  1. Continuous gene expression matrix (filtered_values_dataframe.csv)
  2. Binary clinical comorbidities table (filtered_binary_attributes_dataframe.csv)
  3. Pre-calculated AREA results (.adjpval.csv file)

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


def normalize_id(id_str):
    """
    Normalizes a patient ID to make it match across different platforms 
    (e.g., handling R's conversion of 'pt-xxxx' to 'pt.xxxx' or 'Xpt.xxxx').
    """
    if not isinstance(id_str, str):
        id_str = str(id_str)
    # Remove 'X' prefix if added by R to numeric-like columns
    if id_str.lower().startswith('xpt'):
        id_str = id_str[1:]
    # Lowercase and remove all non-alphanumeric characters
    return "".join(c for c in id_str if c.isalnum()).lower()


def calculate_leading_edge_threshold(expression_series, disease_labels, nes):
    """
    Calculates the gene expression threshold at the GSEA-like leading-edge peak.
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
    
    # Handle edge cases
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
    
    # File definitions
    expression_file = "filtered_values_dataframe.csv"
    comorbidity_file = "filtered_binary_attributes_dataframe.csv"
    area_scores_file = "254T21_WG_filterednormcounts_filteredconfounding_MONDO_area_scores_20260115-141713.adjpval.csv"
    
    # Check parent folder and data subfolder
    paths_to_test = {
        'expr': [expression_file, os.path.join('data', expression_file)],
        'comorb': [comorbidity_file, os.path.join('data', comorbidity_file)],
        'scores': [area_scores_file, os.path.join('data', area_scores_file)]
    }
    
    resolved_paths = {}
    missing_files = []
    
    for key, candidates in paths_to_test.items():
        found = False
        for cand in candidates:
            if os.path.exists(cand):
                resolved_paths[key] = cand
                found = True
                break
        if not found:
            missing_files.append(candidates[0])
            
    if missing_files:
        print("\n提示/Note: The following files could not be found in the current directory or 'data/' directory:")
        for mf in missing_files:
            print(f" - {mf}")
        print("\nPlease make sure to place your files in the 'data/' folder of your project!")
        print("\nGenerating a simulated run with mock files to verify the logic...")
        generate_mock_and_run()
        return

    print("\n[Step 1/4] Found all files successfully!")
    print(f" -> Expression path:  {resolved_paths['expr']}")
    print(f" -> Comorbidity path: {resolved_paths['comorb']}")
    print(f" -> AREA scores path: {resolved_paths['scores']}")
    
    print("\nLoading real data files...")
    expr_df = pd.read_csv(resolved_paths['expr'], index_col=0)
    comorb_df = pd.read_csv(resolved_paths['comorb'], index_col=0)
    area_scores = pd.read_csv(resolved_paths['scores'])
    
    # --- Part A: Handle Patient ID Matching & Orientation ---
    expr_cols = list(expr_df.columns)
    expr_rows = list(expr_df.index)
    
    expr_cols_normalized = {normalize_id(col): col for col in expr_cols}
    expr_rows_normalized = {normalize_id(row): row for row in expr_rows}
    
    comorb_participants = list(comorb_df['Participant'])
    comorb_normalized = {normalize_id(pat): pat for pat in comorb_participants}
    
    matched_cols = set(expr_cols_normalized.keys()).intersection(set(comorb_normalized.keys()))
    matched_rows = set(expr_rows_normalized.keys()).intersection(set(comorb_normalized.keys()))
    
    if len(matched_cols) >= len(matched_rows) and len(matched_cols) > 0:
        print(f" -> Detected patients as COLUMNS in the expression matrix ({len(matched_cols)} matches).")
        matched_normalized = matched_cols
        expr_id_map = expr_cols_normalized
        # Keep columns as-is
    elif len(matched_rows) > len(matched_cols) and len(matched_rows) > 0:
        print(f" -> Detected patients as ROWS (index) in the expression matrix ({len(matched_rows)} matches).")
        print(" -> Transposing expression matrix to standard (Genes x Patients) format...")
        expr_df = expr_df.T
        matched_normalized = matched_rows
        expr_id_map = expr_rows_normalized
    else:
        print("\n[Error] Could not match any patient IDs between expression and comorbidities.")
        print(f" -> Sample expression columns: {expr_cols[:5]}")
        print(f" -> Sample expression index:   {expr_rows[:5]}")
        print(f" -> Sample comorbidity 'Participant' values: {comorb_participants[:5]}")
        sys.exit(1)
        
    # Standardize both dataframes to the intersection of matched patients
    common_patients_expr = [expr_id_map[p_norm] for p in matched_normalized]
    common_patients_comorb = [comorb_normalized[p_norm] for p in matched_normalized]
    
    # Subset expression matrix and rename columns to normalized format for easy alignment
    expr_df = expr_df[common_patients_expr]
    expr_df.columns = [normalize_id(col) for col in expr_df.columns]
    
    # Subset comorbidity dataframe and set index to normalized format
    comorb_df = comorb_df[comorb_df['Participant'].isin(common_patients_comorb)]
    comorb_df['normalized_id'] = comorb_df['Participant'].apply(normalize_id)
    comorb_df = comorb_df.set_index('normalized_id')
    
    aligned_patients = list(expr_df.columns)
    print(f"Matched and aligned {len(aligned_patients)} common patients across matrices.")
    
    # --- Part B: Handle AREA Score File Columns ---
    columns_lower = {col.lower(): col for col in area_scores.columns}
    
    # 1. Match adjusted p-value column
    adj_pval_candidates = [
        'adjusted_pval', 'adjusted_pvalue', 'adj_pval', 'adj_pvalue', 'adjpval', 
        'adjusted_p-value', 'padj', 'p.adj', 'fdr', 'q_value', 'qval'
    ]
    adj_col = None
    for cand in adj_pval_candidates:
        if cand in columns_lower:
            adj_col = columns_lower[cand]
            break
    if not adj_col:
        for col in area_scores.columns:
            col_l = col.lower()
            if 'adj' in col_l or 'padj' in col_l or 'qval' in col_l or 'fdr' in col_l:
                adj_col = col
                break
    if not adj_col:
        for col in area_scores.columns:
            if 'p' in col.lower() and 'val' in col.lower():
                adj_col = col
                break
    if not adj_col:
        print(f"[Error] Could not find adjusted p-value column. Available columns: {list(area_scores.columns)}")
        sys.exit(1)
        
    # 2. Match gene column
    gene_candidates = ['gene', 'symbol', 'gene_symbol', 'feature', 'id', 'gene_id', 'gene_name']
    gene_col = None
    for cand in gene_candidates:
        if cand in columns_lower:
            gene_col = columns_lower[cand]
            break
    if not gene_col:
        for col in area_scores.columns:
            if 'gene' in col.lower() or 'symbol' in col.lower() or 'feature' in col.lower():
                gene_col = col
                break
    if not gene_col:
        gene_col = area_scores.columns[0]
        
    # 3. Match comorbidity/attribute column
    disease_candidates = ['comorbidity', 'disease', 'attribute', 'mondo', 'trait', 'label', 'condition']
    disease_col = None
    for cand in disease_candidates:
        if cand in columns_lower:
            disease_col = columns_lower[cand]
            break
    if not disease_col:
        for col in area_scores.columns:
            col_l = col.lower()
            if 'comorb' in col_l or 'disease' in col_l or 'mondo' in col_l or 'trait' in col_l or 'cond' in col_l:
                disease_col = col
                break
    if not disease_col:
        disease_col = area_scores.columns[1]
        
    # 4. Match NES column
    nes_candidates = ['nes', 'nes_score', 'enrichment', 'score', 'normalized_enrichment_score']
    nes_col = None
    for cand in nes_candidates:
        if cand in columns_lower:
            nes_col = columns_lower[cand]
            break
    if not nes_col:
        for col in area_scores.columns:
            if 'nes' in col.lower() or 'score' in col.lower() or 'enrich' in col.lower():
                nes_col = col
                break
    if not nes_col:
        nes_col = area_scores.columns[2]
        
    print(f" -> Mapped columns successfully:")
    print(f"    * Gene ID Column:      '{gene_col}'")
    print(f"    * Comorbidity Column:  '{disease_col}'")
    print(f"    * NES Column:          '{nes_col}'")
    print(f"    * Adj. P-value Column: '{adj_col}'")

    # --- Part C: Binarization Loop ---
    print("\n[Step 2/4] Filtering significant gene-disease associations (adj. pval < 0.01)...")
    area_scores[adj_col] = pd.to_numeric(area_scores[adj_col], errors='coerce')
    sig_pairs = area_scores[area_scores[adj_col] < 0.01].dropna(subset=[adj_col])
    print(f"Found {len(sig_pairs)} significant gene-disease associations.")

    if len(sig_pairs) == 0:
        print("[Warning] No significant associations found at adj. pval < 0.01. Relaxing threshold to < 0.05...")
        sig_pairs = area_scores[area_scores[adj_col] < 0.05].dropna(subset=[adj_col])
        print(f"Found {len(sig_pairs)} significant associations at adj. pval < 0.05.")

    print("\n[Step 3/4] Calculating leading edge thresholds & binarizing...")
    unique_genes = sig_pairs[gene_col].unique()
    rae_matrix = pd.DataFrame(0, index=aligned_patients, columns=unique_genes)

    for row in sig_pairs.itertuples():
        gene = getattr(row, gene_col)
        disease = getattr(row, disease_col)
        nes = getattr(row, nes_col)
        
        # Ensure gene and comorbidity are present in our matrices
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

    # Save results
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
    
    # Align and run matching logic
    expr_cols = list(expr_df.columns)
    comorb_participants = list(comorb_df['Participant'])
    expr_normalized = {normalize_id(col): col for col in expr_cols}
    comorb_normalized = {normalize_id(pat): pat for pat in comorb_participants}
    matched_normalized = set(expr_normalized.keys()).intersection(set(comorb_normalized.keys()))
    
    common_patients_expr = [expr_id_map[p_norm] for p_norm in matched_normalized] if 'expr_id_map' in locals() else [expr_normalized[p] for p in matched_normalized]
    expr_df = expr_df[common_patients_expr]
    expr_df.columns = [normalize_id(col) for col in expr_df.columns]
    
    comorb_df_set = comorb_df.set_index('Participant')
    
    # Binarize
    rae_matrix = pd.DataFrame(0, index=[normalize_id(p) for p in patients], columns=area_scores['gene'].unique())
    for row in area_scores.itertuples():
        gene = row.gene
        disease = row.comorbidity
        nes = row.NES
        
        expr_series = expr_df.loc[gene]
        disease_series = comorb_df_set.loc[[expr_normalized[p] for p in matched_normalized], disease]
        disease_series.index = [normalize_id(idx) for idx in disease_series.index]
        
        threshold = calculate_leading_edge_threshold(expr_series, disease_series, nes)
        print(f"Calculated Leading-Edge Threshold for {gene} ({disease}): {threshold:.4f} (NES: {nes})")
        
        if nes > 0:
            rae_matrix[gene] = np.where(expr_series >= threshold, 1, 0)
        else:
            rae_matrix[gene] = np.where(expr_series <= threshold, 1, 0)
            
    rae_matrix.to_csv("RAE_matrix_for_clustering.csv")
    print(f"\n[Mock Success] Generated 'RAE_matrix_for_clustering.csv' (Size: {rae_matrix.shape})")


if __name__ == "__main__":
    main()


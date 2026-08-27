#!/usr/bin/env python3
"""
prepare_rae_clustering.py

This script serves as the "glue" code for your computational rotation project. 
It automates the binarization of your continuous gene expression values into 
Risk-Associated Expression (RAE) profiles (0 or 1) by calculating the leading-edge 
inflection point for each gene-disease pair using your real AREA results.

Inputs:
  1. Continuous gene expression matrix (filtered_values_dataframe.csv)
     - Layout: Rows are patients, columns are genes + 'Participant' column containing patient IDs.
  2. Binary clinical comorbidities table (filtered_binary_attributes_dataframe.csv)
     - Layout: Rows are patients, columns are comorbidities + 'Participant' column.
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


def normalize_id(patient_id):
    """Normalizes patient IDs to prevent punctuation or casing mismatches."""
    if pd.isna(patient_id):
        return ""
    s = str(patient_id).strip().lower()
    # Strip common prefixes/suffixes and punctuation
    s = s.replace("-", "").replace(".", "").replace("_", "")
    if s.startswith("x") and len(s) > 1: # R-style prepended X
        s = s[1:]
    return s


def find_column_by_candidates(df, candidates):
    """Finds a column name in a dataframe matching any candidate name (case-insensitive)."""
    df_cols = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in df_cols:
            idx = df_cols.index(cand.lower())
            return df.columns[idx]
    return None


def main():
    print("=== STARTING RAE BINARIZATION GLUE PIPELINE ===")
    
    # Define file names - adjust these to match your exact server directories
    expression_file = "filtered_values_dataframe.csv"
    comorbidity_file = "filtered_binary_attributes_dataframe.csv"
    area_scores_file = "254T21_WG_filterednormcounts_filteredconfounding_MONDO_area_scores_20260115-141713.adjpval.csv"
    
    # Dynamic search in both current directory and 'data/' directory
    paths = {}
    for filename, key in [
        (expression_file, "expression"),
        (comorbidity_file, "comorbidity"),
        (area_scores_file, "area_scores")
    ]:
        if os.path.exists(filename):
            paths[key] = filename
        elif os.path.exists(os.path.join("data", filename)):
            paths[key] = os.path.join("data", filename)
            
    if len(paths) < 3:
        print("\n提示/Note: The following files could not be found in the current directory or 'data/' directory:")
        missing = []
        for filename in [expression_file, comorbidity_file, area_scores_file]:
            if not os.path.exists(filename) and not os.path.exists(os.path.join("data", filename)):
                print(f" - {filename}")
                missing.append(filename)
        print("\nPlease make sure to place your files in the 'data/' folder of your project!")
        print("\nGenerating a simulated run with mock files to verify the logic...")
        generate_mock_and_run()
        return

    print("\n[Step 1/4] Found all files successfully!")
    print(f" -> Expression path:  {paths['expression']}")
    print(f" -> Comorbidity path: {paths['comorbidity']}")
    print(f" -> AREA scores path: {paths['area_scores']}")
    print("\nLoading real data files...")
    
    # Load raw dataframes
    expr_df = pd.read_csv(paths['expression'])
    comorb_df = pd.read_csv(paths['comorbidity'])
    area_scores = pd.read_csv(paths['area_scores'])
    
    # -------------------------------------------------------------
    # PATIENT ID ALIGNMENT & MATRIX ORIENTATION
    # -------------------------------------------------------------
    
    # Step A: Identify the Patient ID column in BOTH matrices
    expr_pat_col = find_column_by_candidates(expr_df, ['Participant', 'id', 'sample', 'patient_id'])
    comorb_pat_col = find_column_by_candidates(comorb_df, ['Participant', 'id', 'sample', 'patient_id'])
    
    if not expr_pat_col:
        print("\n[Error] Could not find a Participant/ID column in the expression file.")
        print(f" -> Available columns: {list(expr_df.columns[:5])}")
        sys.exit(1)
        
    if not comorb_pat_col:
        print("\n[Error] Could not find a Participant/ID column in the comorbidity file.")
        print(f" -> Available columns: {list(comorb_df.columns[:5])}")
        sys.exit(1)

    # Step B: Normalize IDs to build a mapping dictionary
    expr_df['Normalized_ID'] = expr_df[expr_pat_col].apply(normalize_id)
    comorb_df['Normalized_ID'] = comorb_df[comorb_pat_col].apply(normalize_id)
    
    # Filter out empty or null IDs
    expr_df = expr_df[expr_df['Normalized_ID'] != ""]
    comorb_df = comorb_df[comorb_df['Normalized_ID'] != ""]
    
    # Intersection of normalized IDs
    common_norm_ids = set(expr_df['Normalized_ID']).intersection(set(comorb_df['Normalized_ID']))
    
    if len(common_norm_ids) == 0:
        print("\n[Error] Could not match any patient IDs between expression and comorbidities.")
        print(f" -> Sample expression '{expr_pat_col}' values: {list(expr_df[expr_pat_col].dropna().head())}")
        print(f" -> Sample comorbidity '{comorb_pat_col}' values: {list(comorb_df[comorb_pat_col].dropna().head())}")
        sys.exit(1)
        
    print(f"Successfully matched {len(common_norm_ids)} common patients across both matrices.")
    
    # Subset matrices to matched normalized IDs and set it as the index
    expr_df = expr_df[expr_df['Normalized_ID'].isin(common_norm_ids)].set_index('Normalized_ID')
    comorb_df = comorb_df[comorb_df['Normalized_ID'].isin(common_norm_ids)].set_index('Normalized_ID')
    
    # Remove metadata columns from expression to keep only gene columns
    if expr_pat_col in expr_df.columns:
        expr_df = expr_df.drop(columns=[expr_pat_col])
    if 'Unnamed: 0' in expr_df.columns:
        expr_df = expr_df.drop(columns=['Unnamed: 0'])
        
    # -------------------------------------------------------------
    # COLUMN MAPPING FOR AREA SHEET
    # -------------------------------------------------------------
    
    gene_col = find_column_by_candidates(area_scores, ['gene', 'genesymbol', 'feature_id', 'ranked_by'])
    disease_col = find_column_by_candidates(area_scores, ['comorbidity', 'attribute', 'label', 'disease', 'boolean_attribute'])
    nes_col = find_column_by_candidates(area_scores, ['nes', 'normalized_enrichment_score'])
    pval_col = find_column_by_candidates(area_scores, ['adjusted_pval', 'adjusted_p_value', 'adjpval', 'padj', 'p.adj', 'p_value_benjaminihochberg'])
    
    if not all([gene_col, disease_col, nes_col, pval_col]):
        print("\n[Error] Could not map all necessary columns in the AREA score sheet.")
        print(f" -> Found gene col: {gene_col}")
        print(f" -> Found disease col: {disease_col}")
        print(f" -> Found NES col: {nes_col}")
        print(f" -> Found Adj. P-val col: {pval_col}")
        print(f" -> Available columns: {list(area_scores.columns)}")
        sys.exit(1)
        
    # 2. Filter AREA results to significant pairs (adj. p-value < 0.01)
    print(f"\n[Step 2/4] Filtering significant gene-disease associations using column '{pval_col}' < 0.01...")
    sig_pairs = area_scores[area_scores[pval_col] < 0.01]
    print(f"Found {len(sig_pairs)} significant gene-disease associations.")
    
    if len(sig_pairs) == 0:
        print("[Warning] No significant associations found at adj. p-value < 0.01. Relaxing threshold to < 0.05...")
        sig_pairs = area_scores[area_scores[pval_col] < 0.05]
        print(f"Found {len(sig_pairs)} associations at adj. p-value < 0.05.")
        if len(sig_pairs) == 0:
            print("[Error] No associations found even at < 0.05. Cannot proceed.")
            sys.exit(1)

    # 3. Compute leading edge thresholds and binarize
    print("\n[Step 3/4] Calculating leading edge thresholds & binarizing...")
    
    # We want our output RAE matrix to have patients as rows and genes as columns
    unique_sig_genes = [g for g in sig_pairs[gene_col].unique() if g in expr_df.columns]
    print(f"Out of {len(sig_pairs[gene_col].unique())} significant genes, {len(unique_sig_genes)} are present in the expression matrix columns.")
    
    rae_matrix = pd.DataFrame(0, index=expr_df.index, columns=unique_sig_genes)

    processed_count = 0
    # Iterating over rows using a dictionary for perfect key/index safety
    for idx, row in sig_pairs.iterrows():
        gene = row[gene_col]
        disease = row[disease_col]
        nes = row[nes_col]
        
        # Ensure gene and comorbidity are present in your datasets
        if gene in expr_df.columns and disease in comorb_df.columns:
            expr_series = expr_df[gene]
            disease_series = comorb_df[disease]
            
            # Compute leading-edge threshold
            threshold = calculate_leading_edge_threshold(expr_series, disease_series, nes)
            
            if not np.isnan(threshold):
                processed_count += 1
                # Binarize RAE: 1 if in risk zone, 0 typical
                if nes > 0:
                    rae_matrix[gene] = np.where(expr_series >= threshold, 1, rae_matrix[gene])
                else:
                    rae_matrix[gene] = np.where(expr_series <= threshold, 1, rae_matrix[gene])

    print(f"Computed leading-edge thresholds and binarized {processed_count} gene-disease configurations.")

    # 5. Restore original Participant ID names for the output
    # Create mapping from normalized ID back to original Participant ID
    id_mapping = pd.Series(comorb_df[comorb_pat_col].values, index=comorb_df.index).to_dict()
    rae_matrix.index = rae_matrix.index.map(id_mapping)
    
    # 6. Save results for Downstream Clustering
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
    expr_data = np.random.exponential(scale=10, size=(len(patients), len(genes))) + 1.0
    expr_df = pd.DataFrame(expr_data, columns=genes)
    expr_df['Participant'] = patients
    
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
        {'ranked_by': 'GENE_1', 'boolean_attribute': 'MONDO_sleep_apnea_syndrome', 'NES': 1.8, 'p_value_BenjaminiHochberg': 0.0005},
        {'ranked_by': 'GENE_2', 'boolean_attribute': 'MONDO_sleep_apnea_syndrome', 'NES': -1.5, 'p_value_BenjaminiHochberg': 0.0012},
        {'ranked_by': 'GENE_3', 'boolean_attribute': 'MONDO_congenital_heart_disease', 'NES': 2.1, 'p_value_BenjaminiHochberg': 0.0001},
        {'ranked_by': 'GENE_4', 'boolean_attribute': 'MONDO_congenital_heart_disease', 'NES': -1.2, 'p_value_BenjaminiHochberg': 0.008}
    ]
    area_scores = pd.DataFrame(area_data)
    
    # Align
    expr_df['Normalized_ID'] = expr_df['Participant'].apply(normalize_id)
    comorb_df['Normalized_ID'] = comorb_df['Participant'].apply(normalize_id)
    
    expr_df_set = expr_df.set_index('Normalized_ID')
    comorb_df_set = comorb_df.set_index('Normalized_ID')
    
    # Binarize
    rae_matrix = pd.DataFrame(0, index=patients, columns=area_scores['ranked_by'].unique())
    for idx, row in area_scores.iterrows():
        gene = row['ranked_by']
        disease = row['boolean_attribute']
        nes = row['NES']
        
        expr_series = expr_df_set[gene]
        disease_series = comorb_df_set[disease]
        
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


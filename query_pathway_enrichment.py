#!/usr/bin/env python3
"""
query_pathway_enrichment-v2.py

This script parses your comorbidity overlap reports (resilience and alternative 
etiology pairs) and performs a Gene Ontology (GO), KEGG, and Reactome pathway 
enrichment analysis using the Enrichr API. 

It dynamically:
  1. Maps Ensembl transcript IDs (e.g. ENSG00000154153.13) to official HGNC gene 
     symbols using your local gnomAD constraint table.
  2. Submits mapped gene symbols to the Enrichr API.
  3. Queries multiple standard pathway databases:
     - GO Biological Process 2023
     - KEGG 2021 Human
     - Reactome 2022
  4. Formats and prints a highly readable summary table for each pair.
  5. Saves full enrichment results as tab-separated values (TSV) and writes 
     a consolidated text report for downstream biological interpretation.

Usage:
  python3 query_pathway_enrichment-v2.py --report results/rae_only_overlap_report.txt --constraint data/gnomad_constraint.txt

Author: Rotational Student (with Gemini Notebook assistance)
Date: August 2026
"""

import os
import re
import sys
import argparse
import requests
import pandas as pd
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Query public pathway databases for overlap gene sets using Enrichr.")
    parser.add_argument(
        "--report", 
        type=str, 
        default="results/rae_only_overlap_report.txt",
        help="Path to the rae_only_overlap_report.txt or multi_comorbidity_overlap_report.txt"
    )
    parser.add_argument(
        "--constraint", 
        type=str, 
        default="data/gnomad_constraint.csv",
        help="Path to your gnomAD constraint database (used for Ensembl-to-Symbol mapping)"
    )
    parser.add_argument(
        "--out-dir", 
        type=str, 
        default="results/pathway_enrichment",
        help="Output directory for pathway tables and consolidated report"
    )
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="Optional: name of a specific pair to analyze (e.g. 'otitis_media_susceptibility_to_vs_patent_ductus_arteriosus')"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=3,
        help="Minimum number of genes in a set to run enrichment (default: 3)"
    )
    return parser.parse_args()

def clean_ensembl_id(gene_str):
    """Strips version decimals and whitespace from Ensembl IDs (e.g. 'ENSG00000181754.6' -> 'ENSG00000181754')"""
    gene_str = gene_str.strip()
    match = re.search(r"(ENSG\d+)", gene_str)
    if match:
        return match.group(1)
    return gene_str

def parse_overlap_report(report_path):
    """
    Parses rae_only_overlap_report.txt or multi_comorbidity_overlap_report.txt
    to extract Ensembl IDs from the 'Shared Buffered Risks' (Resilience) and 
    'Shared Alternative Risks' (Alternative Etiology) lists.
    """
    if not os.path.exists(report_path):
        print(f"[Error] Overlap report not found at: '{report_path}'")
        return None, None

    print(f"Parsing overlap report: '{report_path}'...")
    resilience_pairs = {}
    alternative_pairs = {}

    # Regular expressions to catch the lists of genes in the report
    res_re = re.compile(r"Shared Buffered Risks \((.+?) vs (.+?)\)\s*:\s*\d+\s*overlapping genes")
    alt_re = re.compile(r"Shared Alternative Risks \((.+?) vs (.+?)\)\s*:\s*\d+\s*overlapping genes")
    alt_re_alt = re.compile(r"Shared alternative genes between (.+?) & (.+?)\s*:\s*\d+\s*overlapping genes")
    
    gene_list_re = re.compile(r"-> (?:Gene Symbols/IDs|Overlapping Gene IDs):\s*(.+)")

    with open(report_path, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        line_str = line.strip()
        
        # Check for Resilience match
        res_match = res_re.search(line_str)
        if res_match:
            d1, d2 = res_match.group(1).strip(), res_match.group(2).strip()
            pair_name = f"{d1}_vs_{d2}"
            
            # Find the gene list on the next line (multiline alignment support)
            if idx + 1 < len(lines):
                next_line = lines[idx + 1].strip()
                gene_match = gene_list_re.search(next_line)
                if gene_match:
                    raw_genes = gene_match.group(1)
                    raw_genes = re.sub(r"\s*\(\s*\.\.\..*?\)", "", raw_genes) # remove trailing "... and X more genes"
                    gene_ids = [clean_ensembl_id(g) for g in raw_genes.split(",") if g.strip()]
                    resilience_pairs[pair_name] = gene_ids
            continue
            
        # Check for Alternative match
        alt_match = alt_re.search(line_str) or alt_re_alt.search(line_str)
        if alt_match:
            d1, d2 = alt_match.group(1).strip(), alt_match.group(2).strip()
            pair_name = f"{d1}_vs_{d2}"
            
            if idx + 1 < len(lines):
                next_line = lines[idx + 1].strip()
                gene_match = gene_list_re.search(next_line)
                if gene_match:
                    raw_genes = gene_match.group(1)
                    raw_genes = re.sub(r"\s*\(\s*\.\.\..*?\)", "", raw_genes)
                    gene_ids = [clean_ensembl_id(g) for g in raw_genes.split(",") if g.strip()]
                    alternative_pairs[pair_name] = gene_ids
            continue

    print(f" -> Extracted {len(resilience_pairs)} shared resilience pairs.")
    print(f" -> Extracted {len(alternative_pairs)} shared alternative etiology pairs.")
    return resilience_pairs, alternative_pairs

def load_ensembl_to_symbol_map(constraint_path):
    """Loads gnomAD constraint file to map Ensembl IDs to official Gene Symbols"""
    if not os.path.exists(constraint_path):
        print(f"[Note] Constraint database '{constraint_path}' not found. Using raw Ensembl IDs as fallback.")
        return {}

    print(f"Loading constraint file for mapping: '{constraint_path}'...")
    try:
        if constraint_path.endswith('.tsv') or constraint_path.endswith('.txt'):
            constraint_df = pd.read_csv(constraint_path, sep='\t')
        else:
            constraint_df = pd.read_csv(constraint_path)

        # Standardize column names to lowercase for robust matching
        constraint_df.columns = [c.lower() for c in constraint_df.columns]
        
        # Identify gene ID column dynamically
        id_col = None
        for cand in ['gene_id', 'ensembl_id', 'gene', 'ensembl', 'id']:
            if cand in constraint_df.columns:
                id_col = cand
                break
                
        if not id_col:
            print(" -> [Warning] Could not find gene ID column in constraint file. Fallback enabled.")
            return {}

        constraint_df['clean_id'] = constraint_df[id_col].astype(str).apply(clean_ensembl_id)
        constraint_df = constraint_df.set_index('clean_id')

        # Identify gene symbol column dynamically
        symbol_col = None
        for cand in ['gene_name', 'gene', 'symbol', 'gene_symbol']:
            if cand in constraint_df.columns:
                symbol_col = cand
                break
                
        if symbol_col:
            ensembl_to_symbol = constraint_df[symbol_col].dropna().to_dict()
            print(f" -> Successfully loaded gene symbol mapping from column '{symbol_col}'.")
            return ensembl_to_symbol
        else:
            print(" -> [Warning] No gene symbol column found in constraint database. Fallback enabled.")
            return {}
    except Exception as e:
        print(f" -> [Warning] Error loading constraint database for mapping: {e}. Fallback enabled.")
        return {}

def query_enrichr(gene_list, list_description, libraries=['GO_Biological_Process_2023', 'KEGG_2021_Human', 'Reactome_2022']):
    """Submits gene symbols to Enrichr and retrieves top enriched terms for specified libraries"""
    if not gene_list:
        return {}

    # Step 1: Add gene list to Enrichr
    add_list_url = 'https://maayanlab.cloud/Enrichr/addList'
    payload = {
        'list': (None, '\n'.join(gene_list)),
        'description': (None, list_description)
    }

    try:
        response = requests.post(add_list_url, files=payload, timeout=15)
        if response.status_code != 200:
            print(f"   [Error] Enrichr API failed to add gene list. Code: {response.status_code}")
            return None
        
        data = response.json()
        user_list_id = data['userListId']
    except requests.exceptions.ConnectionError:
        return "offline"
    except Exception as e:
        print(f"   [Error] Unexpected exception during Enrichr POST: {type(e).__name__}: {e}")
        return None

    # Step 2: Retrieve enrichment results for each library
    enrich_url = 'https://maayanlab.cloud/Enrichr/enrich'
    results = {}

    for library in libraries:
        try:
            res = requests.get(f"{enrich_url}?userListId={user_list_id}&backgroundType={library}", timeout=15)
            if res.status_code == 200:
                results_data = res.json()
                # Parse output lists into structured pandas DataFrames
                # Columns: Rank, Term name, P-value, Z-score, Combined score, Overlapping genes, Adjusted p-value
                terms_list = results_data.get(library, [])
                records = []
                for term in terms_list:
                    records.append({
                        'Term': term[1],
                        'P-value': term[2],
                        'Z-score': term[3],
                        'Combined_Score': term[4],
                        'Overlap_Genes': ", ".join(term[5]),
                        'Adjusted_P-value': term[6]
                    })
                df = pd.DataFrame(records)
                if not df.empty:
                    df = df.sort_values('Adjusted_P-value')
                results[library] = df
            else:
                print(f"   [Warning] Failed to fetch enrichment for {library}. Code: {res.status_code}")
        except Exception as e:
            print(f"   [Warning] Exception fetching enrichment for {library}: {e}")

    return results

def main():
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Parse Overlap Report
    resilience_pairs, alternative_pairs = parse_overlap_report(args.report)
    if resilience_pairs is None:
        sys.exit(1)

    # 2. Load Mapping Dictionary
    ensembl_to_symbol = load_ensembl_to_symbol_map(args.constraint)

    # Combine pairs to analyze
    target_pairs = []
    
    # Optional filtering to single pair
    if args.pair:
        if args.pair in resilience_pairs:
            target_pairs.append(('Resilience Overlap', args.pair, resilience_pairs[args.pair]))
        elif args.pair in alternative_pairs:
            target_pairs.append(('Alternative Overlap', args.pair, alternative_pairs[args.pair]))
        else:
            print(f"[Error] Specified pair '{args.pair}' not found in parsed report.")
            sys.exit(1)
    else:
        for pair, genes in resilience_pairs.items():
            target_pairs.append(('Resilience Overlap', pair, genes))
        for pair, genes in alternative_pairs.items():
            target_pairs.append(('Alternative Overlap', pair, genes))

    summary_lines = []
    summary_lines.append("=================================================================")
    summary_lines.append("MULTI-COMORBIDITY OVERLAP FUNCTIONAL PATHWAY ENRICHMENT REPORT")
    summary_lines.append("=================================================================\\n")
    summary_lines.append(f"Parsed Overlap Report: '{args.report}'")
    summary_lines.append(f"gnomAD Constraint Mapping: '{args.constraint}'\\n")

    libraries = ['GO_Biological_Process_2023', 'KEGG_2021_Human', 'Reactome_2022']
    any_offline = False

    for cohort_type, pair_name, genes in target_pairs:
        # Check size constraints
        if len(genes) < args.min_size:
            print(f"\\nSkipping {cohort_type}: {pair_name} ({len(genes)} genes) - below minimum size threshold of {args.min_size}")
            continue

        print(f"\\nAnalyzing {cohort_type}: '{pair_name}' ({len(genes)} genes)...")
        
        # Map Ensembl IDs to symbols
        mapped_symbols = []
        unmapped_ids = []
        for g in genes:
            symbol = ensembl_to_symbol.get(g)
            if symbol and not symbol.startswith("MOCK_"): # Ignore placeholder symbols
                mapped_symbols.append(symbol)
            else:
                unmapped_ids.append(g)

        # Print mapping stats
        print(f" -> Mapped {len(mapped_symbols)} / {len(genes)} genes to official symbols.")
        if unmapped_ids:
            print(f" -> Unmapped IDs: {', '.join(unmapped_ids[:5])}" + (f" (... and {len(unmapped_ids)-5} more)" if len(unmapped_ids) > 5 else ""))

        if not mapped_symbols:
            # Fallback to raw IDs if mapping completely failed
            mapped_symbols = genes
            print(" -> [Note] Using raw Ensembl IDs directly for enrichment query.")

        # Query Enrichr API
        list_description = f"T21 {cohort_type} genes: {pair_name}"
        enrichment_data = query_enrichr(mapped_symbols, list_description, libraries)

        if enrichment_data == "offline":
            any_offline = True
            print(" -> [Notice] Sandbox is offline. online pathway enrichment skipped.")
            continue
        elif not enrichment_data:
            print(" -> [Warning] Enrichr returned no results or query failed.")
            continue

        # Save individual TSV table
        out_table_path = os.path.join(args.out_dir, f"{cohort_type.lower().replace(' ', '_')}_{pair_name}_enrichment.tsv")
        combined_dfs = []
        for lib, df in enrichment_data.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy['Library'] = lib
                combined_dfs.append(df_copy)
        
        if combined_dfs:
            master_df = pd.concat(combined_dfs).sort_values('Adjusted_P-value')
            master_df.to_csv(out_table_path, sep='\t', index=False)
            print(f" -> Full pathway table written to: '{out_table_path}'")

        # Compile Consolidated Report
        summary_lines.append("-----------------------------------------------------------------")
        summary_lines.append(f"COHORT: {cohort_type}")
        summary_lines.append(f"PAIR:   {pair_name}")
        summary_lines.append(f"GENES:  {len(genes)} genes ({len(mapped_symbols)} mapped)")
        summary_lines.append(f"SYMBOLS: {', '.join(mapped_symbols[:15])}" + (f" (... and {len(mapped_symbols)-15} more)" if len(mapped_symbols) > 15 else ""))
        summary_lines.append("-----------------------------------------------------------------")

        for lib in libraries:
            df = enrichment_data.get(lib, pd.DataFrame())
            summary_lines.append(f"\\n * Database: {lib}")
            if df.empty:
                summary_lines.append("   No significant terms found.")
                continue

            # Display top 5 enriched terms
            top_5 = df.head(5)
            summary_lines.append(f"   {'%-50s' % 'Enriched Pathway Term'} | {'%-12s' % 'Adj. P-val'} | {'%-15s' % 'Overlapping Genes'}")
            summary_lines.append(f"   {'-'*50} | {'-'*12} | {'-'*15}")
            
            for _, row in top_5.iterrows():
                term_truncated = row['Term'][:48] + ".." if len(row['Term']) > 50 else row['Term']
                summary_lines.append(f"   {'%-50s' % term_truncated} | {row['Adjusted_P-value']:.2e}   | {row['Overlap_Genes'][:30]}")
            
            # Print a quick summary to the console as well
            print(f"\\n   -> Top Enriched {lib} Terms:")
            for _, row in top_5.head(3).iterrows():
                print(f"      - {row['Term']} (Adj p-val: {row['Adjusted_P-value']:.2e}, overlap: {row['Overlap_Genes'][:40]}...)")
        summary_lines.append("\\n\\n")

    # 4. Save Final Consolidated Report
    if not any_offline:
        report_out_path = os.path.join(args.out_dir, "overlap_pathway_enrichment_report.txt")
        with open(report_out_path, 'w') as f_out:
            f_out.write("\\n".join(summary_lines))
        print(f"\\n[Success] Consolidated pathway report saved to: '{report_out_path}'")
    else:
        print("\\n[Notice] Pathway report was not saved because execution occurred in the offline sandbox environment.")
        print("When you run this script locally on your Mac, it will successfully fetch the live data and write the report!")

if __name__ == "__main__":
    main()


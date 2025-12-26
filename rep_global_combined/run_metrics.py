import pandas as pd
import os
import sys
import argparse
import re

# Add current directory to path so we can import metrics
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.insert(0, current_dir)

import metrics

# 1. Define Organs
# We try to import from data.py, otherwise default to the standard list
try:
    from data import ALL_TARGET_KEYS
    # Ensure they are upper case for matching headers
    ALL_ORGANS = [k.upper() for k in ALL_TARGET_KEYS]
except ImportError:
    # Fallback list if data.py is missing
    ALL_ORGANS = [
        "LUNG", "HEART", "AORTA", "ESOPHAGUS", "TRACHEA", 
        "RIB", "LIVER", "GALLBLADDER", "STOMACH", "PANCREAS", 
        "SPLEEN", "KIDNEY"
    ]

def normalize_text(text):
    """Simple cleanup to ensure consistent comparisons."""
    if not isinstance(text, str): return ""
    return " ".join(text.split()).strip()

def extract_organ_sections(text):
    """
    Parses a report string like "LUNG: text... HEART: text..." into a dictionary.
    Returns: {'LUNG': 'text...', 'HEART': 'text...', ...}
    """
    text = normalize_text(text)
    if not text:
        return {k: "" for k in ALL_ORGANS}
        
    sections = {k: "" for k in ALL_ORGANS}
    
    # Create a regex pattern that looks for "ORGAN:" (case insensitive)
    # matching the known organ keys
    pattern = r"(" + "|".join(ALL_ORGANS) + r"):"
    
    # Split text. Result example: ['', 'LUNG', 'text...', 'HEART', 'text...']
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    # Iterate pairs: [Header, Content]
    # We skip parts[0] as it is text before the first header
    for i in range(1, len(parts)-1, 2):
        header = parts[i].upper().strip()
        content = parts[i+1].strip()
        
        # If the content runs into the next header, the split handles it,
        # but we should ensure we assign it to the correct known organ
        if header in sections:
            sections[header] = content

    return sections

def run_evaluation_in_folder(folder_path):
    # --- 1. Locate Input File ---
    csv_filename = "generated_reports.csv"
    csv_path = os.path.join(folder_path, csv_filename)
    
    if not os.path.exists(csv_path):
        fallback_path = os.path.join(folder_path, "generated_report.csv")
        if os.path.exists(fallback_path):
            print(f"'{csv_filename}' not found. Using '{os.path.basename(fallback_path)}'.")
            csv_path = fallback_path
        else:
            print(f"Error: Could not find '{csv_filename}' in: {folder_path}")
            return

    print(f"Loading reports from: {csv_path}")
    
    # --- 2. Load Data ---
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Handle NaNs and normalize types
    df['pred'] = df['pred'].fillna("").astype(str)
    df['ref'] = df['ref'].fillna("").astype(str)
    
    # --- 3. Prepare Data Buckets ---
    global_preds = []
    global_refs = []
    
    organ_buckets = {k: {'preds': [], 'refs': []} for k in ALL_ORGANS}
    
    print("Parsing reports...")
    
    for idx, row in df.iterrows():
        p_text = row['pred']
        r_text = row['ref']
        
        # A. GLOBAL: Add everything that has a valid reference
        if len(r_text.strip()) > 2:
            global_preds.append(p_text)
            global_refs.append(r_text)
        
        # B. ORGAN: Parse and Filter
        p_parts = extract_organ_sections(p_text)
        r_parts = extract_organ_sections(r_text)
        
        for organ in ALL_ORGANS:
            ref_content = r_parts[organ]
            pred_content = p_parts[organ]
            
            # CRITICAL FIX: Only evaluate this organ if the REFERENCE has content.
            # If the ground truth doesn't mention the Gallbladder, we shouldn't 
            # evaluate the model on Gallbladder for this patient.
            if len(ref_content) > 2:
                organ_buckets[organ]['refs'].append(ref_content)
                organ_buckets[organ]['preds'].append(pred_content)

    summary = []

    # --- 4. Compute Global Metrics ---
    print(f"\nComputing GLOBAL Metrics (N={len(global_preds)})...")
    if global_preds:
        try:
            g_stats = metrics.compute_all_metrics(global_preds, global_refs)
            g_stats['Organ'] = 'GLOBAL'
            g_stats['N'] = len(global_preds)
            summary.append(g_stats)
            print(f"Global BLEU-4: {g_stats.get('BLEU-4', 0):.4f} | GREEN: {g_stats.get('GREEN', 0):.4f}")
        except Exception as e:
            print(f"Error computing global metrics: {e}")
            import traceback
            traceback.print_exc()

    # --- 5. Compute Per-Organ Metrics ---
    for organ in ALL_ORGANS:
        preds = organ_buckets[organ]['preds']
        refs = organ_buckets[organ]['refs']
        
        n_samples = len(refs)
        
        if n_samples > 0:
            print(f"Computing Metrics for {organ} (N={n_samples})...")
            try:
                o_stats = metrics.compute_all_metrics(preds, refs)
                o_stats['Organ'] = organ
                o_stats['N'] = n_samples
                summary.append(o_stats)
            except Exception as e:
                print(f"Error computing {organ}: {e}")
        else:
            # print(f"Skipping {organ} (No references found)")
            pass

    # --- 6. Save Outputs ---
    print("\n" + "="*30)
    print("       SAVING RESULTS       ")
    print("="*30)
    
    # Save Text Summary
    results_txt_path = os.path.join(folder_path, 'final_metrics_summary.txt')
    with open(results_txt_path, 'w') as f:
        for res in summary:
            f.write(f"--- {res['Organ']} (N={res['N']}) ---\n")
            for k, v in res.items():
                if k not in ['Organ', 'N']:
                    f.write(f"{k}: {v}\n")
            f.write("\n")
    print(f"Text summary saved to: {results_txt_path}")
    
    # Generate Visuals
    if summary:
        try:
            metrics.create_metrics_table_plot(summary, folder_path)
            print(f"Visual table saved to: {folder_path}/metrics_table.png")
            print(f"CSV breakdown saved to: {folder_path}/metrics_breakdown.csv")
        except Exception as e:
            print(f"Could not generate plot: {e}")
    else:
        print("No results to plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Global and Organ-based metrics from CSV.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing generated_reports.csv")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.folder):
        run_evaluation_in_folder(args.folder)
    else:
        print(f"Error: '{args.folder}' is not a valid directory.")
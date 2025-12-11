#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, cpu_count

# --- 1. RAW TOTAL SEGMENTATOR LABEL MAP (Verified 1-117) ---
TOTAL_SEG_MAP = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "liver",
    6: "stomach", 7: "pancreas", 8: "adrenal_gland_right", 9: "adrenal_gland_left",
    10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left", 12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right", 15: "esophagus",
    16: "trachea", 17: "thyroid_gland", 18: "small_bowel", 19: "duodenum",
    20: "colon", 21: "urinary_bladder", 22: "prostate", 23: "kidney_cyst_left",
    24: "kidney_cyst_right", 25: "sacrum", 26: "vertebrae_S1", 27: "vertebrae_L5",
    28: "vertebrae_L4", 29: "vertebrae_L3", 30: "vertebrae_L2", 31: "vertebrae_L1",
    32: "vertebrae_T12", 33: "vertebrae_T11", 34: "vertebrae_T10", 35: "vertebrae_T9",
    36: "vertebrae_T8", 37: "vertebrae_T7", 38: "vertebrae_T6", 39: "vertebrae_T5",
    40: "vertebrae_T4", 41: "vertebrae_T3", 42: "vertebrae_T2", 43: "vertebrae_T1",
    44: "vertebrae_C7", 45: "vertebrae_C6", 46: "vertebrae_C5", 47: "vertebrae_C4",
    48: "vertebrae_C3", 49: "vertebrae_C2", 50: "vertebrae_C1", 51: "heart",
    52: "aorta", 53: "pulmonary_vein", 54: "brachiocephalic_trunk",
    55: "subclavian_artery_right", 56: "subclavian_artery_left",
    57: "common_carotid_artery_right", 58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left", 60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left", 62: "superior_vena_cava",
    63: "inferior_vena_cava", 64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left", 66: "iliac_artery_right", 67: "iliac_vena_left",
    68: "iliac_vena_right", 69: "humerus_left", 70: "humerus_right",
    71: "scapula_left", 72: "scapula_right", 73: "clavicula_left",
    74: "clavicula_right", 75: "femur_left", 76: "femur_right",
    77: "hip_left", 78: "hip_right", 79: "spinal_cord",
    80: "gluteus_maximus_left", 81: "gluteus_maximus_right",
    82: "gluteus_medius_left", 83: "gluteus_medius_right",
    84: "gluteus_minimus_left", 85: "gluteus_minimus_right",
    86: "autochthon_left", 87: "autochthon_right", 88: "iliopsoas_left",
    89: "iliopsoas_right", 90: "brain", 91: "skull", 92: "rib_left_1",
    93: "rib_left_2", 94: "rib_left_3", 95: "rib_left_4",
    96: "rib_left_5", 97: "rib_left_6", 98: "rib_left_7",
    99: "rib_left_8", 100: "rib_left_9", 101: "rib_left_10",
    102: "rib_left_11", 103: "rib_left_12", 104: "rib_right_1",
    105: "rib_right_2", 106: "rib_right_3", 107: "rib_right_4",
    108: "rib_right_5", 109: "rib_right_6", 110: "rib_right_7",
    111: "rib_right_8", 112: "rib_right_9", 113: "rib_right_10",
    114: "rib_right_11", 115: "rib_right_12", 116: "sternum",
    117: "costal_cartilages"
}

# --- 2. MERGED GROUP MAPPING (Updated & Complete) ---
MERGED_GROUPS = {
    # --- MAJOR ORGANS ---
    'lung': [10, 11, 12, 13, 14],
    'heart': [51, 61],
    'aorta': [52],
    'liver': [5],
    'spleen': [1],
    'pancreas': [7],
    'kidney': [2, 3], # Excluding cysts
    'adrenal': [8, 9],
    'gallbladder': [4],
    'stomach': [6],
    'esophagus': [15],
    'trachea': [16],
    'thyroid': [17],
    'colon': [20],
    'small_bowel': [18, 19], # Small bowel + Duodenum
    'bladder': [21],
    'brain': [90],

    # --- VASCULAR (Grouping major veins) ---
    'vena_cava': [62, 63], # SVC + IVC
    'pulmonary_vein': [53],
    'portal_vein': [64],

    # --- BONES ---
    'spine': list(range(26, 51)), # C1-C7, T1-T12, L1-L5, S1
    'rib': list(range(92, 116)) + [117], # Ribs + Costal Cartilages
    'sternum': [116],
    'clavicula': [73, 74],
    'scapula': [71, 72],
    'humerus': [69, 70],
    'femur': [75, 76],
    'hip': [77, 78],
    'sacrum': [25],
    'face': [91], # Skull

    # --- MUSCLES ---
    'gluteus': [80, 81, 82, 83, 84, 85],
    'iliopsoas': [88, 89],
    'autochthon': [86, 87]
}

# --- WORKER FUNCTION ---
def process_single_mask(filepath):
    """
    Analyzes one file for both Raw IDs and Merged Groups.
    """
    try:
        # Extract ID
        filename = os.path.basename(filepath)
        pid = filename.replace(".nii.gz", "").replace(".nii", "")
        
        # Load Data
        img = nib.load(filepath)
        data = img.get_fdata()
        
        # Get unique IDs present in this volume
        unique_ids = np.unique(data).astype(int)
        unique_ids_set = set(unique_ids[unique_ids != 0]) # Remove background
        
        # 1. Analyze Raw Labels
        found_raw = []
        for uid in unique_ids_set:
            if uid in TOTAL_SEG_MAP:
                found_raw.append(TOTAL_SEG_MAP[uid])
            else:
                found_raw.append(f"unknown_{uid}")
        
        # 2. Analyze Merged Groups
        found_merged = []
        for group_name, group_ids in MERGED_GROUPS.items():
            # If ANY ID from the group is present, count it
            if not set(group_ids).isdisjoint(unique_ids_set):
                found_merged.append(group_name)
                
        return (pid, found_raw, found_merged)
        
    except Exception as e:
        return (None, f"Error processing {filepath}: {str(e)}", None)

def analyze_dataset(mask_folder, output_dir, file_extension="*.nii.gz", num_workers=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Find Files
    search_path = os.path.join(mask_folder, file_extension)
    files = glob.glob(search_path)
    if not files:
        search_path = os.path.join(mask_folder, "**", file_extension)
        files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(files)} mask files in {mask_folder}")
    if len(files) == 0: return

    if num_workers is None:
        num_workers = cpu_count()
        if num_workers > 16: num_workers = 16 
        
    print(f"Starting analysis with {num_workers} parallel workers...")
    
    raw_records = []
    merged_records = []
    
    global_raw_counts = Counter()
    global_merged_counts = Counter()
    
    errors = []

    # Parallel Processing
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_mask, files), total=len(files)))

    print("Aggregating results...")
    for pid, raw_list, merged_list in results:
        if pid is None:
            errors.append(raw_list) # raw_list contains error msg here
            continue
            
        # Raw Record
        r_rec = {'patient_id': pid}
        for organ in raw_list:
            r_rec[organ] = 1
            global_raw_counts[organ] += 1
        raw_records.append(r_rec)
        
        # Merged Record
        m_rec = {'patient_id': pid}
        for organ in merged_list:
            m_rec[organ] = 1
            global_merged_counts[organ] += 1
        merged_records.append(m_rec)

    if errors:
        print(f"Encountered {len(errors)} errors. See errors.log")
        with open(os.path.join(output_dir, "errors.log"), "w") as f:
            for e in errors: f.write(e + "\n")

    # Helper function to save and plot
    def save_and_plot(records, counts, prefix):
        if not records: return
        
        # Create DataFrame
        df = pd.DataFrame(records).fillna(0)
        
        # FIX: Convert only organ columns to int, NOT patient_id
        organ_cols = [c for c in df.columns if c != 'patient_id']
        df[organ_cols] = df[organ_cols].astype(int)
        
        # Sort cols
        df = df[['patient_id'] + sorted(organ_cols)]
        
        # Summary
        df_sum = pd.DataFrame.from_dict(counts, orient='index', columns=['count']).reset_index()
        df_sum.rename(columns={'index': 'organ'}, inplace=True)
        df_sum = df_sum.sort_values(by='count', ascending=False)
        
        # Save CSVs
        df.to_csv(os.path.join(output_dir, f"{prefix}_presence_matrix.csv"), index=False)
        df_sum.to_csv(os.path.join(output_dir, f"{prefix}_frequency_summary.csv"), index=False)
        print(f"Saved {prefix} CSVs.")
        
        # Plot
        plt.figure(figsize=(12, 18 if prefix == 'raw' else 12))
        # Top 50 is enough for raw, all for merged
        plot_df = df_sum.head(50 if prefix == 'raw' else 100) 
        sns.barplot(data=plot_df, x='count', y='organ', palette='viridis')
        plt.title(f"{prefix.capitalize()} Organ Frequency (N={len(records)})", fontsize=16)
        plt.xlabel("Count", fontsize=12)
        plt.ylabel("Organ", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_frequency_plot.png"), dpi=200)
        plt.close()

    # --- SAVE RAW DATA ---
    print("\nProcessing RAW Labels...")
    save_and_plot(raw_records, global_raw_counts, "raw")
    
    # --- SAVE MERGED DATA ---
    print("\nProcessing MERGED Labels...")
    save_and_plot(merged_records, global_merged_counts, "merged")
    
    print(f"\nAnalysis Complete. Results in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_folder', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./mask_analysis')
    parser.add_argument('--num_workers', type=int, default=None)
    
    args = parser.parse_args()
    
    analyze_dataset(args.mask_folder, args.output_dir, num_workers=args.num_workers)
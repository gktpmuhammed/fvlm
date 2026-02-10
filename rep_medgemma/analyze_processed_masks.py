
import os
import argparse
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from monai.transforms import Compose, LoadImaged, SpatialPadd, CenterSpatialCropd, Transposed, EnsureChannelFirstd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random
import torch

# --- 1. MERGED GROUP MAPPING (Same as before) ---
MERGED_GROUPS = {
    'lung': [10, 11, 12, 13, 14],
    'heart': [51, 61],
    'esophagus': [15],
    'liver': [5],
    'gallbladder': [4],
    'stomach': [6],
    'pancreas': [7],
    'spleen': [1],
    'kidney': [2, 3],
    'aorta': [52],
    'trachea': [16],
    'rib': list(range(92, 116)) + [117],
}

def build_transforms():
    # Exact replica of train.py (excluding ScaleIntensity for mask)
    return Compose([
        LoadImaged(keys=['mask'], reader='ITKReader', image_only=True), # image_only=True usually returns tensor/array directly if single key? 
        # Wait, if image_only=True with multiple keys, it returns list?
        # In train.py: LoadImaged(keys=['image', 'mask'], ...)
        # Here we only need mask.
        # Let's use standard dictionary load.
        EnsureChannelFirstd(keys=['mask']),
        # train.py: Transposed(keys=['image', 'mask'], indices=(0, 3, 2, 1))
        # Input (Mask): (H, W, D) -> (C, H, W, D) via EnsureChannelFirst
        # Transposed: (C, D, H, W)
        Transposed(keys=['mask'], indices=(0, 3, 2, 1)),
        # SpatialPadd and Crop on (D, H, W)
        SpatialPadd(keys=['mask'], spatial_size=(112, 256, 352), mode='constant', constant_values=0),
        CenterSpatialCropd(keys=['mask'], roi_size=(112, 256, 352)),
    ])

# Global transform for workers (pickling)
# But transforms might not pickle well if they contain C++ objects.
# Better to instantiate inside worker.

def process_single_mask(filepath):
    try:
        # Instantiate transform per worker to avoid pickle issues with ITKReader
        transforms = Compose([
            LoadImaged(keys=['mask'], reader='ITKReader'), # Returns dict
            EnsureChannelFirstd(keys=['mask']),
            Transposed(keys=['mask'], indices=(0, 3, 2, 1)),
            SpatialPadd(keys=['mask'], spatial_size=(112, 256, 352), mode='constant', constant_values=0),
            CenterSpatialCropd(keys=['mask'], roi_size=(112, 256, 352)),
        ])

        filename = os.path.basename(filepath)
        pid = filename.replace(".nii.gz", "").replace(".nii", "")
        
        # Apply Transform
        data_dict = {'mask': filepath}
        out = transforms(data_dict)
        mask_tensor = out['mask'] # (C, D, H, W)
        
        # Convert to numpy
        if isinstance(mask_tensor, torch.Tensor):
            data = mask_tensor.numpy()
        else:
            data = mask_tensor

        # Analyze
        unique_ids = np.unique(data).astype(int)
        unique_ids_set = set(unique_ids[unique_ids != 0])
        
        found = []
        for group_name, group_ids in MERGED_GROUPS.items():
            if not set(group_ids).isdisjoint(unique_ids_set):
                found.append(group_name)
                
        return (pid, found)
    except Exception as e:
        return (None, str(e))

def analyze_dataset(mask_folder, output_dir, file_extension="*.nii.gz", num_workers=None, sample_size=100):
    os.makedirs(output_dir, exist_ok=True)
    search_path = os.path.join(mask_folder, file_extension)
    files = glob.glob(search_path)
    if not files:
        search_path = os.path.join(mask_folder, "**", file_extension)
        files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(files)} mask files in {mask_folder}")
    if len(files) == 0: return

    # Random Sample
    if len(files) > sample_size:
        print(f"Sampling {sample_size} files...")
        files = random.sample(files, sample_size)
    
    if num_workers is None: num_workers = 16
        
    print(f"Starting Processed Analysis with {num_workers} workers...")
    
    records = []
    global_counts = {k:0 for k in MERGED_GROUPS.keys()}
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_mask, files), total=len(files)))

    print("Aggregating results...")
    valid_count = 0
    for pid, found_list in results:
        if pid is None: continue # Error
        valid_count += 1
        rec = {'patient_id': pid}
        for organ in found_list:
            rec[organ] = 1
            global_counts[organ] += 1
        records.append(rec)

    # Save
    df = pd.DataFrame.from_dict(global_counts, orient='index', columns=['count']).reset_index()
    df.rename(columns={'index': 'organ'}, inplace=True)
    df['percentage'] = (df['count'] / valid_count) * 100
    df = df.sort_values(by='count', ascending=False)
    
    print("\n=== ORGAN FREQUENCY (AFTER PREPROCESSING) ===")
    print(df.to_string())
    
    df.to_csv(os.path.join(output_dir, "processed_frequency_summary.csv"), index=False)
    print(f"\nAnalysis Complete. Results in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_folder', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./mask_analysis_processed')
    
    args = parser.parse_args()
    analyze_dataset(args.mask_folder, args.output_dir, sample_size=100)


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

# --- MERGED GROUP MAPPING ---
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

def get_organ_counts(data_array):
    """Counts voxels for each organ group in the array."""
    unique, counts = np.unique(data_array, return_counts=True)
    counts_dict = dict(zip(unique.astype(int), counts))
    
    organ_counts = {}
    for group_name, group_ids in MERGED_GROUPS.items():
        total_voxels = 0
        for gid in group_ids:
            total_voxels += counts_dict.get(gid, 0)
        organ_counts[group_name] = total_voxels
    return organ_counts

def process_single_samples(filepath):
    try:
        # 1. Load Original & Count
        # We use MONAI LoadImaged for consistency, but we need the data before cropping.
        # But wait, Transposed and ChannelFirst change dimensions but preserve Voxel Count (mostly).
        # We should calculate "Original" on the loaded, transposed data *before* padding/cropping.
        
        loader = LoadImaged(keys=['mask'], reader='ITKReader')
        reorienteer = Compose([
            EnsureChannelFirstd(keys=['mask']),
            Transposed(keys=['mask'], indices=(0, 3, 2, 1))
        ])
        
        cropper = Compose([
            SpatialPadd(keys=['mask'], spatial_size=(112, 256, 352), mode='constant', constant_values=0),
            CenterSpatialCropd(keys=['mask'], roi_size=(112, 256, 352))
        ])

        # Load
        data_dict = loader({'mask': filepath})
        
        # Reorient (C, D, H, W)
        # We compare counts here as "Original" because Transpose doesn't change content.
        data_dict = reorienteer(data_dict)
        mask_original = data_dict['mask']
        if isinstance(mask_original, torch.Tensor): mask_original = mask_original.numpy()
        
        counts_original = get_organ_counts(mask_original)
        
        # Crop
        data_dict = cropper(data_dict)
        mask_cropped = data_dict['mask']
        if isinstance(mask_cropped, torch.Tensor): mask_cropped = mask_cropped.numpy()
        
        counts_cropped = get_organ_counts(mask_cropped)
        
        # Return Stats
        # Structure: {organ: (original_vol, cropped_vol)}
        stats = {}
        for organ in MERGED_GROUPS:
            orig = counts_original[organ]
            crop = counts_cropped[organ]
            stats[organ] = (orig, crop)
            
        filename = os.path.basename(filepath)
        pid = filename.replace(".nii.gz", "").replace(".nii", "")
        
        return (pid, stats)

    except Exception as e:
        return (None, str(e))

def analyze_retention(mask_folder, output_dir, file_extension="*.nii.gz", num_workers=None, sample_size=100):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Gather Files
    search_path = os.path.join(mask_folder, file_extension)
    files = glob.glob(search_path)
    if not files:
        search_path = os.path.join(mask_folder, "**", file_extension)
        files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(files)} mask files.")
    if len(files) == 0: return

    if len(files) > sample_size:
        print(f"Sampling {sample_size} files...")
        files = random.sample(files, sample_size)
    
    if num_workers is None: num_workers = 16
    print(f"Starting Retention Analysis with {num_workers} workers...")
    
    # 2. Run Parallel
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_samples, files), total=len(files)))
        
    # 3. Aggregate
    # We want: Average Retention % per organ (averaged over patients who HAVE the organ originally)
    
    organ_ratios = {k: [] for k in MERGED_GROUPS}
    
    for pid, stats in results:
        if pid is None: continue
        
        for organ, (orig_vol, crop_vol) in stats.items():
            if orig_vol > 0:
                ratio = crop_vol / orig_vol
                organ_ratios[organ].append(ratio)
    
    # 4. summary
    summary = []
    for organ, ratios in organ_ratios.items():
        if ratios:
            avg_retention = np.mean(ratios) * 100
            median_retention = np.median(ratios) * 100
            count_present = len(ratios)
        else:
            avg_retention = 0.0
            median_retention = 0.0
            count_present = 0
            
        summary.append({
            'organ': organ,
            'avg_retention_pct': avg_retention,
            'median_retention_pct': median_retention,
            'samples_with_organ': count_present
        })
        
    df = pd.DataFrame(summary).sort_values(by='avg_retention_pct', ascending=False)
    
    print("\n=== ORGAN RETENTION ANALYSIS (Avg % Volume Kept) ===")
    print(df.to_string(float_format="%.1f"))
    
    df.to_csv(os.path.join(output_dir, "organ_retention_stats.csv"), index=False)
    print(f"\nSaved analysis to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_folder', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./mask_retention_analysis')
    parser.add_argument('--sample_size', type=int, default=100)
    
    args = parser.parse_args()
    analyze_retention(args.mask_folder, args.output_dir, sample_size=args.sample_size)

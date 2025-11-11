import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
from monai import transforms
from multiprocessing import Pool, cpu_count
from functools import partial

# ----- Start: Copied Custom Pipeline Classes from analyze_organ_detection.py -----
class SpacingNormalization:
    """Custom transform to normalize spacing to reference spacing"""
    def __init__(self, ref_spacing=(1.0, 1.0, 3.0)):
        self.ref_spacing = ref_spacing
    def __call__(self, data):
        affine = data["image_meta_dict"]["affine"]
        spacing = (
            abs(affine[0, 0].item()), abs(affine[1, 1].item()), abs(affine[2, 2].item())
        )
        _, h, w, d = data["image"].shape
        scale = [spacing[i] / self.ref_spacing[i] for i in range(3)]
        target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]

        if target_size != [h, w, d]:
            resize_transform = transforms.Compose([
                transforms.Resized(spatial_size=target_size, keys=["image"], mode="trilinear"),
                transforms.Resized(spatial_size=target_size, keys=["label"], mode="nearest"),
            ])
            result = resize_transform(data)
            for key in ["image", "label"]:
                if key in result and hasattr(result[key], 'meta') and result[key].meta:
                    if 'pixdim' in result[key].meta:
                        result[key].meta['pixdim'][1:4] = self.ref_spacing
                    if 'affine' in result[key].meta:
                        for i in range(3):
                            result[key].meta['affine'][i, i] = self.ref_spacing[i] * (
                                1 if result[key].meta['affine'][i, i] > 0 else -1
                            )
            return result
        else:
            return data

class Original_ROI_Crop_d:
    """
    Manual ROI cropping with fixed extension (copied from analyze_organ_detection.py)
    """
    def __init__(self, keys, extend_d=5, extend_hw=20):
        self.keys = keys
        self.extend_d = extend_d
        self.extend_hw = extend_hw
    def __call__(self, data):
        import warnings
        d = data.copy()
        image = d["image"]
        label = d["label"]
        if not isinstance(label, torch.Tensor):
            label = torch.as_tensor(label)
        if label.dtype not in (torch.int, torch.long, torch.int8, torch.int16):
            label = label.long()
        if torch.sum(label) > 0:
            # Use torch.nonzero to find bounding box
            roi_coords_tuple = torch.nonzero(label[0], as_tuple=True)
            if len(roi_coords_tuple) != 3:
                warnings.warn(f"ROI crop: label shape is not 3D (got {len(roi_coords_tuple)}D); skipping crop.")
                return d
            min_dhw = torch.tensor([torch.min(coords) for coords in roi_coords_tuple])
            max_dhw = torch.tensor([torch.max(coords) for coords in roi_coords_tuple])
            min_dhw = torch.maximum(
                min_dhw - torch.tensor([self.extend_d, self.extend_hw, self.extend_hw]),
                torch.tensor([0, 0, 0]),
            )
            max_dhw = torch.minimum(
                max_dhw + torch.tensor([self.extend_d, self.extend_hw, self.extend_hw]),
                torch.tensor([image.shape[1], image.shape[2], image.shape[3]]),
            )
            for key in self.keys:
                d[key] = d[key][
                    :, min_dhw[0] : max_dhw[0], min_dhw[1] : max_dhw[1], min_dhw[2] : max_dhw[2]
                ]
        return d

class ToRegularTensor:
    """Convert MetaTensor to regular PyTorch tensor, preserving metadata if present."""
    def __init__(self, keys):
        self.keys = keys
    def __call__(self, data):
        import torch
        import numpy as np
        for key in self.keys:
            if key in data:
                item = data[key]
                original_meta = None
                if hasattr(item, 'meta') and item.meta is not None:
                    original_meta = dict(item.meta)
                try:
                    if hasattr(item, 'array'):
                        array_data = np.array(item.array, copy=True)
                        new_tensor = torch.from_numpy(array_data).clone()
                    elif hasattr(item, 'data'):
                        array_data = np.array(item.data, copy=True)
                        new_tensor = torch.from_numpy(array_data).clone()
                    elif hasattr(item, 'detach'):
                        new_tensor = torch.tensor(item.detach().cpu().numpy(), dtype=item.dtype)
                    else:
                        array_data = np.array(item, copy=True)
                        new_tensor = torch.from_numpy(array_data).clone()
                    if original_meta is not None:
                        from monai.data import MetaTensor
                        data[key] = MetaTensor(new_tensor, meta=original_meta)
                    else:
                        data[key] = new_tensor
                except Exception as e:
                    array_data = np.array(item, copy=True)
                    data[key] = torch.from_numpy(array_data).contiguous()
        return data
# ----- End: Custom Classes -----

def apply_merge_labels(label, num_organs):
    # --- Now uses full merging logic as analyze_organ_detection.py ---
    class_map = {
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

    merged_organ_id_4 = {
        "lung_upper_lobe_left": 0, "lung_lower_lobe_left": 0,
        "lung_upper_lobe_right": 0, "lung_middle_lobe_right": 0,
        "lung_lower_lobe_right": 0,
        "heart": 1, "atrial_appendage_left": 1,
        "esophagus": 2,
        "aorta": 3,
    }
    merged_organ_id_24 = {
        # Face/Head
        "skull": 0, "brain": 1,
        # Thoracic
        "esophagus": 2, "trachea": 3,
        "lung_upper_lobe_left": 4, "lung_lower_lobe_left": 4,
        "lung_upper_lobe_right": 4, "lung_middle_lobe_right": 4,
        "lung_lower_lobe_right": 4,
        "heart": 5, "atrial_appendage_left": 5,
        # Abdominal
        "kidney_right": 6, "kidney_left": 6,
        "stomach": 7, "liver": 8, "gallbladder": 9,
        "pancreas": 10, "spleen": 11, "colon": 12,
        # Vascular
        "aorta": 13,
        # Ribs (grouped)
        "rib_left_1": 14, "rib_left_2": 14, "rib_left_3": 14, "rib_left_4": 14,
        "rib_left_5": 14, "rib_left_6": 14, "rib_left_7": 14, "rib_left_8": 14,
        "rib_left_9": 14, "rib_left_10": 14, "rib_left_11": 14, "rib_left_12": 14,
        "rib_right_1": 14, "rib_right_2": 14, "rib_right_3": 14, "rib_right_4": 14,
        "rib_right_5": 14, "rib_right_6": 14, "rib_right_7": 14, "rib_right_8": 14,
        "rib_right_9": 14, "rib_right_10": 14, "rib_right_11": 14, "rib_right_12": 14,
        # Bones
        "humerus_left": 15, "humerus_right": 15,
        "scapula_left": 16, "scapula_right": 16,
        "clavicula_left": 17, "clavicula_right": 17,
        "femur_left": 18, "femur_right": 18,
        "hip_left": 19, "hip_right": 19,
        "sacrum": 20, "vertebrae_S1": 20,
        # Muscles
        "gluteus_maximus_left": 21, "gluteus_maximus_right": 21,
        "gluteus_medius_left": 21, "gluteus_medius_right": 21,
        "gluteus_minimus_left": 21, "gluteus_minimus_right": 21,
        "iliopsoas_left": 22, "iliopsoas_right": 22,
        "autochthon_left": 23, "autochthon_right": 23
    }
    if num_organs == 4:
        merged_organ_id = merged_organ_id_4
    else:
        merged_organ_id = merged_organ_id_24
    fused_mask = np.zeros_like(label)
    for original_id, organ_name in class_map.items():
        if organ_name in merged_organ_id:
            merged_id = merged_organ_id[organ_name]
            fused_mask[label == original_id] = merged_id + 1
    return fused_mask

def load_annotations(desc_path='data/desc_info.json', conc_path='data/conc_info.json'):
    desc_info = json.load(open(desc_path))
    conc_info = json.load(open(conc_path))
    return desc_info, conc_info

def create_pipeline(organs_list):
    # COMPLETE pipeline as in analyze_organ_detection.py (keep steps exactly the same)
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True),
        transforms.Lambdad(keys=["label"], func=lambda x: apply_merge_labels(x, len(organs_list))),
        SpacingNormalization(ref_spacing=(1.0, 1.0, 3.0)),
        transforms.Transposed(keys=["image", "label"], indices=(0, 3, 2, 1)),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-1150, a_max=350,
            b_min=0.0, b_max=1.0, clip=True
        ),
        Original_ROI_Crop_d(keys=["image", "label"]),
        transforms.SpatialPadd(
            keys=["image", "label"],
            spatial_size=(112, 256, 352),
            mode="constant",
            constant_values=0
        ),
        transforms.CenterSpatialCropd(
            keys=["image", "label"],
            roi_size=(112, 256, 352)
        ),
        ToRegularTensor(keys=["image", "label"])
    ])

def analyze_patient_voxel_percentages(image_path, mask_path, patient_id, organs_list):
    # Build pipeline up to (and including) SpacingNormalization & Transpose
    pre_crop_pipeline = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True),
        transforms.Lambdad(keys=["label"], func=lambda x: apply_merge_labels(x, len(organs_list))),
        SpacingNormalization(ref_spacing=(1.0, 1.0, 3.0)),
        transforms.Transposed(keys=["image", "label"], indices=(0, 3, 2, 1)),
        ToRegularTensor(keys=["image", "label"])
    ])
    # Check original mask BEFORE merging to see if ribs exist
    # Load just the label without merging first using nibabel
    import nibabel as nib
    try:
        nii_img = nib.load(mask_path)
        orig_array = np.array(nii_img.get_fdata())
        # Check for rib labels in original mask (rib labels are 92-115 in class_map)
        rib_labels = list(range(92, 116))  # rib_left_1 (92) through rib_right_12 (115)
        original_rib_count = int(np.sum(np.isin(orig_array, rib_labels)))
    except Exception as e:
        # If loading fails, skip diagnostic
        original_rib_count = 0
    
    # Run up to pre-crop
    data = pre_crop_pipeline({'image': image_path, 'label': mask_path})
    label_mid = data['label'].numpy() if hasattr(data['label'], 'numpy') else data['label']
    if isinstance(label_mid, torch.Tensor):
        label_mid = label_mid.cpu().numpy()
    # Defensive: ensure label is (C, D, H, W)
    if label_mid.ndim == 3:
        label_mid = np.expand_dims(label_mid, axis=0)
    elif label_mid.ndim == 4:
        pass
    else:
        print(f"Warning: label for patient {patient_id} has unexpected shape {label_mid.shape}, skipping patient.")
        return None
    # Now label_mid is safe for the rest of pipeline
    # Remove channel dimension for counting (label_mid is (C, D, H, W), we want (D, H, W))
    if label_mid.ndim == 4:
        label_mid_3d = label_mid[0]  # Remove channel dimension
    else:
        label_mid_3d = label_mid
    # Count voxels for each organ after spacing normalization + transpose
    pre_crop_counts = {}
    for organ_id, organ_name in enumerate(organs_list, 1):
        count = int(np.count_nonzero(label_mid_3d == organ_id))
        pre_crop_counts[organ_name] = count
        # Diagnostic: check if we're looking at the right value for ribs
        if organ_name == 'rib' and count == 0:
            # Check if ribs might be at a different value (merged_id=14 means value=15)
            rib_at_15 = int(np.count_nonzero(label_mid_3d == 15))
            if rib_at_15 > 0:
                print(f"Warning: Patient {patient_id}: Found {rib_at_15} rib voxels at value 15, but expected at value 14. Mapping issue!")
            # Check if ribs existed in original mask but got lost
            if original_rib_count > 0:
                # Additional diagnostic: check what values actually exist in the merged label
                unique_values = np.unique(label_mid_3d)
                print(f"Warning: Patient {patient_id}: Found {original_rib_count} rib voxels in ORIGINAL mask, but 0 after merging/normalization.")
                print(f"  Unique values in merged label: {unique_values[:20]}... (showing first 20)")
                print(f"  Looking for organ_id={organ_id} (rib), but ribs should be at merged_id=14+1=15")
                # Check if value 15 exists
                if 15 in unique_values:
                    count_at_15 = int(np.count_nonzero(label_mid_3d == 15))
                    print(f"  Found {count_at_15} voxels at value 15 - this should be ribs!")
    # Now run full pipeline after pre-crop
    full_pipeline = transforms.Compose([
        Original_ROI_Crop_d(keys=["image", "label"]),
        transforms.SpatialPadd(
            keys=["image", "label"],
            spatial_size=(112, 256, 352),
            mode="constant",
            constant_values=0
        ),
        transforms.CenterSpatialCropd(
            keys=["image", "label"],
            roi_size=(112, 256, 352)
        ),
        ToRegularTensor(keys=["image", "label"])
    ])
    data = {'image': data['image'], 'label': label_mid}  # use mid-output as input
    data = full_pipeline(data)
    label_post = data['label'].numpy() if hasattr(data['label'], 'numpy') else data['label']
    if isinstance(label_post, torch.Tensor):
        label_post = label_post.cpu().numpy()
    if label_post.ndim == 4 and label_post.shape[0] == 1:
        label_post = label_post[0]
    # Count voxels for each organ after all cropping
    post_crop_counts = {}
    for organ_id, organ_name in enumerate(organs_list, 1):
        post_crop_counts[organ_name] = int(np.count_nonzero(label_post == organ_id))
    # Compute retained voxel percentage
    result = {'patient_id': patient_id}
    for organ in organs_list:
        result[f'{organ}_pre_crop'] = pre_crop_counts[organ]
        result[f'{organ}_post_crop'] = post_crop_counts[organ]
        retained = (post_crop_counts[organ] / pre_crop_counts[organ] * 100) if pre_crop_counts[organ] > 0 else 0.0
        result[f'{organ}_retained_pct'] = retained
    return result

def find_image_and_mask_paths(patient_path):
    import os
    files = [f for f in os.listdir(patient_path) if f.endswith('.nii.gz')]
    if not files:
        return None, None
    image_path = os.path.join(patient_path, files[0])
    # Assume 'images' vs 'masks' in the full path; for detection, images are inside the patient_path
    mask_path = image_path.replace('images', 'masks')
    if not os.path.exists(mask_path):
        return None, None
    return image_path, mask_path

def analyze_patient_voxel_percentages_worker(args):
    import os
    patient_path, organs_list = args
    patient_id = os.path.basename(patient_path)
    image_path, mask_path = find_image_and_mask_paths(patient_path)
    if not (image_path and mask_path):
        print(f"Warning: missing file for patient {patient_id} - Skipping.")
        return None
    return analyze_patient_voxel_percentages(image_path, mask_path, patient_id, organs_list)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze per-organ voxel percentages after preprocessing.")
    parser.add_argument('--data_root', type=str, default='data/train/images/train', help='Root dir of training data')
    parser.add_argument('--all_organs', action='store_true', help='Use all 24 organs')
    parser.add_argument('--output_dir', type=str, default='output/organ_voxel_pct_analysis_debug', help='Output dir')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples (for testing)')
    parser.add_argument('--num_workers', type=int, default=4, help='Parallel workers')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.all_organs:
        organs_list = ['face', 'brain', 'esophagus', 'trachea', 'lung', 'heart', \
            'kidney', 'stomach', 'liver', 'gallbladder', 'pancreas', 'spleen', \
            'colon', 'aorta', 'rib', 'humerus', 'scapula', 'clavicula', 'femur', \
            'hip', 'sacrum', 'gluteus', 'iliopsoas', 'autochthon']
    else:
        organs_list = ['lung', 'heart', 'esophagus', 'aorta']
    print(f"Analyzing {len(organs_list)} organs: {organs_list}")
    # Find all patient directories as before
    patient_paths = [
        os.path.join(args.data_root, f1, f2)
        for f1 in os.listdir(args.data_root)
        for f2 in os.listdir(os.path.join(args.data_root, f1))
        if os.path.isdir(os.path.join(args.data_root, f1, f2))
    ]
    if args.max_samples:
        patient_paths = patient_paths[:args.max_samples]
    print(f"Analyzing {len(patient_paths)} patients...")
    # Prepare jobs with paths, not with image/mask files
    job_args = [ (pp, organs_list) for pp in patient_paths ]
    if args.num_workers > 0:
        with Pool(args.num_workers) as pool:
            results = list(tqdm(pool.imap(analyze_patient_voxel_percentages_worker, job_args), total=len(patient_paths)))
    else:
        results = [analyze_patient_voxel_percentages_worker(arg) for arg in tqdm(job_args)]
    results = [r for r in results if r is not None]
    df_result = pd.DataFrame(results)
    df_result.to_csv(os.path.join(args.output_dir, 'per_scan_organ_voxel_percentages.csv'), index=False)
    stats = {}
    valid_organs = [organ for organ in organs_list if f"{organ}_retained_pct" in df_result.columns]
    for organ in valid_organs:
        values = df_result[f"{organ}_retained_pct"].values
        stats[organ] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'percentile_10': np.percentile(values, 10),
            'percentile_25': np.percentile(values, 25),
            'percentile_75': np.percentile(values, 75),
            'percentile_90': np.percentile(values, 90),
        }
    df_stats = pd.DataFrame(stats).T
    df_stats.to_csv(os.path.join(args.output_dir, 'per_organ_voxel_percentage_stats.csv'))
    # Create plots for _retained_pct columns
    plt.figure(figsize=(18, 6))
    for i, organ in enumerate(valid_organs):
        plt.subplot(1, len(valid_organs), i + 1)
        sns.histplot(df_result[f"{organ}_retained_pct"], bins=30, kde=True)
        plt.title(f"{organ}")
        plt.xlabel("Retained Voxel Percentage (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'organ_voxel_retention_histograms.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_result[[f"{organ}_retained_pct" for organ in valid_organs]])
    plt.ylabel("Retained Voxel Percentage (%)")
    plt.xticks(ticks=range(len(valid_organs)), labels=valid_organs, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'organ_voxel_retention_boxplot.png'))
    print(f"Saved all output to {args.output_dir}/")
if __name__ == "__main__":
    main()

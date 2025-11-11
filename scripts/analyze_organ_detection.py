"""
Script to analyze organ detection after preprocessing and boundary checking.
Compares detected complete organs vs. ground truth annotations from reports.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from tqdm import tqdm
from monai import transforms
from multiprocessing import Pool, cpu_count
from functools import partial

def apply_merge_labels(label, num_organs):
    """Apply label merging based on number of organs"""
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
    
    # For 4 organs
    merged_organ_id_4 = {
        "lung_upper_lobe_left": 0, "lung_lower_lobe_left": 0,
        "lung_upper_lobe_right": 0, "lung_middle_lobe_right": 0,
        "lung_lower_lobe_right": 0,
        "heart": 1, "atrial_appendage_left": 1,
        "esophagus": 2,
        "aorta": 3,
    }
    
    # For all 24 organs
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
    
    # Choose mapping based on num_organs
    if num_organs == 4:
        merged_organ_id = merged_organ_id_4
    else:
        merged_organ_id = merged_organ_id_24
    
    # Apply merging
    fused_mask = np.zeros_like(label)
    for original_id, organ_name in class_map.items():
        if organ_name in merged_organ_id:
            merged_id = merged_organ_id[organ_name]
            fused_mask[label == original_id] = merged_id + 1
    return fused_mask

def check_boundary_organs(seg_mask, organ_mapping):
    """
    Check which organs touch the boundaries (same logic as training)
    Returns: detected_organs (not touching boundary), boundary_organs (touching boundary)
    """
    boundaries = [
        seg_mask[0], seg_mask[-1],
        seg_mask[:, 0], seg_mask[:, -1],
        seg_mask[:, :, 0], seg_mask[:, :, -1]
    ]
    
    non_zero_boundaries = [b[b != 0].flatten() for b in boundaries if b.size > 0 and (b != 0).any()]
    boundary_values = np.concatenate(non_zero_boundaries) if len(non_zero_boundaries) > 0 else np.array([])
    boundary_organ_ids = np.unique(boundary_values) if len(boundary_values) > 0 else np.array([])
    
    # Get all organ IDs present in the mask
    all_organ_ids = np.unique(seg_mask)
    all_organ_ids = all_organ_ids[all_organ_ids > 0]  # Remove background
    
    # Separate into detected (complete) and boundary organs
    detected_organ_ids = [oid for oid in all_organ_ids if oid not in boundary_organ_ids]
    boundary_organ_ids = [oid for oid in all_organ_ids if oid in boundary_organ_ids]
    
    return detected_organ_ids, boundary_organ_ids

def load_annotations(desc_path='data/desc_info.json', conc_path='data/conc_info.json'):
    """Load ground truth annotations from reports"""
    desc_info = json.load(open(desc_path))
    conc_info = json.load(open(conc_path))
    
    return desc_info, conc_info

class SpacingNormalization:
    """Custom transform to normalize spacing to reference spacing"""
    def __init__(self, ref_spacing=(1.0, 1.0, 3.0)):
        self.ref_spacing = ref_spacing
    
    def __call__(self, data):
        # Get original spacing from affine matrix
        affine = data["image_meta_dict"]["affine"]
        spacing = (abs(affine[0, 0].item()), abs(affine[1, 1].item()), abs(affine[2, 2].item()))
        
        # Calculate scale and target size
        _, h, w, d = data["image"].shape
        scale = [spacing[i] / self.ref_spacing[i] for i in range(3)]
        target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]
        
        # Apply resizing if needed
        if target_size != [h, w, d]:
            resize_transform = transforms.Compose([
                transforms.Resized(spatial_size=target_size, keys=["image"], mode="trilinear"),
                transforms.Resized(spatial_size=target_size, keys=["label"], mode="nearest"),
            ])
            result = resize_transform(data)
            
            # Manually update spacing metadata after resize
            for key in ["image", "label"]:
                if key in result and hasattr(result[key], 'meta') and result[key].meta:
                    # Update pixdim with new spacing
                    if 'pixdim' in result[key].meta:
                        result[key].meta['pixdim'][1:4] = self.ref_spacing
                    
                    # Update affine matrix diagonal with new spacing  
                    if 'affine' in result[key].meta:
                        for i in range(3):
                            result[key].meta['affine'][i, i] = self.ref_spacing[i] * (1 if result[key].meta['affine'][i, i] > 0 else -1)
            
            return result
        else:
            return data

class Original_ROI_Crop_d:
    """
    Custom dictionary-based transform to replicate the original pipeline's
    manual ROI cropping with fixed extensions.
    """
    def __init__(self, keys, extend_d=5, extend_hw=20):
        self.keys = keys
        self.extend_d = extend_d
        self.extend_hw = extend_hw

    def __call__(self, data):
        d = data.copy()
        image = d["image"]
        label = d["label"]

        # Ensure label is integer type for nonzero operation
        if not isinstance(label, torch.Tensor):
            label = torch.as_tensor(label)
        
        if label.dtype not in (torch.int, torch.long, torch.int8, torch.int16):
             label = label.long()

        if torch.sum(label) > 0:
            # Use torch.nonzero to find bounding box
            roi_coords_tuple = torch.nonzero(label[0], as_tuple=True)
            
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
    """Custom transform to convert MetaTensor to regular PyTorch tensor while preserving metadata."""
    def __init__(self, keys):
        self.keys = keys
        
    def __call__(self, data):
        import torch
        import numpy as np
        
        for key in self.keys:
            if key in data:
                item = data[key]
                
                # Store metadata before conversion
                original_meta = None
                if hasattr(item, 'meta') and item.meta is not None:
                    original_meta = dict(item.meta)  # Make a copy
                
                # Force conversion to regular PyTorch tensor with proper storage
                try:
                    if hasattr(item, 'array'):
                        # MetaTensor case - create completely new tensor
                        array_data = np.array(item.array, copy=True)
                        new_tensor = torch.from_numpy(array_data).clone()
                    elif hasattr(item, 'data'):
                        # Other MONAI tensor types
                        array_data = np.array(item.data, copy=True) 
                        new_tensor = torch.from_numpy(array_data).clone()
                    elif hasattr(item, 'detach'):
                        # Already a tensor but might be MetaTensor
                        new_tensor = torch.tensor(item.detach().cpu().numpy(), dtype=item.dtype)
                    else:
                        # Convert any other format
                        array_data = np.array(item, copy=True)
                        new_tensor = torch.from_numpy(array_data).clone()
                        
                    # Restore metadata if it existed
                    if original_meta is not None:
                        # Create a new MetaTensor with preserved metadata
                        from monai.data import MetaTensor
                        data[key] = MetaTensor(new_tensor, meta=original_meta)
                    else:
                        data[key] = new_tensor
                        
                except Exception as e:
                    # Fallback: force numpy conversion
                    array_data = np.array(item, copy=True)
                    data[key] = torch.from_numpy(array_data).contiguous()
        return data

def create_pipeline(organs_list):
    """Create the preprocessing pipeline (same as training)"""
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

def analyze_patient_worker(args):
    """Worker function for parallel processing"""
    patient_path, desc_info, conc_info, organs_list = args
    
    patient_id = patient_path.split('/')[-1]
    
    # Get a sample image from this patient
    try:
        files = [f for f in os.listdir(patient_path) if f.endswith('.nii.gz')]
        if not files:
            return None
            
        image_path = os.path.join(patient_path, files[0])
        mask_path = image_path.replace('images', 'masks')
        
        if not os.path.exists(mask_path):
            return None
        
        return analyze_patient(image_path, mask_path, patient_id, desc_info, conc_info, organs_list)
    except Exception as e:
        # Silently skip errors in parallel processing
        return None

def analyze_patient(image_path, mask_path, patient_id, desc_info, conc_info, organs_list):
    """Analyze a single patient's data"""
    pipeline = create_pipeline(organs_list)
    
    try:
        data = pipeline({'image': image_path, 'label': mask_path})
        
        # Extract the processed label (after all transforms)
        if hasattr(data['label'], 'numpy'):
            label = data['label'].numpy()
        elif isinstance(data['label'], torch.Tensor):
            label = data['label'].cpu().numpy()
        else:
            label = np.array(data['label'])
        
        # Remove channel dimension if present
        if label.ndim == 4 and label.shape[0] == 1:
            label = label[0]
        
        # The label is already merged by the pipeline
        fused_mask = label
        
        # Define organ map based on organs_list
        if len(organs_list) == 4:
            organ_map = ['lung', 'heart', 'esophagus', 'aorta']
        else:
            organ_map = [
                'face', 'brain', 'esophagus', 'trachea', 'lung', 'heart', 
                'kidney', 'stomach', 'liver', 'gallbladder', 'pancreas', 'spleen', 
                'colon', 'aorta', 'rib', 'humerus', 'scapula', 'clavicula', 
                'femur', 'hip', 'sacrum', 'gluteus', 'iliopsoas', 'autochthon'
            ]
        
        # Check which organs are detected vs boundary
        detected_ids, boundary_ids = check_boundary_organs(fused_mask, organ_map)
        
        detected_organs = [organ_map[int(oid) - 1] for oid in detected_ids]
        boundary_organs = [organ_map[int(oid) - 1] for oid in boundary_ids]
        
        # Get ground truth organs from reports
        gt_organs = set()
        abnormal_organs = set()
        
        if patient_id in desc_info:
            for organ in organs_list:
                if organ in desc_info[patient_id]:
                    gt_organs.add(organ)
                    # Check if it's abnormal (has findings)
                    if desc_info[patient_id][organ].strip():
                        abnormal_organs.add(organ)
        
        if patient_id in conc_info:
            for organ in organs_list:
                if organ in conc_info[patient_id]:
                    gt_organs.add(organ)
                    # Check if it's abnormal
                    conc_text = conc_info[patient_id][organ]
                    if not conc_text.startswith(f'{organ} shows no significant abnormalities'):
                        abnormal_organs.add(organ)
        
        return {
            'patient_id': patient_id,
            'detected_organs': detected_organs,
            'boundary_organs': boundary_organs,
            'gt_organs': list(gt_organs),
            'abnormal_organs': list(abnormal_organs),
            'total_organs_in_mask': len(detected_ids) + len(boundary_ids),
            'detected_count': len(detected_ids),
            'boundary_count': len(boundary_ids)
        }
        
    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze organ detection vs ground truth")
    parser.add_argument('--data_root', type=str, default='data/train/images/train',
                       help='Root directory of training data')
    parser.add_argument('--all_organs', action='store_true',
                       help='Analyze all 24 organs instead of default 4')
    parser.add_argument('--output_dir', type=str, default='output/organ_detection_analysis',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to analyze (for testing)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of parallel workers (default: 4, use 0 for sequential)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define organs based on flag
    if args.all_organs:
        organs_list = [
            'face', 'brain', 'esophagus', 'trachea', 'lung', 'heart', 
            'kidney', 'stomach', 'liver', 'gallbladder', 'pancreas', 'spleen', 
            'colon', 'aorta', 'rib', 'humerus', 'scapula', 'clavicula', 
            'femur', 'hip', 'sacrum', 'gluteus', 'iliopsoas', 'autochthon'
        ]
        print(f"Analyzing all {len(organs_list)} organs")
    else:
        organs_list = ['lung', 'heart', 'esophagus', 'aorta']
        print(f"Analyzing default {len(organs_list)} organs")
    
    # Load annotations
    print("Loading annotations...")
    desc_info, conc_info = load_annotations()
    
    # Get all patient paths
    print("Collecting patient data...")
    patient_paths = [
        os.path.join(args.data_root, f1, f2)
        for f1 in os.listdir(args.data_root)
        for f2 in os.listdir(os.path.join(args.data_root, f1))
        if os.path.isdir(os.path.join(args.data_root, f1, f2))
    ]
    
    if args.max_samples:
        patient_paths = patient_paths[:args.max_samples]
    
    print(f"Analyzing {len(patient_paths)} patients...")
    
    # Analyze all patients (parallel or sequential)
    if args.num_workers > 0:
        print(f"Using {args.num_workers} parallel workers")
        
        # Prepare arguments for worker function
        worker_args = [(path, desc_info, conc_info, organs_list) for path in patient_paths]
        
        # Use multiprocessing pool
        with Pool(processes=args.num_workers) as pool:
            # Use imap for progress bar
            results = []
            for result in tqdm(pool.imap(analyze_patient_worker, worker_args), 
                             total=len(patient_paths), 
                             desc="Processing patients"):
                if result:
                    results.append(result)
    else:
        print("Using sequential processing")
        results = []
        for patient_path in tqdm(patient_paths, desc="Processing patients"):
            patient_id = patient_path.split('/')[-1]
            
            # Get a sample image from this patient
            files = [f for f in os.listdir(patient_path) if f.endswith('.nii.gz')]
            if not files:
                continue
                
            image_path = os.path.join(patient_path, files[0])
            mask_path = image_path.replace('images', 'masks')
            
            if not os.path.exists(mask_path):
                continue
            
            result = analyze_patient(image_path, mask_path, patient_id, desc_info, conc_info, organs_list)
            if result:
                results.append(result)
    
    print(f"\nAnalyzed {len(results)} patients successfully")
    
    # Compute statistics
    print("\n" + "="*80)
    print("ORGAN DETECTION ANALYSIS")
    print("="*80)
    
    # Overall statistics
    total_detected = sum(r['detected_count'] for r in results)
    total_boundary = sum(r['boundary_count'] for r in results)
    total_organs = total_detected + total_boundary
    
    print(f"\nOverall Statistics:")
    print(f"  Total organ instances in masks: {total_organs}")
    if total_organs > 0:
        print(f"  Detected (complete): {total_detected} ({100*total_detected/total_organs:.1f}%)")
        print(f"  Boundary (discarded): {total_boundary} ({100*total_boundary/total_organs:.1f}%)")
    else:
        print("  WARNING: No organs found in analyzed data. Check your data paths.")
        return
    
    # Per-organ statistics
    organ_stats = defaultdict(lambda: {'detected': 0, 'boundary': 0, 'gt_present': 0, 
                                       'gt_abnormal': 0, 'lost_abnormal': 0})
    
    for result in results:
        for organ in result['detected_organs']:
            organ_stats[organ]['detected'] += 1
        for organ in result['boundary_organs']:
            organ_stats[organ]['boundary'] += 1
        for organ in result['gt_organs']:
            if organ in organs_list:
                organ_stats[organ]['gt_present'] += 1
        for organ in result['abnormal_organs']:
            if organ in organs_list:
                organ_stats[organ]['gt_abnormal'] += 1
                # Check if this abnormal organ was lost due to boundary
                if organ in result['boundary_organs']:
                    organ_stats[organ]['lost_abnormal'] += 1
    
    print(f"\nPer-Organ Detection Rates:")
    print(f"{'Organ':<15} {'Detected':<10} {'Boundary':<10} {'Detection %':<12} {'GT Present':<12} {'GT Abnormal':<12} {'Lost Abnormal':<15}")
    print("-" * 115)
    
    organ_detection_data = []
    for organ in organs_list:
        stats = organ_stats[organ]
        total = stats['detected'] + stats['boundary']
        detection_rate = 100 * stats['detected'] / total if total > 0 else 0
        
        print(f"{organ:<15} {stats['detected']:<10} {stats['boundary']:<10} {detection_rate:<11.1f}% "
              f"{stats['gt_present']:<12} {stats['gt_abnormal']:<12} {stats['lost_abnormal']:<15}")
        
        organ_detection_data.append({
            'organ': organ,
            'detected': stats['detected'],
            'boundary': stats['boundary'],
            'detection_rate': detection_rate,
            'gt_present': stats['gt_present'],
            'gt_abnormal': stats['gt_abnormal'],
            'lost_abnormal': stats['lost_abnormal']
        })
    
    # Save detailed results
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(args.output_dir, 'patient_level_results.csv'), index=False)
    
    df_organ_stats = pd.DataFrame(organ_detection_data)
    df_organ_stats.to_csv(os.path.join(args.output_dir, 'organ_detection_stats.csv'), index=False)
    
    print(f"\n Saved detailed results to {args.output_dir}/")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot 1: Detection rates by organ
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Detection rate
    ax = axes[0, 0]
    organs = [d['organ'] for d in organ_detection_data]
    detection_rates = [d['detection_rate'] for d in organ_detection_data]
    bars = ax.barh(organs, detection_rates, color='steelblue')
    ax.set_xlabel('Detection Rate (%)')
    ax.set_title('Organ Detection Rate (Not Touching Boundaries)')
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax.legend()
    ax.set_xlim(0, 100)
    
    # Add value labels
    for i, (organ, rate) in enumerate(zip(organs, detection_rates)):
        ax.text(rate + 1, i, f'{rate:.1f}%', va='center', fontsize=8)
    
    # Detected vs Boundary counts
    ax = axes[0, 1]
    detected = [d['detected'] for d in organ_detection_data]
    boundary = [d['boundary'] for d in organ_detection_data]
    x = np.arange(len(organs))
    width = 0.35
    ax.barh(x - width/2, detected, width, label='Detected (Complete)', color='green', alpha=0.7)
    ax.barh(x + width/2, boundary, width, label='Boundary (Discarded)', color='red', alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(organs)
    ax.set_xlabel('Count')
    ax.set_title('Detected vs Boundary Organs')
    ax.legend()
    
    # Ground truth comparison
    ax = axes[1, 0]
    gt_present = [d['gt_present'] for d in organ_detection_data]
    gt_abnormal = [d['gt_abnormal'] for d in organ_detection_data]
    x = np.arange(len(organs))
    ax.barh(x - width/2, gt_present, width, label='GT Present', color='blue', alpha=0.7)
    ax.barh(x + width/2, gt_abnormal, width, label='GT Abnormal', color='orange', alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(organs)
    ax.set_xlabel('Count')
    ax.set_title('Ground Truth Organ Mentions')
    ax.legend()
    
    # Lost abnormal organs
    ax = axes[1, 1]
    lost_abnormal = [d['lost_abnormal'] for d in organ_detection_data]
    loss_rates = [100 * d['lost_abnormal'] / d['gt_abnormal'] if d['gt_abnormal'] > 0 else 0 
                  for d in organ_detection_data]
    bars = ax.barh(organs, loss_rates, color='darkred', alpha=0.7)
    ax.set_xlabel('Loss Rate (%)')
    ax.set_title('Abnormal Organ Loss Rate (Due to Boundaries)')
    ax.axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
    ax.legend()
    
    # Add value labels
    for i, (organ, rate, count) in enumerate(zip(organs, loss_rates, lost_abnormal)):
        if count > 0:
            ax.text(rate + 1, i, f'{rate:.1f}% ({count})', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'organ_detection_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {args.output_dir}/organ_detection_analysis.png")
    
    # Summary statistics
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    
    # Most problematic organs (high boundary rate)
    high_boundary_organs = sorted(organ_detection_data, key=lambda x: x['boundary']/(x['detected'] + x['boundary'] + 1e-6), reverse=True)[:5]
    print("\nOrgans with highest boundary collision rate:")
    for i, organ_data in enumerate(high_boundary_organs, 1):
        total = organ_data['detected'] + organ_data['boundary']
        rate = 100 * organ_data['boundary'] / total if total > 0 else 0
        print(f"  {i}. {organ_data['organ']}: {rate:.1f}% boundary rate ({organ_data['boundary']}/{total})")
    
    # Most lost abnormal findings
    high_loss_organs = sorted(organ_detection_data, key=lambda x: x['lost_abnormal'], reverse=True)[:5]
    print("\nOrgans losing most abnormal findings:")
    for i, organ_data in enumerate(high_loss_organs, 1):
        if organ_data['lost_abnormal'] > 0:
            loss_rate = 100 * organ_data['lost_abnormal'] / organ_data['gt_abnormal']
            print(f"  {i}. {organ_data['organ']}: {organ_data['lost_abnormal']} lost ({loss_rate:.1f}% of {organ_data['gt_abnormal']} abnormal)")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    main()


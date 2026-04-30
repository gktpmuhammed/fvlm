#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import torch
from monai import transforms

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
REFERENCE_FVLM_ROOT = Path(
    os.getenv("REFERENCE_FVLM_ROOT", PROJECT_ROOT.parent / "test" / "fvlm")
)

# --- SHARED UTILITY FUNCTIONS ---

class SpacingNormalization:
    """Custom transform to normalize spacing to reference spacing"""
    def __init__(self, ref_spacing=(1.0, 1.0, 3.0), debug=False):
        self.ref_spacing = ref_spacing
        self.debug = debug
    
    def __call__(self, data):
        affine = data["image_meta_dict"]["affine"]
        spacing = (abs(affine[0, 0].item()), abs(affine[1, 1].item()), abs(affine[2, 2].item()))
        
        _, h, w, d = data["image"].shape
        scale = [spacing[i] / self.ref_spacing[i] for i in range(3)]
        target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]
        
        if target_size != [h, w, d]:
            resize_transform = transforms.Compose([
                transforms.Resized(spatial_size=target_size, keys=["image"], mode="trilinear"),
                transforms.Resized(spatial_size=target_size, keys=["label"], mode="nearest")
            ])
            result = resize_transform(data)
            return result
        else:
            return data

class Original_ROI_Crop_d:
    """Custom transform to replicate original ROI cropping"""
    def __init__(self, keys, extend_d=5, extend_hw=20):
        self.keys = keys
        self.extend_d = extend_d
        self.extend_hw = extend_hw

    def __call__(self, data):
        d = data.copy()
        image = d["image"]
        label = d["label"]

        if not isinstance(label, torch.Tensor):
            label = torch.as_tensor(label)
        
        if label.dtype not in (torch.int, torch.long, torch.int8, torch.int16):
            label = label.long()

        if torch.sum(label) > 0:
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

# --- LABEL FUNCTIONS ---

def get_full_merged_labels(label):
    """Full merge_labels function from your pipeline"""
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
    
    merged_organ_id = {
        "skull": 0, "brain": 1, "esophagus": 2, "trachea": 3,
        "lung_upper_lobe_left": 4, "lung_lower_lobe_left": 4, "lung_upper_lobe_right": 4,
        "lung_middle_lobe_right": 4, "lung_lower_lobe_right": 4,
        "heart": 5, "atrial_appendage_left": 5, "kidney_right": 6, "kidney_left": 6,
        "stomach": 7, "liver": 8, "gallbladder": 9, "pancreas": 10,
        "spleen": 11, "colon": 12, "aorta": 13,
        "rib_left_1": 14, "rib_left_2": 14, "rib_left_3": 14, "rib_left_4": 14, "rib_left_5": 14,
        "rib_left_6": 14, "rib_left_7": 14, "rib_left_8": 14, "rib_left_9": 14, "rib_left_10": 14,
        "rib_left_11": 14, "rib_left_12": 14, "rib_right_1": 14, "rib_right_2": 14, "rib_right_3": 14,
        "rib_right_4": 14, "rib_right_5": 14, "rib_right_6": 14, "rib_right_7": 14, "rib_right_8": 14,
        "rib_right_9": 14, "rib_right_10": 14, "rib_right_11": 14, "rib_right_12": 14,
        "humerus_left": 15, "humerus_right": 15, "scapula_left": 16, "scapula_right": 16,
        "clavicula_left": 17, "clavicula_right": 17, "femur_left": 18, "femur_right": 18,
        "hip_left": 19, "hip_right": 19, "sacrum": 20, "vertebrae_S1": 20,
        "gluteus_maximus_left": 21, "gluteus_maximus_right": 21, "gluteus_medius_left": 21,
        "gluteus_medius_right": 21, "gluteus_minimus_left": 21, "gluteus_minimus_right": 21,
        "iliopsoas_left": 22, "iliopsoas_right": 22, "autochthon_left": 23, "autochthon_right": 23
    }
    
    fused_mask = np.zeros_like(label)
    for original_id, organ_name in class_map.items():
        if organ_name in merged_organ_id:
            merged_id = merged_organ_id[organ_name]
            fused_mask[label == original_id] = merged_id + 1
    return fused_mask

def get_original_limited_labels(label):
    """Keep only the 4 labels that were in the original processed data"""
    # Based on our previous analysis, the original had only 4 unique labels
    # We'll simulate this by keeping only a subset of organs
    limited_mask = np.zeros_like(label)
    # Keep spleen (1), kidney_left (3), liver (5), stomach (6)
    for organ_id in [1, 3, 5, 6]:
        limited_mask[label == organ_id] = organ_id
    return limited_mask

# --- COMPARISON FUNCTIONS ---

def print_step_comparison(step_name, orig_data, user_data):
    """Print detailed comparison at each step"""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    
    orig_img = orig_data["image"]
    orig_lbl = orig_data["label"]
    user_img = user_data["image"] 
    user_lbl = user_data["label"]
    
    print(f"IMAGE SHAPES:")
    print(f"  Original Pipeline: {orig_img.shape}")
    print(f"  Your Pipeline:     {user_img.shape}")
    shape_match = orig_img.shape == user_img.shape
    print(f"  Match: {'YES' if shape_match else 'NO'}")
    
    print(f"\nLABEL INFO:")
    orig_unique = torch.unique(orig_lbl).tolist()
    user_unique = torch.unique(user_lbl).tolist()
    print(f"  Original unique labels: {orig_unique} (count: {len(orig_unique)})")
    print(f"  Your unique labels:     {user_unique} (count: {len(user_unique)})")
    
    print(f"\nIMAGE STATS:")
    print(f"  Original - min: {orig_img.min():.4f}, max: {orig_img.max():.4f}, mean: {orig_img.mean():.4f}")
    print(f"  Yours    - min: {user_img.min():.4f}, max: {user_img.max():.4f}, mean: {user_img.mean():.4f}")
    
    if shape_match:
        diff = torch.abs(orig_img - user_img)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"  Max pixel difference: {max_diff:.6f}")
        print(f"  Mean pixel difference: {mean_diff:.6f}")
        
        if max_diff < 1e-6:
            print("  Images are IDENTICAL!")
        elif max_diff < 0.001:
            print("  Small differences (likely numerical precision)")
        else:
            print("  Significant differences detected!")
    else:
        print("  Cannot compare pixel values - shapes differ!")
    
    if not shape_match:
        print(f"\nDIVERGENCE DETECTED AT THIS STEP!")

def get_limited_merged_labels(label):
    """Limited merge_labels function matching the preprocessing pipeline"""
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
    
    # This matches the limited merging from resize.py
    merged_organ_id = {
        "lung_upper_lobe_left": 0,
        "lung_lower_lobe_left": 0,
        "lung_upper_lobe_right": 0,
        "lung_middle_lobe_right": 0,
        "lung_lower_lobe_right": 0,
        "heart": 1,
        "atrial_appendage_left": 1,
        "esophagus": 2,
        "aorta": 3,
    }
    
    fused_mask = np.zeros_like(label)
    for original_id, organ_name in class_map.items():
        if organ_name in merged_organ_id:
            merged_id = merged_organ_id[organ_name]
            fused_mask[label == original_id] = merged_id + 1
    return fused_mask

def main():
    # Paths for different stages
    reference_base_path = str(REFERENCE_FVLM_ROOT / "data")
    user_base_path = str(PROJECT_ROOT / 'data')
    sample_id = "train_10/train_10_a/train_10_a_1.nii.gz"
    
    # User input paths (semi-preprocessed files)
    user_image_path = f"{user_base_path}/images/train/{sample_id}"
    user_mask_path = f"{user_base_path}/masks/train/{sample_id}"
    
    # Reference pre-processed paths (for comparison)
    merged_mask_path = f"{reference_base_path}/merged_train_masks/{sample_id}"
    resized_image_path = f"{reference_base_path}/resized_train_images/{sample_id}"
    resized_mask_path = f"{reference_base_path}/resized_train_masks/{sample_id}"
    processed_image_path = f"{reference_base_path}/processed_train_images/{sample_id}"
    processed_mask_path = f"{reference_base_path}/processed_train_masks/{sample_id}"
    
    print("PREPROCESSING PIPELINE COMPARISON")
    print("="*60)
    print("Comparing user pipeline (starting from semi-preprocessed data) against reference data")
    print(f"User input: Semi-preprocessed files from {PROJECT_ROOT / 'data'}")
    print(f"Reference: Pre-processed results from {REFERENCE_FVLM_ROOT / 'data'}")
    
    try:
        # Initialize user stream with semi-preprocessed data
        print(f"\nLoading semi-preprocessed data for user stream...")
        print(f"  Image: {user_image_path}")
        print(f"  Mask:  {user_mask_path}")
        loader = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True)
        ])
        
        user_data = loader({"image": user_image_path, "label": user_mask_path})
        user_stream = {
            "image": user_data["image"].clone(),
            "label": user_data["label"].clone(), 
            "image_meta_dict": user_data["image_meta_dict"],
            "label_meta_dict": user_data["label_meta_dict"]
        }
        
        # Load reference data at each stage
        print("Loading reference data at each preprocessing stage...")
        
        # Reference: After label merging (we'll load the raw image for this comparison)
        raw_image_path = f"{reference_base_path}/train_fixed/{sample_id}"
        ref_merged = loader({"image": raw_image_path, "label": merged_mask_path})
        
        # Reference: After resizing
        ref_resized = loader({"image": resized_image_path, "label": resized_mask_path})
        
        # Reference: Final processed
        ref_processed = loader({"image": processed_image_path, "label": processed_mask_path})
        
        print("All reference data loaded successfully!")
        
        # COMPARISON 1: After label merging
        print("\n" + "="*60)
        print("COMPARISON 1: After Label Merging")
        print("="*60)
        
        # Apply label merging to user stream
        user_stream["label"] = torch.from_numpy(get_limited_merged_labels(user_stream["label"].numpy())).long()
        
        print_step_comparison("After Label Merging", ref_merged, user_stream)
        
        # COMPARISON 2: After spacing normalization (resizing)
        print("\n" + "="*60)
        print("COMPARISON 2: After Spacing Normalization")
        print("="*60)
        
        # Apply spacing normalization to user stream
        spacing_transform = SpacingNormalization(ref_spacing=(1.0, 1.0, 3.0))
        user_stream = spacing_transform(user_stream)
        
        print_step_comparison("After Spacing Normalization", ref_resized, user_stream)
        
        # COMPARISON 3: After full preprocessing pipeline
        print("\n" + "="*60)
        print("COMPARISON 3: After Full Preprocessing Pipeline")
        print("="*60)
        
        # Apply remaining transforms to user stream
        remaining_transforms = transforms.Compose([
            transforms.Transposed(keys=["image", "label"], indices=(0, 3, 2, 1)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1150, a_max=350, b_min=0.0, b_max=1.0, clip=True
            )
        ])
        user_stream = remaining_transforms(user_stream)
        
        # Apply ROI cropping
        crop_transform = Original_ROI_Crop_d(keys=["image", "label"])
        user_stream = crop_transform(user_stream)
        
        # Apply spatial padding
        pad_transform = transforms.SpatialPadd(
            keys=["image", "label"],
            spatial_size=(112, 256, 352),
            mode="constant",
            constant_values=0
        )
        user_stream = pad_transform(user_stream)
        
        print_step_comparison("After Full Pipeline (FINAL)", ref_processed, user_stream)
        
        # Final summary
        print(f"\n\n{'='*60}")
        print("FINAL ANALYSIS")
        print(f"{'='*60}")
        
        final_ref = ref_processed
        final_user = user_stream
        
        if final_ref["image"].shape == final_user["image"].shape:
            print("SUCCESS: User pipeline produces the same final shape as reference!")
            diff = torch.abs(final_ref["image"] - final_user["image"]).max().item()
            if diff < 1e-6:
                print("SUCCESS: Images are pixel-perfect identical to reference!")
            else:
                print(f"Images have small differences from reference (max diff: {diff:.6f})")
                
            # Check label similarity
            label_diff = torch.abs(final_ref["label"].float() - final_user["label"].float()).max().item()
            if label_diff < 1e-6:
                print("SUCCESS: Labels are identical to reference!")
            else:
                print(f"Labels have differences from reference (max diff: {label_diff:.6f})")
        else:
            print("FAILURE: User pipeline produces different shapes from reference.")
            print(f"Reference shape: {final_ref['image'].shape}")
            print(f"User shape: {final_user['image'].shape}")
        
    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

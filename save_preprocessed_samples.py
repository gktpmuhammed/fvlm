#!/usr/bin/env python3

import os
import glob
import numpy as np
import torch
from monai import transforms
from pathlib import Path

def merge_labels(label):
    """Same label merging function from FVLMImageTrainProcessor"""
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
        # Face/Head
        "skull": 0, "brain": 1,
        
        # Thoracic
        "esophagus": 2, "trachea": 3,
        "lung_upper_lobe_left": 4, "lung_lower_lobe_left": 4, "lung_upper_lobe_right": 4,
        "lung_middle_lobe_right": 4, "lung_lower_lobe_right": 4,
        "heart": 5, "atrial_appendage_left": 5,
        
        # Abdominal (removed: adrenal gland, small bowel, urinary bladder)
        "kidney_right": 6, "kidney_left": 6,
        "stomach": 7, "liver": 8, "gallbladder": 9, "pancreas": 10,
        "spleen": 11, "colon": 12,
        
        # Vascular (removed: inferior vena cava, portal vein, pulmonary artery, iliac vessels)
        "aorta": 13,
        
        # Ribs (grouped)
        "rib_left_1": 14, "rib_left_2": 14, "rib_left_3": 14, "rib_left_4": 14, "rib_left_5": 14,
        "rib_left_6": 14, "rib_left_7": 14, "rib_left_8": 14, "rib_left_9": 14, "rib_left_10": 14,
        "rib_left_11": 14, "rib_left_12": 14, "rib_right_1": 14, "rib_right_2": 14, "rib_right_3": 14,
        "rib_right_4": 14, "rib_right_5": 14, "rib_right_6": 14, "rib_right_7": 14, "rib_right_8": 14,
        "rib_right_9": 14, "rib_right_10": 14, "rib_right_11": 14, "rib_right_12": 14,
        
        # Bones  
        "humerus_left": 15, "humerus_right": 15,
        "scapula_left": 16, "scapula_right": 16,
        "clavicula_left": 17, "clavicula_right": 17,
        "femur_left": 18, "femur_right": 18,
        "hip_left": 19, "hip_right": 19,
        "sacrum": 20, "vertebrae_S1": 20,
        
        # Muscles
        "gluteus_maximus_left": 21, "gluteus_maximus_right": 21, "gluteus_medius_left": 21,
        "gluteus_medius_right": 21, "gluteus_minimus_left": 21, "gluteus_minimus_right": 21,
        "iliopsoas_left": 22, "iliopsoas_right": 22,
        "autochthon_left": 23, "autochthon_right": 23
    }
    
    fused_mask = np.zeros_like(label)
    for original_id, organ_name in class_map.items():
        if organ_name in merged_organ_id:
            merged_id = merged_organ_id[organ_name]
            fused_mask[label == original_id] = merged_id + 1
    return fused_mask


def create_preprocessing_pipeline():
    """Create preprocessing pipeline similar to FVLMImageTrainProcessor but without random augmentations"""
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
        transforms.Lambdad(keys=["label"], func=merge_labels),
        # Convert HU values to [0, 1] range
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-1150, a_max=350,
            b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.CropForegroundd(keys=["image", "label"], source_key="label"),
        transforms.SpatialPadd(
            keys=["image", "label"],
            spatial_size=(112, 256, 352),
            method='end',
            mode="constant",
            constant_values=0
        ),
        transforms.CenterSpatialCropd(
            keys=["image", "label"],
            roi_size=(112, 256, 352)
        ),
        # Skip random flips for consistent output
        transforms.ToTensord(keys=["image", "label"])
    ])


def convert_to_regular_tensor(data):
    """Convert MetaTensor to regular PyTorch tensor"""
    for key in ["image", "label"]:
        if key in data:
            item = data[key]
            try:
                if hasattr(item, 'array'):
                    # MetaTensor case - create completely new tensor
                    array_data = np.array(item.array, copy=True)
                    data[key] = torch.from_numpy(array_data).clone()
                elif hasattr(item, 'data'):
                    # Other MONAI tensor types
                    array_data = np.array(item.data, copy=True) 
                    data[key] = torch.from_numpy(array_data).clone()
                elif hasattr(item, 'detach'):
                    # Already a tensor but might be MetaTensor
                    data[key] = torch.tensor(item.detach().cpu().numpy(), dtype=item.dtype)
                else:
                    # Convert any other format
                    array_data = np.array(item, copy=True)
                    data[key] = torch.from_numpy(array_data).clone()
            except Exception as e:
                # Fallback: force numpy conversion
                array_data = np.array(item, copy=True)
                data[key] = torch.from_numpy(array_data).contiguous()
    return data


def main():
    # Set up paths
    data_dir = "/home/muhammedg/fvlm/data"
    image_dir = os.path.join(data_dir, "images", "train")
    mask_dir = os.path.join(data_dir, "masks", "train")
    output_dir = os.path.join(data_dir, "preprocessed_samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    # Find sample files
    image_files = sorted(glob.glob(os.path.join(image_dir, "**", "*.nii.gz"), recursive=True))
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    # Take first 5 samples for demonstration
    sample_files = image_files[:5]
    
    # Create preprocessing pipeline
    preprocess = create_preprocessing_pipeline()
    
    print(f"Processing {len(sample_files)} samples...")
    
    for i, image_path in enumerate(sample_files):
        try:
            # Find corresponding mask
            rel_path = os.path.relpath(image_path, image_dir)
            mask_path = os.path.join(mask_dir, rel_path)
            
            if not os.path.exists(mask_path):
                print(f"Mask not found for {image_path}, skipping...")
                continue
            
            print(f"Processing sample {i+1}: {os.path.basename(image_path)}")
            
            # Load and preprocess
            data_dict = {"image": image_path, "label": mask_path}
            processed_data = preprocess(data_dict)
            processed_data = convert_to_regular_tensor(processed_data)
            
            # Save processed results as NIfTI files
            sample_name = f"sample_{i+1:02d}_{Path(image_path).stem}"
            
            # Save as NIfTI files (.nii.gz)
            image_output_path = os.path.join(output_dir, "images", f"{sample_name}_image.nii.gz")
            mask_output_path = os.path.join(output_dir, "masks", f"{sample_name}_mask.nii.gz")
            
            # Use MONAI's SaveImage to save as NIfTI
            image_saver = transforms.SaveImage(
                output_dir=os.path.join(output_dir, "images"),
                output_postfix="image",
                output_ext=".nii.gz",
                separate_folder=False,
                resample=False
            )
            
            mask_saver = transforms.SaveImage(
                output_dir=os.path.join(output_dir, "masks"), 
                output_postfix="mask",
                output_ext=".nii.gz",
                separate_folder=False,
                resample=False
            )
            
            # Save with proper sample naming
            temp_data = {
                "image": processed_data["image"],
                "label": processed_data["label"],
                "image_meta_dict": {"filename_or_obj": f"{sample_name}_image"},
                "label_meta_dict": {"filename_or_obj": f"{sample_name}_mask"}
            }
            
            image_saver(processed_data["image"], {"filename_or_obj": f"{sample_name}_image"})
            mask_saver(processed_data["label"], {"filename_or_obj": f"{sample_name}_mask"})
            
            # Print stats
            image_shape = processed_data["image"].shape
            mask_shape = processed_data["label"].shape
            unique_labels = torch.unique(processed_data["label"])
            
            print(f"  Image shape: {image_shape}")
            print(f"  Mask shape: {mask_shape}")
            print(f"  Unique mask values: {unique_labels.tolist()}")
            print(f"  Image range: [{processed_data['image'].min():.3f}, {processed_data['image'].max():.3f}]")
            print(f"  Saved to: {image_output_path}")
            print(f"  Saved to: {mask_output_path}")
            print()
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    print("Done! You can now load the preprocessed samples with:")
    print(f"  from monai import transforms")
    print(f"  loader = transforms.LoadImage(image_only=True)")
    print(f"  image = loader('{output_dir}/images/sample_01_<name>_image.nii.gz')")
    print(f"  mask = loader('{output_dir}/masks/sample_01_<name>_mask.nii.gz')")
    print()
    print("Or view them in any medical image viewer like:")
    print("  - ITK-SNAP, 3D Slicer, FSLeyes")
    print("  - Or load with nibabel: nibabel.load('sample_01_<name>_image.nii.gz')")
    print()
    print("For visualization in Python:")
    print("  import matplotlib.pyplot as plt")
    print("  plt.imshow(image[slice_idx, :, :], cmap='gray')  # Axial slice")
    print("  plt.imshow(mask[slice_idx, :, :], cmap='tab20')   # Mask overlay")


if __name__ == "__main__":
    main()
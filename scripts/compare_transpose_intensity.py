#!/usr/bin/env python3

import os
import numpy as np
import torch
from monai import transforms
from pathlib import Path
import nibabel as nib

class SpacingNormalization:
    """Custom transform to normalize spacing to reference spacing"""
    def __init__(self, ref_spacing=(1.0, 1.0, 3.0), debug=False):
        self.ref_spacing = ref_spacing
        self.debug = debug
    
    def __call__(self, data):
        # Get original spacing from affine matrix
        affine = data["image_meta_dict"]["affine"]
        spacing = (abs(affine[0, 0].item()), abs(affine[1, 1].item()), abs(affine[2, 2].item()))
        
        # Calculate scale and target size
        _, h, w, d = data["image"].shape
        scale = [spacing[i] / self.ref_spacing[i] for i in range(3)]
        target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]
        
        if self.debug:
            print(f"  [SpacingNorm] Original spacing: {spacing}")
            print(f"  [SpacingNorm] Target spacing: {self.ref_spacing}")
            print(f"  [SpacingNorm] Original size: ({h}, {w}, {d})")
            print(f"  [SpacingNorm] Scale factors: {scale}")
            print(f"  [SpacingNorm] Target size: {target_size}")
        
        # Apply resizing if needed
        if target_size != [h, w, d]:
            if self.debug:
                print(f"  [SpacingNorm] Applying resize from {[h, w, d]} to {target_size}")
            
            resize_transform = transforms.Compose([
                transforms.Resized(spatial_size=target_size, keys=["image"], mode="trilinear"),
                transforms.Resized(spatial_size=target_size, keys=["label"], mode="nearest") if "label" in data else transforms.Identity()
            ])
            result = resize_transform(data)
            
            # Update spacing metadata
            for key in ["image", "label"]:
                if key in result and hasattr(result[key], 'meta') and result[key].meta:
                    if 'pixdim' in result[key].meta:
                        result[key].meta['pixdim'][1:4] = self.ref_spacing
                    if 'affine' in result[key].meta:
                        for i in range(3):
                            result[key].meta['affine'][i, i] = self.ref_spacing[i] * (1 if result[key].meta['affine'][i, i] > 0 else -1)
            
            return result
        else:
            if self.debug:
                print(f"  [SpacingNorm] No resize needed - spacing already matches target")
            return data

def apply_original_pipeline_steps(resized_image_path):
    """Apply transpose + intensity scaling steps from original pipeline"""
    print(f"\n=== Original Pipeline: Transpose + Intensity Scaling ===")
    
    # This matches the loader from the original preprocess.py (lines 127-135)
    pipeline = transforms.Compose([
        transforms.LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
        transforms.Transposed(keys=["image"], indices=(0, 3, 2, 1)),  # Transpose
        transforms.ScaleIntensityRanged(  # Intensity scaling
            keys=["image"], a_min=-1150, a_max=350,
            b_min=0.0, b_max=1.0, clip=True
        )
    ])
    
    # Apply pipeline
    data_dict = {"image": resized_image_path}
    result = pipeline(data_dict)
    
    # Extract metrics
    image_data = result["image"]
    min_val = float(image_data.min())
    max_val = float(image_data.max())
    mean_val = float(image_data.mean())
    std_val = float(image_data.std())
    
    print(f"  Shape after transpose: {image_data.shape}")
    print(f"  Intensity range after scaling: [{min_val:.4f}, {max_val:.4f}]")
    print(f"  Mean±Std after scaling: {mean_val:.4f}±{std_val:.4f}")
    
    # Check some specific values to verify transpose worked correctly
    print(f"  Sample values: {image_data[0, :3, :3, 0].flatten()[:6].tolist()}")
    
    return {
        'data': result,
        'shape': image_data.shape,
        'min_val': min_val,
        'max_val': max_val,
        'mean_val': mean_val,
        'std_val': std_val,
        'sample_values': image_data[0, :3, :3, 0].flatten()[:10].tolist()
    }

def apply_user_pipeline_steps(raw_image_path, raw_mask_path=None):
    """Apply SpacingNormalization + transpose + intensity scaling from user's pipeline"""
    print(f"\n=== User Pipeline: SpacingNorm + Transpose + Intensity Scaling ===")
    
    # This matches the relevant steps from save_preprocessed_samples.py (lines 164-173)
    # Skip label merging since we're comparing images only
    pipeline = transforms.Compose([
        transforms.LoadImaged(keys=["image"] + (["label"] if raw_mask_path else []), 
                             image_only=False, ensure_channel_first=True),
        SpacingNormalization(ref_spacing=(1.0, 1.0, 3.0), debug=False),  # Step 1
        transforms.Transposed(keys=["image"] + (["label"] if raw_mask_path else []), indices=(0, 3, 2, 1)),  # Step 2
        # Skip label merging (step 3) for image comparison
        transforms.ScaleIntensityRanged(  # Step 4 (equivalent to step 3 in comparison)
            keys=["image"], a_min=-1150, a_max=350,
            b_min=0.0, b_max=1.0, clip=True
        ),
    ])
    
    # Create data dict
    data_dict = {"image": raw_image_path}
    if raw_mask_path:
        data_dict["label"] = raw_mask_path
    
    # Apply pipeline
    result = pipeline(data_dict)
    
    # Extract metrics
    image_data = result["image"]
    min_val = float(image_data.min())
    max_val = float(image_data.max())
    mean_val = float(image_data.mean())
    std_val = float(image_data.std())
    
    print(f"  Shape after transpose: {image_data.shape}")
    print(f"  Intensity range after scaling: [{min_val:.4f}, {max_val:.4f}]")
    print(f"  Mean±Std after scaling: {mean_val:.4f}±{std_val:.4f}")
    
    # Check same specific values to verify transpose worked correctly
    print(f"  Sample values: {image_data[0, :3, :3, 0].flatten()[:6].tolist()}")
    
    return {
        'data': result,
        'shape': image_data.shape,
        'min_val': min_val,
        'max_val': max_val,
        'mean_val': mean_val,
        'std_val': std_val,
        'sample_values': image_data[0, :3, :3, 0].flatten()[:10].tolist()
    }

def compare_results(original_result, user_result):
    """Compare the two preprocessing approaches after transpose + intensity scaling"""
    print(f"\n{'='*60}")
    print("COMPARISON: TRANSPOSE + INTENSITY SCALING")
    print(f"{'='*60}")
    
    print(f"\nSHAPES:")
    print(f"  Original Pipeline:  {original_result['shape']}")
    print(f"  User Pipeline:      {user_result['shape']}")
    
    if original_result['shape'] == user_result['shape']:
        print("  Shapes match perfectly!")
    else:
        print("  Shapes differ!")
    
    print(f"\nINTENSITY VALUES (after scaling to [0,1]):")
    print(f"  Original Pipeline:  [{original_result['min_val']:.4f}, {original_result['max_val']:.4f}] (mean: {original_result['mean_val']:.4f})")
    print(f"  User Pipeline:      [{user_result['min_val']:.4f}, {user_result['max_val']:.4f}] (mean: {user_result['mean_val']:.4f})")
    
    # Check if intensity values are similar
    intensity_diff = abs(original_result['mean_val'] - user_result['mean_val'])
    range_diff = abs((original_result['max_val'] - original_result['min_val']) - (user_result['max_val'] - user_result['min_val']))
    
    if intensity_diff < 0.001 and range_diff < 0.001:
        print("  Intensity scaling is identical!")
    elif intensity_diff < 0.01 and range_diff < 0.01:
        print("  Very small intensity differences (likely numerical precision)")
    else:
        print(f"  Significant intensity differences (mean diff: {intensity_diff:.4f}, range diff: {range_diff:.4f})")
    
    print(f"\nTRANSPOSE VERIFICATION:")
    print(f"  Original sample values: {original_result['sample_values']}")
    print(f"  User sample values:     {user_result['sample_values']}")
    
    # Check if transpose worked the same way
    if original_result['sample_values'] == user_result['sample_values']:
        print("  Transpose applied identically!")
    else:
        print("  Transpose results differ!")
        # Check if it's just floating point precision
        orig_vals = np.array(original_result['sample_values'])
        user_vals = np.array(user_result['sample_values'])
        max_diff = np.max(np.abs(orig_vals - user_vals))
        print(f"      Max difference in sample values: {max_diff:.6f}")
        if max_diff < 1e-5:
            print("      Differences are negligible (floating point precision)")
    
    # Pixel-wise comparison if possible
    if original_result['shape'] == user_result['shape']:
        print(f"\nPIXEL-WISE COMPARISON:")
        
        # Extract data arrays
        orig_data = original_result['data']['image'].numpy()
        user_data = user_result['data']['image'].numpy()
        
        # Compute differences
        diff = np.abs(orig_data - user_data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        
        # Check percentage of identical pixels
        identical_pixels = np.sum(diff < 1e-6)
        total_pixels = diff.size
        identical_percentage = (identical_pixels / total_pixels) * 100
        
        print(f"  Identical pixels: {identical_pixels}/{total_pixels} ({identical_percentage:.2f}%)")
        
        if max_diff < 1e-5:
            print("  Images are nearly identical (within floating point precision)!")
        elif max_diff < 0.001:
            print("  Very small differences detected")
        else:
            print("  Significant differences found!")
            
        # Show where the biggest differences are
        if max_diff > 1e-5:
            max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"  Largest difference at index {max_diff_idx}:")
            print(f"    Original: {orig_data[max_diff_idx]:.6f}")
            print(f"    User:     {user_data[max_diff_idx]:.6f}")

def main():
    # Define paths
    original_resized_path = "/home/muhammedg/test/fvlm/data/resized_train_images/train_1/train_1_a/train_1_a_1.nii.gz"
    raw_image_path = "/home/muhammedg/test/fvlm/data/train_fixed/train_1/train_1_a/train_1_a_1.nii.gz"
    raw_mask_path = "/home/muhammedg/test/fvlm/data/train_mask/train_1/train_1_a/train_1_a_1.nii.gz"
    
    print("Comparing Transpose + Intensity Scaling Steps")
    print("="*60)
    
    # Check if files exist
    for path, name in [(original_resized_path, "Original resized"), 
                      (raw_image_path, "Raw image"), 
                      (raw_mask_path, "Raw mask")]:
        if not os.path.exists(path):
            print(f"{name} not found: {path}")
            return
        else:
            print(f"{name} found")
    
    try:
        # Apply original pipeline steps (transpose + intensity scaling to resized image)
        original_result = apply_original_pipeline_steps(original_resized_path)
        
        # Apply user pipeline steps (spacing + transpose + intensity scaling to raw image)
        user_result = apply_user_pipeline_steps(raw_image_path, raw_mask_path)
        
        # Compare results
        compare_results(original_result, user_result)
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

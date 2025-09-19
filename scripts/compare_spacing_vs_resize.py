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
            
            if self.debug:
                # Check final spacing after resize and metadata update
                if hasattr(result["image"], 'meta') and result["image"].meta and 'pixdim' in result["image"].meta:
                    final_pixdim = result["image"].meta['pixdim']
                    final_spacing = (final_pixdim[1], final_pixdim[2], final_pixdim[3])
                    print(f"  [SpacingNorm] Final spacing after resize: {final_spacing}")
                print(f"  [SpacingNorm] Final shape after resize: {result['image'].shape}")
            
            return result
        else:
            if self.debug:
                print(f"  [SpacingNorm] No resize needed - spacing already matches target")
            return data

def load_and_analyze_image(filepath, label=""):
    """Load image and extract key metrics"""
    print(f"\n=== {label} ===")
    print(f"Loading: {filepath}")
    
    # Load with nibabel for metadata
    nii_img = nib.load(filepath)
    spacing = nii_img.header.get_zooms()[:3]
    shape = nii_img.get_fdata().shape
    
    # Load with MONAI
    loader = transforms.LoadImage(image_only=False, ensure_channel_first=True)
    data = loader(filepath)
    
    # Get intensity stats
    image_data = data[0] if isinstance(data, tuple) else data
    min_val = float(image_data.min())
    max_val = float(image_data.max())
    mean_val = float(image_data.mean())
    std_val = float(image_data.std())
    
    print(f"  Shape: {shape}")
    print(f"  Spacing: {spacing}")
    print(f"  Intensity range: [{min_val:.2f}, {max_val:.2f}]")
    print(f"  Mean±Std: {mean_val:.2f}±{std_val:.2f}")
    
    return {
        'data': data,
        'shape': shape,
        'spacing': spacing,
        'min_val': min_val,
        'max_val': max_val,
        'mean_val': mean_val,
        'std_val': std_val,
        'filepath': filepath
    }

def apply_spacing_normalization(raw_image_path, raw_mask_path=None, target_spacing=(1.0, 1.0, 3.0)):
    """Apply only spacing normalization from your pipeline"""
    print(f"\n=== Applying SpacingNormalization ===")
    
    # Create minimal pipeline with just spacing normalization
    pipeline = transforms.Compose([
        transforms.LoadImaged(keys=["image"] + (["label"] if raw_mask_path else []), 
                             image_only=False, ensure_channel_first=True),
        SpacingNormalization(ref_spacing=target_spacing, debug=True)
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
    
    print(f"  Final shape: {image_data.shape}")
    print(f"  Intensity range: [{min_val:.2f}, {max_val:.2f}]")
    print(f"  Mean±Std: {mean_val:.2f}±{std_val:.2f}")
    
    return {
        'data': result,
        'shape': image_data.shape,
        'spacing': target_spacing,  # Should be target spacing after normalization
        'min_val': min_val,
        'max_val': max_val,
        'mean_val': mean_val,
        'std_val': std_val
    }

def compare_results(original_result, spacing_result):
    """Compare the two preprocessing approaches"""
    print(f"\n{'='*50}")
    print("COMPARISON RESULTS")
    print(f"{'='*50}")
    
    print(f"\n📏 SHAPES:")
    print(f"  Original Resized:     {original_result['shape']}")
    print(f"  SpacingNormalization: {spacing_result['shape']}")
    
    if original_result['shape'] == spacing_result['shape']:
        print("  ✅ Shapes match!")
    else:
        print("  ❌ Shapes differ!")
    
    print(f"\n📐 SPACING:")
    print(f"  Original Resized:     {original_result['spacing']}")
    print(f"  SpacingNormalization: {spacing_result['spacing']}")
    
    print(f"\n🎨 INTENSITY VALUES:")
    print(f"  Original Resized:     [{original_result['min_val']:.2f}, {original_result['max_val']:.2f}] (mean: {original_result['mean_val']:.2f})")
    print(f"  SpacingNormalization: [{spacing_result['min_val']:.2f}, {spacing_result['max_val']:.2f}] (mean: {spacing_result['mean_val']:.2f})")
    
    # Check if intensity values are similar
    intensity_diff = abs(original_result['mean_val'] - spacing_result['mean_val'])
    if intensity_diff < 1.0:
        print("  ✅ Intensity values are similar!")
    else:
        print(f"  ⚠️  Intensity values differ significantly (diff: {intensity_diff:.2f})")
    
    # Pixel-wise comparison if possible
    if original_result['shape'] == spacing_result['shape']:
        print(f"\n🔍 PIXEL-WISE COMPARISON:")
        
        # Extract data arrays
        orig_data = original_result['data']
        if isinstance(orig_data, tuple):
            orig_array = orig_data[0].numpy()
        else:
            orig_array = orig_data.numpy()
            
        spacing_data = spacing_result['data']['image'].numpy()
        
        # Compute differences
        diff = np.abs(orig_array - spacing_data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"  Max absolute difference: {max_diff:.4f}")
        print(f"  Mean absolute difference: {mean_diff:.4f}")
        
        # Check percentage of identical pixels
        identical_pixels = np.sum(diff < 1e-6)
        total_pixels = diff.size
        identical_percentage = (identical_pixels / total_pixels) * 100
        
        print(f"  Identical pixels: {identical_pixels}/{total_pixels} ({identical_percentage:.2f}%)")
        
        if max_diff < 1e-3:
            print("  ✅ Images are nearly identical!")
        elif max_diff < 0.01:
            print("  ⚠️  Small differences detected")
        else:
            print("  ❌ Significant differences found!")

def main():
    # Define paths
    original_resized_path = "/home/muhammedg/test/fvlm/data/resized_train_images/train_1/train_1_a/train_1_a_1.nii.gz"
    raw_image_path = "/home/muhammedg/test/fvlm/data/train_fixed/train_1/train_1_a/train_1_a_1.nii.gz"
    raw_mask_path = "/home/muhammedg/test/fvlm/data/train_mask/train_1/train_1_a/train_1_a_1.nii.gz"
    
    print("Comparing SpacingNormalization vs Original Resizing")
    print("="*60)
    
    # Check if files exist
    for path, name in [(original_resized_path, "Original resized"), 
                      (raw_image_path, "Raw image"), 
                      (raw_mask_path, "Raw mask")]:
        if not os.path.exists(path):
            print(f"❌ {name} not found: {path}")
            return
        else:
            print(f"✅ {name} found: {path}")
    
    try:
        # Load and analyze original resized image
        original_result = load_and_analyze_image(original_resized_path, "Original Resized Image")
        
        # Apply spacing normalization to raw image
        spacing_result = apply_spacing_normalization(raw_image_path, raw_mask_path)
        
        # Compare results
        compare_results(original_result, spacing_result)
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

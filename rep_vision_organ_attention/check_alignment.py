import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# Import your actual classes to ensure we test the EXACT pipeline
from train import build_transforms, OnePassOrganDataset

def visualize_alignment(csv_file, json_file, output_dir="alignment_checks"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Checking Alignment ---")
    print(f"CSV: {csv_file}")
    
    # 1. Setup minimal requirements to load the dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Build the exact transforms used in training
    transforms = build_transforms()
    
    # 3. Load the Dataset
    ds = OnePassOrganDataset(
        csv_file=csv_file,
        json_file=json_file,
        tokenizer=tokenizer,
        transform=transforms,
        subset_size=10,
        split='validation'
    )

    print(f"Generating visualizations for {len(ds)} samples...")

    for i in range(len(ds)):
        sample = ds[i]
        if sample is None:
            continue
            
        # Shapes coming in:
        # img_tensor: (1, D, H, W)
        # mask_tensor: (N_organs, 1, D, H, W)
        
        img_tensor = sample['pixel_values']
        mask_tensor = sample['organ_masks']
        
        # Collapse organs -> (1, D, H, W)
        combined_mask, _ = torch.max(mask_tensor, dim=0)
        
        # Squeeze to remove Channel dimension -> (D, H, W)
        img_np = img_tensor.squeeze().numpy()
        mask_np = combined_mask.squeeze().numpy()
        
        # Safety Check
        if img_np.ndim != 3 or mask_np.ndim != 3:
            print(f"Skipping {i}: Shape mismatch. Img: {img_np.shape}, Mask: {mask_np.shape}")
            continue

        # Pick middle slices
        mid_d = img_np.shape[0] // 2
        mid_h = img_np.shape[1] // 2
        mid_w = img_np.shape[2] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # View 1: Axial (Depth)
        axes[0].imshow(img_np[mid_d, :, :], cmap='gray')
        # Use a masked array to make 0 values transparent
        masked_overlay = np.ma.masked_where(mask_np[mid_d, :, :] == 0, mask_np[mid_d, :, :])
        axes[0].imshow(masked_overlay, cmap='jet', alpha=0.5)
        axes[0].set_title(f"Axial Slice {mid_d}")
        axes[0].axis('off')

        # View 2: Coronal (Height)
        axes[1].imshow(img_np[:, mid_h, :], cmap='gray')
        masked_overlay = np.ma.masked_where(mask_np[:, mid_h, :] == 0, mask_np[:, mid_h, :])
        axes[1].imshow(masked_overlay, cmap='jet', alpha=0.5)
        axes[1].set_title(f"Coronal Slice {mid_h}")
        axes[1].axis('off')
        
        # View 3: Sagittal (Width)
        axes[2].imshow(img_np[:, :, mid_w], cmap='gray')
        masked_overlay = np.ma.masked_where(mask_np[:, :, mid_w] == 0, mask_np[:, :, mid_w])
        axes[2].imshow(masked_overlay, cmap='jet', alpha=0.5)
        axes[2].set_title(f"Sagittal Slice {mid_w}")
        axes[2].axis('off')
        
        plt.suptitle(f"Sample {i} Alignment Check")
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"patient_{i}_alignment.png")
        plt.savefig(save_path)
        plt.close()
        
    print(f"Done! Check the folder: {output_dir}")

if __name__ == "__main__":
    CSV_FILE = '/home/muhammedg/fvlm/data/image_first_dataset.csv'
    JSON_FILE = '/home/muhammedg/fvlm/data/combined_desc_conc.json'
    visualize_alignment(CSV_FILE, JSON_FILE)
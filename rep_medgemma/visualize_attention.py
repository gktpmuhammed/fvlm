
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
from medical_vlm import MedicalVLM
from train import build_transforms, OnePassOrganDataset
from tqdm import tqdm

def visualize_attention(args):
    # 1. Load Model
    print(f"Loading model from {args.checkpoint_path}")
    # We need to construct the model first, then load weights
    # Assuming the checkpoint folder contains the bin files
    # Note: For V3 architecture, queries_per_organ is default 8.
    model = MedicalVLM(
        vision_encoder_path="dummy", # Initialize from scratch
        decoder_model_name="google/medgemma-4b-it",
        queries_per_organ=8
    )
    
    # Load custom weights
    vision_path = os.path.join(args.checkpoint_path, "vision_encoder.bin")
    projector_path = os.path.join(args.checkpoint_path, "projector.bin")
    ln_path = os.path.join(args.checkpoint_path, "projector_layernorm.bin")
    pos_path = os.path.join(args.checkpoint_path, "visual_pos_embed.bin")
    
    if os.path.exists(vision_path):
        model.vision_encoder.load_state_dict(torch.load(vision_path))
        print("Loaded Vision Encoder")
    if os.path.exists(projector_path):
        model.visual_projection.load_state_dict(torch.load(projector_path))
        print("Loaded Projector")
    if os.path.exists(ln_path):
        model.projector_layernorm.load_state_dict(torch.load(ln_path))
    if os.path.exists(pos_path):
        model.visual_pos_embed = torch.load(pos_path)
        
    model.eval()
    model.cuda()
    
    # 2. Load Data
    transform = build_transforms()
    val_ds = OnePassOrganDataset(
        csv_file='/home/muhammedg/fvlm/data_sym/image_first_dataset.csv',
        json_file='/home/muhammedg/fvlm/data_sym/combined_desc_conc.json',
        tokenizer=model.tokenizer,
        transform=transform,
        split='validation',
        subset_size=None
    )
    
    # 3. Get Sample (Patient ID or Index)
    if args.patient_id:
        # Search for patient
        idx = -1
        for i, row in enumerate(val_ds.valid_patients):
            fname = os.path.basename(row['image_path'])
            base_id = fname.replace('.nii.gz', '').replace('.nii', '')
            # Logic from train.py
            pid = base_id
            if pid not in val_ds.reports_json and len(pid.split('_')) > 1:
                pid = pid.rsplit('_', 1)[0]
            
            if pid == args.patient_id:
                idx = i
                break
        if idx == -1:
            print(f"Patient {args.patient_id} not found.")
            return
    else:
        idx = 0
        
    row = val_ds.valid_patients[idx]
    fname = os.path.basename(row['image_path'])
    pid = fname.replace('.nii.gz', '').replace('.nii', '')
    if pid not in val_ds.reports_json and len(pid.split('_')) > 1:
        pid = pid.rsplit('_', 1)[0]
    print(f"Visualizing Patient: {pid}")
    sample = val_ds[idx]
    
    # Unpack sample
    pixel_values = sample['pixel_values'].unsqueeze(0).cuda() # (1, C, D, H, W)
    organ_masks = sample['organ_masks'].unsqueeze(0).cuda()   # (1, 12, D, H, W)
    
    # 4. Forward Pass
    with torch.no_grad():
        # returns (img_feats, attn_weights) from vision_encoder
        visual_feats, attn_weights = model.vision_encoder(pixel_values, organ_masks)
    
    # attn_weights shape: (Batch, TargetSeq, SourceSeq)
    # TargetSeq = 12 Organs * 8 Queries = 96
    # SourceSeq = D*H*W of ViT patch grid = 7*16*11 = 1232
    print(f"Attention Weights Shape: {attn_weights.shape}")
    
    # 5. Process Attention Map
    # Reshape source to grid (7, 16, 11)
    # Check ViT config in MedicalVLM: image_size=(112, 256, 352), patch=(16, 16, 32)
    # 112/16=7, 256/16=16, 352/32=11
    D_vit, H_vit, W_vit = 7, 16, 11
    
    attn_map = attn_weights[0] # (96, 1232)
    attn_map = attn_map.view(96, D_vit, H_vit, W_vit) # (96, 7, 16, 11)
    
    # Average over the 8 queries per organ -> (12, 7, 16, 11)
    attn_map = attn_map.view(12, 8, D_vit, H_vit, W_vit)
    attn_map = attn_map.mean(dim=1) # (12, 7, 16, 11)
    
    # Get original CT image (normalize back to 0-1 for display)
    ct_vol = pixel_values[0, 0].cpu().numpy()
    
    print(f"CT Volume Stats: Min={ct_vol.min():.4f}, Max={ct_vol.max():.4f}, Shape={ct_vol.shape}")
    print(f"Attn Map Stats: Min={attn_map.min().item():.6f}, Max={attn_map.max().item():.6f}")

    # 6. Overlay on CT
    # We want to visualize specific organs.
    target_organs = ['lung', 'heart', 'liver', 'trachea']
    ALL_ORGANS = sorted(val_ds.target_keys) # Ensure correct order
    print(f"Target Organs: {target_organs}")
    print(f"Dataset Organs: {ALL_ORGANS}")
    
    # Get original CT image (normalize back to 0-1 for display)
    ct_vol = pixel_values[0, 0].cpu().numpy()
    # Normalize approx (-1, 1) to (0, 1)
    ct_vol = (ct_vol - ct_vol.min()) / (ct_vol.max() - ct_vol.min())
    
    # Create plot
    # Show 4 organs, each with 3 slices (Axial, Coronal, Sagittal)
    # Choose slices with max attention
    
    fig, axes = plt.subplots(len(target_organs), 4, figsize=(20, 5 * len(target_organs)))
    
    for row_idx, organ in enumerate(target_organs):
        if organ not in ALL_ORGANS: 
            print(f"Skipping {organ} (not in dataset)")
            continue
        print(f"Plotting {organ}...")
        organ_idx = ALL_ORGANS.index(organ)
        
        att = attn_map[organ_idx] # (7, 16, 11)
        
        # Upsample attention to match CT
        D, H, W = ct_vol.shape # (112, 256, 352)
        att_up = F.interpolate(
            att.unsqueeze(0).unsqueeze(0), 
            size=(D, H, W), 
            mode='trilinear', 
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # Find max attention coordinate
        max_loc = np.unravel_index(np.argmax(att_up), att_up.shape)
        z, y, x = max_loc
        
        # Plot Axial (XY)
        ax = axes[row_idx, 0]
        ax.imshow(ct_vol[z, :, :], cmap='gray')
        ax.imshow(att_up[z, :, :], cmap='jet', alpha=0.5)
        ax.set_title(f"{organ} - Axial (Z={z})")
        ax.axis('off')

        # Plot Coronal (XZ) -> resliced for display
        # ct shape (D, H, W). Coronal is usually (D, W) if viewing front?
        # Let's verify standard axes: Z=Depth, Y=Height, X=Width.
        # Coronal: Fixed Y (mid-coronal slice?) Or fixed row?
        # Usually Coronal is Front view (X-Z plane if Y is depth? No. Z is head-toe usually).
        # In medical: Z=Axial (Head-Toe), Y=Posterior-Anterior, X=Left-Right.
        # Axial: X-Y plane (Slice at Z). Correct.
        # Coronal: X-Z plane (Slice at Y). 
        # Sagittal: Y-Z plane (Slice at X).
        
        # Coronal (Slice at Y)
        ax = axes[row_idx, 1]
        ax.imshow(ct_vol[:, y, :], cmap='gray', aspect='auto') # Z, X
        ax.imshow(att_up[:, y, :], cmap='jet', alpha=0.5, aspect='auto')
        ax.set_title(f"Coronal (Y={y})")
        ax.axis('off')
        
        # Sagittal (Slice at X)
        ax = axes[row_idx, 2]
        ax.imshow(ct_vol[:, :, x], cmap='gray', aspect='auto') # Z, Y
        ax.imshow(att_up[:, :, x], cmap='jet', alpha=0.5, aspect='auto')
        ax.set_title(f"Sagittal (X={x})")
        ax.axis('off')
        
        # Show text attention?
        # Maybe show the Organ Mask (Ground Truth) to compare
        # Assuming GT mask is available in organ_masks
        # organ_masks shape (12, D, H, W) or (12, 1, D, H, W)
        gt_mask = organ_masks[0, organ_idx].cpu().numpy()
        if gt_mask.ndim == 4 and gt_mask.shape[0] == 1:
            gt_mask = gt_mask.squeeze(0)
            
        ax = axes[row_idx, 3]
        # Show GT on Axial
        ax.imshow(ct_vol[z, :, :], cmap='gray')
        ax.imshow(gt_mask[z, :, :], cmap='spring', alpha=0.5) # Different colormap for GT
        ax.set_title(f"GT Mask (Z={z})")
        ax.axis('off')

    plt.tight_layout()
    result_path = "attention_vis.png"
    plt.savefig(result_path)
    print(f"Saved visualization to {result_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--patient_id', type=str, default=None)
    args = parser.parse_args()
    visualize_attention(args)

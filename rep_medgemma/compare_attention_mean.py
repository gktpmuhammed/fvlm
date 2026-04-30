"""
Attention Map Comparison (Head-Averaged / Mean approach)
Base-8T vs Multiscale-ViT-FPN vs CNN-Stem-V3

Uses standard head-averaged attention (default nn.MultiheadAttention behavior)
and mean across query tokens per organ. This shows the "true" attention
distribution as used by the model during inference.

Generates publication-quality figures with quantitative metrics per panel:
  - attention-in-mask (%)
  - off-target attention (%)
  - attention entropy (normalized)
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from scipy import ndimage

# Set priority so they can find lavis locally without shadowing local `train.py`
sys.path.insert(1, "/home/muhammedg/fvlm")

# Architecture-specific module copies (each returns attn_weights)
import medical_vlm_base8t
import medical_vlm_multiscale
import medical_vlm_v3

# Dataset builder
import train


# ---------------------------------------------------------------------------
# Quantitative metrics
# ---------------------------------------------------------------------------

def compute_metrics(attn_map_3d, gt_mask_3d):
    """
    Compute per-organ attention metrics.
    
    Args:
        attn_map_3d: (D, H, W) attention map (upsampled to CT resolution), values >= 0
        gt_mask_3d:  (D, H, W) binary ground-truth organ mask
    
    Returns:
        dict with attention_in_mask_pct, off_target_pct, entropy_norm
    """
    attn = attn_map_3d.copy().astype(np.float64)
    mask = (gt_mask_3d > 0.5).astype(np.float64)
    
    # Normalize attention to a probability distribution
    attn_sum = attn.sum()
    if attn_sum < 1e-12:
        return {"attention_in_mask_pct": 0.0, "off_target_pct": 100.0, "entropy_norm": 1.0}
    
    p = attn / attn_sum  # probability distribution
    
    # 1. Attention-in-mask (%)
    aim = p[mask > 0.5].sum() * 100.0
    
    # 2. Off-target attention (%)
    off_target = 100.0 - aim
    
    # 3. Normalized entropy: H / log(N)
    p_flat = p.flatten()
    p_flat = p_flat[p_flat > 1e-15]  # avoid log(0)
    H = -np.sum(p_flat * np.log(p_flat))
    H_max = np.log(p.size)  # uniform distribution entropy
    entropy_norm = H / H_max if H_max > 0 else 0.0
    
    return {
        "attention_in_mask_pct": aim,
        "off_target_pct": off_target,
        "entropy_norm": entropy_norm,
    }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_dataset():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-it")
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    special_tokens_dict = {'additional_special_tokens': ['<vis>', '<end_vis>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    transform = train.build_transforms()
    val_ds = train.OnePassOrganDataset(
        csv_file='/home/muhammedg/fvlm/data_sym/image_first_dataset.csv',
        json_file='/home/muhammedg/fvlm/data_sym/combined_desc_conc.json',
        tokenizer=tokenizer,
        transform=transform,
        split='validation',
        subset_size=None
    )
    return val_ds


def get_sample_for_patient(val_ds, patient_id):
    if patient_id:
        idx = -1
        for i, row in enumerate(val_ds.valid_patients):
            fname = os.path.basename(row['image_path'])
            base_id = fname.replace('.nii.gz', '').replace('.nii', '')
            pid = base_id
            if pid not in val_ds.reports_json and len(pid.split('_')) > 1:
                pid = pid.rsplit('_', 1)[0]
            if pid == patient_id:
                idx = i
                break
        if idx == -1:
            print(f"Patient {patient_id} not found.")
            return None, None
    else:
        idx = 0
        
    row = val_ds.valid_patients[idx]
    fname = os.path.basename(row['image_path'])
    pid = fname.replace('.nii.gz', '').replace('.nii', '')
    if pid not in val_ds.reports_json and len(pid.split('_')) > 1:
        pid = pid.rsplit('_', 1)[0]
    print(f"Visualizing Patient: {pid}")
    return pid, val_ds[idx]


# ---------------------------------------------------------------------------
# Model instantiation & attention extraction
# ---------------------------------------------------------------------------

def get_medical_model(name, kwargs):
    if name == "Base-8T":
        return medical_vlm_base8t.MedicalVLM(**kwargs)
    elif name == "Multiscale-ViT-FPN":
        return medical_vlm_multiscale.MedicalVLM(**kwargs)
    elif name == "CNN-Stem-V3":
        return medical_vlm_v3.MedicalVLM(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")


def get_model_attention(name, checkpoint_path, pixel_values, organ_masks):
    """
    Instantiate model, load vision-only weights, extract cross-attention.
    
    Uses default head-averaged attention (average_attn_weights=True).
    Returns: attn (Q, 1232) where Q = 96 (8 tokens * 12 organs)
    """
    kwargs = {
        "vision_encoder_path": "dummy",
        "decoder_model_name": "google/medgemma-4b-it",
        "queries_per_organ": 8
    }

    model = get_medical_model(name, kwargs)
    
    # Load trainable vision weights from checkpoint
    vision_path = os.path.join(checkpoint_path, "vision_encoder.bin")
    stem_path = os.path.join(checkpoint_path, "stem.bin")
    projector_path = os.path.join(checkpoint_path, "projector.bin")
    ln_path = os.path.join(checkpoint_path, "projector_layernorm.bin")
    pos_path = os.path.join(checkpoint_path, "visual_pos_embed.bin")
    
    # Load CNN Stem if present (V3 architecture)
    has_stem = hasattr(model, 'stem') and os.path.exists(stem_path)
    if has_stem:
        model.stem.load_state_dict(torch.load(stem_path, map_location='cpu'))
        print(f"  Loaded CNN Stem for {name}")
    
    if os.path.exists(vision_path):
        model.vision_encoder.load_state_dict(torch.load(vision_path, map_location='cpu'))
    if os.path.exists(projector_path):
        model.visual_projection.load_state_dict(torch.load(projector_path, map_location='cpu'))
    if os.path.exists(ln_path):
        model.projector_layernorm.load_state_dict(torch.load(ln_path, map_location='cpu'))
    if os.path.exists(pos_path):
        model.visual_pos_embed = torch.load(pos_path, map_location='cpu')
        
    model.eval()
    model.cuda()
    
    # Use default head-averaged attention (average_attn_weights=True is the default)
    # This returns (B, Q, S) — the true attention distribution used by the model
    
    with torch.no_grad():
        # For V3: run CNN stem first, then vision encoder on stem features
        if has_stem:
            stem_feats = model.stem(pixel_values)
            visual_feats, attn_weights = model.vision_encoder(stem_feats, organ_masks)
        else:
            visual_feats, attn_weights = model.vision_encoder(pixel_values, organ_masks)
        
        attn = attn_weights[0]  # (Q, S) — first sample in batch
        
    # Free memory
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
        
    return attn


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def find_best_slice(gt_mask_3d):
    """Find the axial slice with maximum organ coverage."""
    if gt_mask_3d.ndim == 4 and gt_mask_3d.shape[0] == 1:
        gt_mask_3d = gt_mask_3d.squeeze(0)
    slice_sums = gt_mask_3d.sum(axis=(1, 2))
    return int(np.argmax(slice_sums))


def compare_attention(args):
    # ---- Model configs ----
    checkpoints = [
        ("Base-8T",
         "/home/muhammedg/fvlm/rep_medgemma/checkpoints_retrain/medical_vlm_8_tokens_full/final"),
        ("Multiscale-ViT-FPN",
         "/home/muhammedg/fvlm/rep_medgemma/checkpoints_retrain/multiscale_vit_fpn/final"),
        ("CNN-Stem-V3",
         "/home/muhammedg/fvlm/rep_medgemma/checkpoints_retrain/medgemma_architecture_v3/final"),
    ]

    # ---- Load data ----
    val_ds = load_dataset()
    target_organs = [o.lower() for o in args.target_organs] if args.target_organs else ["lung", "kidney"]

    print(f"Finding patient...")
    pid, sample = get_sample_for_patient(val_ds, args.patient_id)
    if sample is None:
        return
    
    pixel_values = sample['pixel_values'].unsqueeze(0).cuda()
    organ_masks = sample['organ_masks'].unsqueeze(0).cuda()
    
    ct_vol = pixel_values[0, 0].cpu().numpy()
    ct_vol = (ct_vol - ct_vol.min()) / (ct_vol.max() - ct_vol.min())
    D_ct, H_ct, W_ct = ct_vol.shape
    
    ALL_ORGANS = val_ds.target_keys
    D_vit, H_vit, W_vit = 7, 16, 11

    # ---- Extract attention for all models ----
    model_maps = []
    for name, cp in checkpoints:
        print(f"Evaluating {name}...")
        attn = get_model_attention(name, cp, pixel_values, organ_masks)
        Q = attn.shape[0]
        attn = attn.view(Q, D_vit, H_vit, W_vit)
        
        if Q == 96:  # 8 tokens per organ -> mean across tokens
            attn = attn.view(12, 8, D_vit, H_vit, W_vit).mean(dim=1)  # (12, 7, 16, 11)
        elif Q != 12:
            print(f"Warning: Unexpected number of queries {Q}")
        model_maps.append((name, attn))

    # ---- Plot per organ ----
    for organ in target_organs:
        if organ not in ALL_ORGANS:
            print(f"Organ {organ} not found in target_keys. Skipping.")
            continue
        print(f"\n{'='*60}")
        print(f"  Plotting for {organ}")
        print(f"{'='*60}")
        organ_idx = ALL_ORGANS.index(organ)
        
        # GT mask for this organ
        gt_mask = organ_masks[0, organ_idx].cpu().numpy()
        if gt_mask.ndim == 4 and gt_mask.shape[0] == 1:
            gt_mask = gt_mask.squeeze(0)
        
        # Find the best axial slice (max organ presence)
        z = find_best_slice(gt_mask)
        
        n_models = len(checkpoints)
        
        # --- Figure layout: n_models + 1 columns (models + GT), with metrics table below ---
        fig = plt.figure(figsize=(4.5 * (n_models + 1), 5.5))
        gs = gridspec.GridSpec(2, n_models + 1, height_ratios=[4, 1], hspace=0.05, wspace=0.15)
        
        # Collect metrics for the table
        all_metrics = []
        
        for i, (name, att_grid) in enumerate(model_maps):
            att = att_grid[organ_idx]
            att_up = F.interpolate(
                att.unsqueeze(0).unsqueeze(0), size=(D_ct, H_ct, W_ct),
                mode='trilinear', align_corners=False
            ).squeeze().cpu().numpy()
            
            # Compute quantitative metrics
            metrics = compute_metrics(att_up, gt_mask)
            all_metrics.append(metrics)
            
            # Print metrics
            print(f"  {name}:")
            print(f"    Attn-in-mask:  {metrics['attention_in_mask_pct']:.1f}%")
            print(f"    Off-target:    {metrics['off_target_pct']:.1f}%")
            print(f"    Entropy (norm): {metrics['entropy_norm']:.3f}")
            
            # Attention map panel
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(ct_vol[z, :, :], cmap='gray')
            ax.imshow(att_up[z, :, :], cmap='jet', alpha=0.5)
            ax.set_title(f"{name}", fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Metrics text below
            ax_m = fig.add_subplot(gs[1, i])
            ax_m.axis('off')
            metric_text = (
                f"In-mask: {metrics['attention_in_mask_pct']:.1f}%\n"
                f"Off-target: {metrics['off_target_pct']:.1f}%\n"
                f"Entropy: {metrics['entropy_norm']:.3f}"
            )
            ax_m.text(0.5, 0.8, metric_text, ha='center', va='top',
                     fontsize=10, fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                     transform=ax_m.transAxes)
        
        # GT Mask panel
        ax = fig.add_subplot(gs[0, n_models])
        ax.imshow(ct_vol[z, :, :], cmap='gray')
        ax.imshow(gt_mask[z, :, :], cmap='spring', alpha=0.5)
        ax.set_title(f"GT Mask", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Empty metrics cell under GT
        ax_m = fig.add_subplot(gs[1, n_models])
        ax_m.axis('off')
        ax_m.text(0.5, 0.8, f"Organ: {organ}\nSlice: z={z}",
                 ha='center', va='top', fontsize=10, fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8),
                 transform=ax_m.transAxes)
        
        fig.suptitle(f"Cross-Attention Maps (head-averaged) — {organ}", fontsize=14, fontweight='bold', y=0.98)
        
        out_path = f"attention_mean_{organ}_{pid}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved {out_path}")
        plt.close()
    
    # ---- Summary table across all organs ----
    print(f"\n{'='*60}")
    print(f"  QUANTITATIVE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Organ':<12} {'Model':<22} {'In-Mask%':>10} {'Off-Target%':>12} {'Entropy':>10}")
    print("-" * 66)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention Map Comparison (Head-Averaged)")
    parser.add_argument('--patient_id', type=str, default=None,
                        help="Patient ID to visualize (default: first in validation)")
    parser.add_argument('--target_organs', type=str, nargs='+', default=None,
                        help="Organs to visualize (default: LUNG KIDNEY)")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    compare_attention(args)

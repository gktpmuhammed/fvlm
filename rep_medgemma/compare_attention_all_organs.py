"""
All-Organs Attention Map Comparison (Head-Averaged) — Single Combined Plot
Base-8T vs Multiscale-ViT-FPN vs CNN-Stem-V3

Generates ONE figure with:
  Rows    = 12 organs
  Columns = 3 models + GT mask
Each cell shows the attention overlay on the best axial slice for that organ.
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

sys.path.insert(1, "/home/muhammedg/fvlm")

import medical_vlm_base8t
import medical_vlm_multiscale
import medical_vlm_v3
import train


# ---------------------------------------------------------------------------
# Quantitative metrics
# ---------------------------------------------------------------------------

def compute_metrics(attn_map_3d, gt_mask_3d):
    attn = attn_map_3d.copy().astype(np.float64)
    mask = (gt_mask_3d > 0.5).astype(np.float64)
    attn_sum = attn.sum()
    if attn_sum < 1e-12:
        return {"attention_in_mask_pct": 0.0, "off_target_pct": 100.0, "entropy_norm": 1.0}
    p = attn / attn_sum
    aim = p[mask > 0.5].sum() * 100.0
    off_target = 100.0 - aim
    p_flat = p.flatten()
    p_flat = p_flat[p_flat > 1e-15]
    H = -np.sum(p_flat * np.log(p_flat))
    H_max = np.log(p.size)
    entropy_norm = H / H_max if H_max > 0 else 0.0
    return {"attention_in_mask_pct": aim, "off_target_pct": off_target, "entropy_norm": entropy_norm}


# ---------------------------------------------------------------------------
# Data loading
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
        tokenizer=tokenizer, transform=transform, split='validation', subset_size=None
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
# Model attention extraction
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
    kwargs = {
        "vision_encoder_path": "dummy",
        "decoder_model_name": "google/medgemma-4b-it",
        "queries_per_organ": 8
    }
    model = get_medical_model(name, kwargs)
    
    vision_path = os.path.join(checkpoint_path, "vision_encoder.bin")
    stem_path = os.path.join(checkpoint_path, "stem.bin")
    projector_path = os.path.join(checkpoint_path, "projector.bin")
    ln_path = os.path.join(checkpoint_path, "projector_layernorm.bin")
    pos_path = os.path.join(checkpoint_path, "visual_pos_embed.bin")
    
    has_stem = hasattr(model, 'stem') and os.path.exists(stem_path)
    if has_stem:
        model.stem.load_state_dict(torch.load(stem_path, map_location='cpu'))
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
    
    with torch.no_grad():
        if has_stem:
            stem_feats = model.stem(pixel_values)
            visual_feats, attn_weights = model.vision_encoder(stem_feats, organ_masks)
        else:
            visual_feats, attn_weights = model.vision_encoder(pixel_values, organ_masks)
        attn = attn_weights[0]
    
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return attn


# ---------------------------------------------------------------------------
# Slice selection
# ---------------------------------------------------------------------------

def find_best_slice(gt_mask_3d):
    if gt_mask_3d.ndim == 4 and gt_mask_3d.shape[0] == 1:
        gt_mask_3d = gt_mask_3d.squeeze(0)
    D = gt_mask_3d.shape[0]
    slice_voxels = np.array([(gt_mask_3d[z] > 0.5).sum() for z in range(D)])
    best_z = int(np.argmax(slice_voxels))
    return best_z, int(slice_voxels[best_z])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare_all_organs(args):
    checkpoints = [
        ("Base-8T",
         "/home/muhammedg/fvlm/rep_medgemma/checkpoints_retrain/medical_vlm_8_tokens_full/final"),
        ("Multiscale-ViT-FPN",
         "/home/muhammedg/fvlm/rep_medgemma/checkpoints_retrain/multiscale_vit_fpn/final"),
        ("CNN-Stem-V3",
         "/home/muhammedg/fvlm/rep_medgemma/checkpoints_retrain/medgemma_architecture_v3/final"),
    ]

    val_ds = load_dataset()
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
    n_models = len(checkpoints)
    n_organs = len(ALL_ORGANS)
    n_cols = n_models + 1  # models + GT mask

    # ---- Extract attention for all models ----
    model_maps = []
    for name, cp in checkpoints:
        print(f"Evaluating {name}...")
        attn = get_model_attention(name, cp, pixel_values, organ_masks)
        Q = attn.shape[0]
        attn = attn.view(Q, D_vit, H_vit, W_vit)
        if Q == 96:
            attn = attn.view(12, 8, D_vit, H_vit, W_vit).mean(dim=1)
        model_maps.append((name, attn))

    # ---- Pre-compute best slices and check which organs have masks ----
    organ_slices = []
    valid_organs = []
    for organ_idx, organ in enumerate(ALL_ORGANS):
        gt_mask = organ_masks[0, organ_idx].cpu().numpy()
        if gt_mask.ndim == 4 and gt_mask.shape[0] == 1:
            gt_mask = gt_mask.squeeze(0)
        z, voxel_count = find_best_slice(gt_mask)
        if voxel_count > 0:
            valid_organs.append((organ_idx, organ, z, voxel_count, gt_mask))
            print(f"  {organ}: z={z} ({voxel_count} voxels)")
        else:
            print(f"  {organ}: no mask voxels — skipping")

    n_valid = len(valid_organs)
    if n_valid == 0:
        print("No organs with valid masks found.")
        return

    # ---- Create combined figure ----
    # Each cell ~ 3x2.5 inches
    cell_w, cell_h = 3.0, 2.5
    fig_w = cell_w * n_cols + 1.0
    fig_h = cell_h * n_valid + 1.5

    fig, axes = plt.subplots(n_valid, n_cols, figsize=(fig_w, fig_h))
    if n_valid == 1:
        axes = axes[np.newaxis, :]

    # Column headers
    model_names = [name for name, _ in checkpoints] + ["GT Mask"]

    summary_rows = []

    for row_i, (organ_idx, organ, z, voxel_count, gt_mask) in enumerate(valid_organs):
        for col_i in range(n_cols):
            ax = axes[row_i, col_i]

            if col_i < n_models:
                # Model attention panel
                name, att_grid = model_maps[col_i]
                att = att_grid[organ_idx]
                att_up = F.interpolate(
                    att.unsqueeze(0).unsqueeze(0), size=(D_ct, H_ct, W_ct),
                    mode='trilinear', align_corners=False
                ).squeeze().cpu().numpy()

                metrics = compute_metrics(att_up, gt_mask)
                summary_rows.append({
                    'organ': organ, 'model': name,
                    'in_mask': metrics['attention_in_mask_pct'],
                    'off_target': metrics['off_target_pct'],
                    'entropy': metrics['entropy_norm']
                })

                ax.imshow(ct_vol[z, :, :], cmap='gray')
                ax.imshow(att_up[z, :, :], cmap='jet', alpha=0.5)

                # Show in-mask % as small text in corner
                ax.text(0.98, 0.02, f"{metrics['attention_in_mask_pct']:.0f}%",
                       ha='right', va='bottom', fontsize=8, color='white',
                       fontweight='bold', transform=ax.transAxes,
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6))
            else:
                # GT Mask panel
                ax.imshow(ct_vol[z, :, :], cmap='gray')
                ax.imshow(gt_mask[z, :, :], cmap='spring', alpha=0.5)
                ax.text(0.98, 0.02, f"z={z}",
                       ha='right', va='bottom', fontsize=8, color='white',
                       fontweight='bold', transform=ax.transAxes,
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6))

            ax.axis('off')

            # Column headers (top row only)
            if row_i == 0:
                ax.set_title(model_names[col_i], fontsize=11, fontweight='bold', pad=4)

        # Row label (organ name on the left)
        axes[row_i, 0].text(-0.05, 0.5, organ.capitalize(),
                           ha='right', va='center', fontsize=11, fontweight='bold',
                           transform=axes[row_i, 0].transAxes, rotation=90)

    fig.suptitle(f"Cross-Attention Maps (head-averaged) — Patient {pid}",
                fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0.03, 0.0, 1.0, 0.97])

    out_path = f"attention_all_organs_{pid}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out_path}")
    plt.close()

    # ---- Print summary table ----
    print(f"\n{'='*80}")
    print(f"  QUANTITATIVE SUMMARY — Patient {pid}")
    print(f"{'='*80}")
    print(f"{'Organ':<14} {'Model':<22} {'In-Mask%':>10} {'Off-Target%':>12} {'Entropy':>10}")
    print("-" * 70)
    for r in summary_rows:
        print(f"{r['organ']:<14} {r['model']:<22} {r['in_mask']:>9.1f}% {r['off_target']:>11.1f}% {r['entropy']:>10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="All-Organs Attention Comparison (combined)")
    parser.add_argument('--patient_id', type=str, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    compare_all_organs(args)

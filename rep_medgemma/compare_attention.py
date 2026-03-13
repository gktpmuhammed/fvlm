import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# Set priority so they can find lavis locally without shadowing local `train.py`
sys.path.insert(1, "/home/muhammedg/fvlm")

# We copied all medical_vlm.py files here so they can all load at once natively
import medical_vlm_baseline
import medical_vlm_lora
import medical_vlm_perceiver
import medical_vlm_multiscale

# They also need `train` to build the generic dataset
import train

def get_medical_model(name, kwargs):
    # Depending on the name, use the specific architecture import
    if name == "Baseline-8T":
        return medical_vlm_baseline.MedicalVLM(**kwargs)
    elif name == "Prefix-Stable-8T":
        return medical_vlm_lora.MedicalVLM(**kwargs)
    elif name == "Perceiver":
        return medical_vlm_perceiver.MedicalVLM(**kwargs)
    elif name == "Multiscale":
        return medical_vlm_multiscale.MedicalVLM(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")

def load_dataset():
    from transformers import AutoTokenizer
    # We just need the tokenizer
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

def get_model_attention(name, checkpoint_path, pixel_values, organ_masks, is_perceiver=False, is_multiscale=False):
    kwargs = {
        "vision_encoder_path": "dummy",
        "decoder_model_name": "google/medgemma-4b-it",
        "queries_per_organ": 8
    }
    
    if is_perceiver:
        kwargs["use_perceiver"] = True
    if is_multiscale:
        kwargs["use_multiscale"] = True

    model = get_medical_model(name, kwargs)
    
    # Load custom weights
    vision_path = os.path.join(checkpoint_path, "vision_encoder.bin")
    stem_path = os.path.join(checkpoint_path, "stem.bin")
    projector_path = os.path.join(checkpoint_path, "projector.bin")
    ln_path = os.path.join(checkpoint_path, "projector_layernorm.bin")
    pos_path = os.path.join(checkpoint_path, "visual_pos_embed.bin")
    
    if os.path.exists(stem_path): model.stem.load_state_dict(torch.load(stem_path))
    if os.path.exists(vision_path): model.vision_encoder.load_state_dict(torch.load(vision_path))
    if os.path.exists(projector_path): model.visual_projection.load_state_dict(torch.load(projector_path))
    if os.path.exists(ln_path): model.projector_layernorm.load_state_dict(torch.load(ln_path))
    if os.path.exists(pos_path): model.visual_pos_embed = torch.load(pos_path)
        
    model.eval()
    model.cuda()
    
    with torch.no_grad():
        if hasattr(model, 'stem') and model.stem is not None:
             stem_feats = model.stem(pixel_values)
        else:
             stem_feats = pixel_values # Fallback if no stem
             
        visual_feats, attn_weights = model.vision_encoder(stem_feats, organ_masks)
        attn = attn_weights[0] # (Q, 1232)
        
    # Free memory before next model
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
        
    return attn

def compare_attention(args):
    # Setup Checkpoints to compare based on case
    if args.case == "A" or args.case == "B":
        checkpoints = [
            ("Baseline-8T", "/home/muhammedg/fvlm/rep_medgemma/checkpoints_retrain/medical_vlm_8_tokens_full", False, False),
            ("Prefix-Stable-8T", "/home/muhammedg/fvlm/rep_medgemma/checkpoints_retrain/medgemma_lora_vis_token_pos_embed", False, False)
        ]
    elif args.case == "C":
        checkpoints = [
            ("Baseline-8T", "/home/muhammedg/fvlm/rep_medgemma/checkpoints_retrain/medical_vlm_8_tokens_full", False, False),
            ("Perceiver", "/home/muhammedg/fvlm/rep_medgemma/checkpoints_retrain/perceiver_resampler", True, False),
            ("Multiscale", "/home/muhammedg/fvlm/rep_medgemma/checkpoints_retrain/multiscale_vit_fpn", False, True)
        ]
    else:
        print("Invalid case. Must be A, B, or C.")
        return

    # Load Data
    val_ds = load_dataset()
    target_organs = [args.target_organ] if args.target_organ else ["LUNG", "HEART", "LIVER"]

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
    
    model_maps = []
    
    for name, cp, is_perc, is_multi in checkpoints:
        print(f"Evaluating {name}...")
        attn = get_model_attention(name, cp, pixel_values, organ_masks, is_perc, is_multi)
        D_vit, H_vit, W_vit = 7, 16, 11
        Q = attn.shape[0]
        attn = attn.view(Q, D_vit, H_vit, W_vit)
        
        if Q == 96: # 8 tokens per organ
            attn = attn.view(12, 8, D_vit, H_vit, W_vit).mean(dim=1) # (12, 7, 16, 11)
        elif Q != 12:
            print(f"Warning: Unexpected number of queries {Q}")
        model_maps.append((name, attn))

    for organ in target_organs:
        if organ not in ALL_ORGANS:
            continue
        print(f"Plotting for {organ}...")
        organ_idx = ALL_ORGANS.index(organ)
        
        base_attn = model_maps[0][1][organ_idx]
        base_att_up = F.interpolate(
            base_attn.unsqueeze(0).unsqueeze(0), size=(D_ct, H_ct, W_ct), mode='trilinear', align_corners=False
        ).squeeze().cpu().numpy()
        z, y, x = np.unravel_index(np.argmax(base_att_up), base_att_up.shape)
        
        n_models = len(checkpoints)
        fig, axes = plt.subplots(1, n_models + 1, figsize=(4*(n_models + 1), 4))
        
        for i, (name, att_grid) in enumerate(model_maps):
            att = att_grid[organ_idx]
            att_up = F.interpolate(
                att.unsqueeze(0).unsqueeze(0), size=(D_ct, H_ct, W_ct), mode='trilinear', align_corners=False
            ).squeeze().cpu().numpy()
            
            ax = axes[i]
            ax.imshow(ct_vol[z, :, :], cmap='gray')
            ax.imshow(att_up[z, :, :], cmap='jet', alpha=0.5)
            ax.set_title(f"{name} ({organ})")
            ax.axis('off')
            
        gt_mask = organ_masks[0, organ_idx].cpu().numpy()
        if gt_mask.ndim == 4 and gt_mask.shape[0] == 1:
            gt_mask = gt_mask.squeeze(0)
        ax = axes[n_models]
        ax.imshow(ct_vol[z, :, :], cmap='gray')
        ax.imshow(gt_mask[z, :, :], cmap='spring', alpha=0.5)
        ax.set_title(f"GT Mask ({organ})")
        ax.axis('off')
        
        plt.tight_layout()
        out_path = f"attention_vis_case{args.case}_{organ}_{pid}.png"
        plt.savefig(out_path)
        print(f"Saved {out_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, required=True, choices=['A', 'B', 'C'])
    parser.add_argument('--patient_id', type=str, default=None)
    parser.add_argument('--target_organ', type=str, default=None, help="E.g. LUNG, LIVER, HEART")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    compare_attention(args)

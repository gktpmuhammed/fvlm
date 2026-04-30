import os
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

# Set priority so they can find lavis locally
sys.path.insert(0, str(PROJECT_ROOT / 'rep_medgemma')) 
parent_dir = os.path.dirname(os.path.abspath(__file__))

import medical_vlm_baseline
import medical_vlm_lora
import medical_vlm_perceiver
import medical_vlm_multiscale

# They also need `train` to build the generic dataset
import train

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
        csv_file=str(PROJECT_ROOT / 'data_sym/image_first_dataset.csv'),
        json_file=str(PROJECT_ROOT / 'data_sym/combined_desc_conc.json'),
        tokenizer=tokenizer,
        transform=transform,
        split='validation',
        subset_size=None
    )
    return val_ds

def get_model_attention(name, checkpoint_path, pixel_values, organ_masks, is_perceiver=False, is_multiscale=False):
    if is_perceiver:
        model = medical_vlm_perceiver.MedicalVLM(vision_encoder_path="dummy")
    elif is_multiscale:
        model = medical_vlm_multiscale.MedicalVLM(vision_encoder_path="dummy")
    elif name == "Prefix-Stable-8T":
        model = medical_vlm_lora.MedicalVLM(vision_encoder_path="dummy")
    else:
        model = medical_vlm_baseline.MedicalVLM(vision_encoder_path="dummy")

    if os.path.isdir(checkpoint_path):
        vision_path = os.path.join(checkpoint_path, "vision_encoder.bin")
        if not os.path.exists(vision_path):
            dirs = [d for d in os.listdir(checkpoint_path) if d.startswith("checkpoint")]
            if dirs:
                dirs.sort(key=lambda x: int(x.split("-")[-1]))
                checkpoint_path = os.path.join(checkpoint_path, dirs[-1])
                vision_path = os.path.join(checkpoint_path, "vision_encoder.bin")
        vision_state = torch.load(vision_path, map_location='cpu')
    else:
        vision_state = torch.load(checkpoint_path, map_location='cpu')
    model.vision_encoder.load_state_dict(vision_state, strict=False)
    
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        if hasattr(model.vision_encoder.vit, 'forward_features'):
             stem_feats = model.vision_encoder.vit.forward_features(pixel_values)
        else:
             stem_feats = pixel_values 
             
        visual_feats, attn_weights = model.vision_encoder(stem_feats, organ_masks)
        attn = attn_weights[0] 
        
    del model
    torch.cuda.empty_cache()
    
    return attn

def evaluate_models_entropy(args):
    val_ds = load_dataset()
    ALL_ORGANS = val_ds.target_keys
    target_organ = args.target_organ
    
    if target_organ not in ALL_ORGANS:
        print(f"Organ {target_organ} not found.")
        return
        
    organ_idx = ALL_ORGANS.index(target_organ)
    
    if args.case == "A" or args.case == "B":
        checkpoints = [
            ("Baseline-8T", str(PROJECT_ROOT / 'rep_medgemma/checkpoints_retrain/medical_vlm_8_tokens_full'), False, False),
            ("Prefix-Stable-8T", str(PROJECT_ROOT / 'rep_medgemma/checkpoints_retrain/medgemma_lora_vis_token_pos_embed'), False, False)
        ]
    elif args.case == "C":
         checkpoints = [
            ("Baseline-8T", str(PROJECT_ROOT / 'rep_medgemma/checkpoints_retrain/medical_vlm_8_tokens_full'), False, False),
            ("Perceiver", str(PROJECT_ROOT / 'rep_medgemma/checkpoints_retrain/perceiver_resampler'), True, False),
            ("Multiscale", str(PROJECT_ROOT / 'rep_medgemma/checkpoints_retrain/multiscale_vit_fpn'), False, True)
        ]
        
    print(f"Finding best examples for Case {args.case} target: {target_organ} using Entropy")
    
    results = []
    
    limit = min(args.limit, len(val_ds))
    
    for i in tqdm(range(limit)):
        sample = val_ds[i]
        pid = sample.get('patient_id', f'sample_{i}')
        
        # Check if organ is actually present in this scan
        mask_tensor = sample['organ_masks']
        organ_mask = mask_tensor[organ_idx].squeeze() # Force (D, H, W)
        if organ_mask.max() == 0:
            continue
            
        pixel_values = sample['pixel_values'].unsqueeze(0).cuda()
        organ_masks_gpu = sample['organ_masks'].unsqueeze(0).cuda()
        
        # Downsample ground truth mask to match attention map shape (7, 16, 11)
        D_m, H_m, W_m = organ_mask.shape
        flat_mask = organ_mask.view(1, 1, D_m, H_m, W_m).cuda()
        mask_down = F.adaptive_max_pool3d(flat_mask, output_size=(7, 16, 11)).squeeze() # (7, 16, 11)
        
        patient_scores = {'patient_id': pid}
        
        for name, cp, is_perc, is_multi in checkpoints:
            attn = get_model_attention(name, cp, pixel_values, organ_masks_gpu, is_perc, is_multi)
            
            # Format attention
            Q = attn.shape[0]
            attn = attn.view(Q, 7, 16, 11)
            if Q == 96:
                attn = attn.view(12, 8, 7, 16, 11).mean(dim=1)
                
            organ_attn = attn[organ_idx] # (7, 16, 11)
            organ_attn = organ_attn.squeeze()
            
            # Select only the attention inside the mask
            mask_binary = (mask_down > 0)
            attn_in_mask = organ_attn[mask_binary]
            
            # Normalize to sum to 1 over the mask to get probability distribution
            attn_sum = attn_in_mask.sum().item() + 1e-8
            p = attn_in_mask / attn_sum
            
            # Compute Entropy (H = -sum(p * log(p)))
            # Lower entropy = sharper focus, Higher entropy = more diffuse
            entropy = - torch.sum(p * torch.log(p + 1e-8)).item()
            patient_scores[name] = entropy
            
        results.append(patient_scores)
        
    # Sort results
    if args.case == "A" or args.case == "B":
        # We want Prefix-Stable-8T to have lower entropy (sharper) than Baseline (more diffuse).
        # Maximizing Baseline_Entropy - Prefix_Entropy
        results.sort(key=lambda x: x["Baseline-8T"] - x["Prefix-Stable-8T"], reverse=True)
        
        print(f"\nTop Examples where Prefix-Stable-8T focuses more sharply than Baseline ({target_organ}):")
        for j, res in enumerate(results[:10]):
            diff = res["Baseline-8T"] - res["Prefix-Stable-8T"]
            print(f"{j+1}. Patient {res['patient_id']} | Prefix Entropy: {res['Prefix-Stable-8T']:.3f} | Baseline Entropy: {res['Baseline-8T']:.3f} | Diff: +{diff:.3f}")
            
    elif args.case == "C":
        # We want proposed models to have lower entropy (sharper) than Baseline.
        # Maximizing Baseline_Entropy - min(Proposed_Entropy)
        results.sort(key=lambda x: x["Baseline-8T"] - min(x["Multiscale"], x["Perceiver"]), reverse=True)
        
        print(f"\nTop Examples where Proposed Models focus more sharply than Baseline ({target_organ}):")
        for j, res in enumerate(results[:10]):
            best_prop = min(res["Multiscale"], res["Perceiver"])
            diff = res["Baseline-8T"] - best_prop
            print(f"{j+1}. Patient {res['patient_id']} | Multiscale Entropy: {res['Multiscale']:.3f} | Perceiver Entropy: {res['Perceiver']:.3f} | Baseline Entropy: {res['Baseline-8T']:.3f} | Diff: +{diff:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, required=True, choices=["A", "B", "C"])
    parser.add_argument("--target_organ", type=str, required=True)
    parser.add_argument("--limit", type=int, default=20, help="Number of files to scan")
    args = parser.parse_args()
    evaluate_models_entropy(args)

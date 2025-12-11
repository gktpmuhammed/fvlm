#!/usr/bin/env python3
import sys
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import json
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, Transposed, EnsureChannelFirstd

# Fix path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)

from medical_vlm import MedicalVLM
from train import get_organ_ids_for_key, ALL_TARGET_KEYS, FULL_BODY_ID, build_transforms

class EvalDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform, subset_size=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == 'validation'].reset_index(drop=True)
        
        # --- SUBSET LOGIC ADDED ---
        if subset_size is not None and subset_size > 0:
            print(f"Subsetting validation to first {subset_size} samples.")
            self.df = self.df.head(subset_size)
            
        self.transform = transform
        self.tokenizer = tokenizer
        
    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            mask_path = row['image_path'].replace('images', 'masks')
            data = self.transform({'image': row['image_path'], 'mask': mask_path})
            
            # Convert to Tensor (MetaTensor Fix)
            img = data['image']
            img = img.as_tensor().float() if hasattr(img, 'as_tensor') else torch.from_numpy(img).float()
            
            mask = data['mask']
            mask = mask.as_tensor() if hasattr(mask, 'as_tensor') else torch.from_numpy(mask)
            
            return {
                'pixel_values': img,
                'full_mask': mask,
                'patient_id': os.path.basename(row['image_path']).split('.')[0]
            }
        except: return None

def evaluate(args):
    print("Loading Model...")
    model = MedicalVLM(args.vision_encoder_path, args.decoder_model)
    
    # Load weights logic
    if os.path.isdir(args.model_path):
        w_path = os.path.join(args.model_path, "pytorch_model.bin")
        if not os.path.exists(w_path): # Check for safetensors if bin missing
            w_path = os.path.join(args.model_path, "model.safetensors")
            if os.path.exists(w_path):
                from safetensors.torch import load_file
                state = load_file(w_path)
            else:
                raise FileNotFoundError(f"No weights found in {args.model_path}")
        else:
            state = torch.load(w_path, map_location='cpu')
    else:
        state = torch.load(args.model_path, map_location='cpu')
    
    model.model.load_state_dict(state, strict=False)
    model.model.eval().cuda()
    
    # Pass subset_size to Dataset
    ds = EvalDataset(args.csv_file, model.tokenizer, build_transforms(), args.subset_size)
    
    # Batch size must be 1 for easy result matching
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    results = []
    
    print("Generating Reports...")
    with torch.no_grad():
        for batch in tqdm(dl):
            if batch is None: continue
            
            pixel_values = batch['pixel_values'].cuda() # (1, 1, D, H, W)
            full_mask = batch['full_mask'].cuda()
            pid = batch['patient_id'][0]
            
            # 1. Create Masks for ALL targets
            mask_stack = []
            prompts = []
            
            for key in ALL_TARGET_KEYS:
                # Prompt
                p_text = f"Describe {key}: " if key != "Conclusion" else "Conclusion: "
                prompts.append(p_text)
                
                # Mask
                tids = get_organ_ids_for_key(key)
                if tids == [FULL_BODY_ID]:
                    m = torch.ones_like(full_mask)
                elif len(tids) > 0:
                    m = torch.zeros_like(full_mask)
                    for t in tids: m[full_mask == t] = 1.0
                else:
                    m = torch.zeros_like(full_mask)
                
                mask_stack.append(m)
            
            # Stack: (1, N_Targets, D, H, W)
            organ_masks = torch.stack(mask_stack, dim=1).float()
            
            # 2. One-Pass Generation
            # Output: (N_Targets, Seq_Len)
            
            prompt_ids = model.tokenizer(prompts, return_tensors="pt", padding=True).input_ids.cuda()
            
            # --- FIX: Use decoder_input_ids instead of input_ids ---
            outputs = model.generate(
                pixel_values=pixel_values,
                organ_masks=organ_masks,
                decoder_input_ids=prompt_ids,  # <--- CHANGED HERE
                max_length=150,
                num_beams=4,
                repetition_penalty=2.0
            )
            
            # 3. Decode
            decoded = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # 4. Format Result
            report_dict = {}
            full_text = f"Patient: {pid}\n"
            
            for key, text, p in zip(ALL_TARGET_KEYS, decoded, prompts):
                clean_text = text.replace(p, "").strip()
                report_dict[key] = clean_text
                full_text += f"[{key.upper()}]: {clean_text}\n"
            
            results.append({"patient_id": pid, "report": full_text})
            
            # Optional: Print first few to verify
            # print(f"\n--- {pid} ---\n{full_text}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, "generated_full_reports.csv"), index=False)
    print(f"Saved results to {args.output_dir}/generated_full_reports.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--decoder_model', type=str, default='gpt2') # Fixed default to match train
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='./results')
    # --- ARGUMENT ADDED ---
    parser.add_argument('--subset_size', type=int, default=None, help="Number of patients to evaluate")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    evaluate(args)
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

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)

from medical_vlm import MedicalVLM
# Import configuration from training script to ensure consistency
from train import get_organ_ids_for_key, ALL_TARGET_KEYS, FULL_BODY_ID, build_transforms

class EvalDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == 'validation'].reset_index(drop=True)
        self.transform = transform
        self.tokenizer = tokenizer
        
    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            # Assumes masks are in a parallel 'masks' folder
            mask_path = row['image_path'].replace('images', 'masks')
            
            # Use the exact same transforms as training
            data = self.transform({'image': row['image_path'], 'mask': mask_path})
            
            # Handle Monai MetaTensor to PyTorch Tensor conversion
            img = data['image']
            if hasattr(img, 'as_tensor'): 
                img = img.as_tensor().float()
            elif not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img).float()
            
            mask = data['mask']
            if hasattr(mask, 'as_tensor'): 
                mask = mask.as_tensor()
            elif not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask)
            
            return {
                'pixel_values': img,
                'full_mask': mask,
                'patient_id': os.path.basename(row['image_path']).split('.')[0]
            }
        except Exception as e: 
            print(f"Error loading {row['image_path']}: {e}")
            return None

def evaluate(args):
    print(f"--- Starting Evaluation on {args.csv_file} ---")
    
    # 1. Load Model
    print(f"Loading MedicalVLM (Decoder: {args.decoder_model})...")
    model = MedicalVLM(args.vision_encoder_path, args.decoder_model)
    
    # Ensure Tokenizer has pad token for prompt batching
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.model.config.pad_token_id = model.tokenizer.eos_token_id

    # 2. Load Weights
    print(f"Loading weights from {args.model_path}...")
    if os.path.isdir(args.model_path):
        weights_path = os.path.join(args.model_path, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location='cpu')
    else:
        state_dict = torch.load(args.model_path, map_location='cpu')
    
    model.model.load_state_dict(state_dict, strict=False)
    
    model.cuda()
    model.eval()
    
    # 3. Setup Data
    transform = build_transforms()
    ds = EvalDataset(args.csv_file, model.tokenizer, transform)
    
    # Batch size 1 is required because we expand one patient into N organ queries
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    
    results = []
    
    print("Generating Reports...")
    # Get device from model parameters
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in tqdm(dl):
            if batch is None: continue
            
            # Inputs: (1, 1, D, H, W)
            pixel_values = batch['pixel_values'].to(device)
            full_mask = batch['full_mask'].to(device)
            pid = batch['patient_id'][0]
            
            # 4. Prepare Multi-Organ Input
            mask_stack = []
            prompts = []
            
            for key in ALL_TARGET_KEYS:
                # A. Construct Prompt
                if key == "Conclusion":
                    p_text = "Conclusion: "
                else:
                    p_text = f"Describe {key}: "
                prompts.append(p_text)
                
                # B. Construct Binary Mask
                tids = get_organ_ids_for_key(key)
                
                if tids == [FULL_BODY_ID]:
                    m = torch.ones_like(full_mask)
                elif len(tids) > 0:
                    m = torch.zeros_like(full_mask)
                    for t in tids:
                        m[full_mask == t] = 1.0
                else:
                    m = torch.zeros_like(full_mask)
                
                mask_stack.append(m)
            
            # Stack masks: (1, N_Targets, 1, D, H, W) -> 6D Tensor
            organ_masks = torch.stack(mask_stack, dim=1).float()
            
            # Tokenize Prompts: (N_Targets, Seq_Len)
            # FIX: Use `device` variable instead of `model.device`
            prompt_inputs = model.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(device)
            
            # 5. Generate
            try:
                outputs = model.generate(
                    pixel_values=pixel_values,
                    organ_masks=organ_masks,
                    input_ids=prompt_inputs.input_ids,
                    attention_mask=prompt_inputs.attention_mask,
                    max_length=150,
                    num_beams=4,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=3
                )
                
                # 6. Decode
                decoded = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # 7. Format Output
                full_text = f"Patient: {pid}\n"
                
                for key, text, p_text in zip(ALL_TARGET_KEYS, decoded, prompts):
                    clean_text = text.replace(p_text, "").strip()
                    full_text += f"[{key.upper()}]: {clean_text}\n"
                
                results.append({"patient_id": pid, "report": full_text})
                
            except RuntimeError as e:
                print(f"Skipping {pid} due to error: {e}")
                continue

    # 8. Save Results
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "generated_full_reports.csv")
    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    print(f"Done! Results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to checkpoint folder or .bin file")
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--decoder_model', type=str, default='gpt2', help="HuggingFace model name or path for decoder")
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv', help="CSV file with image paths and splits")
    parser.add_argument('--output_dir', type=str, default='./results')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    evaluate(args)
#!/usr/bin/env python3
import sys
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
from tqdm import tqdm
import argparse
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, Transposed, EnsureChannelFirstd

# Add parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Fix Import Conflict:
# 1. Ensure current_dir is FIRST so `import train` picks up local train.py
# 2. Ensure parent_dir is DIRECTLY AFTER so `import lavis` picks up custom lavis (before site-packages)
if parent_dir in sys.path: sys.path.remove(parent_dir)
if current_dir in sys.path: sys.path.remove(current_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from medical_vlm import MedicalVLM
from train import get_organ_ids_for_key, ALL_TARGET_KEYS, build_transforms

class EvalDataset(Dataset):
    def __init__(self, csv_file, transform, subset_size=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == 'validation'].reset_index(drop=True)
        if subset_size: self.df = self.df.head(subset_size)
        self.transform = transform
        
    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            mask_path = row['image_path'].replace('images', 'masks')
            data = self.transform({'image': row['image_path'], 'mask': mask_path})
            
            img = data['image'].as_tensor().float() if hasattr(data['image'], 'as_tensor') else torch.tensor(data['image']).float()
            mask = data['mask'].as_tensor() if hasattr(data['mask'], 'as_tensor') else torch.tensor(data['mask'])
            
            return {
                'pixel_values': img,
                'full_mask': mask,
                'patient_id': os.path.basename(row['image_path']).split('.')[0]
            }
        except: return None

def evaluate(args):
    print(f"Loading Model: {args.decoder_model}")
    # 1. Load Model (Architecture handles loading weights)
    # Note: We need to load the saved projector and encoder weights
    model = MedicalVLM(args.vision_encoder_path, args.decoder_model)
    
    # Load trained weights
    print(f"Loading trained weights from {args.checkpoint_dir}...")
    enc_path = os.path.join(args.checkpoint_dir, "vision_encoder.bin")
    proj_path = os.path.join(args.checkpoint_dir, "projector.bin")
    
    if os.path.exists(enc_path):
        model.vision_encoder.load_state_dict(torch.load(enc_path, map_location='cpu'))
    if os.path.exists(proj_path):
        model.visual_projection.load_state_dict(torch.load(proj_path, map_location='cpu'))
    
    ln_path = os.path.join(args.checkpoint_dir, "projector_layernorm.bin")
    if os.path.exists(ln_path):
        print("Loading Projector LayerNorm...")
        model.projector_layernorm.load_state_dict(torch.load(ln_path, map_location='cpu'))
        
    model.cuda()
    model.eval()
    
    # 2. Data
    transform = build_transforms()
    ds = EvalDataset(args.csv_file, transform, args.subset_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    
    results = []
    
    print("Generating...")
    with torch.no_grad():
        for batch in tqdm(dl):
            if batch is None: continue
            
            pixel_values = batch['pixel_values'].cuda()
            full_mask = batch['full_mask'].cuda()
            pid = batch['patient_id'][0]
            
            # Prepare Prompts for all organs
            prompts = []
            mask_stack = []
            
            for key in ALL_TARGET_KEYS:
                # Format: Chat Template (User turn only)
                p = f"<start_of_turn>user\nAnalyze the specific image feature. Describe the findings for the {key}.<end_of_turn>\n<start_of_turn>model"
                prompts.append(p)
                
                # Mask
                tids = get_organ_ids_for_key(key)
                m = torch.zeros_like(full_mask)
                for t in tids: m[full_mask == t] = 1.0
                mask_stack.append(m)
            
            organ_masks = torch.stack(mask_stack, dim=1).float()
            
            # Tokenize Prompts
            # Tokenize Prompts
            inputs = model.tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')
            outputs = model.generate(
                pixel_values=pixel_values,
                organ_masks=organ_masks,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=False, 
                num_beams=3,            # Standard for medical reports
                repetition_penalty=1.2, # Penalize repetition
                no_repeat_ngram_size=3  # Stop 3-gram loops
            )
            
            decoded = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Parse results
            report = ""
            for key, text in zip(ALL_TARGET_KEYS, decoded):
                # Clean up the prompt from the output if necessary
                # Clean up the prompt from the output if necessary
                # Gemma typically outputs the full string, so we split by 'model\n'
                if "model\n" in text:
                    clean_text = text.split("model\n")[-1].strip()
                else:
                    clean_text = text.strip()
                    
                report += f"{key.upper()}: {clean_text}\n"
            
            results.append({'patient_id': pid, 'prediction': report})
            
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "generated_reports_gemma.csv"), index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/mae_pretrain_vit_base.pth')
    parser.add_argument('--decoder_model', type=str, default='google/medgemma-4b-it')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data_sym/image_first_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='./results/medgemma_vlm')
    parser.add_argument('--subset_size', type=int, default=None)
    args = parser.parse_args()
    evaluate(args)
#!/usr/bin/env python3
import sys
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import argparse
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, Transposed, EnsureChannelFirstd
import numpy as np

# Local Imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from medical_vlm import MedicalVLM
from train import get_organ_ids_for_key, build_transforms

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, tokenizer, transform):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == 'validation']
        self.tokenizer = tokenizer
        self.transform = transform
        
        # We define a standard list of organs to evaluate for EVERY patient
        self.target_organs = ["heart", "lung", "liver", "kidney", "aorta"]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        mask_path = img_path.replace('images', 'masks')
        
        try:
            data = self.transform({'image': img_path, 'mask': mask_path})
            img_tensor = torch.from_numpy(data['image']).float().unsqueeze(0)
            mask_tensor = torch.from_numpy(data['mask']).unsqueeze(0) # (1, 1, D, H, W)
            
            return {
                'pixel_values': img_tensor,
                'full_mask': mask_tensor,
                'patient_id': os.path.basename(img_path)
            }
        except:
            return None

def evaluate(args):
    model = MedicalVLM(args.vision_encoder_path, args.decoder_model)
    state_dict = torch.load(os.path.join(args.model_path, "pytorch_model.bin"), map_location='cpu')
    model.model.load_state_dict(state_dict)
    model.model.eval().cuda()
    
    transform = build_transforms()
    ds = EvalDataset(args.csv_file, model.tokenizer, transform)
    
    # Organs to generate
    organs_to_test = ["heart", "lung", "liver", "kidney"]
    results = []
    
    print("Generating Organ Reports...")
    with torch.no_grad():
        for item in tqdm(ds):
            if item is None: continue
            
            pixel_values = item['pixel_values'].cuda()
            full_mask = item['full_mask'] # Keep on CPU until needed or move to CUDA
            pid = item['patient_id']
            
            generated_report_parts = []
            
            for organ in organs_to_test:
                # 1. Create specific mask for this organ
                tids = get_organ_ids_for_key(organ)
                binary_mask = torch.zeros_like(full_mask)
                for tid in tids:
                    binary_mask[full_mask == tid] = 1.0
                
                binary_mask = binary_mask.cuda().float()
                
                # 2. Prompt
                prompt_text = f"Describe the {organ}: "
                input_ids = model.tokenizer(prompt_text, return_tensors="pt").input_ids.cuda()
                
                # 3. Generate
                # Pass mask to generate
                output_ids = model.generate(
                    pixel_values=pixel_values,
                    pixel_mask=binary_mask, 
                    input_ids=input_ids,
                    max_length=100,
                    num_beams=4,
                    repetition_penalty=1.5
                )
                
                text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                # Clean up prompt from output if present
                text = text.replace(prompt_text, "").strip()
                
                generated_report_parts.append(f"{organ.upper()}: {text}")
            
            full_generated_report = "\n".join(generated_report_parts)
            results.append({"patient_id": pid, "report": full_generated_report})
            
            # Simple print to check progress
            print(f"\n--- {pid} ---\n{full_generated_report}\n")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, "generated_reports.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vision_encoder_path', type=str, default='')
    parser.add_argument('--decoder_model', type=str, default='microsoft/biogpt')
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()
    evaluate(args)
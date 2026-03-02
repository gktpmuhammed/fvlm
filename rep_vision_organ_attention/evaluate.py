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
import nltk
import math
import csv
import matplotlib
matplotlib.use('Agg') # Fix for headless servers
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# NLTK Setup
try:
    nltk.data.find('wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, Transposed, EnsureChannelFirstd

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)

from medical_vlm import MedicalVLM
from train import get_organ_ids_for_key, ALL_TARGET_KEYS, build_transforms

# --- DATASET ---

class EvalDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform, subset_size=None):
        self.df = pd.read_csv(csv_file)
        # Assuming we evaluate on the validation set
        self.df = self.df[self.df['split'] == 'validation'].reset_index(drop=True)
        if subset_size:
            self.df = self.df.head(subset_size)
            print(f"Subset enabled: Evaluating on {len(self.df)} samples.")
            
        self.transform = transform
        self.tokenizer = tokenizer
        
    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            mask_path = row['image_path'].replace('images', 'masks')
            data = self.transform({'image': row['image_path'], 'mask': mask_path})
            
            img = data['image']
            if hasattr(img, 'as_tensor'): img = img.as_tensor().float()
            elif not isinstance(img, torch.Tensor): img = torch.from_numpy(img).float()
            
            mask = data['mask']
            if hasattr(mask, 'as_tensor'): mask = mask.as_tensor()
            elif not isinstance(mask, torch.Tensor): mask = torch.from_numpy(mask)
            
            return {
                'pixel_values': img,
                'full_mask': mask,
                'patient_id': os.path.basename(row['image_path']).split('.')[0]
            }
        except Exception as e:
            print(f"Error loading {row['image_path']}: {e}")
            return None

# --- EVALUATION LOGIC ---

def evaluate(args):
    print(f"--- Starting Evaluation ---")
    print(f"Model: {args.decoder_model}")
    print(f"Dataset: {args.csv_file}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Model
    model = MedicalVLM(args.vision_encoder_path, args.decoder_model, queries_per_organ=args.queries_per_organ)
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
        
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded weights. Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    
    model.cuda()
    model.eval()
    
    # 3. Setup Data
    transform = build_transforms()
    ds = EvalDataset(args.csv_file, model.tokenizer, transform, args.subset_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    
    # 4. Load Reference Text (JSON)
    # This is where the Ground Truth comes from
    with open(args.json_file, 'r') as f:
        ref_json = json.load(f)

    # Storage
    # Global: Concatenated strings per patient
    full_predictions = []
    full_references = []
    patient_ids = []
    
    # Organ-Specific: Lists of individual organ strings
    organ_specific_data = defaultdict(lambda: {'preds': [], 'refs': []})
    
    device = next(model.parameters()).device
    
    print("Generating Reports...")
    with torch.no_grad():
        for batch in tqdm(dl):
            if batch is None: continue
            
            pixel_values = batch['pixel_values'].to(device)
            full_mask = batch['full_mask'].to(device)
            pid = batch['patient_id'][0]
            
            # Prepare inputs
            mask_stack = []
            prompts = []
            
            for key in ALL_TARGET_KEYS:
                p_text = f"Describe {key}: "
                prompts.append(p_text)
                
                tids = get_organ_ids_for_key(key)
                if len(tids) > 0:
                    m = torch.zeros_like(full_mask)
                    for t in tids: m[full_mask == t] = 1.0
                else:
                    m = torch.zeros_like(full_mask)
                mask_stack.append(m)
            
            organ_masks = torch.stack(mask_stack, dim=1).float()
            
            prompt_inputs = model.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(device)
            
            # Prepend BOS token (decoder_start_token_id) to the prompted input_ids
            # to match the shifted labels used during training.
            batch_size_prompt = prompt_inputs.input_ids.shape[0]
            bos_token_id = model.model.config.decoder_start_token_id
            bos_tensor = torch.full((batch_size_prompt, 1), bos_token_id, dtype=torch.long, device=device)
            bos_attn = torch.ones((batch_size_prompt, 1), dtype=torch.long, device=device)
            
            decoder_input_ids = torch.cat([bos_tensor, prompt_inputs.input_ids], dim=1)
            decoder_attention_mask = torch.cat([bos_attn, prompt_inputs.attention_mask], dim=1)
            
            # Generate
            try:
                outputs = model.generate(
                    pixel_values=pixel_values,
                    organ_masks=organ_masks,
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attention_mask,
                    max_length=120,
                    num_beams=4,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=3
                )
                
                decoded = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # --- DATA COLLECTION ---
                # Retrieve Reference for this specific patient ID
                base_id = pid.replace('.nii.gz', '').replace('.nii', '')
                p_ref_dict = {}
                
                # Try exact match or substring match for ID
                if base_id in ref_json: 
                    p_ref_dict = ref_json[base_id]
                elif '_' in base_id:
                    short = base_id.rsplit('_', 1)[0]
                    if short in ref_json: 
                        p_ref_dict = ref_json[short]
                
                patient_pred_concat = ""
                patient_ref_concat = ""
                
                for key, text, p_text in zip(ALL_TARGET_KEYS, decoded, prompts):
                    # Clean Prediction (Remove Prompt)
                    clean_pred = text.replace(p_text, "").strip()
                    
                    # Get specific organ reference from JSON
                    ref_sent = p_ref_dict.get(key, "")
                    if not ref_sent:
                        ref_sent = p_ref_dict.get(key.lower(), "")
                    
                    # Store if Valid Reference Exists
                    if ref_sent and len(ref_sent) > 2:
                        organ_specific_data[key]['preds'].append(clean_pred)
                        organ_specific_data[key]['refs'].append(ref_sent)
                        
                        # Build Concatenated Report
                        if clean_pred:
                            patient_pred_concat += f"{key.upper()}: {clean_pred}\n"
                        patient_ref_concat += f"{key.upper()}: {ref_sent}\n"

                # Store Full Report
                if patient_ref_concat.strip():
                    full_predictions.append(patient_pred_concat.strip())
                    full_references.append(patient_ref_concat.strip())
                    patient_ids.append(pid)
                
            except Exception as e:
                print(f"Skipping {pid} error: {e}")
                continue

    # Save Generated Text
    out_csv = os.path.join(args.output_dir, "generated_reports.csv")
    df = pd.DataFrame({
        'patient_id': patient_ids, 
        'prediction': full_predictions, 
        'reference': full_references
    })
    df.to_csv(out_csv, index=False)
    
    print("\n" + "-"*50)
    print(f"Full reports saved to: {out_csv}")
    print(f"Evaluation Complete. Please run radeval_metrics.py on the output CSV.")
    print("-"*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--vision_encoder_path", type=str, default="/home/muhammedg/fvlm/checkpoints/model.pth", help="Path to pre-trained ViT")
    parser.add_argument("--decoder_model", type=str, default="gpt2", help="Decoder model to evaluate (gpt2 or GanjinZero/biobart-v2-base)")
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data_sym/image_first_dataset.csv')
    parser.add_argument('--json_file', type=str, default='../../data_sym/combined_desc_conc_v2.json')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--subset_size', type=int, default=None)
    parser.add_argument('--queries_per_organ', type=int, default=8)
    
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Allow shell override
    evaluate(args)
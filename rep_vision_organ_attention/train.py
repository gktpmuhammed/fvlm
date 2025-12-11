#!/usr/bin/env python3
import sys
import os
import logging
import json
import pandas as pd
import torch
import numpy as np
import argparse
import SimpleITK as sitk
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, Transposed, EnsureChannelFirstd

# Fix path for lavis
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from medical_vlm import MedicalVLM
import wandb

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "thesis"
os.environ["WANDB_ENTITY"] = "gktp-thesis"

# --- CONFIGURATION ---
FULL_BODY_ID = -999

# The fixed list of targets (Organs + Conclusion)
# The model will output N embeddings per patient in one pass.
ALL_TARGET_KEYS = [
    'Conclusion', # The "Whole Body" Summary
    'brain', 'face', 'lung', 'heart', 'aorta', 
    'esophagus', 'trachea', 'spine', 'rib', 'sternum',
    'liver', 'gallbladder', 'stomach', 'pancreas', 'spleen', 'kidney', 'adrenal',
    'colon', 'intestine', 'bladder', 'prostate', 'uterus',
    'humerus', 'scapula', 'clavicula', 'femur', 'hip', 'sacrum',
    'gluteus', 'iliopsoas', 'autochthon'
]

# --- MAPPING LOGIC ---
def get_organ_ids_for_key(report_key):
    key = report_key.lower().strip()
    
    if "conclusion" in key or "impression" in key: return [FULL_BODY_ID]

    if "lung" in key: return [10, 11, 12, 13, 14] 
    if "heart" in key: return [51, 61] 
    if "kidney" in key: return [2, 3] 
    if "liver" in key: return [5]
    if "gallbladder" in key: return [4]
    if "pancreas" in key: return [7]
    if "spleen" in key: return [1]
    if "stomach" in key: return [6]
    if "esophagus" in key: return [15]
    if "trachea" in key: return [16]
    if "colon" in key: return [20]
    if "aorta" in key: return [52]
    if "brain" in key: return [90]
    
    if "face" in key or "skull" in key: return [91]
    if "humerus" in key: return [69, 70] 
    if "scapula" in key: return [71, 72] 
    if "clavicula" in key: return [73, 74] 
    if "femur" in key: return [75, 76] 
    if "hip" in key: return [77, 78] 
    if "sacrum" in key: return [25]
    if "rib" in key: return list(range(92, 116)) 
    if "sternum" in key: return [116]
    
    if "gluteus" in key: return [80, 81, 82, 83, 84, 85]
    if "iliopsoas" in key: return [88, 89]
    if "autochthon" in key: return [86, 87]

    return []

@dataclass
class OrganCollator:
    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features: raise ValueError("Empty batch")
        
        # Stack Patients
        # pixel_values: (Batch, 1, D, H, W)
        pixel_values = torch.stack([f['pixel_values'] for f in features])
        
        # organ_masks: (Batch, N_Targets, D, H, W)
        organ_masks = torch.stack([f['organ_masks'] for f in features])
        
        # labels: (Batch, N_Targets, Seq_Len)
        labels = torch.stack([f['labels'] for f in features])
        
        return {'pixel_values': pixel_values, 'organ_masks': organ_masks, 'labels': labels}

def build_transforms():
    return Compose([
        LoadImaged(keys=['image', 'mask'], reader='ITKReader', image_only=True),
        EnsureChannelFirstd(keys=['image', 'mask']),
        Transposed(keys=['image', 'mask'], indices=(0, 3, 2, 1)),
        ScaleIntensityRanged(keys=['image'], a_min=-1150, a_max=350, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=['image', 'mask'], spatial_size=(112, 256, 352), mode='constant', constant_values=0),
        CenterSpatialCropd(keys=['image', 'mask'], roi_size=(112, 256, 352)),
    ])

class OnePassOrganDataset(Dataset):
    def __init__(self, csv_file, json_file, tokenizer, transform, max_length=128, subset_size=None, split='training'):
        print(f"--- Loading One-Pass Dataset ({split}) ---")
        
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        if subset_size: self.df = self.df.head(subset_size)

        # Load Single JSON
        with open(json_file, 'r') as f: 
            self.reports_json = json.load(f)

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.target_keys = ALL_TARGET_KEYS
        
        # Filter patients that exist in JSON
        self.valid_patients = []
        for _, row in self.df.iterrows():
            fname = os.path.basename(row['image_path'])
            base_id = fname.replace('.nii.gz', '').replace('.nii', '')
            
            # ID Matching
            target_pid = None
            if base_id in self.reports_json: 
                target_pid = base_id
            elif '_' in base_id:
                short_id = base_id.rsplit('_', 1)[0]
                if short_id in self.reports_json: 
                    target_pid = short_id
            
            if target_pid:
                self.valid_patients.append({
                    'image_path': row['image_path'],
                    'mask_path': row['image_path'].replace('images', 'masks'),
                    'pid': target_pid
                })

        print(f" Found {len(self.valid_patients)} valid patients for training.")

    def __len__(self): return len(self.valid_patients)

    def __getitem__(self, idx):
        item = self.valid_patients[idx]
        try:
            if not os.path.exists(item['mask_path']): return None

            # 1. Load Image & Mask (ONCE per patient)
            data = self.transform({'image': item['image_path'], 'mask': item['mask_path']})
            
            # Tensor Conversion (MetaTensor Fix)
            img_data = data['image']
            if hasattr(img_data, 'as_tensor'): image_tensor = img_data.as_tensor().float()
            elif isinstance(img_data, torch.Tensor): image_tensor = img_data.float()
            else: image_tensor = torch.from_numpy(img_data).float()

            mask_data = data['mask']
            if hasattr(mask_data, 'as_tensor'): full_mask_tensor = mask_data.as_tensor()
            elif isinstance(mask_data, torch.Tensor): full_mask_tensor = mask_data
            else: full_mask_tensor = torch.from_numpy(mask_data)
            
            # 2. Iterate ALL Target Organs
            mask_stack = []
            label_stack = []
            
            patient_data = self.reports_json.get(item['pid'], {})
            
            for key in self.target_keys:
                target_ids = get_organ_ids_for_key(key)
                
                # A. Prepare MASK
                if len(target_ids) == 0:
                    binary_mask = torch.zeros_like(full_mask_tensor)
                elif target_ids == [FULL_BODY_ID]:
                    # Conclusion -> Full Mask
                    binary_mask = torch.ones_like(full_mask_tensor)
                else:
                    # Specific Organ
                    binary_mask = torch.zeros_like(full_mask_tensor)
                    for tid in target_ids:
                        binary_mask[full_mask_tensor == tid] = 1.0
                
                mask_stack.append(binary_mask)
                
                # B. Prepare TEXT (Single Source)
                text = patient_data.get(key, "").strip()
                
                # C. Tokenize
                if len(text) > 3:
                    prompt = f"Describe {key}: " if key != "Conclusion" else "Conclusion: "
                    full_input = prompt + text
                    
                    tokens = self.tokenizer(
                        full_input, 
                        max_length=self.max_length, 
                        padding='max_length', 
                        truncation=True, 
                        return_tensors='pt'
                    )['input_ids'].squeeze(0)
                    
                    tokens[tokens == self.tokenizer.pad_token_id] = -100
                else:
                    # MISSING DATA -> PAD with -100
                    tokens = torch.full((self.max_length,), -100, dtype=torch.long)
                
                label_stack.append(tokens)

            # Stack tensors
            return {
                'pixel_values': image_tensor,
                'organ_masks': torch.stack(mask_stack).float(),
                'labels': torch.stack(label_stack)
            }

        except Exception as e:
            logger.error(f"Error {e}")
            return None

def main(args):
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    
    model = MedicalVLM(
        vision_encoder_path=args.vision_encoder_path,
        decoder_model_name=args.decoder_model
    )

    transform = build_transforms()
    
    train_dataset = OnePassOrganDataset(
        args.csv_file, args.json_file, model.tokenizer, transform, 
        args.max_length, args.subset_size, 'training'
    )
    val_dataset = OnePassOrganDataset(
        args.csv_file, args.json_file, model.tokenizer, transform, 
        args.max_length, args.subset_size, 'validation'
    )

    run_name = f"{args.decoder_model.split('/')[-1]}_onepass_organ_guided_attention"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,
        report_to="wandb",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size, # Recommended: 1
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8, 
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=400,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=OrganCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()
    model.save_pretrained(f"{args.output_dir}/final_model")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder_model', type=str, default='gpt2')
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv')
    parser.add_argument('--json_file', type=str, default='/home/muhammedg/fvlm/data/combined_desc_conc.json')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/medical_vlm')
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=4) # Low batch size for One-Pass
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--subset_size', type=int, default=None)
    
    args = parser.parse_args()
    main(args)
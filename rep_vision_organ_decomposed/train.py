#!/usr/bin/env python3
import sys
import os
import logging
import json # Added
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Restored WandB Settings
os.environ["WANDB_PROJECT"] = "thesis"
os.environ["WANDB_ENTITY"] = "gktp-thesis"

def get_organ_ids_for_key(report_key):
    """
    Maps report keys to TotalSegmentator Integer IDs.
    """
    key = report_key.lower().strip()
    
    # 1. SOFT TISSUE / ORGANS
    if "lung" in key: return [10, 11, 12, 13, 14] # All lobes
    if "heart" in key: return [51, 61] # Heart + Left Atrial Appendage
    if "kidney" in key: return [2, 3, 23, 24] # Right, Left, Cysts
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
    
    # 2. BONE / SKELETAL
    if "face" in key or "skull" in key: return [91]
    if "humerus" in key: return [69, 70] # Left, Right
    if "scapula" in key: return [71, 72] # Left, Right
    if "clavicula" in key: return [73, 74] # Left, Right
    if "femur" in key: return [75, 76] # Left, Right
    if "hip" in key: return [77, 78] # Left, Right
    if "sacrum" in key: return [25]
    
    # Ribs: TotalSegmentator has IDs 92-115 for individual ribs
    if "rib" in key: return list(range(92, 116)) 
    
    # 3. MUSCLES
    # Gluteus: Maximus (80,81), Medius (82,83), Minimus (84,85)
    if "gluteus" in key: return [80, 81, 82, 83, 84, 85]
    # Iliopsoas: Left (88), Right (89)
    if "iliopsoas" in key: return [88, 89]
    # Autochthon: Left (86), Right (87)
    if "autochthon" in key: return [86, 87]

    return []

@dataclass
class OrganCollator:
    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features: raise ValueError("Empty batch")
        pixel_values = torch.stack([f['pixel_values'] for f in features])
        pixel_masks = torch.stack([f['pixel_mask'] for f in features]) # Added
        labels = torch.stack([f['labels'] for f in features])
        return {'pixel_values': pixel_values, 'pixel_mask': pixel_masks, 'labels': labels}

def build_transforms():
    # Modified to include mask loading and alignment
    return Compose([
        LoadImaged(keys=['image', 'mask'], reader='ITKReader', image_only=True),
        EnsureChannelFirstd(keys=['image', 'mask']),
        Transposed(keys=['image', 'mask'], indices=(0, 3, 2, 1)),
        ScaleIntensityRanged(keys=['image'], a_min=-1150, a_max=350, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=['image', 'mask'], spatial_size=(112, 256, 352), mode='constant', constant_values=0),
        CenterSpatialCropd(keys=['image', 'mask'], roi_size=(112, 256, 352)),
    ])

class OrganReportDataset(Dataset):
    def __init__(self, csv_file, json_file, tokenizer, transform, max_length=512, subset_size=None, split='training'):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        with open(json_file, 'r') as f:
            self.reports_json = json.load(f)
            
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.samples = []

        # Flatten dataset logic
        for _, row in self.df.iterrows():
            if subset_size and len(self.samples) >= subset_size: break
            fname = os.path.basename(row['image_path'])
            # Assuming filename format matches decomposed json keys
            pid = fname.replace('.nii.gz', '').replace('.nii', '')
            pid = pid.rsplit('_', 1)[0]  # need to remove last part from underscore if any
            if pid in self.reports_json:
                for key, text in self.reports_json[pid].items():
                    target_ids = get_organ_ids_for_key(key)
                    if target_ids and len(str(text)) > 5:
                        self.samples.append({
                            'image_path': row['image_path'],
                            # Assuming mask path convention:
                            'mask_path': row['image_path'].replace('images', 'masks'), 
                            'organ_key': key,
                            'target_ids': target_ids,
                            'text': text
                        })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            if not os.path.exists(item['mask_path']): return None

            data = self.transform({'image': item['image_path'], 'mask': item['mask_path']})
            
            # --- FIX: Handle MONAI MetaTensor ---
            # image_tensor = torch.from_numpy(data['image']).float()  <-- THIS WAS THE ERROR
            
            # Check if it's already a tensor (MetaTensor is a Tensor)
            img_data = data['image']
            if hasattr(img_data, 'as_tensor'):
                image_tensor = img_data.as_tensor().float()
            elif isinstance(img_data, torch.Tensor):
                image_tensor = img_data.float()
            else:
                image_tensor = torch.from_numpy(img_data).float()

            # Same logic for Mask
            mask_data = data['mask']
            if hasattr(mask_data, 'as_tensor'):
                mask_tensor = mask_data.as_tensor()
            elif isinstance(mask_data, torch.Tensor):
                mask_tensor = mask_data
            else:
                mask_tensor = torch.from_numpy(mask_data)
            # ------------------------------------

            # Create binary mask for specific organ
            binary_mask = torch.zeros_like(mask_tensor)
            for tid in item['target_ids']:
                binary_mask[mask_tensor == tid] = 1.0
            
            text_input = f"Describe the {item['organ_key']}: {item['text']}"
            
            encoding = self.tokenizer(text_input, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            labels = encoding['input_ids'].squeeze(0)
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'pixel_values': image_tensor,
                'pixel_mask': binary_mask.float(),
                'labels': labels
            }
        except Exception as e:
            logger.error(f"Error processing {item['image_path']}: {e}")
            return None

def main(args):
    # Setup logging same as before
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    
    model = MedicalVLM(
        vision_encoder_path=args.vision_encoder_path,
        decoder_model_name=args.decoder_model,
        use_qformer=args.use_qformer,
        num_query_tokens=args.num_query_tokens
    )

    transform = build_transforms()
    # Updated Dataset Class
    train_dataset = OrganReportDataset(args.csv_file, args.json_file, model.tokenizer, transform, args.max_length, args.subset_size, 'training')
    val_dataset = OrganReportDataset(args.csv_file, args.json_file, model.tokenizer, transform, args.max_length, args.subset_size, 'validation')

    run_name = f"{args.decoder_model.split('/')[-1]}_organ_guided_{'with_qformer' if args.use_qformer else 'no_qformer'}_bs{args.batch_size}_ep{args.num_epochs}"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,
        report_to="wandb",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8, # Restored
        learning_rate=2e-4, # Restored
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
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
        data_collator=OrganCollator(), # Updated Collator
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()
    model.save_pretrained(f"{args.output_dir}/final_model")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Keeping arguments consistent + new json_file
    parser.add_argument('--decoder_model', type=str, default='microsoft/biogpt')
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv')
    parser.add_argument('--json_file', type=str, default='/home/muhammedg/fvlm/data/combined_desc_conc.json', help="Path to decomposed reports JSON")
    parser.add_argument('--output_dir', type=str, default='./checkpoints/medical_vlm')
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--subset_size', type=int, default=None)
    parser.add_argument('--logging_steps', type=int, default=5)
    parser.add_argument('--use_qformer', action='store_true')
    parser.add_argument('--num_query_tokens', type=int, default=32)
    
    args = parser.parse_args()
    main(args)
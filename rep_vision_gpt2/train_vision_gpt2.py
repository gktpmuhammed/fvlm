#!/usr/bin/env python3
"""
Training Script for Medical Vision-GPT2
"""

import sys
import os

# ------------------------------------------------------------------
# FIX: Insert parent directory at the BEGINNING of sys.path
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import torch
import numpy as np
import argparse
import logging
import SimpleITK as sitk
from dataclasses import dataclass
from typing import Dict, List
from torch.utils.data import Dataset

from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    SpatialPadd,
    CenterSpatialCropd,
    Transposed,
    EnsureChannelFirstd,
)

from medical_vision_gpt2 import MedicalVisionGPT2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure CUDA is available
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

@dataclass
class VisionEncoderDecoderCollator:
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        features = [f for f in features if f is not None]
        if not features:
            raise ValueError("All features in batch were None!")
            
        pixel_values = torch.stack([f['pixel_values'] for f in features])
        labels = torch.stack([f['labels'] for f in features])
        return {'pixel_values': pixel_values, 'labels': labels}

def build_transforms():
    return Compose([
        LoadImaged(keys=['image'], reader='ITKReader', image_only=True),
        EnsureChannelFirstd(keys=['image']),
        Transposed(keys=['image'], indices=(0, 3, 2, 1)),
        ScaleIntensityRanged(
            keys=['image'], a_min=-1150, a_max=350, b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=['image'], spatial_size=(112, 256, 352), mode='constant', constant_values=0),
        CenterSpatialCropd(keys=['image'], roi_size=(112, 256, 352)),
    ])

class MedicalReportDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform, max_length=512, subset_size=None, split='training'):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['split'] == split].reset_index(drop=True)
        
        if subset_size:
            self.data = self.data.head(subset_size)
            
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            image_dict = self.transform({'image': row['image_path']})
            image = image_dict['image']
            if isinstance(image, sitk.Image):
                image = sitk.GetArrayFromImage(image)
            image_tensor = torch.from_numpy(np.array(image)).float()
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

            text = f"{row['findings']} {row['impressions']}"
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            labels = encoding['input_ids'].squeeze(0)
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {'pixel_values': image_tensor, 'labels': labels}
            
        except Exception as e:
            logger.error(f"Error processing {row['image_path']}: {e}")
            return None

def main(args):
    logger.info("Initializing Model...")
    
    model = MedicalVisionGPT2(
        vision_encoder_path=args.vision_encoder_path,
        decoder_model_name="gpt2"
    )

    transform = build_transforms()
    
    logger.info("Loading Datasets...")
    train_dataset = MedicalReportDataset(
        csv_file=args.csv_file,
        tokenizer=model.tokenizer,
        transform=transform,
        max_length=args.max_length,
        split='training',
        subset_size=args.subset_size
    )
    
    val_subset = int(args.subset_size/5) if args.subset_size else None
    val_dataset = MedicalReportDataset(
        csv_file=args.csv_file,
        tokenizer=model.tokenizer,
        transform=transform,
        max_length=args.max_length,
        split='validation',
        subset_size=val_subset
    )
    
    logger.info(f"Train Size: {len(train_dataset)}")
    logger.info(f"Val Size: {len(val_dataset)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8, 
        
        learning_rate=2e-4, 
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=VisionEncoderDecoderCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    logger.info("Starting Training...")
    trainer.train()
    
    logger.info(f"Saving model to {args.output_dir}/final_model")
    model.save_pretrained(f"{args.output_dir}/final_model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/vision_gpt2')
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--subset_size', type=int, default=None)
    args = parser.parse_args()
    main(args)
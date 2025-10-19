#!/usr/bin/env python3

"""
Train Medical BLIP-2 Model with BioGPT Decoder
Updated with correct 3D transforms (112, 256, 352) and cosine scheduler
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import Trainer, TrainingArguments
from medical_blip2_biogpt import MedicalBLIP2BioGPT
import argparse
import os
from pathlib import Path
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Transposed,
    ScaleIntensityRanged,
    SpatialPadd,
    CenterSpatialCropd,
)
import SimpleITK as sitk


def build_transforms():
    """Transform pipeline for 3D medical images - CORRECTED"""
    return Compose([
        LoadImaged(keys=['image'], reader='ITKReader', image_only=True),
        EnsureChannelFirstd(keys=['image']),
        Transposed(keys=['image'], indices=(0, 3, 2, 1)),
        ScaleIntensityRanged(
            keys=['image'],
            a_min=-1150,
            a_max=350,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        SpatialPadd(keys=['image'], spatial_size=(112, 256, 352), mode='constant', constant_values=0),
        CenterSpatialCropd(keys=['image'], roi_size=(112, 256, 352)),
    ])


class MedicalReportDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform, max_length=256, subset_size=None, split='training'):
        df = pd.read_csv(csv_file)
        df = df[df['split'] == split].reset_index(drop=True)

        if subset_size and subset_size > 0:
            df = df.head(subset_size)

        self.data = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load 3D medical image
        image_dict = self.transform({'image': row['image_path']})
        image = image_dict['image']

        if isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)
        image = torch.from_numpy(np.array(image)).float()

        # Ensure proper dimensions (B, C, D, H, W) - add channel if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Prepare text - combine findings and impressions
        report_text = f"{row['findings']} {row['impressions']}"

        encoding = self.tokenizer(
            report_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'pixel_values': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0),
        }


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def main(args):
    print("="*80)
    print("MEDICAL BLIP-2 TRAINING WITH BIOGPT (3D ViT)")
    print("="*80)

    # Initialize model
    print(f"\nInitializing Medical BLIP-2 with BioGPT...")
    print(f"  Vision encoder: {args.vision_encoder_path}")
    print(f"  Image size: {args.image_size}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Use Q-Former: {args.use_qformer}")

    # Parse tuple arguments
    image_size = tuple(map(int, args.image_size.split(',')))
    patch_size = tuple(map(int, args.patch_size.split(',')))

    model = MedicalBLIP2BioGPT(
        vision_encoder_path=args.vision_encoder_path,
        decoder_model_name="microsoft/biogpt",
        image_size=image_size,  # 3D: (D, H, W)
        patch_size=patch_size,  # 3D patches
        num_query_tokens=args.num_query_tokens,
        freeze_encoder=args.freeze_vision,
        freeze_decoder_base=args.freeze_decoder_base,
        use_qformer=args.use_qformer,
    )

    # Prepare datasets
    print(f"\nLoading datasets...")
    print(f"  CSV file: {args.csv_file}")

    transform = build_transforms()

    train_dataset = MedicalReportDataset(
        csv_file=args.csv_file,
        tokenizer=model.tokenizer,
        transform=transform,
        max_length=args.max_length,
        subset_size=args.subset_size,
        split='training',
    )

    val_dataset = MedicalReportDataset(
        csv_file=args.csv_file,
        tokenizer=model.tokenizer,
        transform=transform,
        max_length=args.max_length,
        subset_size=args.val_subset_size,
        split='validation',
    )

    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Data collator
    data_collator = DataCollator(model.tokenizer)

    # Training arguments with cosine scheduler
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        #  Learning rate strategy (cosine scheduler)
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,           # Dynamic warmup based on total steps
        lr_scheduler_type=args.lr_scheduler_type,  # Cosine decay by default
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,

        # Evaluation strategy
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,

        # Logging
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=args.logging_steps,

        # Performance
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        seed=42,
        remove_unused_columns=False,
        report_to="tensorboard" if args.use_tensorboard else "none",
    )

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LR scheduler: {args.lr_scheduler_type}")
    print(f"  Warmup ratio: {args.warmup_ratio}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Max grad norm: {args.max_grad_norm}")
    print(f"  FP16: {args.fp16}")
    print(f"  Output dir: {args.output_dir}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print(f"\n{'='*80}")
    print("Starting Training...")
    print(f"{'='*80}\n")

    # Train
    trainer.train()

    # Save final model
    print(f"\nSaving final model to {args.output_dir}/final_model")
    model.save_pretrained(f"{args.output_dir}/final_model")

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medical BLIP-2 with BioGPT")

    # Model arguments
    parser.add_argument('--vision_encoder_path', type=str, 
                       default='/home/muhammedg/fvlm/checkpoints/model.pth',
                       help='Path to pretrained 3D ViT checkpoint')
    parser.add_argument('--image_size', type=str, default='112,256,352',
                       help='3D image size as D,H,W')
    parser.add_argument('--patch_size', type=str, default='16,16,32',
                       help='3D patch size as D,H,W')
    parser.add_argument('--num_query_tokens', type=int, default=32,
                       help='Number of query tokens for Q-Former')
    parser.add_argument('--freeze_vision', action='store_true', default=True,
                       help='Freeze vision encoder')
    parser.add_argument('--freeze_decoder_base', action='store_true', default=True,
                       help='Freeze decoder base (unfreeze last 3 layers)')
    parser.add_argument('--use_qformer', action='store_true', default=True,
                       help='Use Q-Former compression (disable for direct connection)')

    # Data arguments
    parser.add_argument('--csv_file', type=str, 
                       default='/home/muhammedg/fvlm/data/image_first_dataset.csv',
                       help='Path to CSV file with "split" column (training/validation)')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length for tokenization')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Use subset of training data for testing')
    parser.add_argument('--val_subset_size', type=int, default=None,
                       help='Use subset of validation data for testing')

    # Training arguments
    parser.add_argument('--output_dir', type=str, default='./blip2_biogpt_output',
                       help='Output directory for checkpoints')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient accumulation steps (effective batch = batch_size * this)')

    # Learning rate arguments (with cosine scheduler)
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Peak learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio (fraction of total steps for warmup)')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                       choices=['linear', 'cosine', 'cosine_with_restarts', 
                               'polynomial', 'constant', 'constant_with_warmup'],
                       help='Learning rate scheduler type (default: cosine)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay for AdamW optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')

    # Logging and evaluation
    parser.add_argument('--logging_steps', type=int, default=50,
                       help='Log every N steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save checkpoint every N steps')

    # Performance
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision (FP16) training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--use_tensorboard', action='store_true',
                       help='Enable tensorboard logging')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)

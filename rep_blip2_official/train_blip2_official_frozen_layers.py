#!/usr/bin/env python3
"""
Train Medical BLIP-2 with ENCODER FINE-TUNING
Supports differential learning rates for optimal adaptation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import Trainer, TrainingArguments
from medical_blip2_official import MedicalBLIP2Official
import argparse
import os
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
    """Transform pipeline for 3D medical images"""
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
    def __init__(self, csv_file, transform, split='training', subset_size=None):
        df = pd.read_csv(csv_file)
        df = df[df['split'] == split].reset_index(drop=True)

        if subset_size and subset_size > 0:
            df = df.head(subset_size)

        self.data = df
        self.transform = transform

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

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Prepare text
        text_output = f"{row['findings']} {row['impressions']}"

        return {
            'image': image,
            'text_output': text_output,
        }


class BLIP2Collator:
    """Data collator for BLIP-2 training"""
    def __call__(self, batch):
        images = torch.stack([item['image'] for item in batch])
        text_outputs = [item['text_output'] for item in batch]

        return {
            'image': images,
            'text_output': text_outputs,
        }


class BLIP2Trainer(Trainer):
    """Custom trainer for BLIP-2 with differential LR support"""

    def create_optimizer(self):
        """
        OVERRIDE: Create optimizer with differential learning rates
        """
        if self.optimizer is None:
            # Get parameter groups with different LRs
            param_groups = self.model.get_param_groups_with_lr(
                lr_vision=self.args.encoder_lr,      # Very low for encoder
                lr_qformer=self.args.qformer_lr,     # High for Q-Former
                lr_proj=self.args.projection_lr,     # High for projection
                lr_opt=self.args.learning_rate,      # Normal for OPT
            )

            print("\n" + "="*80)
            print("OPTIMIZER CONFIGURATION (Differential Learning Rates):")
            print("="*80)
            for group in param_groups:
                num_params = sum(p.numel() for p in group['params'])
                print(f"  {group['name']:<20}: LR={group['lr']:.2e}  Params={num_params:,}")
            print("="*80 + "\n")

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # Remove learning_rate from kwargs since we're using custom LRs
            optimizer_kwargs.pop('lr', None)

            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss for BLIP-2 training"""
        outputs = model(
            image=inputs['image'],
            text_output=inputs['text_output'],
        )

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def main(args):
    print("="*80)
    print("MEDICAL BLIP-2 TRAINING (WITH ENCODER FINE-TUNING)")
    print("="*80)

    # Initialize model
    print(f"\nInitializing Medical BLIP-2...")
    print(f"  Vision encoder: {args.vision_encoder_path}")
    print(f"  OPT model: {args.opt_model}")
    print(f"  Image size: {args.image_size}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Query tokens: {args.num_query_tokens}")
    print(f"  Unfrozen encoder layers: {args.num_unfrozen_layers}")

    image_size = tuple(map(int, args.image_size.split(',')))
    patch_size = tuple(map(int, args.patch_size.split(',')))

    model = MedicalBLIP2Official(
        vision_encoder_path=args.vision_encoder_path,
        opt_model=args.opt_model,
        image_size=image_size,
        patch_size=patch_size,
        num_query_tokens=args.num_query_tokens,
        freeze_vision=True,  # We'll selectively unfreeze
        freeze_opt=args.freeze_opt,
        num_unfrozen_layers=args.num_unfrozen_layers,  # NEW
    )

    # Prepare datasets
    print(f"\nLoading datasets...")
    print(f"  CSV file: {args.csv_file}")

    transform = build_transforms()

    train_dataset = MedicalReportDataset(
        csv_file=args.csv_file,
        transform=transform,
        split='training',
        subset_size=args.subset_size,
    )

    val_dataset = MedicalReportDataset(
        csv_file=args.csv_file,
        transform=transform,
        split='validation',
        subset_size=args.val_subset_size,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Data collator
    data_collator = BLIP2Collator()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Learning rate strategy
        learning_rate=args.learning_rate,  # For OPT
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,

        # Evaluation
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,  # Keep 2 checkpoints
        load_best_model_at_end=False,

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

    # Add custom LR arguments to training_args for our custom optimizer
    training_args.encoder_lr = args.encoder_lr
    training_args.qformer_lr = args.qformer_lr
    training_args.projection_lr = args.projection_lr

    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION:")
    print("="*80)
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"\nLearning Rates (Differential):")
    print(f"  Vision Encoder: {args.encoder_lr:.2e}  {'(frozen)' if args.num_unfrozen_layers == 0 else f'({args.num_unfrozen_layers} layers)'}")
    print(f"  Q-Former:       {args.qformer_lr:.2e}")
    print(f"  Projection:     {args.projection_lr:.2e}")
    print(f"  OPT Decoder:    {args.learning_rate:.2e}  {'(frozen)' if args.freeze_opt else ''}")
    print(f"\nScheduler:")
    print(f"  Type: {args.lr_scheduler_type}")
    print(f"  Warmup ratio: {args.warmup_ratio}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"\nOther:")
    print(f"  FP16: {args.fp16}")
    print(f"  Output dir: {args.output_dir}")
    print("="*80)

    # Initialize trainer
    trainer = BLIP2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print(f"\n{'='*80}")
    print("Starting Training...")
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    print(f"{'='*80}\n")

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    print(f"\nSaving final model to {args.output_dir}/final_model")
    model.save_pretrained(f"{args.output_dir}/final_model")

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medical BLIP-2 with Encoder Fine-tuning")

    # Model arguments
    parser.add_argument('--vision_encoder_path', type=str,
                       default='/home/muhammedg/fvlm/checkpoints/model.pth',
                       help='Path to pretrained 3D ViT')
    parser.add_argument('--opt_model', type=str, default='facebook/opt-2.7b',
                       choices=['facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b'],
                       help='OPT model size')
    parser.add_argument('--image_size', type=str, default='112,256,352',
                       help='3D image size as D,H,W')
    parser.add_argument('--patch_size', type=str, default='16,16,32',
                       help='3D patch size as D,H,W')
    parser.add_argument('--num_query_tokens', type=int, default=32,
                       help='Number of query tokens for Q-Former')

    # NEW: Encoder fine-tuning arguments
    parser.add_argument('--num_unfrozen_layers', type=int, default=0,
                       help='Number of encoder layers to unfreeze (0=fully frozen, 6=last 6 layers)')
    parser.add_argument('--encoder_lr', type=float, default=1e-5,
                       help='Learning rate for vision encoder (if unfrozen)')
    parser.add_argument('--qformer_lr', type=float, default=1e-4,
                       help='Learning rate for Q-Former')
    parser.add_argument('--projection_lr', type=float, default=1e-4,
                       help='Learning rate for projection layer')

    parser.add_argument('--freeze_opt', action='store_true', default=True,
                       help='Freeze OPT decoder')

    # Data arguments
    parser.add_argument('--csv_file', type=str,
                       default='/home/muhammedg/fvlm/data/image_first_dataset.csv',
                       help='Path to CSV file')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Use subset of training data')
    parser.add_argument('--val_subset_size', type=int, default=None,
                       help='Use subset of validation data')

    # Training arguments
    parser.add_argument('--output_dir', type=str, default='./blip2_finetuned_encoder',
                       help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=15,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient accumulation steps')

    # Learning rate arguments
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Peak learning rate for OPT')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                       help='LR scheduler type')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm')

    # Logging
    parser.add_argument('--logging_steps', type=int, default=50,
                       help='Logging steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluation steps')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save steps')

    # Performance
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Dataloader workers')
    parser.add_argument('--use_tensorboard', action='store_true',
                       help='Use tensorboard')

    # Resume training
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)

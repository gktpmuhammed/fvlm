#!/usr/bin/env python3

"""
Train Medical BLIP-2 with Official LAVIS Architecture
Uses proven BLIP-2 components with your 3D medical ViT
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
from typing import Optional, Sequence, Union

# Hardcoded standardization values (edit these to use your computed mean/std).
# Set to None to disable standardization.
# Example: STANDARDIZE_MEAN = 0.45; STANDARDIZE_STD = 0.22
STANDARDIZE_MEAN: Optional[Union[float, Sequence[float]]] = -0.028866397216916084
STANDARDIZE_STD: Optional[Union[float, Sequence[float]]] = 0.5328696370124817


def build_transforms():
    """Transform pipeline for 3D medical images.

    Uses hardcoded STANDARDIZE_MEAN/STD defined above to optionally include a standardization
    transform at the end of the pipeline.
    """
    mean = STANDARDIZE_MEAN
    std = STANDARDIZE_STD

    transforms = [
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
    ]


    # Append standardization if requested (scaling removed per request)
    if mean is not None or std is not None:
        transforms.append(StandardizeAndScale(mean=mean, std=std))

    return Compose(transforms)


class StandardizeAndScale:
    """MONAI-style transform that applies dataset standardization using provided mean/std.

    The transform performs: image = (image - mean) / (std + eps)
    Scaling/jittering has been removed per request.
    """

    def __init__(self,
                 mean: Optional[Union[float, Sequence[float]]] = None,
                 std: Optional[Union[float, Sequence[float]]] = None,
                 eps: float = 1e-8):
        self.mean = np.array(mean, dtype=np.float32) if mean is not None else None
        self.std = np.array(std, dtype=np.float32) if std is not None else None
        self.eps = float(eps)

    def __call__(self, data):
        image = data['image']

        # Convert to numpy array and float32
        arr = np.array(image).astype(np.float32)

        # Apply standardization if mean/std provided
        if self.mean is not None:
            try:
                arr = (arr - self.mean) / (self.std + self.eps)
            except Exception:
                # Try broadcasting along channel dimension if shapes differ
                if hasattr(self.mean, 'ndim') and self.mean.ndim > 0:
                    mean = self.mean.reshape((1,) + self.mean.shape)
                else:
                    mean = self.mean
                if self.std is not None and hasattr(self.std, 'ndim') and self.std.ndim > 0:
                    std = self.std.reshape((1,) + self.std.shape)
                else:
                    std = self.std
                arr = (arr - mean) / (std + self.eps)

        data['image'] = arr
        return data


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

        # Ensure proper dimensions (B, C, D, H, W)
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
    """Custom trainer for BLIP-2"""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for BLIP-2 training
        """
        outputs = model(
            image=inputs['image'],
            text_output=inputs['text_output'],
        )

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        """
        Override to save only trainable parameters in checkpoints
        This reduces checkpoint size from 11GB to ~885MB
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save only trainable parameters (exclude frozen OPT model)
        if state_dict is None:
            state_dict = {
                k: v for k, v in self.model.state_dict().items() 
                if any(k.startswith(prefix) for prefix in 
                       ['visual_encoder', 'Qformer', 'query_tokens', 'opt_proj'])
            }
        
        # Save trainable weights
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save optimizer and scheduler (still needed for resuming)
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        
        # Save training state
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        
        print(f"✓ Checkpoint saved (trainable params only): {output_dir}")


def main(args):
    print("="*80)
    print("MEDICAL BLIP-2 TRAINING (OFFICIAL ARCHITECTURE)")
    print("="*80)

    # Initialize model
    print(f"\nInitializing Medical BLIP-2...")
    print(f"  Vision encoder: {args.vision_encoder_path}")
    print(f"  OPT model: {args.opt_model}")
    print(f"  Image size: {args.image_size}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Query tokens: {args.num_query_tokens}")

    image_size = tuple(map(int, args.image_size.split(',')))
    patch_size = tuple(map(int, args.patch_size.split(',')))

    model = MedicalBLIP2Official(
        vision_encoder_path=args.vision_encoder_path,
        opt_model=args.opt_model,
        image_size=image_size,
        patch_size=patch_size,
        num_query_tokens=args.num_query_tokens,
        freeze_vision=args.freeze_vision,
        freeze_opt=args.freeze_opt,
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
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,

        # Evaluation
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        load_best_model_at_end=False,
        # metric_for_best_model="loss",
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
    print(f"  FP16: {args.fp16}")
    print(f"  Output dir: {args.output_dir}")

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
    parser = argparse.ArgumentParser(description="Train Medical BLIP-2 (Official)")

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
    parser.add_argument('--freeze_vision', action='store_true', default=True,
                       help='Freeze vision encoder')
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
    parser.add_argument('--output_dir', type=str, default='./blip2_official_output',
                       help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient accumulation steps')

    # Learning rate arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Peak learning rate')
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

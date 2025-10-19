#!/usr/bin/env python3
"""
Training Script for Medical VisionEncoderDecoder - FIXED
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator
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
import SimpleITK as sitk
import numpy as np
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Import our model
from medical_vision_encoder_decoder import MedicalVisionEncoderDecoder


# ==================== Custom Data Collator ====================
@dataclass
class VisionEncoderDecoderCollator:
    """
    Custom data collator for VisionEncoderDecoderModel
    Handles pixel_values and labels without trying to tokenize them
    """
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Stack pixel_values
        pixel_values = torch.stack([f['pixel_values'] for f in features])

        # Stack labels
        labels = torch.stack([f['labels'] for f in features])

        return {
            'pixel_values': pixel_values,
            'labels': labels
        }


# ==================== Data Transforms ====================
def build_transforms():
    """Build MONAI transforms for medical images"""
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


# ==================== Dataset ====================
class MedicalReportDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform, max_length=512, subset_size=None, split='training'):
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

        # Load and transform image
        image_dict = self.transform({'image': row['image_path']})
        image = image_dict['image']

        if isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)
        image = torch.from_numpy(np.array(image)).float()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Prepare text
        report_text = f"{row['findings']} {row['impressions']}"

        # Tokenize
        encoding = self.tokenizer(
            report_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'pixel_values': image,
            'labels': encoding['input_ids'].squeeze(0)
        }


# ==================== Main Training ====================
def main(args):
    logger.info("="*80)
    logger.info("MEDICAL VISIONENCODERDECODER TRAINING")
    logger.info("="*80)

    # Initialize model with YOUR pretrained ViT
    logger.info("\nInitializing model...")
    model = MedicalVisionEncoderDecoder(
        vision_encoder_path=args.vision_encoder_path,
        decoder_model_name="microsoft/biogpt"
    )

    # Build transforms
    transform = build_transforms()

    # Create datasets
    logger.info("\nLoading datasets...")
    if args.subset_size:
        logger.info(f"Using subset: {args.subset_size} training samples")

    train_dataset = MedicalReportDataset(
        csv_file=args.csv_file,
        tokenizer=model.tokenizer,
        transform=transform,
        max_length=args.max_length,
        split='training',
        subset_size=args.subset_size
    )

    val_subset = int(args.subset_size / 4) if args.subset_size else None
    val_dataset = MedicalReportDataset(
        csv_file=args.csv_file,
        tokenizer=model.tokenizer,
        transform=transform,
        max_length=args.max_length,
        split='validation',
        subset_size=val_subset
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create custom data collator
    data_collator = VisionEncoderDecoderCollator()

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,

        # Batch size
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,

        # Learning rate
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',

        # Regularization
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Evaluation
        evaluation_strategy='steps',
        eval_steps=50,
        save_strategy='steps',
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',

        # Seq2Seq specific
        predict_with_generate=False,  # Disable for now due to complexity

        # Logging
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=10,

        # Optimization
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # Misc
        seed=42,
        remove_unused_columns=False,
    )

    # Initialize Seq2SeqTrainer with custom collator
    logger.info("\nInitializing Seq2SeqTrainer...")
    trainer = Seq2SeqTrainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=model.tokenizer,
        data_collator=data_collator,  # Use custom collator
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.01
            )
        ]
    )

    # Train
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    trainer.train()

    # Save final model
    logger.info("\nSaving final model...")
    model.save_pretrained(f'{args.output_dir}/final_model')

    logger.info("\n Training completed successfully!")
    logger.info(f"Model saved to: {args.output_dir}/final_model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Medical VisionEncoderDecoder')

    # Model config
    parser.add_argument('--vision_encoder_path', type=str,
                       default='/home/muhammedg/fvlm/checkpoints/model.pth',
                       help='Path to YOUR pretrained medical vision encoder')

    # Data config
    parser.add_argument('--csv_file', type=str,
                       default='/home/muhammedg/fvlm/data/image_first_dataset.csv',
                       help='Path to data CSV')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Use subset for debugging (e.g., 400)')

    # Training config
    parser.add_argument('--output_dir', type=str,
                       default='./checkpoints/vision_encoder_decoder',
                       help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Per-device batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')

    args = parser.parse_args()
    main(args)

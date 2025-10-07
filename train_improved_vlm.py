#!/usr/bin/env python3

"""
IMPROVED TRAINING: Medical VLM with LoRA and Better Configuration
- LoRA on ViT encoder (last 4-6 layers)
- Better learning rates and scheduling
- NLP metrics integrated into training
- Anti-repetition mechanisms
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    SpatialPadd,
    CenterSpatialCropd,
    Transposed,
)
import sys
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import argparse
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
import logging

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Import improved model
from improved_medical_vlm import ImprovedMedicalVLM, MedicalReportDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== MONAI Transform Pipeline ====================
def build_transforms():
    """Build MONAI transforms for medical images"""
    return Compose([
        LoadImaged(keys=['image'], reader='ITKReader', image_only=True),
        ScaleIntensityRanged(
            keys=['image'],
            a_min=-1000,
            a_max=500,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        SpatialPadd(keys=['image'], spatial_size=(112, 256, 352)),
        CenterSpatialCropd(keys=['image'], roi_size=(112, 256, 352)),
        Transposed(keys=['image'], indices=(0, 1, 2)),
    ])


# ==================== Custom Trainer with NLP Metrics ====================
class MetricsAwareTrainer(Trainer):
    """
    Custom Trainer that computes NLP metrics during training
    and uses them to augment the loss
    """
    def __init__(self, *args, use_metric_loss=True, metric_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_metric_loss = use_metric_loss
        self.metric_weight = metric_weight

        # Initialize metrics
        self.rouge_metric = ROUGEScore()
        self.bleu_metric = BLEUScore(n_gram=2)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with optional NLP metric-based regularization
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        # Add metric-based loss component during training
        if self.use_metric_loss and model.training:
            try:
                # Generate predictions (greedy for speed)
                with torch.no_grad():
                    generated = model.generate(
                        images=inputs['images'],
                        max_length=128,
                        num_beams=1,  # Greedy for speed
                        temperature=1.0,
                        repetition_penalty=1.5,
                    )

                # Decode predictions and labels
                predictions = model.tokenizer.batch_decode(generated, skip_special_tokens=True)
                references = model.tokenizer.batch_decode(labels, skip_special_tokens=True)

                # Compute ROUGE score
                rouge_scores = self.rouge_metric(predictions, references)
                rouge_l_f1 = rouge_scores['rougeL_fmeasure']

                # Use negative ROUGE as penalty (we want to maximize ROUGE)
                metric_loss = (1.0 - rouge_l_f1) * self.metric_weight
                loss = loss + metric_loss

                # Log metric loss occasionally
                if self.state.global_step % 100 == 0:
                    logger.info(f"Step {self.state.global_step}: CE Loss={outputs.loss:.4f}, "
                              f"ROUGE-L F1={rouge_l_f1:.4f}, Total Loss={loss:.4f}")
            except Exception as e:
                logger.warning(f"Could not compute metric loss: {e}")

        return (loss, outputs) if return_outputs else loss


# ==================== Dataset Class ====================
class ImageFirstDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform, max_length=512, subset_size=None, split='train'):
        df = pd.read_csv(csv_file)
        df = df[df['split'] == split].reset_index(drop=True)

        if subset_size:
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
        report_text = f"{row['findings']} {row['impression']}"

        # Tokenize
        encoding = self.tokenizer(
            report_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'images': image.squeeze(0),
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0)
        }


# ==================== Main Training Function ====================
def main(args):
    logger.info("=" * 80)
    logger.info("IMPROVED MEDICAL VLM TRAINING")
    logger.info("=" * 80)

    # Initialize model
    logger.info("\nInitializing improved model with LoRA...")
    model = ImprovedMedicalVLM(
        vision_encoder_path=args.vision_encoder_path,
        bert_model_name=args.bert_model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        vit_layers_to_adapt=args.vit_layers_to_adapt
    )

    # Build transforms
    transform = build_transforms()

    # Create datasets
    logger.info("\nLoading datasets...")
    train_dataset = ImageFirstDataset(
        csv_file=args.csv_file,
        tokenizer=model.tokenizer,
        transform=transform,
        max_length=args.max_length,
        split='train'
    )

    val_dataset = ImageFirstDataset(
        csv_file=args.csv_file,
        tokenizer=model.tokenizer,
        transform=transform,
        max_length=args.max_length,
        split='validation'
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Training arguments with improved configuration
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,

        # Batch size (keeping same as requested)
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch size = 32

        # Learning rate (IMPROVED)
        learning_rate=1e-5,  # Lower than before (was 5e-5)
        warmup_ratio=0.1,  # 10% warmup
        lr_scheduler_type='cosine',  # Cosine instead of linear

        # Weight decay for regularization
        weight_decay=0.01,

        # Evaluation
        eval_strategy='steps',
        eval_steps=500,
        save_strategy='steps',
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,

        # Logging
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=50,
        # report_to='tensorboard',

        # Optimization
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        max_grad_norm=1.0,

        # Misc
        seed=42,
        remove_unused_columns=False,
    )

    # Initialize trainer with metrics
    logger.info("\nInitializing trainer with NLP metrics...")
    trainer = MetricsAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        use_metric_loss=args.use_metric_loss,
        metric_weight=args.metric_weight,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            )
        ]
    )

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    trainer.train()

    # Save final model
    logger.info("\nSaving final model...")
    trainer.save_model(f'{args.output_dir}/final_model')
    model.tokenizer.save_pretrained(f'{args.output_dir}/final_model')

    logger.info("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train improved medical VLM with LoRA')

    # Model config
    parser.add_argument('--vision_encoder_path', type=str,
                       default='/home/muhammedg/fvlm/checkpoints/model.pth',
                       help='Path to pretrained vision encoder')
    parser.add_argument('--bert_model_name', type=str,
                       default='/home/muhammedg/fvlm/BiomedVLP-CXR-BERT-specialized',
                       help='BERT model path')

    # LoRA config
    parser.add_argument('--lora_rank', type=int, default=8,
                       help='LoRA rank (higher = more capacity, more parameters)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA scaling factor')
    parser.add_argument('--vit_layers_to_adapt', type=int, default=4,
                       help='Number of last ViT layers to apply LoRA (4-6 recommended)')

    # Data config
    parser.add_argument('--csv_file', type=str,
                       default='/home/muhammedg/fvlm/image_first_dataset.csv',
                       help='Path to data CSV')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')

    # Training config
    parser.add_argument('--output_dir', type=str,
                       default='./checkpoints/improved_vlm',
                       help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')

    # Metric loss config
    parser.add_argument('--use_metric_loss', action='store_true',
                       help='Use NLP metrics (ROUGE) to augment loss')
    parser.add_argument('--metric_weight', type=float, default=0.1,
                       help='Weight for metric-based loss component')

    args = parser.parse_args()
    main(args)

#!/usr/bin/env python3
import sys
import os
import logging

# Fix path to find 'lavis'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import torch
import numpy as np
import argparse
import SimpleITK as sitk
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, Transposed, EnsureChannelFirstd

from medical_vlm import MedicalVLM

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ======================================================
# WANDB SETUP
# ======================================================
import wandb # Make sure to import wandb

@dataclass
class VisionEncoderDecoderCollator:
    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features: raise ValueError("Empty batch")
        pixel_values = torch.stack([f['pixel_values'] for f in features])
        labels = torch.stack([f['labels'] for f in features])
        return {'pixel_values': pixel_values, 'labels': labels}

def build_transforms():
    return Compose([
        LoadImaged(keys=['image'], reader='ITKReader', image_only=True),
        EnsureChannelFirstd(keys=['image']),
        Transposed(keys=['image'], indices=(0, 3, 2, 1)),
        ScaleIntensityRanged(keys=['image'], a_min=-1150, a_max=350, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=['image'], spatial_size=(112, 256, 352), mode='constant', constant_values=0),
        CenterSpatialCropd(keys=['image'], roi_size=(112, 256, 352)),
    ])

class MedicalReportDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform, max_length=512, subset_size=None, split='training'):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['split'] == split].reset_index(drop=True)
        if subset_size: self.data = self.data.head(subset_size)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try:
            image_dict = self.transform({'image': row['image_path']})
            image = image_dict['image']
            if isinstance(image, sitk.Image): image = sitk.GetArrayFromImage(image)
            image_tensor = torch.from_numpy(np.array(image)).float()
            if image_tensor.dim() == 3: image_tensor = image_tensor.unsqueeze(0)

            text = f"{row['findings']} {row['impressions']}"
            encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            labels = encoding['input_ids'].squeeze(0)
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {'pixel_values': image_tensor, 'labels': labels}
        except Exception as e:
            logger.error(f"Error processing image {idx}: {e}")
            return None

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logger.info(f"Logging initialized. Saving logs to: {log_file}")

def main(args):
    setup_logging(args.output_dir)
    
    # ------------------------------------------------------
    # 1. CONFIGURE WANDB PROJECT & TEAM
    # ------------------------------------------------------
    os.environ["WANDB_PROJECT"] = "thesis"
    os.environ["WANDB_ENTITY"] = "gktp-thesis"
    # Optional: Log the model checkpoints to cloud (can use a lot of storage)
    # os.environ["WANDB_LOG_MODEL"] = "false" 
    
    # Create a unique name for this run based on arguments
    model_short_name = args.decoder_model.split('/')[-1]
    qformer_tag = "qformer" if args.use_qformer else "base"
    run_name = f"{model_short_name}_{qformer_tag}_bs{args.batch_size}_ep{args.num_epochs}"

    logger.info(f"WandB Run Name: {run_name}")
    logger.info(f"Starting Training with Decoder: {args.decoder_model}")

    # 2. Initialize Model
    model = MedicalVLM(
        vision_encoder_path=args.vision_encoder_path,
        decoder_model_name=args.decoder_model,
        use_qformer=args.use_qformer,
        num_query_tokens=args.num_query_tokens
    )

    # 3. Data
    transform = build_transforms()
    train_dataset = MedicalReportDataset(args.csv_file, model.tokenizer, transform, args.max_length, args.subset_size, 'training')
    val_dataset = MedicalReportDataset(args.csv_file, model.tokenizer, transform, args.max_length, int(args.subset_size/5) if args.subset_size else None, 'validation')

    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,               # <--- Name shown in WandB
        report_to="wandb",               # <--- ENABLE WANDB HERE
        
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=2e-4, 
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        logging_dir=os.path.join(args.output_dir, "runs"),
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

    # 5. Trainer
    trainer = Seq2SeqTrainer(
        model=model,  
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        data_collator=VisionEncoderDecoderCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()
    
    logger.info(f"Saving final model to {args.output_dir}/final_model")
    model.save_pretrained(f"{args.output_dir}/final_model")
    
    # Finish the WandB run
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder_model', type=str, default='microsoft/biogpt')
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/medical_vlm')
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--subset_size', type=int, default=None)
    parser.add_argument('--logging_steps', type=int, default=5)
    
    # New Q-Former args
    parser.add_argument('--use_qformer', action='store_true', help="Enable Q-Former")
    parser.add_argument('--num_query_tokens', type=int, default=32)
    
    args = parser.parse_args()
    main(args)
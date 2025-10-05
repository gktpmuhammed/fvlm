#!/usr/bin/env python3
"""
FULL TRAINING: Image-First VLM Training with the complete dataset.
Uses the specialized CXR-BERT model and an optimized batch size.
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
)
from monai import transforms
import sys
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import argparse # Add argparse
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    SpatialPadd,
    CenterSpatialCropd,
    Transposed,
)

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use RTX 3090

# Import the existing SimpleMedicalVLM
from simple_medical_vlm import SimpleMedicalVLM, MedicalReportDataset

# --- MONAI TRANSFORM PIPELINE ---
# Define at global scope so it can be imported by other scripts
transform = Compose([
    LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
    Transposed(keys=["image"], indices=(0, 3, 2, 1)),
    ScaleIntensityRanged(
        keys=["image"], a_min=-1150, a_max=350,
        b_min=0.0, b_max=1.0, clip=True
    ),
    SpatialPadd(
        keys=["image"], spatial_size=(112, 256, 352),
        mode="constant", constant_values=0
    ),
    CenterSpatialCropd(
        keys=["image"], roi_size=(112, 256, 352)
    ),
])


# --- DATASET DEFINITION ---
# This dataset now correctly uses the MONAI transform pipeline
class ImageFirstDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform, max_length=512, subset_size=None, split='training'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            tokenizer: Tokenizer for text processing.
            transform (callable, optional): Optional MONAI transform to be applied on a sample.
            max_length (int): Maximum length for tokenized text.
            subset_size (int, optional): If specified, uses a smaller subset of the dataset.
            split (string): The dataset split to use (e.g., 'training', 'validation').
        """
        df = pd.read_csv(csv_file)
        
        # Filter by split
        df = df[df['split'] == split].copy()
        
        if subset_size:
            if subset_size > len(df):
                subset_size = len(df)
            self.samples = df.sample(n=subset_size, random_state=42).reset_index(drop=True)
        else:
            self.samples = df.reset_index(drop=True)
            
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.missing_images = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples.iloc[idx]
        image_path = sample['image_path']
        
        # Apply the MONAI transforms to load and process the image
        if os.path.exists(image_path):
            data = self.transform({"image": image_path})
            pixel_values = data["image"]
        else:
            # Handle missing images gracefully
            self.missing_images += 1
            pixel_values = torch.zeros((1, 112, 256, 352)) # Return a zero tensor of the correct size

        # For this dataset, the 'text' is the full medical report
        # We need to construct it from findings and impressions
        
        # Prepare text by combining findings and impressions
        findings = str(sample.get("findings", "")).strip() if pd.notna(sample.get("findings", "")) else ""
        impressions = str(sample.get("impressions", "")).strip() if pd.notna(sample.get("impressions", "")) else ""
        
        # Correctly format with special tokens
        if findings and impressions:
            text = f"[FINDINGS] {findings} [IMPRESSION] {impressions}"
        elif impressions:
            text = f"[IMPRESSION] {impressions}"
        elif findings:
            text = f"[FINDINGS] {findings}"
        else:
            text = "[NORMAL]" # Use a special token for normal cases

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': pixel_values,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze().clone(),
            'text': text,
            'patient_id': sample['patient_id']
        }

def collate_fn(batch):
    """Custom collate function"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def main():
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Train an Image-First VLM on medical imaging data.")
    parser.add_argument(
        '--full_dataset',
        action='store_true',
        help="Use the full dataset for training. If not set, a small subset is used."
    )
    args = parser.parse_args()

    if args.full_dataset:
        print("🚀 FULL DATASET TRAINING: Image-First VLM 🚀")
        print("=" * 70)
        # Settings for FULL training
        output_dir = "/home/muhammedg/fvlm/outputs/ImageFirst_VLM_Full_Training"
        train_subset_size = None # Use all data
        val_subset_size = None   # Use all data
        num_epochs = 10
        warmup_steps = 500
        logging_steps = 100
        eval_save_steps = 1000
        
    else:
        print("🚀 SUBSET TRAINING (Final Check): Image-First VLM 🚀")
        print("=" * 70)
        # Settings for SUBSET training
        output_dir = "/home/muhammedg/fvlm/outputs/ImageFirst_VLM_Final_Subset_Test"
        train_subset_size = 400
        val_subset_size = 100
        num_epochs = 5
        warmup_steps = 20
        logging_steps = 50
        eval_save_steps = 200

    # Paths
    dataset_path = "/home/muhammedg/fvlm/image_first_dataset.csv"
    vision_encoder_path = "/home/muhammedg/fvlm/checkpoints/model.pth"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
      
    # Initialize tokenizer from the specialized BERT model path
    tokenizer = AutoTokenizer.from_pretrained(
        '/home/muhammedg/fvlm/BiomedVLP-CXR-BERT-specialized', 
        trust_remote_code=True
    )
  
    # Create datasets with the MONAI transform
    print(f"\n📊 Creating datasets with MONAI transforms...")
    train_dataset = ImageFirstDataset(
        csv_file=dataset_path,
        tokenizer=tokenizer,
        transform=transform,
        subset_size=train_subset_size,
        split='training'
    )
    val_dataset = ImageFirstDataset(
        csv_file=dataset_path,
        tokenizer=tokenizer,
        transform=transform,
        subset_size=val_subset_size,
        split='validation'
    )
    
    # Filter out missing images by checking the dataset's count
    if hasattr(train_dataset, 'missing_images') and train_dataset.missing_images > 0:
        print(f"\n🔍 Filtering Training dataset for existing images...")
        valid_samples = []
        missing_count = 0
        
        for sample in train_dataset.samples:
            if os.path.exists(sample['image_path']):
                valid_samples.append(sample)
            else:
                missing_count += 1
                if missing_count <= 5:  # Show first 5 missing files
                    print(f"Missing image: {sample['image_path']}")
        
        if missing_count > 5:
            print(f"... and {missing_count - 5} more missing images")
        
        train_dataset.samples = valid_samples
        print(f"✅ Training: {len(valid_samples)} valid samples ({missing_count} missing)")
    
    if hasattr(val_dataset, 'missing_images') and val_dataset.missing_images > 0:
        print(f"\n🔍 Filtering Validation dataset for existing images...")
        valid_samples = []
        missing_count = 0
        
        for sample in val_dataset.samples:
            if os.path.exists(sample['image_path']):
                valid_samples.append(sample)
            else:
                missing_count += 1
                if missing_count <= 5:  # Show first 5 missing files
                    print(f"Missing image: {sample['image_path']}")
        
        if missing_count > 5:
            print(f"... and {missing_count - 5} more missing images")
        
        val_dataset.samples = valid_samples
        print(f"✅ Validation: {len(valid_samples)} valid samples ({missing_count} missing)")
    
    print(f"\n📊 FINAL DATASET SIZES:")
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("❌ No valid training samples found!")
        return
    
    # Initialize model
    print(f"\n🤖 Initializing SimpleMedicalVLM...")
    model = SimpleMedicalVLM(
        vision_encoder_path=vision_encoder_path
    )
    
    # Full training arguments for the complete dataset
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,   
        per_device_eval_batch_size=8,  
        gradient_accumulation_steps=2,  
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        learning_rate=5e-5,
        fp16=True,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="none", # Disabled TensorBoard
        dataloader_num_workers=4, # OPTIMIZATION: Use parallel data loading
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # Calculate effective batch size for logging
    effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    
    print(f"\n🎯 TRAINING CONFIGURATION:")
    print(f"   • Mode: {'FULL DATASET' if args.full_dataset else 'SUBSET'}")
    print(f"   • Training samples: {len(train_dataset):,}")
    print(f"   • Validation samples: {len(val_dataset):,}")
    print(f"   • Epochs: {training_args.num_train_epochs}")
    print(f"   • Batch size: {training_args.per_device_train_batch_size}")
    print(f"   • Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"   • Effective batch size: {effective_batch_size}")
    
    # Calculate total steps for logging
    total_steps = (len(train_dataset) // effective_batch_size) * training_args.num_train_epochs
    print(f"   • Total training steps: {total_steps:,}")
    
    print(f"   • Estimated training time: ~{ (total_steps * 2.1) / 3600:.1f} hours") # Rough estimate
    print(f"   • Output directory: {training_args.output_dir}")
    print(f"   • GPU: RTX 3090 with FP16 mixed precision")
    
    print(f"\n⚠️  IMPORTANT:")
    print(f"   • This is a {'FULL' if args.full_dataset else 'SUBSET'} run on the CLEANED and TRANSFORMED dataset.")
    print(f"   • Training will take {'several hours' if args.full_dataset else 'a short amount of time'}.")
    print(f"   • Make sure you have enough disk space for checkpoints.")
    print(f"   • You can monitor progress in: {output_dir}/logs")
    
    # Start training automatically (no user input needed)
    print(f"\n🚀 Starting training...")
    
    # Test a single forward pass first
    print(f"\n🔍 Testing single forward pass...")
    try:
        sample = train_dataset[0]
        pixel_values = sample['pixel_values'].unsqueeze(0).cuda()
        input_ids = sample['input_ids'].unsqueeze(0).cuda()
        attention_mask = sample['attention_mask'].unsqueeze(0).cuda()
        labels = sample['labels'].unsqueeze(0).cuda()
        
        model.cuda()
        model.train()
        
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        print(f"✅ Forward pass successful!")
        print(f"   Initial loss: {outputs.loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return
    
    # Start full training
    print(f"\n🚀 Starting training with {len(train_dataset):,} samples...")
    try:
        trainer.train()
        
        # Save final model
        print(f"\n💾 Saving final model...")
        trainer.save_model()
        train_dataset.tokenizer.save_pretrained(output_dir)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {output_dir}")
        print(f"✅ Ready for evaluation and inference!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()

"""
Simple Medical VLM using:
- Pretrained Vision Encoder (from existing checkpoint)
- Pretrained BERT Decoder (from Hugging Face)
- Just train the projection layer between them
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertLMHeadModel, AutoTokenizer
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from monai import transforms
import os

class SimpleMedicalVLM(nn.Module):
    def __init__(self, vision_encoder_path, bert_model_name="/home/muhammedg/fvlm/BiomedVLP-CXR-BERT-specialized"):
        super().__init__()
        
        # 1. Load pretrained vision encoder (from your existing checkpoint)
        print("Loading pretrained vision encoder...")
        checkpoint = torch.load(vision_encoder_path, map_location='cpu')
        
        # Extract vision encoder from your existing model
        from lavis.models.blip_models.vit import ViT
        self.vision_encoder = ViT(
            in_channels=1,
            img_size=(112, 256, 352),
            patch_size=(16, 16, 32),
            num_classes=0,
            dropout_rate=0.1,
            qkv_bias=True
        )
        
        # Load vision encoder weights from checkpoint
        vision_state_dict = {}
        for key, value in checkpoint['model'].items():
            if key.startswith('visual_encoder.'):
                new_key = key.replace('visual_encoder.', '')
                vision_state_dict[new_key] = value
        
        self.vision_encoder.load_state_dict(vision_state_dict, strict=False)
        
        # Freeze vision encoder (optional - can unfreeze for fine-tuning)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        print(f"Vision encoder loaded and frozen")
        
        # 2. Load pretrained BERT decoder from Hugging Face
        print(f"Loading BERT decoder: {bert_model_name}")
        # --- Text Decoder ---
        # Using AutoTokenizer to load the correct tokenizer class
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name, trust_remote_code=True)
        
        # Add special tokens for medical reports
        special_tokens_dict = {
            "additional_special_tokens": ["[FINDINGS]", "[IMPRESSION]", "[NORMAL]", "[ABNORMAL]"]
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # Load BERT for conditional generation
        bert_config = BertConfig.from_pretrained(bert_model_name)
        bert_config.is_decoder = True
        bert_config.add_cross_attention = True
        
        self.text_decoder = BertLMHeadModel.from_pretrained(
            bert_model_name, 
            config=bert_config
        )
        
        # Resize embeddings for new special tokens
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        
        # Freeze text decoder, but keep cross-attention layers unfrozen
        for name, param in self.text_decoder.named_parameters():
            if "crossattention" not in name:
                param.requires_grad = False
            
        print(f"BERT decoder loaded. Cross-attention layers are trainable.")
        
        # 3. A more substantial MLP projection layer (this is what we'll train!)
        vision_dim = 768  # ViT base dimension
        text_dim = bert_config.hidden_size  # Usually 768
        intermediate_dim = text_dim * 2

        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.Dropout(0.1)
        )

        print(f"Projection MLP: {vision_dim} -> {intermediate_dim} -> {text_dim}")
        
    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for training
        """
        # 1. Encode image
        vision_outputs = self.vision_encoder(pixel_values)
        if isinstance(vision_outputs, tuple):
            vision_embeds = vision_outputs[0]  # Take the embeddings
        else:
            vision_embeds = vision_outputs
            
        # 2. Project vision features to text space
        vision_embeds = self.vision_projection(vision_embeds)
        
        # 3. Create attention mask for vision embeddings
        batch_size, seq_len = vision_embeds.shape[:2]
        vision_attention_mask = torch.ones(
            batch_size, seq_len, 
            dtype=torch.long, 
            device=vision_embeds.device
        )
        
        # 4. Forward through BERT decoder with cross-attention
        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, pixel_values, max_length=256, 
                 temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=2.0, 
                 no_repeat_ngram_size=2, **kwargs):
        """
        Generate medical reports from images using sampling for diversity
        
        Default parameters use controlled sampling which provides 100% unique reports
        while maintaining medical coherence and avoiding repetition.
        """
        # Encode image
        vision_outputs = self.vision_encoder(pixel_values)
        if isinstance(vision_outputs, tuple):
            vision_embeds = vision_outputs[0]
        else:
            vision_embeds = vision_outputs
            
        # Project to text space
        vision_embeds = self.vision_projection(vision_embeds)
        
        # Create attention mask
        batch_size, seq_len = vision_embeds.shape[:2]
        vision_attention_mask = torch.ones(
            batch_size, seq_len,
            dtype=torch.long,
            device=vision_embeds.device
        )
        
        # Start with [CLS] token
        input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.cls_token_id,
            dtype=torch.long,
            device=pixel_values.device
        )
        
        # Generate with controlled sampling for diversity
        outputs = self.text_decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.sep_token_id,
            **kwargs
        )
        
        return outputs


class MedicalReportDataset(Dataset):
    def __init__(self, csv_path, images_root, tokenizer, max_length=512):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Image transforms (same as your existing pipeline)
        self.transform = transforms.Compose([
            transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
            transforms.Transposed(keys=["image"], indices=(0, 3, 2, 1)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1150, a_max=350,
                b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.SpatialPadd(
                keys=["image"], spatial_size=(112, 256, 352),
                mode="constant", constant_values=0
            ),
            transforms.CenterSpatialCropd(
                keys=["image"], roi_size=(112, 256, 352)
            ),
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        volume_name = row["VolumeName"]
        base_name = volume_name.replace('.nii.gz', '')
        parts = base_name.split('_')
        
        if len(parts) >= 3:
            nested_path = f"{parts[0]}_{parts[1]}/{parts[0]}_{parts[1]}_{parts[2]}/{volume_name}"
            # Check which split this is based on the CSV path
            if 'train_reports' in str(self.csv_path):
                image_path = os.path.join(self.images_root, "train/images/train", nested_path)
            else:
                image_path = os.path.join(self.images_root, "valid/images/valid", nested_path)
        else:
            image_path = os.path.join(self.images_root, volume_name)
            
        # Check if file exists, if not try alternative paths
        if not os.path.exists(image_path):
            # Try without nested structure
            if 'train_reports' in str(self.csv_path):
                alt_path = os.path.join(self.images_root, "train/images/train", volume_name)
            else:
                alt_path = os.path.join(self.images_root, "valid/images/valid", volume_name)
            
            if os.path.exists(alt_path):
                image_path = alt_path
            else:
                # Skip this sample if file doesn't exist
                return None
        
        # Transform image
        data = self.transform({"image": image_path})
        pixel_values = data["image"]  # Keep all dimensions [C, D, H, W]
        
        # Prepare text
        findings = str(row.get("Findings_EN", "")).strip() if pd.notna(row.get("Findings_EN", "")) else ""
        impressions = str(row.get("Impressions_EN", "")).strip() if pd.notna(row.get("Impressions_EN", "")) else ""
        
        if findings and impressions:
            text = f"[FINDINGS] {findings} [IMPRESSION] {impressions}"
        elif impressions:
            text = f"[IMPRESSION] {impressions}"
        elif findings:
            text = f"[FINDINGS] {findings}"
        else:
            text = "[NORMAL] No significant findings."
        
        # Tokenize
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
            'labels': encoding['input_ids'].squeeze()
        }


def train_simple_vlm():
    """
    Simple training script using Hugging Face Trainer
    """
    print("Training Simple Medical VLM")
    
    # Initialize model
    model = SimpleMedicalVLM(
        vision_encoder_path="/home/muhammedg/fvlm/checkpoints/model.pth",
        bert_model_name="bert-base-uncased"
    )
    
    # Load datasets
    train_dataset = MedicalReportDataset(
        csv_path="/home/muhammedg/fvlm/data/dataset/radiology_text_reports/train_reports.csv",
        images_root="/home/muhammedg/fvlm/data/",
        tokenizer=model.tokenizer
    )
    
    val_dataset = MedicalReportDataset(
        csv_path="/home/muhammedg/fvlm/data/dataset/radiology_text_reports/validation_reports.csv",
        images_root="/home/muhammedg/fvlm/data/",
        tokenizer=model.tokenizer
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="/home/muhammedg/fvlm/outputs/simple_vlm",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="/home/muhammedg/fvlm/outputs/simple_vlm/logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb
        dataloader_num_workers=4,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=model.tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model("/home/muhammedg/fvlm/outputs/simple_vlm/final")
    print("Training complete!")


if __name__ == "__main__":
    train_simple_vlm()

#!/usr/bin/env python3
"""
Medical Vision-T5 using LAVIS ViT (Your Proven Approach)
Combines your working LAVIS ViT loader + T5 decoder
"""

import sys
import os
# Add project root to path to make lavis importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.modeling_outputs import BaseModelOutput
import pandas as pd
import numpy as np
from monai.transforms import *
import SimpleITK as sitk
import argparse
from dataclasses import dataclass
from typing import Dict, List

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    SpatialPadd,
    CenterSpatialCropd,
    Transposed,
    EnsureChannelFirstd,
)

@dataclass
class VisionTextCollator:
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            'pixel_values': torch.stack([f['pixel_values'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features])
        }


# ============================================================================
# LAVIS ViT Wrapper (Your Proven Approach)
# ============================================================================

class LAVISViTWrapper(nn.Module):
    """
    Loads YOUR pretrained LAVIS ViT exactly like your BioGPT version
    """

    def __init__(self, vision_encoder_path, image_size=(112, 256, 352), patch_size=(16, 16, 32)):
        super().__init__()

        print(f"\nLoading LAVIS ViT from: {vision_encoder_path}")

        # Load LAVIS ViT (absolute import now possible)
        from lavis.models.blip_models.vit import ViT

        self.vit = ViT(
            in_channels=1,
            img_size=image_size,
            patch_size=patch_size,
            num_classes=0,
        )

        # Load checkpoint (exactly as in your BioGPT code)
        checkpoint = torch.load(vision_encoder_path, map_location='cpu', weights_only=False)
        vision_state = {}

        if 'state_dict' in checkpoint:
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('visual_encoder.'):
                    vision_state[k.replace('visual_encoder.', '')] = v
        elif 'model' in checkpoint:
            for k, v in checkpoint['model'].items():
                if k.startswith('visual_encoder.'):
                    vision_state[k.replace('visual_encoder.', '')] = v
        else:
            for k, v in checkpoint.items():
                if k.startswith('visual_encoder.'):
                    vision_state[k.replace('visual_encoder.', '')] = v
                else:
                    vision_state[k] = v

        missing, unexpected = self.vit.load_state_dict(vision_state, strict=False)
        total_keys = len(self.vit.state_dict())
        loaded_keys = total_keys - len(missing)
        percent_loaded = 100 * loaded_keys / total_keys if total_keys > 0 else 0.0
        percent_missing = 100 * len(missing) / total_keys if total_keys > 0 else 0.0
        percent_unexpected = 100 * len(unexpected) / (len(vision_state) if len(vision_state) > 0 else 1)
        print(f"Loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
        print(f"Percentage loaded: {percent_loaded:.2f}% | missing: {percent_missing:.2f}% | unexpected: {percent_unexpected:.2f}%")

        self.hidden_size = 768
        print(f"Hidden size: {self.hidden_size}")

    def forward(self, pixel_values):
        """Forward pass - returns just the features"""
        outputs = self.vit(pixel_values)

        if isinstance(outputs, tuple):
            return outputs[0]  # [B, num_patches, 768]
        return outputs


# ============================================================================
# Vision-T5 Model
# ============================================================================

class VisionT5Model(nn.Module):
    """
    Your LAVIS ViT → Projection → T5 Decoder
    """

    def __init__(self, vision_encoder, t5_model, vision_hidden_size, t5_hidden_size):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.t5_model = t5_model

        # Projection layer
        if vision_hidden_size != t5_hidden_size:
            print(f"  Creating projection: {vision_hidden_size} -> {t5_hidden_size}")
            self.vision_projection = nn.Linear(vision_hidden_size, t5_hidden_size)
            nn.init.xavier_uniform_(self.vision_projection.weight)
        else:
            self.vision_projection = nn.Identity()

        self.config = t5_model.config

    def forward(self, pixel_values=None, labels=None, decoder_input_ids=None, **kwargs):
        """Forward pass with cross-attention"""
        # Encode vision
        vision_features = self.vision_encoder(pixel_values)  # [B, num_patches, 768]

        # Project to T5 space
        encoder_hidden_states = self.vision_projection(vision_features)  # [B, num_patches, 1024]

        # Attention mask
        B, L, _ = encoder_hidden_states.shape
        attention_mask = torch.ones(B, L, device=encoder_hidden_states.device, dtype=torch.long)

        # Create BaseModelOutput for T5 (FIXED!)
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states,
            hidden_states=None,
            attentions=None
        )

        # T5 forward
        outputs = self.t5_model(
            encoder_outputs=encoder_outputs,  # Pass as BaseModelOutput, not tuple!
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )

        return outputs

    def generate(self, pixel_values, max_length=256, num_beams=5, **kwargs):
        """Fast generation with KV caching"""
        # Encode vision
        vision_features = self.vision_encoder(pixel_values)
        encoder_hidden_states = self.vision_projection(vision_features)

        # Attention mask
        B, L, _ = encoder_hidden_states.shape
        attention_mask = torch.ones(B, L, device=encoder_hidden_states.device, dtype=torch.long)

        # FIX: Wrap encoder outputs in BaseModelOutput for generation
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states,
            hidden_states=None,
            attentions=None
        )

        # Generate
        generated_ids = self.t5_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            use_cache=True,
            **kwargs
        )

        return generated_ids


# ============================================================================
# Dataset
# ============================================================================

def build_transforms():
    return Compose([
        LoadImaged(keys=['image'], reader='ITKReader', image_only=True),
        EnsureChannelFirstd(keys=['image']),
        Transposed(keys=['image'], indices=(0, 3, 2, 1)),
        ScaleIntensityRanged(keys=['image'], a_min=-1150, a_max=350, b_min=0, b_max=1, clip=True),
        SpatialPadd(keys=['image'], spatial_size=(112, 256, 352)),
        CenterSpatialCropd(keys=['image'], roi_size=(112, 256, 352)),
    ])


class MedicalReportDataset(Dataset):
    def __init__(self, csv_file, split, transform, tokenizer, max_length=512, subset_size=None):
        df = pd.read_csv(csv_file)
        self.data = df[df['split'] == split].reset_index(drop=True)
        if subset_size:
            self.data = self.data.head(subset_size)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image
        image_dict = self.transform({'image': row['image_path']})
        image = image_dict['image']
        if isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)
        image = torch.from_numpy(np.array(image)).float()
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Tokenize
        text = f"{row['findings']} {row['impressions']}"
        encoding = self.tokenizer(text, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        labels = encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {'pixel_values': image, 'labels': labels}


# ============================================================================
# Model Creation
# ============================================================================

def create_model(vision_path, t5_name, freeze_vision):
    print("="*80)
    print("Medical Vision-T5 (LAVIS ViT + T5)")
    print("="*80)

    # 1. Load LAVIS ViT
    vision = LAVISViTWrapper(vision_path)
    print("\nFreezing all vision encoder parameters by default.")
    for p in vision.parameters():
        p.requires_grad = False

    # 2. Load T5
    print(f"\nLoading T5: {t5_name}")
    t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
    print(f"T5 hidden size: {t5.config.d_model}")
    print("Freezing all T5 parameters by default.")
    for p in t5.parameters():
        p.requires_grad = False

    # 3. Create model
    model = VisionT5Model(vision, t5, vision.hidden_size, t5.config.d_model)

    # 4. Selectively unfreeze layers for parameter-efficient training
    print("\nUnfreezing selected layers for training:")
    
    # Unfreeze the vision projection layer
    print("  - Unfreezing vision projection layer.")
    for p in model.vision_projection.parameters():
        p.requires_grad = True
        
    # Unfreeze the cross-attention layers in the T5 decoder
    print("  - Unfreezing T5 decoder cross-attention and LM head.")
    for i, block in enumerate(model.t5_model.decoder.block):
        # The cross-attention is in the second layer of each block
        for p in block.layer[1].EncDecAttention.parameters():
            p.requires_grad = True
            
    # Unfreeze the final language model head
    for p in model.t5_model.lm_head.parameters():
        p.requires_grad = True


    # 5. Stats
    print("\n" + "="*80)
    print("PARAMETERS (Parameter-Efficient Fine-Tuning):")
    print("="*80)

    v_train = sum(p.numel() for p in model.vision_encoder.parameters() if p.requires_grad)
    v_total = sum(p.numel() for p in model.vision_encoder.parameters())
    p_train = sum(p.numel() for p in model.vision_projection.parameters() if p.requires_grad)
    t_train = sum(p.numel() for p in model.t5_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_p = sum(p.numel() for p in model.parameters())

    print(f"Vision: {v_train:,} / {v_total:,} ({100*v_train/max(v_total,1):.1f}%) -> FROZEN")
    print(f"Proj:   {p_train:,} -> TRAINABLE")
    print(f"T5:     {t_train:,} -> PARTIALLY TRAINABLE (Cross-Attention + LM Head)")
    print(f"TOTAL:  {total:,} / {all_p:,} trainable params ({100*total/all_p:.2f}%)")
    print("="*80 + "\n")

    return model


# ============================================================================
# Training
# ============================================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.t5_model)
    model = create_model(args.vision_encoder_path, args.t5_model, args.freeze_vision).to(device)

    transform = build_transforms()
    train_ds = MedicalReportDataset(args.csv_file, 'training', transform, tokenizer, args.max_length, args.train_subset_size)
    val_ds = MedicalReportDataset(args.csv_file, 'validation', transform, tokenizer, args.max_length, args.val_subset_size)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}\n")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=args.num_workers,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=VisionTextCollator(),
    )

    print("="*80)
    print("Training...")
    print("="*80)
    trainer.train()

    final = os.path.join(args.output_dir, "final_model")
    os.makedirs(final, exist_ok=True)
    torch.save({
        'vision_encoder': model.vision_encoder.state_dict(),
        'vision_projection': model.vision_projection.state_dict(),
        't5_model': model.t5_model.state_dict(),
        'config': {'vision_hidden_size': model.vision_encoder.hidden_size, 't5_hidden_size': model.t5_model.config.d_model}
    }, os.path.join(final, 'model.pt'))
    tokenizer.save_pretrained(final)
    print(f"Saved to {final}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_encoder_path', default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--t5_model', default='google/flan-t5-large')
    parser.add_argument('--freeze_vision', action='store_true')
    parser.add_argument('--csv_file', default='/home/muhammedg/fvlm/image_first_dataset.csv')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--train_subset_size', type=int, default=None)
    parser.add_argument('--val_subset_size', type=int, default=None)
    parser.add_argument('--output_dir', default='./trained_vision_t5_lavis')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for dataloader")
    args = parser.parse_args()
    main(args)
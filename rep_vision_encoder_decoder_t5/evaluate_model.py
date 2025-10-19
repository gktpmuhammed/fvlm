#!/usr/bin/env python3
"""
Evaluation Script for Vision-T5 Model

This script loads a trained model checkpoint and runs evaluation on the validation set
to debug issues like NaN loss and inspect generated outputs.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
import pandas as pd
import SimpleITK as sitk
from transformers import AutoTokenizer, T5ForConditionalGeneration
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    SpatialPadd,
    CenterSpatialCropd,
    Transposed,
    EnsureChannelFirstd,
)
from torch.utils.data import Dataset, DataLoader
from transformers.modeling_outputs import BaseModelOutput

# Add project root to path to make lavis importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================================================
# DATASET AND TRANSFORMS
# ============================================================================

def build_transforms():
    """Builds the MONAI transforms for preprocessing the images."""
    return Compose([
        LoadImaged(keys=['image'], reader='ITKReader', image_only=True),
        EnsureChannelFirstd(keys=['image']),
        Transposed(keys=['image'], indices=(0, 3, 2, 1)),
        ScaleIntensityRanged(keys=['image'], a_min=-1150, a_max=350, b_min=0, b_max=1, clip=True),
        SpatialPadd(keys=['image'], spatial_size=(112, 256, 352)),
        CenterSpatialCropd(keys=['image'], roi_size=(112, 256, 352)),
    ])

class MedicalReportDataset(Dataset):
    """Dataset for loading medical reports and images."""
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
        image_dict = self.transform({'image': row['image_path']})
        image = image_dict['image']
        if isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)
        image = torch.from_numpy(np.array(image)).float()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        text = f"Findings: {row['findings']} Impressions: {row['impressions']}"
        encoding = self.tokenizer(text, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        labels = encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {'pixel_values': image, 'labels': labels, 'text': text}

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class LAVISViTWrapper(nn.Module):
    """Loads the pretrained LAVIS ViT model."""
    def __init__(self, vision_encoder_path, image_size=(112, 256, 352), patch_size=(16, 16, 32)):
        super().__init__()
        from lavis.models.blip_models.vit import ViT
        self.vit = ViT(in_channels=1, img_size=image_size, patch_size=patch_size, num_classes=0)
        
        # Load the base ViT checkpoint, not the trainer one
        print(f"Loading base ViT weights from: {vision_encoder_path}")
        checkpoint = torch.load(vision_encoder_path, map_location='cpu', weights_only=False)
        vision_state = {}
        # This loading logic is specific to the original medical ViT checkpoint
        if 'state_dict' in checkpoint:
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('visual_encoder.'):
                    vision_state[k.replace('visual_encoder.', '')] = v
        else: # Handle raw model state dict
             for k, v in checkpoint.get('model', checkpoint).items():
                if k.startswith('visual_encoder.'):
                    vision_state[k.replace('visual_encoder.', '')] = v
                else:
                    vision_state[k] = v

        msg = self.vit.load_state_dict(vision_state, strict=False)
        print(f"ViT Loading Message: {msg}")

        self.hidden_size = 768

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        return outputs[0] if isinstance(outputs, tuple) else outputs

class VisionT5Model(nn.Module):
    """The main Vision-T5 model architecture."""
    def __init__(self, vision_encoder, t5_model, vision_hidden_size, t5_hidden_size):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.t5_model = t5_model
        
        if vision_hidden_size != t5_hidden_size:
            self.vision_projection = nn.Linear(vision_hidden_size, t5_hidden_size)
            nn.init.xavier_uniform_(self.vision_projection.weight)
        else:
            self.vision_projection = nn.Identity()
        self.config = t5_model.config

    def forward(self, pixel_values=None, labels=None, **kwargs):
        vision_features = self.vision_encoder(pixel_values)
        encoder_hidden_states = self.vision_projection(vision_features)
        attention_mask = torch.ones(encoder_hidden_states.shape[:2], device=encoder_hidden_states.device)

        # Pass inputs to T5
        return self.t5_model(
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def generate(self, pixel_values, **kwargs):
        vision_features = self.vision_encoder(pixel_values)
        encoder_hidden_states = self.vision_projection(vision_features)
        attention_mask = torch.ones(encoder_hidden_states.shape[:2], device=encoder_hidden_states.device)
        
        # FIX: Wrap encoder outputs in BaseModelOutput for generation
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states,
            hidden_states=None,
            attentions=None
        )

        return self.t5_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **kwargs
        )

# ============================================================================
# EVALUATION LOGIC
# ============================================================================

def run_evaluation(args):
    """Main function to run the evaluation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 2. Reconstruct Model Architecture
    print("\nReconstructing model architecture...")
    # Important: Load the original, pretrained ViT, not from the training checkpoint
    vision_encoder = LAVISViTWrapper(args.vision_encoder_path).to(device)
    t5_model = T5ForConditionalGeneration.from_pretrained(args.t5_name).to(device)
    
    model = VisionT5Model(
        vision_encoder=vision_encoder,
        t5_model=t5_model,
        vision_hidden_size=vision_encoder.hidden_size,
        t5_hidden_size=t5_model.config.d_model
    )

    # 3. Load Trained Weights from Checkpoint
    model_checkpoint_path = os.path.join(args.model_path, 'pytorch_model.bin')
    print(f"Loading trained weights from: {model_checkpoint_path}")
    try:
        model_state_dict = torch.load(model_checkpoint_path, map_location='cpu')
        msg = model.load_state_dict(model_state_dict)
        print(f"Model Loading Message: {msg}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_checkpoint_path}. Make sure the path is correct.")
        return
    except Exception as e:
        print(f"ERROR: Failed to load model weights: {e}")
        return
        
    model.to(device)
    model.eval()

    # 4. Load Dataset
    print(f"\nLoading validation data from: {args.csv_path}")
    transform = build_transforms()
    dataset = MedicalReportDataset(
        csv_file=args.csv_path,
        split='validation',
        transform=transform,
        tokenizer=tokenizer,
        subset_size=args.num_samples
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # 5. Run Inference and Check for NaNs
    print("\n" + "="*80)
    print("RUNNING INFERENCE AND CHECKING OUTPUTS")
    print("="*80)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            original_texts = batch['text']

            print(f"\n--- Processing Batch {i+1} ---")
            
            # Check forward pass for NaN loss
            outputs = model(pixel_values=pixel_values, labels=labels, return_dict=True)
            loss = outputs.loss
            print(f"Loss: {loss.item()}")
            if torch.isnan(loss):
                print("NaN DETECTED IN LOSS!")
            
            # Check generation
            generated_ids = model.generate(
                pixel_values,
                max_length=150,
                num_beams=4,
                repetition_penalty=1.5,
            )
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Print comparison
            for j in range(len(generated_texts)):
                print(f"\nSample {i*args.batch_size + j + 1}:")
                print(f"  GT:      {original_texts[j][:200]}...")
                print(f"  GEN:     {generated_texts[j]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Vision-T5 model.")
    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/muhammedg/fvlm/rep_vision_encoder_decoder_t5/trained_vision_t5_lavis/checkpoint-8000',
        help='Path to the trained model checkpoint directory.'
    )
    parser.add_argument(
        '--vision_encoder_path',
        type=str,
        default='/home/muhammedg/fvlm/checkpoints/model.pth',
        help='Path to the original pretrained vision encoder weights (model.pth).'
    )
    parser.add_argument(
        '--t5_name',
        type=str,
        default='google/flan-t5-large',
        help='Name of the T5 model used during training.'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default='/home/muhammedg/fvlm/image_first_dataset.csv',
        help='Path to the dataset CSV file.'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=8,
        help='Number of validation samples to evaluate.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size for evaluation.'
    )
    args = parser.parse_args()
    run_evaluation(args)

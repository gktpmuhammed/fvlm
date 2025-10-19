#!/usr/bin/env python3
"""
Integrated Debug Script - Add to end of your training script or run separately
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from transformers import AutoTokenizer, T5ForConditionalGeneration
import pandas as pd
import SimpleITK as sitk
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
import argparse

# Add project root to path to make lavis importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================================================
# DATASET AND TRANSFORMS (Copied from training script for standalone use)
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
        image_dict = self.transform({'image': row['image_path']})
        image = image_dict['image']
        if isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)
        image = torch.from_numpy(np.array(image)).float()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        text = f"{row['findings']} {row['impressions']}"
        encoding = self.tokenizer(text, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        labels = encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {'pixel_values': image, 'labels': labels}

# ============================================================================
# MODEL DEFINITIONS (Copied from training script for standalone use)
# ============================================================================

class LAVISViTWrapper(nn.Module):
    """ Loads YOUR pretrained LAVIS ViT """
    def __init__(self, vision_encoder_path, image_size=(112, 256, 352), patch_size=(16, 16, 32)):
        super().__init__()
        print(f"\nLoading LAVIS ViT from: {vision_encoder_path}")
        from lavis.models.blip_models.vit import ViT
        self.vit = ViT(in_channels=1, img_size=image_size, patch_size=patch_size, num_classes=0)
        
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
        
        self.vit.load_state_dict(vision_state, strict=False)
        self.hidden_size = 768
        print(f"Hidden size set to: {self.hidden_size}")

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        return outputs[0] if isinstance(outputs, tuple) else outputs

class VisionT5Model(nn.Module):
    """ Your LAVIS ViT → Projection → T5 Decoder """
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
        
        return self.t5_model.generate(
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=attention_mask,
            **kwargs
        )

# ============================================================================
# DIAGNOSTIC CHECKS
# ============================================================================

def debug_checkpoint_loading(checkpoint_path, vision_encoder):
    """Analyze what weights were loaded from checkpoint"""
    print("\n" + "="*80)
    print("CHECKPOINT LOADING ANALYSIS")
    print("="*80)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
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

    model_state = vision_encoder.vit.state_dict()
    matched_keys = model_state.keys() & vision_state.keys()
    missing_keys = model_state.keys() - vision_state.keys()
    unexpected_keys = vision_state.keys() - model_state.keys()
    loaded_params = sum(model_state[k].numel() for k in matched_keys)
    total_params = sum(p.numel() for p in vision_encoder.vit.parameters())

    print(f"\n Matched keys: {len(matched_keys)}")
    print(f" Missing keys in model: {len(missing_keys)}")
    print(f" Unexpected keys in checkpoint: {len(unexpected_keys)}")
    print(f" Parameters loaded: {loaded_params:,} / {total_params:,} ({100*loaded_params/max(1, total_params):.1f}%)")

    if loaded_params / max(1, total_params) < 0.5:
        print("\  WARNING: Less than 50% of parameters loaded!")
        print("   This may impact performance. Consider checking key naming.")
        if missing_keys:
            print(f"   Sample missing keys: {list(missing_keys)[:5]}")
    elif loaded_params / max(1, total_params) < 0.99:
        print("\  Note: Some parameters are randomly initialized or from the base model.")
    else:
        print("\n Good checkpoint loading! Most parameters were loaded.")


def debug_encoder_diversity(model, dataloader):
    """Test encoder output diversity using real data"""
    print("\n" + "="*80)
    print("ENCODER DIVERSITY CHECK (REAL DATA)")
    print("="*80)

    model.eval()
    device = next(model.parameters()).device
    
    try:
        batch = next(iter(dataloader))
        pixel_values = batch['pixel_values'].to(device)
        num_samples = pixel_values.shape[0]
        print(f"\nUsing a batch of {num_samples} CT volumes from the dataset...")
    except StopIteration:
        print("\ Could not get data from the dataloader. Skipping diversity check.")
        return

    with torch.no_grad():
        features = model.vision_encoder(pixel_values)
        outputs = features.mean(dim=1)  # Pool to [B, 768]

    outputs_norm = outputs / (outputs.norm(dim=1, keepdim=True) + 1e-8)
    sims = outputs_norm @ outputs_norm.t()
    mask = ~torch.eye(num_samples, dtype=bool, device=device)
    avg_sim = sims[mask].mean().item()

    print(f"\ Average cosine similarity between samples: {avg_sim:.4f}")

    if avg_sim > 0.95:
        print("  WARNING: Outputs are very similar! Encoder may be collapsed.")
    elif avg_sim > 0.7:
        print("  Moderate similarity - encoder produces somewhat similar outputs")
    else:
        print(" Good diversity - encoder produces distinct outputs")

    mean = outputs.mean().item()
    std = outputs.std().item()

    print(f"\ Feature statistics:")
    print(f"   Mean: {mean:.4f}")
    print(f"   Std:  {std:.4f}")

    if std < 0.1:
        print("  WARNING: Low std - features may lack variance")
    else:
        print(" Reasonable variance in features")


def debug_cross_attention(model, tokenizer, dataloader):
    """Test cross-attention mechanism using real data"""
    print("\n" + "="*80)
    print("CROSS-ATTENTION CHECK (REAL DATA)")
    print("="*80)

    model.eval()
    device = next(model.parameters()).device

    try:
        batch = next(iter(dataloader))
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
    except StopIteration:
        print("Could not get data from the dataloader. Skipping cross-attention check.")
        return False
        
    print(f"\nTesting forward pass with a batch of {pixel_values.shape[0]} samples...")

    with torch.no_grad():
        try:
            vision_feat = model.vision_encoder(pixel_values)
            proj_feat = model.vision_projection(vision_feat)

            print(f"Vision encoding: {vision_feat.shape}")
            print(f"Projection: {proj_feat.shape}")
            
            # Use the actual labels from the batch
            labels[labels == -100] = tokenizer.pad_token_id 
            
            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
                return_dict=True,
                output_attentions=True
            )

            print(f"T5 forward pass successful")
            print(f"   Loss: {outputs.loss.item():.4f}")

            if outputs.cross_attentions:
                first_attn = outputs.cross_attentions[0]
                print(f"Cross-attention shape: {first_attn.shape}")
                print(f"   [batch={first_attn.shape[0]}, heads={first_attn.shape[1]}, tgt_len={first_attn.shape[2]}, src_len={first_attn.shape[3]}]")

                attn_mean = first_attn.mean().item()
                attn_std = first_attn.std().item()

                print(f"\n   Attention mean: {attn_mean:.6f}")
                print(f"   Attention std:  {attn_std:.6f}")

                uniform_val = 1.0 / first_attn.shape[-1]
                if abs(attn_mean - uniform_val) < 0.001:
                    print("Attention is uniform - may not be learning yet")
                else:
                    print("Attention shows variation")

        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\nTesting generation...")
    try:
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=15, num_beams=3)
            texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            print("Generation successful!")
            for i, t in enumerate(texts):
                # Decode original text for comparison
                original_label = labels[i]
                original_text = tokenizer.decode(original_label[original_label != tokenizer.pad_token_id], skip_special_tokens=True)
                print(f"   Sample {i+1}:")
                print(f"     Original:    '{original_text[:100]}...'")
                print(f"     Generated:   '{t}'")

            if all(t == texts[0] for t in texts if t):
                print("All outputs identical")
            else:
                print("Outputs are different")

    except Exception as e:
        print(f"Generation error: {e}")
        return False

    return True


def run_all_diagnostics(model, tokenizer, checkpoint_path, csv_path, batch_size=4):
    """Run all diagnostic checks using real data"""
    print("\n" + "="*80)
    print("RUNNING FULL DIAGNOSTICS")
    print("="*80)

    # 1. Create Dataloader for validation split
    print(f"\nLoading validation data from {csv_path}...")
    try:
        transform = build_transforms()
        dataset = MedicalReportDataset(csv_file=csv_path, split='validation', transform=transform, tokenizer=tokenizer, max_length=512, subset_size=batch_size)
        if len(dataset) == 0:
            raise ValueError("Dataset is empty. Check split name ('validation') and CSV file.")
        dataloader = DataLoader(dataset, batch_size=batch_size)
        print(f"Loaded {len(dataset)} samples for diagnostics.")
    except Exception as e:
        print(f"Failed to create dataloader: {e}")
        return
    
    # 2. Checkpoint loading
    debug_checkpoint_loading(checkpoint_path, model.vision_encoder)

    # 3. Encoder diversity
    debug_encoder_diversity(model, dataloader)

    # 4. Cross-attention
    success = debug_cross_attention(model, tokenizer, dataloader)

    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)

    if success:
        print("\nAll checks passed! Model appears to be working correctly.")
    else:
        print("\nSome issues detected. Review the output above.")

    return success


# ============================================================================
# STANDALONE USAGE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diagnostics on a Vision-T5 model.")
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth', help='Path to the original LAVIS ViT checkpoint (e.g., model.pth)')
    parser.add_argument('--t5_model', type=str, default='google/flan-t5-large', help='Name of the T5 model')
    parser.add_argument('--csv_path', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv', help='Path to the dataset CSV file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for diagnostic tests')
    args = parser.parse_args()

    print("\nRunning diagnostics in STANDALONE mode...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.t5_model)
        
        # 2. Create Model
        print("\nReconstructing model...")
        vision_encoder = LAVISViTWrapper(args.vision_encoder_path).to(device)
        t5 = T5ForConditionalGeneration.from_pretrained(args.t5_model).to(device)
        model = VisionT5Model(
            vision_encoder=vision_encoder,
            t5_model=t5,
            vision_hidden_size=vision_encoder.hidden_size,
            t5_hidden_size=t5.config.d_model
        ).to(device)
        
        # 3. Run diagnostics
        run_all_diagnostics(
            model=model,
            tokenizer=tokenizer,
            checkpoint_path=args.vision_encoder_path,
            csv_path=args.csv_path,
            batch_size=args.batch_size
        )
        
    except Exception as e:
        print(f"\nAn error occurred during standalone execution: {e}")
        import traceback
        traceback.print_exc()

#!/usr/bin/env python3
"""
Evaluate Vision-T5 Model
FAST evaluation with KV caching!
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
import sys
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
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
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import BLEUScore
from nltk.translate.meteor_score import meteor_score
import nltk

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


# ============================================================================
# Model Components (adapted from evaluate_model.py)
# ============================================================================

# Add project root to path to make lavis importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
# Dataset
# ============================================================================

def build_transforms():
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


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform, split='validation', subset_size=None):
        df = pd.read_csv(csv_file)
        self.data = df[df['split'] == split].reset_index(drop=True)

        if subset_size:
            self.data = self.data.head(subset_size)

        self.transform = transform

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

        reference_text = f"{row['findings']} {row['impressions']}"

        return {
            'pixel_values': image,
            'reference_text': reference_text,
            'image_path': row['image_path'],
        }


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, dataloader, tokenizer, device, args):
    model.eval()

    all_predictions = []
    all_references = []
    all_image_paths = []

    print("\n" + "="*80)
    print("Generating Predictions (FAST with KV Cache!)")
    print("="*80)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            references = batch['reference_text']
            image_paths = batch['image_path']

            # FAST generation!
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=args.max_length,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty,
                early_stopping=True,
            )

            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            all_predictions.extend(predictions)
            all_references.extend(references)
            all_image_paths.extend(image_paths)

    return all_predictions, all_references, all_image_paths


def calculate_meteor(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        try:
            score = meteor_score([ref.split()], pred.split())
            scores.append(score)
        except:
            scores.append(0.0)
    return np.mean(scores)


def compute_metrics(predictions, references):
    print("\n" + "="*80)
    print("Computing Metrics...")
    print("="*80)

    rouge = ROUGEScore()
    bleu = BLEUScore(n_gram=4)

    rouge_scores = rouge(predictions, references)
    references_list = [[ref] for ref in references]
    bleu_score = bleu(predictions, references_list)
    meteor = calculate_meteor(predictions, references)

    results = {
        'rouge1_f': rouge_scores['rouge1_fmeasure'].item(),
        'rouge2_f': rouge_scores['rouge2_fmeasure'].item(),
        'rougeL_f': rouge_scores['rougeL_fmeasure'].item(),
        'bleu': bleu_score.item(),
        'meteor': meteor,
    }

    return results


def save_results(results, predictions, references, image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"ROUGE-1 F1: {results['rouge1_f']:.4f}")
    print(f"ROUGE-2 F1: {results['rouge2_f']:.4f}")
    print(f"ROUGE-L F1: {results['rougeL_f']:.4f}")
    print(f"BLEU-4: {results['bleu']:.4f}")
    print(f"METEOR: {results['meteor']:.4f}")
    print("="*80)

    # Save metrics
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")

    # Save predictions
    pd.DataFrame({
        'image_path': image_paths,
        'prediction': predictions,
        'reference': references,
    }).to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    print(f"\nResults saved to: {output_dir}")


def main(args):
    print("="*80)
    print("Vision-T5 Model Evaluation")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 2. Reconstruct Model Architecture
    print("\nReconstructing model architecture...")
    # Important: Load the original, pretrained ViT, not from the training checkpoint
    vision_encoder = LAVISViTWrapper(args.vision_encoder_path).to(device)
    t5_model = T5ForConditionalGeneration.from_pretrained(args.t5_model).to(device)
    
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
        # Using strict=False is important as we are only loading the trained weights (projection layer and parts of T5)
        msg = model.load_state_dict(model_state_dict, strict=False)
        print(f"Model Loading Message: {msg}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_checkpoint_path}. This script expects 'pytorch_model.bin'.")
        return
    except Exception as e:
        print(f"ERROR: Failed to load model weights: {e}")
        return
        
    model.to(device)
    model.eval()

    # Data
    print("\nLoading data...")
    transform = build_transforms()
    dataset = EvaluationDataset(args.csv_file, transform, args.split, args.subset_size)
    print(f"Evaluation samples: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Evaluate
    predictions, references, image_paths = evaluate_model(model, dataloader, tokenizer, device, args)

    # Metrics
    results = compute_metrics(predictions, references)

    # Save
    save_results(results, predictions, references, image_paths, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--t5_model', type=str, default='google/flan-t5-large')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/image_first_dataset.csv')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--subset_size', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--repetition_penalty', type=float, default=1.5)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./evaluation_vision_t5')

    args = parser.parse_args()
    main(args)
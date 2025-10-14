#!/usr/bin/env python3

"""
Quick Evaluation of BLIP-2 Checkpoint
Loads directly from Trainer checkpoint
"""

import torch
import pandas as pd
import numpy as np
from medical_blip2_official import MedicalBLIP2Official
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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


def build_transforms():
    """Build image transforms"""
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


class MedicalImageDataset(Dataset):
    """Dataset for evaluation"""

    def __init__(self, csv_file, split='validation', transform=None, subset_size=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        # Apply subset if specified
        if subset_size is not None and subset_size > 0:
            self.df = self.df.head(subset_size).reset_index(drop=True)
        
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_dict = {'image': row['image_path']}
        if self.transform:
            image_dict = self.transform(image_dict)

        image = image_dict['image']
        if isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)
        image = torch.from_numpy(np.array(image)).float()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        reference = f"{row['findings']} {row['impressions']}"

        return {
            'image': image,
            'reference': reference,
            'image_path': row['image_path']
        }


def collate_fn(batch):
    """Custom collate function"""
    images = torch.stack([item['image'] for item in batch])
    references = [item['reference'] for item in batch]
    image_paths = [item['image_path'] for item in batch]

    return {
        'image': images,
        'reference': references,
        'image_path': image_paths
    }


def load_model_from_checkpoint(checkpoint_path, vision_encoder_path, opt_model):
    """Load model from trainer checkpoint"""

    print(f"Loading checkpoint: {checkpoint_path}")

    # Initialize model
    model = MedicalBLIP2Official(
        vision_encoder_path=vision_encoder_path,
        opt_model=opt_model,
        num_query_tokens=32,
        freeze_vision=True,
        freeze_opt=True,
    )

    # Load checkpoint
    checkpoint_file = f"{checkpoint_path}/pytorch_model.bin"
    state_dict = torch.load(checkpoint_file, map_location='cpu')

    # Load state dict
    model.load_state_dict(state_dict, strict=False)

    print(" Checkpoint loaded!")

    return model


def evaluate_checkpoint(args):
    """Evaluate checkpoint"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model_from_checkpoint(
        args.checkpoint_path,
        args.vision_encoder_path,
        args.opt_model
    )
    model = model.to(device)
    model.eval()

    # Load dataset
    print(f"\nLoading dataset: {args.csv_file}")
    transform = build_transforms()
    dataset = MedicalImageDataset(
        args.csv_file, 
        split=args.split, 
        transform=transform,
        subset_size=args.subset_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"Samples: {len(dataset)}")

    # Generate predictions
    print(f"\nGenerating predictions (max_length={args.max_length})...")
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)

            # Generate
            preds = model.generate(
                image=images,
                max_length=args.max_length,
                min_length=10,
                repetition_penalty=1.5,
            )

            predictions.extend(preds)
            references.extend(batch['reference'])

    # Compute metrics
    print("\nComputing metrics...")

    from torchmetrics.text.rouge import ROUGEScore
    from torchmetrics.text import BLEUScore
    from nltk.translate.meteor_score import meteor_score
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    # Initialize metrics
    rouge = ROUGEScore()
    bleu = BLEUScore(n_gram=4)

    # ROUGE
    print("Computing ROUGE scores...")
    rouge_scores = rouge(predictions, references)

    # BLEU
    print("Computing BLEU scores...")
    references_list = [[ref] for ref in references]
    bleu_score = bleu(predictions, references_list)

    # METEOR
    print("Computing METEOR scores...")
    meteor_scores = []
    for pred, ref in zip(predictions, references):
        try:
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            score = meteor_score([ref_tokens], pred_tokens)
            meteor_scores.append(score)
        except:
            meteor_scores.append(0.0)
    meteor = np.mean(meteor_scores)

    # Print results
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS - Checkpoint: {args.checkpoint_path}")
    print("="*80)
    print(f"\nROUGE-1 F1:        {rouge_scores['rouge1_fmeasure'].item():.4f}")
    print(f"ROUGE-1 Precision: {rouge_scores['rouge1_precision'].item():.4f}")
    print(f"ROUGE-1 Recall:    {rouge_scores['rouge1_recall'].item():.4f}")
    print(f"ROUGE-2 F1:        {rouge_scores['rouge2_fmeasure'].item():.4f}")
    print(f"ROUGE-2 Precision: {rouge_scores['rouge2_precision'].item():.4f}")
    print(f"ROUGE-2 Recall:    {rouge_scores['rouge2_recall'].item():.4f}")
    print(f"ROUGE-L F1:        {rouge_scores['rougeL_fmeasure'].item():.4f}")
    print(f"ROUGE-L Precision: {rouge_scores['rougeL_precision'].item():.4f}")
    print(f"ROUGE-L Recall:    {rouge_scores['rougeL_recall'].item():.4f}")
    print(f"BLEU-4:            {bleu_score.item():.4f}")
    print(f"METEOR:            {meteor:.4f}")
    print("="*80)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    results_df = pd.DataFrame({
        'prediction': predictions,
        'reference': references
    })
    results_df.to_csv(f"{args.output_dir}/predictions.csv", index=False)

    # Save sample predictions
    with open(f"{args.output_dir}/sample_predictions.txt", 'w') as f:
        f.write("Sample Predictions (first 5):\n")
        f.write("="*80 + "\n\n")
        for i in range(min(5, len(predictions))):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Prediction: {predictions[i]}\n")
            f.write(f"Reference: {references[i][:200]}...\n")
            f.write("\n" + "-"*80 + "\n\n")

    print(f"\nResults saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--vision_encoder_path', type=str,
                       default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--opt_model', type=str, default='facebook/opt-2.7b')
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--output_dir', type=str, default='./eval_checkpoint')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Number of samples to use for evaluation (default: use all)')

    args = parser.parse_args()
    evaluate_checkpoint(args)


if __name__ == "__main__":
    main()

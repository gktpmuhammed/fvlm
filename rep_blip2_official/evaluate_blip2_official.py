#!/usr/bin/env python3

"""
Evaluate Medical BLIP-2 (Official Architecture)
"""

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
from medical_blip2_official import MedicalBLIP2Official
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

# Metrics
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import BLEUScore
from nltk.translate.meteor_score import meteor_score
import nltk

# Download NLTK data
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def build_transforms():
    """Transform pipeline"""
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
        
        # Apply subset if specified
        if subset_size and subset_size > 0:
            self.data = self.data.head(subset_size)
        
        self.transform = transform

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

        reference_text = f"{row['findings']} {row['impressions']}"

        return {
            'image': image,
            'reference_text': reference_text,
            'image_path': row['image_path'],
        }


def calculate_meteor(predictions, references):
    """Calculate METEOR score"""
    scores = []
    for pred, ref in zip(predictions, references):
        try:
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            score = meteor_score([ref_tokens], pred_tokens)
            scores.append(score)
        except:
            scores.append(0.0)
    return np.mean(scores)


def evaluate_model(model, dataloader, device, args):
    """Run evaluation"""
    model.eval()

    all_predictions = []
    all_references = []
    all_image_paths = []

    print("\n" + "="*80)
    print("Generating Predictions...")
    print("="*80)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            references = batch['reference_text']
            image_paths = batch['image_path']

            # Generate predictions
            predictions = model.generate(
                image=images,
                prompt=args.prompt,
                max_length=args.max_length,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty,
            )

            all_predictions.extend(predictions)
            all_references.extend(references)
            all_image_paths.extend(image_paths)

    return all_predictions, all_references, all_image_paths


def compute_metrics(predictions, references):
    """Compute evaluation metrics"""
    print("\n" + "="*80)
    print("Computing Metrics...")
    print("="*80)

    # Initialize metrics
    rouge = ROUGEScore()
    bleu = BLEUScore(n_gram=4)

    # ROUGE
    print("\nComputing ROUGE scores...")
    rouge_scores = rouge(predictions, references)

    # BLEU
    print("Computing BLEU scores...")
    references_list = [[ref] for ref in references]
    bleu_score = bleu(predictions, references_list)

    # METEOR
    print("Computing METEOR scores...")
    meteor = calculate_meteor(predictions, references)

    results = {
        'rouge1_fmeasure': rouge_scores['rouge1_fmeasure'].item(),
        'rouge1_precision': rouge_scores['rouge1_precision'].item(),
        'rouge1_recall': rouge_scores['rouge1_recall'].item(),
        'rouge2_fmeasure': rouge_scores['rouge2_fmeasure'].item(),
        'rouge2_precision': rouge_scores['rouge2_precision'].item(),
        'rouge2_recall': rouge_scores['rouge2_recall'].item(),
        'rougeL_fmeasure': rouge_scores['rougeL_fmeasure'].item(),
        'rougeL_precision': rouge_scores['rougeL_precision'].item(),
        'rougeL_recall': rouge_scores['rougeL_recall'].item(),
        'bleu': bleu_score.item(),
        'meteor': meteor,
    }

    return results


def save_results(results, predictions, references, image_paths, output_dir):
    """Save results"""
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Medical BLIP-2 (Official) - Evaluation Metrics\n")
        f.write("="*80 + "\n\n")

        f.write("ROUGE Scores:\n")
        f.write("-"*80 + "\n")
        f.write(f"  ROUGE-1 F1:        {results['rouge1_fmeasure']:.4f}\n")
        f.write(f"  ROUGE-1 Precision: {results['rouge1_precision']:.4f}\n")
        f.write(f"  ROUGE-1 Recall:    {results['rouge1_recall']:.4f}\n")
        f.write(f"  ROUGE-2 F1:        {results['rouge2_fmeasure']:.4f}\n")
        f.write(f"  ROUGE-2 Precision: {results['rouge2_precision']:.4f}\n")
        f.write(f"  ROUGE-2 Recall:    {results['rouge2_recall']:.4f}\n")
        f.write(f"  ROUGE-L F1:        {results['rougeL_fmeasure']:.4f}\n")
        f.write(f"  ROUGE-L Precision: {results['rougeL_precision']:.4f}\n")
        f.write(f"  ROUGE-L Recall:    {results['rougeL_recall']:.4f}\n\n")

        f.write("Other Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"  BLEU-4:            {results['bleu']:.4f}\n")
        f.write(f"  METEOR:            {results['meteor']:.4f}\n")
        f.write("="*80 + "\n")

    with open(metrics_file, 'r') as f:
        print(f.read())

    # Save predictions
    predictions_df = pd.DataFrame({
        'image_path': image_paths,
        'prediction': predictions,
        'reference': references,
    })
    predictions_file = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_file, index=False)
    print(f"\nPredictions saved to: {predictions_file}")

    # Save samples
    samples_file = os.path.join(output_dir, 'sample_predictions.txt')
    with open(samples_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Sample Predictions (First 10)\n")
        f.write("="*80 + "\n\n")

        for i in range(min(10, len(predictions))):
            f.write(f"Sample {i+1}:\n")
            f.write("-"*80 + "\n")
            f.write(f"Image: {image_paths[i]}\n\n")
            f.write(f"Prediction:\n{predictions[i]}\n\n")
            f.write(f"Reference:\n{references[i]}\n")
            f.write("="*80 + "\n\n")

    print(f"Sample predictions saved to: {samples_file}")


def main(args):
    print("="*80)
    print("Medical BLIP-2 (Official) - Evaluation")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model
    print(f"\nLoading model from: {args.model_path}")

    image_size = tuple(map(int, args.image_size.split(',')))
    patch_size = tuple(map(int, args.patch_size.split(',')))

    model = MedicalBLIP2Official.from_pretrained(
        args.model_path,
        vision_encoder_path=args.vision_encoder_path,
    )

    model = model.to(device)
    model.eval()
    print(" Model loaded")

    # Load data
    print(f"\nLoading evaluation data...")
    transform = build_transforms()

    dataset = EvaluationDataset(
        csv_file=args.csv_file,
        transform=transform,
        split=args.split,
        subset_size=args.subset_size,
    )

    if args.subset_size:
        print(f"Using subset: {args.subset_size} samples")
    print(f"Evaluation samples: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Evaluate
    predictions, references, image_paths = evaluate_model(model, dataloader, device, args)

    # Compute metrics
    results = compute_metrics(predictions, references)

    # Save
    save_results(results, predictions, references, image_paths, args.output_dir)

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Medical BLIP-2 (Official)")

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--vision_encoder_path', type=str,
                       default='/home/muhammedg/fvlm/checkpoints/model.pth',
                       help='Path to vision encoder')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv',
                       help='Path to CSV file')
    parser.add_argument('--split', type=str, default='validation',
                       help='Which split to evaluate')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Use only a subset of data for quick testing (e.g., 50)')
    parser.add_argument('--image_size', type=str, default='112,256,352',
                       help='Image size')
    parser.add_argument('--patch_size', type=str, default='16,16,32',
                       help='Patch size')

    # Generation arguments
    parser.add_argument('--prompt', type=str, 
                       default="A medical report describing this CT scan: ",
                       help='Generation prompt')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Max generation length')
    parser.add_argument('--num_beams', type=int, default=5,
                       help='Number of beams')
    parser.add_argument('--repetition_penalty', type=float, default=1.5,
                       help='Repetition penalty')
    parser.add_argument('--length_penalty', type=float, default=1.0,
                       help='Length penalty')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Dataloader workers')

    # Output
    parser.add_argument('--output_dir', type=str, default='./evaluation_official',
                       help='Output directory')

    args = parser.parse_args()
    main(args)

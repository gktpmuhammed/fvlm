#!/usr/bin/env python3

"""
Evaluate Medical BLIP-2 with BioGPT Model
Computes BLEU, ROUGE, METEOR scores and generates sample reports
"""

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
from medical_blip2_biogpt import MedicalBLIP2BioGPT
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

# Download required NLTK data
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def build_transforms():
    """Transform pipeline for 3D medical images"""
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


def calculate_accuracy(predictions, references):
    """Calculate exact match accuracy"""
    exact_match = sum([1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower()])
    return exact_match / len(predictions) if len(predictions) > 0 else 0.0


def evaluate_model(model, dataloader, device, args):
    """Run evaluation on dataset"""
    model.eval()

    all_predictions = []
    all_references = []
    all_image_paths = []

    print("\n" + "="*80)
    print("Generating Predictions...")
    print("="*80)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            references = batch['reference_text']
            image_paths = batch['image_path']

            # Generate predictions
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=args.max_length,
                num_beams=args.num_beams,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

            # Decode predictions
            predictions = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            all_predictions.extend(predictions)
            all_references.extend(references)
            all_image_paths.extend(image_paths)

    return all_predictions, all_references, all_image_paths


def compute_metrics(predictions, references):
    """Compute all evaluation metrics"""
    print("\n" + "="*80)
    print("Computing Metrics...")
    print("="*80)

    # Initialize metrics
    rouge = ROUGEScore()
    bleu = BLEUScore(n_gram=4)

    # Compute ROUGE
    print("\nComputing ROUGE scores...")
    rouge_scores = rouge(predictions, references)

    # Compute BLEU
    print("Computing BLEU scores...")
    # BLEU expects list of reference lists
    references_list = [[ref] for ref in references]
    bleu_score = bleu(predictions, references_list)

    # Compute METEOR
    print("Computing METEOR scores...")
    meteor = calculate_meteor(predictions, references)

    # Compute accuracy
    accuracy = calculate_accuracy(predictions, references)

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
        'accuracy': accuracy,
    }

    return results


def save_results(results, predictions, references, image_paths, output_dir):
    """Save evaluation results"""
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Medical BLIP-2 with BioGPT - Evaluation Metrics\n")
        f.write("="*80 + "\n\n")

        # Group metrics
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
        f.write(f"  Exact Match Acc:   {results['accuracy']:.4f}\n")
        f.write("="*80 + "\n")

    # Print to console
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

    # Save sample predictions
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
    print(f"\nAll results saved to: {output_dir}")


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform, split='validation'):
        df = pd.read_csv(csv_file)
        self.data = df[df['split'] == split].reset_index(drop=True)
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

        # Ensure proper dimensions
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Combine findings and impressions as reference
        reference_text = f"{row['findings']} {row['impressions']}"

        return {
            'pixel_values': image,
            'reference_text': reference_text,
            'image_path': row['image_path'],
        }


def main(args):
    print("="*80)
    print("Medical BLIP-2 with BioGPT - Evaluation")
    print("="*80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model
    print(f"\nLoading model from: {args.model_path}")

    if os.path.exists(os.path.join(args.model_path, 'config.pt')):
        # Load from saved checkpoint
        print("Loading from saved checkpoint...")
        model = MedicalBLIP2BioGPT.from_pretrained(
            args.model_path,
            vision_encoder_path=args.vision_encoder_path
        )
    else:
        # Initialize and load weights manually
        print("Initializing model...")
        image_size = tuple(map(int, args.image_size.split(',')))
        patch_size = tuple(map(int, args.patch_size.split(',')))

        model = MedicalBLIP2BioGPT(
            vision_encoder_path=args.vision_encoder_path,
            decoder_model_name="microsoft/biogpt",
            image_size=image_size,
            patch_size=patch_size,
            num_query_tokens=args.num_query_tokens,
            use_qformer=args.use_qformer,
        )

        # Load weights
        print("Loading model weights...")
        if os.path.exists(os.path.join(args.model_path, 'encoder.pt')):
            model.encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder.pt')))
        if os.path.exists(os.path.join(args.model_path, 'projection.pt')):
            model.projection.load_state_dict(torch.load(os.path.join(args.model_path, 'projection.pt')))
        if args.use_qformer and os.path.exists(os.path.join(args.model_path, 'qformer.pt')):
            model.qformer.load_state_dict(torch.load(os.path.join(args.model_path, 'qformer.pt')))
        if args.use_qformer and os.path.exists(os.path.join(args.model_path, 'query_tokens.pt')):
            model.query_tokens = torch.load(os.path.join(args.model_path, 'query_tokens.pt'))

    model = model.to(device)
    print(" Model loaded successfully")

    # Prepare dataset
    print(f"\nLoading evaluation data from: {args.csv_file}")
    transform = build_transforms()

    dataset = EvaluationDataset(
        csv_file=args.csv_file,
        transform=transform,
        split=args.split,
    )

    print(f"Evaluation samples: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Run evaluation
    predictions, references, image_paths = evaluate_model(model, dataloader, device, args)

    # Compute metrics
    results = compute_metrics(predictions, references)

    # Save results
    save_results(results, predictions, references, image_paths, args.output_dir)

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Medical BLIP-2 with BioGPT")

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--vision_encoder_path', type=str,
                       default='/home/muhammedg/fvlm/checkpoints/model.pth',
                       help='Path to pretrained vision encoder')
    parser.add_argument('--image_size', type=str, default='112,256,352',
                       help='3D image size as D,H,W')
    parser.add_argument('--patch_size', type=str, default='16,16,32',
                       help='3D patch size as D,H,W')
    parser.add_argument('--num_query_tokens', type=int, default=32,
                       help='Number of query tokens')
    parser.add_argument('--use_qformer', action='store_true', default=True,
                       help='Model uses Q-Former')

    # Data arguments
    parser.add_argument('--csv_file', type=str, required=True,
                       help='Path to CSV file with evaluation data')
    parser.add_argument('--split', type=str, default='validation',
                       choices=['training', 'validation', 'test'],
                       help='Which split to evaluate')

    # Generation arguments
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum generation length')
    parser.add_argument('--num_beams', type=int, default=4,
                       help='Number of beams for beam search')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')

    args = parser.parse_args()

    main(args)

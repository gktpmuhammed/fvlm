#!/usr/bin/env python3
"""
Evaluation Script for Medical VisionEncoderDecoder
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import BLEUScore
import argparse
import os
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent))

from medical_vision_encoder_decoder import MedicalVisionEncoderDecoder
from train_vision_encoder_decoder import MedicalReportDataset, build_transforms


def evaluate_model(args):
    """Evaluate trained model"""

    print("="*80)
    print("EVALUATION: Medical VisionEncoderDecoder")
    print("="*80)

    # Load trained model
    print(f"\nLoading model from {args.model_path}...")
    model = MedicalVisionEncoderDecoder.from_pretrained(args.model_path)
    model.model.eval()
    model.model.cuda()

    # Build transforms
    transform = build_transforms()

    # Load validation dataset
    print(f"\nLoading validation dataset...")
    val_dataset = MedicalReportDataset(
        csv_file=args.csv_file,
        tokenizer=model.tokenizer,
        transform=transform,
        max_length=512,
        split='validation',
        subset_size=args.subset_size
    )

    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloader
    dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize metrics
    rouge = ROUGEScore()
    bleu = BLEUScore(n_gram=2)

    # Collect results
    results = []

    print("\nGenerating reports...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            try:
                # Move to GPU
                pixel_values = batch['pixel_values'].cuda()
                labels = batch['labels']

                # Generate reports
                generated = model.generate(
                    pixel_values=pixel_values,
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=2.0
                )

                # Decode predictions and references
                predictions = model.tokenizer.batch_decode(generated, skip_special_tokens=True)
                references = model.tokenizer.batch_decode(labels, skip_special_tokens=True)

                # Store results
                for pred, ref in zip(predictions, references):
                    results.append({
                        'generated_report': pred,
                        'ground_truth': ref
                    })

                # Print first few samples
                if batch_idx == 0:
                    print("\n" + "="*80)
                    print("SAMPLE GENERATIONS:")
                    print("="*80)
                    for i in range(min(3, len(predictions))):
                        print(f"\nSample {i+1}:")
                        print(f"Generated: {predictions[i][:300]}")
                        print(f"Reference: {references[i][:300]}")
                    print("="*80 + "\n")

            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                continue

    # Save results
    print(f"\nSaving results to {args.output_csv}...")
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} results")

    # Compute metrics
    print("\nComputing metrics...")
    all_preds = df['generated_report'].tolist()
    all_refs = df['ground_truth'].tolist()

    # Filter out empty predictions
    valid_pairs = [(p, r) for p, r in zip(all_preds, all_refs) if p.strip() and r.strip()]

    if valid_pairs:
        valid_preds, valid_refs = zip(*valid_pairs)

        rouge_scores = rouge(list(valid_preds), list(valid_refs))
        bleu_score = bleu(list(valid_preds), [[ref] for ref in valid_refs])

        # Print results
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"Total samples: {len(results)}")
        print(f"Valid samples: {len(valid_pairs)}")
        print(f"\nROUGE-1: {rouge_scores['rouge1_fmeasure']:.4f}")
        print(f"ROUGE-2: {rouge_scores['rouge2_fmeasure']:.4f}")
        print(f"ROUGE-L: {rouge_scores['rougeL_fmeasure']:.4f}")
        print(f"BLEU-2: {bleu_score:.4f}")
        print(f"{'='*80}\n")

        # Save metrics
        metrics_file = args.output_csv.replace('.csv', '_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"EVALUATION RESULTS\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total samples: {len(results)}\n")
            f.write(f"Valid samples: {len(valid_pairs)}\n")
            f.write(f"\nROUGE-1: {rouge_scores['rouge1_fmeasure']:.4f}\n")
            f.write(f"ROUGE-2: {rouge_scores['rouge2_fmeasure']:.4f}\n")
            f.write(f"ROUGE-L: {rouge_scores['rougeL_fmeasure']:.4f}\n")
            f.write(f"BLEU-2: {bleu_score:.4f}\n")

        print(f"Metrics saved to {metrics_file}")
    else:
        print("\nERROR: No valid predictions!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Medical VisionEncoderDecoder')

    parser.add_argument('--model_path', type=str,
                       default='./checkpoints/vision_encoder_decoder/final_model',
                       help='Path to trained model')
    parser.add_argument('--csv_file', type=str,
                       default='/home/muhammedg/fvlm/image_first_dataset.csv',
                       help='Path to data CSV')
    parser.add_argument('--output_csv', type=str,
                       default='vision_encoder_decoder_results.csv',
                       help='Output CSV file')

    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum generation length')
    parser.add_argument('--num_beams', type=int, default=5,
                       help='Number of beams')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Evaluate on subset')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    evaluate_model(args)

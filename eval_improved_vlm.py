#!/usr/bin/env python3
"""
Evaluation script for Improved Medical VLM with BioGPT - FIXED VERSION
Handles generation errors gracefully
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
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from improved_medical_vlm import ImprovedMedicalVLM
from train_improved_vlm import ImageFirstDataset, build_transforms

def load_model(checkpoint_dir, vision_encoder_path):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_dir}...")

    # Initialize model architecture
    model = ImprovedMedicalVLM(
        vision_encoder_path=vision_encoder_path,
        lora_rank=8,
        lora_alpha=16,
        vit_layers_to_adapt=4
    )

    # Load trained weights
    checkpoint_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_dir}")

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Warning: {len(missing)} missing keys in checkpoint")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys in checkpoint")

    model.eval()
    model.cuda()

    print("Model loaded successfully!")
    return model

def evaluate_model(args):
    """Evaluate model on validation set"""

    # Load model
    model = load_model(args.model_path, args.vision_encoder_path)

    # Build transforms
    transform = build_transforms()

    # Load validation dataset
    print(f"Loading validation dataset from {args.csv_file}...")
    val_dataset = ImageFirstDataset(
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
        num_workers=args.num_workers
    )

    # Initialize metrics
    rouge = ROUGEScore()
    bleu = BLEUScore(n_gram=2)

    # Collect results
    results = []
    successful_batches = 0
    failed_batches = 0

    print("\nGenerating reports...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            try:
                # Move to GPU
                images = batch['images'].cuda()
                labels = batch['labels']

                # Generate reports
                generated = model.generate(
                    images=images,
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    temperature=1.0,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    early_stopping=True,
                    log_attention=args.log_attention
                )

                # Decode predictions and references
                predictions = model.tokenizer.batch_decode(generated, skip_special_tokens=True)
                references = model.tokenizer.batch_decode(labels, skip_special_tokens=True)

                # Store results
                for pred, ref in zip(predictions, references):
                    results.append({
                        'generated_report': pred if pred else "[EMPTY]",
                        'ground_truth': ref
                    })

                successful_batches += 1

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
                print(f"\nError in batch {batch_idx}: {str(e)}")
                failed_batches += 1

                # Add placeholder results for failed batch
                batch_size = len(batch['labels'])
                for i in range(batch_size):
                    results.append({
                        'generated_report': "[GENERATION_FAILED]",
                        'ground_truth': model.tokenizer.decode(batch['labels'][i], skip_special_tokens=True)
                    })
                continue

    print(f"\nGeneration complete:")
    print(f"  Successful batches: {successful_batches}")
    print(f"  Failed batches: {failed_batches}")
    print(f"  Total samples: {len(results)}")

    # Save results to CSV
    if results:
        print(f"\nSaving results to {args.output_csv}...")
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved {len(df)} results")

        # Compute metrics (filter out failed generations)
        print("\nComputing metrics...")
        valid_pairs = [
            (row['generated_report'], row['ground_truth']) 
            for _, row in df.iterrows() 
            if row['generated_report'] not in ["[EMPTY]", "[GENERATION_FAILED]"] 
            and row['generated_report'].strip()
        ]

        if valid_pairs:
            valid_preds, valid_refs = zip(*valid_pairs)

            print(f"Valid predictions: {len(valid_pairs)} / {len(results)}")

            rouge_scores = rouge(list(valid_preds), list(valid_refs))
            bleu_score = bleu(list(valid_preds), [[ref] for ref in valid_refs])

            # Print results
            print(f"\n{'='*80}")
            print("EVALUATION RESULTS")
            print(f"{'='*80}")
            print(f"Total samples: {len(results)}")
            print(f"Valid samples: {len(valid_pairs)}")
            print(f"Failed samples: {failed_batches * args.batch_size}")
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
                f.write(f"Failed samples: {failed_batches * args.batch_size}\n")
                f.write(f"\nROUGE-1: {rouge_scores['rouge1_fmeasure']:.4f}\n")
                f.write(f"ROUGE-2: {rouge_scores['rouge2_fmeasure']:.4f}\n")
                f.write(f"ROUGE-L: {rouge_scores['rougeL_fmeasure']:.4f}\n")
                f.write(f"BLEU-2: {bleu_score:.4f}\n")

            print(f"Metrics saved to {metrics_file}")
        else:
            print("\nERROR: No valid predictions generated!")
            print("All generations either failed or were empty.")
    else:
        print("\nERROR: No results to save!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Improved Medical VLM')

    parser.add_argument('--model_path', type=str,
                       default='./checkpoints/improved_vlm/final_model',
                       help='Path to trained model checkpoint')
    parser.add_argument('--vision_encoder_path', type=str,
                       default='/home/muhammedg/fvlm/checkpoints/model.pth',
                       help='Path to pretrained vision encoder')
    parser.add_argument('--csv_file', type=str,
                       default='/home/muhammedg/fvlm/image_first_dataset.csv',
                       help='Path to data CSV')
    parser.add_argument('--output_csv', type=str,
                       default='improved_results.csv',
                       help='Output CSV file for results')

    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for dataloader')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum length for generated reports')
    parser.add_argument('--num_beams', type=int, default=5,
                       help='Number of beams for beam search')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Evaluate on subset (for debugging)')
    parser.add_argument('--log_attention', action='store_true',
                       help='Log attention weights during generation')

    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    evaluate_model(args)
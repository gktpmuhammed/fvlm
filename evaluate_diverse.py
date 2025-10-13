#!/usr/bin/env python3
"""
Evaluation with DIVERSE generation settings + encoder diagnostics
Tests if model CAN generate different outputs
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import BLEUScore
import argparse
import os
import numpy as np

from train_vision_gpt2 import MedicalReportDataset, build_transforms
from medical_vision_gpt2 import MedicalVisionGPT2


def evaluate_model(args):
    print("="*80)
    print("DIVERSE GENERATION EVALUATION")
    print("="*80)

    # Load model
    print(f"\nLoading model...")
    vision_encoder_path = '/home/muhammedg/fvlm/checkpoints/model.pth'

    full_model = MedicalVisionGPT2(
        vision_encoder_path=vision_encoder_path,
        decoder_model_name="gpt2",
        freeze_encoder=True,
        freeze_decoder_base=True
    )

    # Load trained decoder
    print(f"Loading trained weights from {args.model_path}...")
    from transformers import VisionEncoderDecoderModel
    trained_model = VisionEncoderDecoderModel.from_pretrained(args.model_path)

    full_model.model.decoder = trained_model.decoder
    full_model.model.eval()
    full_model.model.cuda()

    # Build transforms
    transform = build_transforms()

    # Load dataset
    print(f"\nLoading validation dataset...")
    val_dataset = MedicalReportDataset(
        csv_file=args.csv_file,
        tokenizer=full_model.tokenizer,
        transform=transform,
        max_length=512,
        split='validation',
        subset_size=args.subset_size
    )

    print(f"Validation samples: {len(val_dataset)}")

    # Dataloader
    dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Metrics
    rouge = ROUGEScore()
    bleu = BLEUScore(n_gram=2)

    # Collect results and encoder stats
    results = []
    encoder_stats = []

    print("\nGenerating with DIVERSE settings...")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Do sample: {args.do_sample}")
    print(f"  Num beams: {args.num_beams}\n")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            try:
                pixel_values = batch['pixel_values'].cuda()
                labels = batch['labels'].cuda()

                # Encode and collect stats
                encoder_outputs = full_model.model.encoder(pixel_values, return_dict=True)
                encoder_hidden = encoder_outputs.last_hidden_state

                # Encoder statistics
                encoder_mean = encoder_hidden.mean().item()
                encoder_std = encoder_hidden.std().item()
                encoder_max = encoder_hidden.max().item()
                encoder_min = encoder_hidden.min().item()

                encoder_stats.append({
                    'mean': encoder_mean,
                    'std': encoder_std,
                    'max': encoder_max,
                    'min': encoder_min
                })

                # Generate with diverse settings
                if args.do_sample:
                    generated = full_model.model.decoder.generate(
                        encoder_hidden_states=encoder_hidden,
                        max_length=args.max_length,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=True,
                        no_repeat_ngram_size=3,
                        repetition_penalty=2.5,
                        pad_token_id=full_model.tokenizer.pad_token_id,
                        eos_token_id=full_model.tokenizer.eos_token_id,
                    )
                else:
                    generated = full_model.model.decoder.generate(
                        encoder_hidden_states=encoder_hidden,
                        max_length=args.max_length,
                        num_beams=args.num_beams,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        repetition_penalty=2.5,
                        pad_token_id=full_model.tokenizer.pad_token_id,
                        eos_token_id=full_model.tokenizer.eos_token_id,
                    )

                # Decode
                pred_text = full_model.tokenizer.decode(generated[0], skip_special_tokens=True)
                ref_text = full_model.tokenizer.decode(labels[0], skip_special_tokens=True)

                results.append({
                    'generated_report': pred_text,
                    'ground_truth': ref_text,
                    'encoder_mean': encoder_mean,
                    'encoder_std': encoder_std
                })

                # Print first 5 samples
                if batch_idx < 5:
                    if batch_idx == 0:
                        print("\n" + "="*80)
                        print("SAMPLE GENERATIONS (with encoder stats):")
                        print("="*80)
                    print(f"\nSample {batch_idx + 1}:")
                    print(f"Encoder stats: mean={encoder_mean:.4f}, std={encoder_std:.4f}")
                    print(f"Generated: {pred_text[:200]}...")
                    print(f"Reference: {ref_text[:200]}...")
                    if batch_idx == 4:
                        print("="*80 + "\n")

            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    if len(results) == 0:
        print("\n ERROR: No results generated!")
        return

    # Analyze encoder diversity
    print("\n" + "="*80)
    print("ENCODER OUTPUT ANALYSIS:")
    print("="*80)
    encoder_means = [s['mean'] for s in encoder_stats]
    encoder_stds = [s['std'] for s in encoder_stats]

    print(f"Encoder mean range: [{min(encoder_means):.4f}, {max(encoder_means):.4f}]")
    print(f"Encoder mean std: {np.std(encoder_means):.6f}")
    print(f"Encoder std range: [{min(encoder_stds):.4f}, {max(encoder_stds):.4f}]")

    if np.std(encoder_means) < 1e-6:
        print("\n  WARNING: Encoder outputs are IDENTICAL!")
        print("   This means the frozen encoder is not discriminating between images.")
        print("   Solution: Unfreeze last 2-3 encoder layers and retrain.")
    else:
        print("\n Encoder outputs are DIVERSE (good!)")

    # Check output diversity
    unique_outputs = len(set([r['generated_report'] for r in results]))
    print(f"\nUnique generated reports: {unique_outputs} / {len(results)}")
    if unique_outputs == 1:
        print(" Mode collapse: ALL outputs are identical!")
    elif unique_outputs < len(results) * 0.3:
        print("  Low diversity: Many duplicate outputs")
    else:
        print(" Good diversity!")

    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} results to {args.output_csv}")

    # Compute metrics
    print("\nComputing metrics...")
    all_preds = df['generated_report'].tolist()
    all_refs = df['ground_truth'].tolist()

    valid_pairs = [(p, r) for p, r in zip(all_preds, all_refs) 
                   if p and r and p.strip() and r.strip()]

    if len(valid_pairs) == 0:
        print("\n ERROR: No valid pairs!")
        return

    valid_preds, valid_refs = zip(*valid_pairs)

    rouge_scores = rouge(list(valid_preds), list(valid_refs))
    bleu_score = bleu(list(valid_preds), [[ref] for ref in valid_refs])

    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Total samples: {len(results)}")
    print(f"Valid samples: {len(valid_pairs)}")
    print(f"Unique outputs: {unique_outputs}")
    print(f"\nROUGE-1: {rouge_scores['rouge1_fmeasure']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2_fmeasure']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL_fmeasure']:.4f}")
    print(f"BLEU-2: {bleu_score:.4f}")
    print(f"{'='*80}\n")

    # Save metrics
    metrics_file = args.output_csv.replace('.csv', '_diverse_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("DIVERSE GENERATION EVALUATION\n")
        f.write(f"{'='*80}\n")
        f.write(f"Settings: temp={args.temperature}, top_p={args.top_p}, sample={args.do_sample}\n")
        f.write(f"Total: {len(results)}, Valid: {len(valid_pairs)}\n")
        f.write(f"Unique outputs: {unique_outputs}\n")
        f.write(f"\nEncoder diversity: {np.std(encoder_means):.6f}\n")
        f.write(f"\nROUGE-1: {rouge_scores['rouge1_fmeasure']:.4f}\n")
        f.write(f"ROUGE-2: {rouge_scores['rouge2_fmeasure']:.4f}\n")
        f.write(f"ROUGE-L: {rouge_scores['rougeL_fmeasure']:.4f}\n")
        f.write(f"BLEU-2: {bleu_score:.4f}\n")

    print(f"Metrics saved to {metrics_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                       default='./checkpoints/ved_test/final_model')
    parser.add_argument('--csv_file', type=str,
                       default='/home/muhammedg/fvlm/image_first_dataset.csv')
    parser.add_argument('--output_csv', type=str,
                       default='vision_gpt2_diverse_results.csv')

    # Diverse generation settings
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--num_beams', type=int, default=1)

    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--subset_size', type=int, default=None)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    evaluate_model(args)

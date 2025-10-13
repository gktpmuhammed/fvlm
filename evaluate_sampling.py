#!/usr/bin/env python3
"""
SAMPLING-BASED EVALUATION - FIXED
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import argparse
import os
import numpy as np

from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import BLEUScore
from nltk.translate.meteor_score import meteor_score
import nltk

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

from train_vision_gpt2 import MedicalReportDataset, build_transforms
from medical_vision_gpt2 import MedicalVisionGPT2


def calculate_meteor(predictions, references):
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


def calculate_accuracy_green(predictions, references):
    exact_match = sum([1 for p, r in zip(predictions, references) if p.strip() == r.strip()])
    acc = exact_match / len(predictions) if len(predictions) > 0 else 0.0

    green_scores = []
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        if len(ref_words) > 0:
            overlap = len(pred_words & ref_words) / len(ref_words)
            green_scores.append(overlap)
        else:
            green_scores.append(0.0)

    return acc, np.mean(green_scores) if green_scores else 0.0


def evaluate_model(args):
    print("="*80)
    print("SAMPLING-BASED EVALUATION: Testing Output Diversity")
    print("="*80)

    print(f"\n Generation Settings:")
    print(f"  Temperature:     {args.temperature}")
    print(f"  Top-p:           {args.top_p}")
    print(f"  Top-k:           {args.top_k}")
    print(f"  Do sample:       True")
    print(f"  Repetition pen:  {args.repetition_penalty}")
    print("="*80)

    print(f"\nLoading model...")
    vision_encoder_path = '/home/muhammedg/fvlm/checkpoints/model.pth'

    full_model = MedicalVisionGPT2(
        vision_encoder_path=vision_encoder_path,
        decoder_model_name="gpt2",
        freeze_encoder=False,
        freeze_decoder_base=False
    )

    print(f"Loading trained DECODER weights from {args.model_path}...")
    from transformers import VisionEncoderDecoderModel
    trained_model = VisionEncoderDecoderModel.from_pretrained(args.model_path)

    # FIXED: Only load decoder, keep custom 3D ViT encoder!
    full_model.model.decoder = trained_model.decoder
    print("  Loaded trained decoder weights")
    print("  Keeping custom 3D ViT encoder (not from checkpoint)")

    full_model.model.eval()
    full_model.model.cuda()

    transform = build_transforms()

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

    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize metrics
    rouge = ROUGEScore()
    bleu1 = BLEUScore(n_gram=1)
    bleu2 = BLEUScore(n_gram=2)
    bleu3 = BLEUScore(n_gram=3)
    bleu4 = BLEUScore(n_gram=4)

    results = []

    print("\n Generating reports with SAMPLING...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            try:
                pixel_values = batch['pixel_values'].cuda()
                labels = batch['labels'].cuda()

                # Encode with custom 3D ViT
                encoder_outputs = full_model.model.encoder(pixel_values, return_dict=True)

                # SAMPLING-BASED GENERATION
                generated = full_model.model.decoder.generate(
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    max_length=args.max_length,

                    # SAMPLING PARAMETERS
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,

                    # DIVERSITY CONTROLS
                    no_repeat_ngram_size=3,
                    repetition_penalty=args.repetition_penalty,

                    # STANDARD
                    pad_token_id=full_model.tokenizer.pad_token_id,
                    eos_token_id=full_model.tokenizer.eos_token_id,
                )

                # Decode
                pred_text = full_model.tokenizer.decode(generated[0], skip_special_tokens=True)

                # Clean labels
                labels_clean = labels[0].clone()
                labels_clean = labels_clean[labels_clean != -100]
                ref_text = full_model.tokenizer.decode(labels_clean, skip_special_tokens=True)

                results.append({
                    'generated_report': pred_text,
                    'ground_truth': ref_text
                })

                # Print first 5
                if batch_idx < 5:
                    if batch_idx == 0:
                        print("\n" + "="*80)
                        print("SAMPLE GENERATIONS (with sampling):")
                        print("="*80)
                    print(f"\nSample {batch_idx + 1}:")
                    print(f"Generated: {pred_text[:300]}")
                    print(f"Reference: {ref_text[:300]}")
                    if batch_idx == 4:
                        print("="*80 + "\n")

            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                continue

    if len(results) == 0:
        print("\n No results!")
        return

    print(f"\nSaving results to {args.output_csv}...")
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} results")

    # Compute metrics
    print("\nComputing metrics...")
    all_preds = df['generated_report'].tolist()
    all_refs = df['ground_truth'].tolist()

    valid_pairs = [(p, r) for p, r in zip(all_preds, all_refs) 
                   if p and r and p.strip() and r.strip()]

    if len(valid_pairs) == 0:
        print("\n No valid pairs!")
        return

    valid_preds, valid_refs = zip(*valid_pairs)

    print("  Computing scores...")
    rouge_scores = rouge(list(valid_preds), list(valid_refs))
    bleu1_score = bleu1(list(valid_preds), [[ref] for ref in valid_refs])
    bleu2_score = bleu2(list(valid_preds), [[ref] for ref in valid_refs])
    bleu3_score = bleu3(list(valid_preds), [[ref] for ref in valid_refs])
    bleu4_score = bleu4(list(valid_preds), [[ref] for ref in valid_refs])
    meteor = calculate_meteor(list(valid_preds), list(valid_refs))
    acc, green = calculate_accuracy_green(list(valid_preds), list(valid_refs))

    unique_outputs = len(set(all_preds))

    # Analyze diversity
    print("\n" + "="*80)
    print(" DIVERSITY ANALYSIS:")
    print("="*80)
    print(f"  Total samples:      {len(results)}")
    print(f"  Unique outputs:     {unique_outputs}")
    print(f"  Diversity ratio:    {unique_outputs/len(results):.2%}")

    if unique_outputs == 1:
        print("\n   SEVERE MODE COLLAPSE: All outputs identical!")
        print("     → Model fundamentally cannot generate diverse outputs")
        print("     → Need to change training approach (contrastive loss, BLIP-2, etc.)")
    elif unique_outputs < len(results) * 0.3:
        print("\n    LOW DIVERSITY: Many duplicate outputs")
        print("     → Model has learned limited variations")
        print("     → Consider higher temperature or different training")
    elif unique_outputs < len(results) * 0.7:
        print("\n    MODERATE DIVERSITY: Some variations")
        print("     → Model can generate different outputs")
        print("     → Beam search may be suppressing diversity")
    else:
        print("\n   GOOD DIVERSITY: Most outputs unique")
        print("     → Model learned proper image-specific generation!")

    # Display results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS (Sampling-based)")
    print(f"{'='*80}")
    print(f"\n{''*80}")
    print("ACCURACY METRICS:")
    print(f"{''*80}")
    print(f"  ACC:      {acc:.4f}")
    print(f"  GREEN:    {green:.4f}")
    print(f"\n{''*80}")
    print("N-GRAM METRICS:")
    print(f"{''*80}")
    print(f"  BLEU-1:   {bleu1_score:.4f}")
    print(f"  BLEU-2:   {bleu2_score:.4f}")
    print(f"  BLEU-3:   {bleu3_score:.4f}")
    print(f"  BLEU-4:   {bleu4_score:.4f}")
    print(f"\n{''*80}")
    print("SEMANTIC METRICS:")
    print(f"{''*80}")
    print(f"  METEOR:   {meteor:.4f}")
    print(f"  ROUGE-1:  {rouge_scores['rouge1_fmeasure']:.4f}")
    print(f"  ROUGE-2:  {rouge_scores['rouge2_fmeasure']:.4f}")
    print(f"  ROUGE-L:  {rouge_scores['rougeL_fmeasure']:.4f}")
    print(f"{'='*80}\n")

    # Save metrics
    metrics_file = args.output_csv.replace('.csv', '_sampling_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("SAMPLING-BASED EVALUATION\n")
        f.write(f"{'='*80}\n")
        f.write(f"Settings: temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}\n")
        f.write(f"\nDiversity: {unique_outputs}/{len(results)} = {unique_outputs/len(results):.2%}\n")
        f.write(f"\nMETRICS:\n")
        f.write(f"  ACC:      {acc:.4f}\n")
        f.write(f"  GREEN:    {green:.4f}\n")
        f.write(f"  BLEU-1:   {bleu1_score:.4f}\n")
        f.write(f"  BLEU-2:   {bleu2_score:.4f}\n")
        f.write(f"  BLEU-3:   {bleu3_score:.4f}\n")
        f.write(f"  BLEU-4:   {bleu4_score:.4f}\n")
        f.write(f"  METEOR:   {meteor:.4f}\n")
        f.write(f"  ROUGE-1:  {rouge_scores['rouge1_fmeasure']:.4f}\n")
        f.write(f"  ROUGE-2:  {rouge_scores['rouge2_fmeasure']:.4f}\n")
        f.write(f"  ROUGE-L:  {rouge_scores['rougeL_fmeasure']:.4f}\n")

    print(f"Metrics saved to {metrics_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, 
                       default='./checkpoints/gpt2_full_unfrozen/checkpoint-4900')
    parser.add_argument('--csv_file', type=str,
                       default='/home/muhammedg/fvlm/image_first_dataset.csv')
    parser.add_argument('--output_csv', type=str,
                       default='gpt2_sampling_results.csv')

    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--repetition_penalty', type=float, default=2.5)

    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--subset_size', type=int, default=50)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    evaluate_model(args)

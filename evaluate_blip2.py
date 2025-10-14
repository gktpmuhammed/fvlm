#!/usr/bin/env python3
"""
Evaluate Medical BLIP-2 Model
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

from train_blip2 import MedicalReportDataset, build_transforms
from medical_blip2 import MedicalBLIP2


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


def evaluate(args):
    print("="*80)
    print("EVALUATING MEDICAL BLIP-2")
    print("="*80)

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = MedicalBLIP2(
        vision_encoder_path='/home/muhammedg/fvlm/checkpoints/model.pth',
        language_model="facebook/opt-350m",
        num_query_tokens=32,
        freeze_vision_encoder=True,
    )

    # Load trained weights
    model.qformer.load_state_dict(torch.load(f"{args.model_path}/qformer.pt"))
    model.language_projection.load_state_dict(torch.load(f"{args.model_path}/projection.pt"))
    model.query_tokens.data = torch.load(f"{args.model_path}/query_tokens.pt")

    from transformers import OPTForCausalLM, AutoTokenizer
    model.language_model = OPTForCausalLM.from_pretrained(f"{args.model_path}/language_model")
    model.tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}/tokenizer")

    print("  Model loaded successfully!")

    model.eval()
    model.cuda()

    # Load dataset
    transform = build_transforms()
    val_dataset = MedicalReportDataset(
        csv_file=args.csv_file,
        tokenizer=model.tokenizer,
        transform=transform,
        max_length=512,
        split='validation',
        subset_size=args.subset_size
    )

    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize metrics
    rouge = ROUGEScore()
    bleu1 = BLEUScore(n_gram=1)
    bleu2 = BLEUScore(n_gram=2)
    bleu3 = BLEUScore(n_gram=3)
    bleu4 = BLEUScore(n_gram=4)

    results = []

    print(f"\nGenerating reports for {len(val_dataset)} samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            try:
                pixel_values = batch['pixel_values'].cuda()
                labels = batch['labels'].cuda()

                # Generate
                generated_ids = model.generate(
                    pixel_values=pixel_values,
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=2.0,
                )

                # Decode
                pred_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                # Clean reference
                labels_clean = labels[0].clone()
                labels_clean = labels_clean[labels_clean != -100]
                ref_text = model.tokenizer.decode(labels_clean, skip_special_tokens=True)

                results.append({
                    'generated_report': pred_text,
                    'ground_truth': ref_text
                })

                # Print first 3
                if batch_idx < 3:
                    if batch_idx == 0:
                        print("\n" + "="*80)
                        print("SAMPLE GENERATIONS:")
                        print("="*80)
                    print(f"\nSample {batch_idx + 1}:")
                    print(f"Generated: {pred_text[:250]}")
                    print(f"Reference: {ref_text[:250]}")
                    if batch_idx == 2:
                        print("="*80 + "\n")

            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                continue

    if len(results) == 0:
        print("\n No results!")
        return

    # Save results
    print(f"\nSaving results to {args.output_csv}...")
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)

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

    # Calculate metrics
    rouge_scores = rouge(list(valid_preds), list(valid_refs))
    bleu1_score = bleu1(list(valid_preds), [[ref] for ref in valid_refs])
    bleu2_score = bleu2(list(valid_preds), [[ref] for ref in valid_refs])
    bleu3_score = bleu3(list(valid_preds), [[ref] for ref in valid_refs])
    bleu4_score = bleu4(list(valid_preds), [[ref] for ref in valid_refs])
    meteor = calculate_meteor(list(valid_preds), list(valid_refs))
    acc, green = calculate_accuracy_green(list(valid_preds), list(valid_refs))

    unique_outputs = len(set(all_preds))

    # Display
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS - BLIP-2")
    print(f"{'='*80}")
    print(f"Total: {len(results)}, Valid: {len(valid_pairs)}, Unique: {unique_outputs}")
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
    metrics_file = args.output_csv.replace('.csv', '_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"BLIP-2 EVALUATION\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total: {len(results)}, Valid: {len(valid_pairs)}, Unique: {unique_outputs}\n")
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
                       default='./checkpoints/blip2_medical/final_model')
    parser.add_argument('--csv_file', type=str,
                       default='/home/muhammedg/fvlm/image_first_dataset.csv')
    parser.add_argument('--output_csv', type=str,
                       default='blip2_results.csv')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--subset_size', type=int, default=None)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    evaluate(args)

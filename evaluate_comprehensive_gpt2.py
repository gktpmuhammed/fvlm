#!/usr/bin/env python3
"""
Comprehensive Evaluation for Medical Vision-GPT2
Implements all standard medical report generation metrics
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import argparse
import os
import numpy as np

# Metrics
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import BLEUScore
from nltk.translate.meteor_score import meteor_score
import nltk

# Download NLTK data
try:
    nltk.data.find('wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')

from train_vision_gpt2 import MedicalReportDataset, build_transforms
from medical_vision_gpt2 import MedicalVisionGPT2


def calculate_cider(predictions, references):
    """Calculate CIDEr score (consensus-based metric)"""
    try:
        from pycocoevalcap.cider.cider import Cider

        # Format for CIDEr
        gts = {i: [ref] for i, ref in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}

        scorer = Cider()
        score, scores = scorer.compute_score(gts, res)
        return score, scores
    except ImportError:
        print("  CIDEr requires pycocoevalcap: pip install pycocoevalcap")
        return None, None


def calculate_meteor(predictions, references):
    """Calculate METEOR score"""
    scores = []
    for pred, ref in zip(predictions, references):
        try:
            # Tokenize
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            score = meteor_score([ref_tokens], pred_tokens)
            scores.append(score)
        except:
            scores.append(0.0)
    return np.mean(scores)


def calculate_accuracy_green(predictions, references, entity_list=None):
    """
    Calculate accuracy metrics:
    - ACC: Exact match accuracy
    - GREEN: Clinical entity overlap
    """
    exact_match = sum([1 for p, r in zip(predictions, references) if p.strip() == r.strip()])
    acc = exact_match / len(predictions)

    # GREEN score (simplified - measures clinical entity overlap)
    # For full implementation, you'd need medical entity extraction
    green_scores = []
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        if len(ref_words) > 0:
            overlap = len(pred_words & ref_words) / len(ref_words)
            green_scores.append(overlap)
        else:
            green_scores.append(0.0)

    green = np.mean(green_scores)
    return acc, green


def evaluate_model(args):
    print("="*80)
    print("COMPREHENSIVE EVALUATION: Medical Vision-GPT2")
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

    # Load trained weights
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
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize metrics
    rouge = ROUGEScore()
    bleu1 = BLEUScore(n_gram=1)
    bleu2 = BLEUScore(n_gram=2)
    bleu3 = BLEUScore(n_gram=3)
    bleu4 = BLEUScore(n_gram=4)

    # Results
    results = []

    print("\nGenerating reports...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            try:
                pixel_values = batch['pixel_values'].cuda()
                labels = batch['labels'].cuda()

                # Encode
                encoder_outputs = full_model.model.encoder(pixel_values, return_dict=True)

                # Generate
                generated = full_model.model.decoder.generate(
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=2.0,
                    pad_token_id=full_model.tokenizer.pad_token_id,
                    eos_token_id=full_model.tokenizer.eos_token_id,
                    temperature=0.9
                )

                labels_clean = labels[0].clone()
                labels_clean = labels_clean[labels_clean != -100]  # Remove padding mask
                ref_text = full_model.tokenizer.decode(labels_clean, skip_special_tokens=True)

                # Decode
                pred_text = full_model.tokenizer.decode(generated[0], skip_special_tokens=True)
                # ref_text = full_model.tokenizer.decode(labels[0], skip_special_tokens=True)

                results.append({
                    'generated_report': pred_text,
                    'ground_truth': ref_text
                })

                # Print first 3 samples
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
    print(f"Saved {len(df)} results")

    # Compute all metrics
    print("\nComputing comprehensive metrics...")
    all_preds = df['generated_report'].tolist()
    all_refs = df['ground_truth'].tolist()

    # Filter valid pairs
    valid_pairs = [(p, r) for p, r in zip(all_preds, all_refs) 
                   if p and r and p.strip() and r.strip()]

    if len(valid_pairs) == 0:
        print("\n No valid pairs!")
        return

    valid_preds, valid_refs = zip(*valid_pairs)

    # Calculate all metrics
    print("  Computing ROUGE...")
    rouge_scores = rouge(list(valid_preds), list(valid_refs))

    print("  Computing BLEU...")
    bleu1_score = bleu1(list(valid_preds), [[ref] for ref in valid_refs])
    bleu2_score = bleu2(list(valid_preds), [[ref] for ref in valid_refs])
    bleu3_score = bleu3(list(valid_preds), [[ref] for ref in valid_refs])
    bleu4_score = bleu4(list(valid_preds), [[ref] for ref in valid_refs])

    print("  Computing METEOR...")
    meteor = calculate_meteor(list(valid_preds), list(valid_refs))

    print("  Computing ACC & GREEN...")
    acc, green = calculate_accuracy_green(list(valid_preds), list(valid_refs))

    print("  Computing CIDEr...")
    cider_score, _ = calculate_cider(list(valid_preds), list(valid_refs))

    # Count unique outputs
    unique_outputs = len(set(all_preds))

    # Display results
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Total samples: {len(results)}")
    print(f"Valid samples: {len(valid_pairs)}")
    print(f"Unique outputs: {unique_outputs}")
    print(f"\n{''*80}")
    print("ACCURACY METRICS:")
    print(f"{''*80}")
    print(f"  ACC (Exact Match):  {acc:.4f}")
    print(f"  GREEN (Overlap):    {green:.4f}")
    print(f"\n{''*80}")
    print("N-GRAM METRICS:")
    print(f"{''*80}")
    print(f"  BLEU-1:            {bleu1_score:.4f}")
    print(f"  BLEU-2:            {bleu2_score:.4f}")
    print(f"  BLEU-3:            {bleu3_score:.4f}")
    print(f"  BLEU-4:            {bleu4_score:.4f}")
    print(f"\n{''*80}")
    print("SEMANTIC METRICS:")
    print(f"{''*80}")
    print(f"  METEOR:            {meteor:.4f}")
    print(f"  ROUGE-1:           {rouge_scores['rouge1_fmeasure']:.4f}")
    print(f"  ROUGE-2:           {rouge_scores['rouge2_fmeasure']:.4f}")
    print(f"  ROUGE-L:           {rouge_scores['rougeL_fmeasure']:.4f}")
    if cider_score is not None:
        print(f"  CIDEr:             {cider_score:.4f}")
    else:
        print(f"  CIDEr:             N/A (install pycocoevalcap)")
    print(f"{'='*80}\n")

    # Save metrics to file
    metrics_file = args.output_csv.replace('.csv', '_comprehensive_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("COMPREHENSIVE EVALUATION RESULTS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total: {len(results)}, Valid: {len(valid_pairs)}, Unique: {unique_outputs}\n")
        f.write(f"\nACCURACY METRICS:\n")
        f.write(f"  ACC:      {acc:.4f}\n")
        f.write(f"  GREEN:    {green:.4f}\n")
        f.write(f"\nN-GRAM METRICS:\n")
        f.write(f"  BLEU-1:   {bleu1_score:.4f}\n")
        f.write(f"  BLEU-2:   {bleu2_score:.4f}\n")
        f.write(f"  BLEU-3:   {bleu3_score:.4f}\n")
        f.write(f"  BLEU-4:   {bleu4_score:.4f}\n")
        f.write(f"\nSEMANTIC METRICS:\n")
        f.write(f"  METEOR:   {meteor:.4f}\n")
        f.write(f"  ROUGE-1:  {rouge_scores['rouge1_fmeasure']:.4f}\n")
        f.write(f"  ROUGE-2:  {rouge_scores['rouge2_fmeasure']:.4f}\n")
        f.write(f"  ROUGE-L:  {rouge_scores['rougeL_fmeasure']:.4f}\n")
        if cider_score is not None:
            f.write(f"  CIDEr:    {cider_score:.4f}\n")

    print(f"Comprehensive metrics saved to {metrics_file}")

    # Create summary table
    summary = pd.DataFrame([{
        'ACC': acc,
        'GREEN': green,
        'BLEU-1': bleu1_score.item(),
        'BLEU-2': bleu2_score.item(),
        'BLEU-3': bleu3_score.item(),
        'BLEU-4': bleu4_score.item(),
        'METEOR': meteor,
        'ROUGE-1': rouge_scores['rouge1_fmeasure'].item(),
        'ROUGE-2': rouge_scores['rouge2_fmeasure'].item(),
        'ROUGE-L': rouge_scores['rougeL_fmeasure'].item(),
        'CIDEr': cider_score if cider_score else 0.0,
        'Unique': unique_outputs,
        'Total': len(results)
    }])

    summary_file = args.output_csv.replace('.csv', '_summary.csv')
    summary.to_csv(summary_file, index=False)
    print(f"Summary table saved to {summary_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, 
                       default='./checkpoints/gpt2_full/checkpoint-1450')
    parser.add_argument('--csv_file', type=str,
                       default='/home/muhammedg/fvlm/image_first_dataset.csv')
    parser.add_argument('--output_csv', type=str,
                       default='gpt2_comprehensive_results.csv')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Subset size for quick eval (e.g., 50)')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    evaluate_model(args)

#!/usr/bin/env python3
"""
Evaluation for Medical Vision-GPT2 - Final Working Version
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import BLEUScore
import argparse
import os

from train_vision_gpt2 import MedicalReportDataset, build_transforms
from medical_vision_gpt2 import MedicalVisionGPT2


def evaluate_model(args):
    print("="*80)
    print("EVALUATION: Medical Vision-GPT2")
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

    # Load trained decoder weights
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

    # Dataloader with batch_size=1 to avoid issues
    dataloader = DataLoader(
        val_dataset,
        batch_size=1,  # Process one at a time
        shuffle=False,
        num_workers=0
    )

    # Metrics
    rouge = ROUGEScore()
    bleu = BLEUScore(n_gram=2)

    # Collect results
    results = []
    sample_count = 0

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
                )

                # Decode (batch_size=1 so just index 0)
                pred_text = full_model.tokenizer.decode(generated[0], skip_special_tokens=True)
                ref_text = full_model.tokenizer.decode(labels[0], skip_special_tokens=True)

                results.append({
                    'generated_report': pred_text,
                    'ground_truth': ref_text
                })

                # Print first few samples
                if sample_count < 5:
                    if sample_count == 0:
                        print("\n" + "="*80)
                        print("SAMPLE GENERATIONS:")
                        print("="*80)
                    print(f"\nSample {sample_count + 1}:")
                    print(f"Generated: {pred_text[:250]}")
                    print(f"Reference: {ref_text[:250]}")
                    if sample_count == 4:
                        print("="*80 + "\n")
                    sample_count += 1

            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    if len(results) == 0:
        print("\n ERROR: No results generated!")
        return

    # Save results
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
        print("\n ERROR: No valid prediction pairs!")
        return

    valid_preds, valid_refs = zip(*valid_pairs)

    rouge_scores = rouge(list(valid_preds), list(valid_refs))
    bleu_score = bleu(list(valid_preds), [[ref] for ref in valid_refs])

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
        f.write("EVALUATION RESULTS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total: {len(results)}, Valid: {len(valid_pairs)}\n")
        f.write(f"\nROUGE-1: {rouge_scores['rouge1_fmeasure']:.4f}\n")
        f.write(f"ROUGE-2: {rouge_scores['rouge2_fmeasure']:.4f}\n")
        f.write(f"ROUGE-L: {rouge_scores['rougeL_fmeasure']:.4f}\n")
        f.write(f"BLEU-2: {bleu_score:.4f}\n")

    print(f"Metrics saved to {metrics_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                       default='./checkpoints/vision_gpt2/final_model')
    parser.add_argument('--csv_file', type=str,
                       default='/home/muhammedg/fvlm/image_first_dataset.csv')
    parser.add_argument('--output_csv', type=str,
                       default='vision_gpt2_results.csv')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--subset_size', type=int, default=None)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    evaluate_model(args)

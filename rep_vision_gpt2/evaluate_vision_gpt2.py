#!/usr/bin/env python3
"""
Evaluation for Medical Vision-GPT2
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
from transformers import VisionEncoderDecoderModel

def evaluate_model(args):
    print(f"Loading model from {args.model_path}...")
    
    # Load model structure
    full_model = MedicalVisionGPT2(
        vision_encoder_path='/home/muhammedg/fvlm/checkpoints/model.pth', # Dummy path, purely for init
        decoder_model_name="gpt2"
    )
    
    # Load Trained Weights
    trained_model = VisionEncoderDecoderModel.from_pretrained(args.model_path)
    full_model.model = trained_model
    full_model.model.eval()
    full_model.model.cuda()

    # Data
    transform = build_transforms()
    val_dataset = MedicalReportDataset(
        csv_file=args.csv_file,
        tokenizer=full_model.tokenizer,
        transform=transform,
        max_length=512,
        split='validation',
        subset_size=args.subset_size
    )
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    results = []
    
    print("Generating reports...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pixel_values = batch['pixel_values'].cuda()
            labels = batch['labels'].cuda()
            
            # Generate with anti-repetition settings
            generated_ids = full_model.model.generate(
                pixel_values,
                max_length=args.max_length,
                num_beams=4,
                no_repeat_ngram_size=3, # Stops "Trachea trachea trachea"
                repetition_penalty=2.0, # Penalizes repeating words generally
                early_stopping=True,
                length_penalty=1.0,
                pad_token_id=full_model.tokenizer.pad_token_id,
                eos_token_id=full_model.tokenizer.eos_token_id,
            )
            
            pred_text = full_model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            ref_text = full_model.tokenizer.decode(labels[0], skip_special_tokens=True)
            
            results.append({'generated': pred_text, 'ground_truth': ref_text})

    # Save
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved results to {args.output_csv}")
    
    # Calculate Metrics
    rouge = ROUGEScore()
    bleu = BLEUScore(n_gram=2)
    
    preds = df['generated'].tolist()
    refs = df['ground_truth'].tolist()
    
    # Filter empty
    valid_data = [(p, r) for p, r in zip(preds, refs) if len(str(p)) > 5 and len(str(r)) > 5]
    if not valid_data:
        print("No valid data for metrics")
        return
        
    p_valid, r_valid = zip(*valid_data)
    
    print("\nMetrics:")
    print(f"ROUGE-L: {rouge(list(p_valid), list(r_valid))['rougeL_fmeasure']:.4f}")
    print(f"BLEU-2: {bleu(list(p_valid), [[r] for r in r_valid]):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/vision_gpt2/final_model')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv')
    parser.add_argument('--output_csv', type=str, default='vision_gpt2_results.csv')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--subset_size', type=int, default=None)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    evaluate_model(args)
#!/usr/bin/env python3
"""
Evaluation for Medical Vision-GPT2
Fixed: Handling -100 in labels before decoding.
"""

import sys
import os

# ------------------------------------------------------------------
# FIX: Insert parent directory at the BEGINNING of sys.path
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import BLEUScore
import argparse

from train_vision_gpt2 import MedicalReportDataset, build_transforms
from medical_vision_gpt2 import MedicalVisionGPT2
from transformers import VisionEncoderDecoderModel

def evaluate_model(args):
    print(f"Loading model structure...")
    
    # 1. Initialize structure
    full_model = MedicalVisionGPT2(
        vision_encoder_path='/home/muhammedg/fvlm/checkpoints/model.pth', 
        decoder_model_name="gpt2"
    )
    
    # 2. Load Trained Weights MANUALLY
    print(f"Loading trained weights from {args.model_path}...")
    
    if os.path.isdir(args.model_path):
        weights_path = os.path.join(args.model_path, "pytorch_model.bin")
    else:
        weights_path = args.model_path

    if not os.path.exists(weights_path):
        weights_path = os.path.join(args.model_path, "model.safetensors")
        if os.path.exists(weights_path):
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            raise FileNotFoundError(f"Could not find weights at {args.model_path}")
    else:
        state_dict = torch.load(weights_path, map_location='cpu')

    msg = full_model.model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}")

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
            if batch is None: continue
            
            pixel_values = batch['pixel_values'].cuda()
            labels = batch['labels'].cuda()
            
            # Anti-repetition settings
            generated_ids = full_model.model.generate(
                pixel_values,
                max_length=args.max_length,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=2.0,
                early_stopping=True,
                length_penalty=1.0,
                pad_token_id=full_model.tokenizer.pad_token_id,
                eos_token_id=full_model.tokenizer.eos_token_id,
            )
            
            pred_text = full_model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # --- FIX: Handle -100 in labels ---
            # Create a copy to avoid modifying original tensor if needed
            label_ids = labels[0].clone()
            # Replace -100 with pad_token_id so tokenizer doesn't crash
            label_ids[label_ids == -100] = full_model.tokenizer.pad_token_id
            
            ref_text = full_model.tokenizer.decode(label_ids, skip_special_tokens=True)
            # ----------------------------------
            
            results.append({'generated': pred_text, 'ground_truth': ref_text})

    # Save results immediately
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
    parser.add_argument('--output_csv', type=str, default='vision_gpt2_results_attention.csv')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--subset_size', type=int, default=None)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    evaluate_model(args)
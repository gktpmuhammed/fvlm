#!/usr/bin/env python3
"""
Improved evaluation with anti-repetition mechanisms
"""
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import BLEUScore
from improved_medical_vlm import ImprovedMedicalVLM
from train_improved_vlm import ImageFirstDataset, build_transforms
import os

def evaluate_model(model_path, csv_file, output_csv='improved_results.csv'):
    # Manually initialize model and load state dict
    model = ImprovedMedicalVLM(
        vision_encoder_path='/home/muhammedg/fvlm/checkpoints/model.pth' # Dummy path, weights will be overwritten
    )
    model_weights_path = os.path.join(model_path, 'pytorch_model.bin')
    model.load_state_dict(torch.load(model_weights_path, map_location='cuda'))
    
    model.eval()
    model.cuda()
    
    transform = build_transforms()
    val_dataset = ImageFirstDataset(
        csv_file=csv_file,
        tokenizer=model.tokenizer,
        transform=transform,
        split='validation',
        subset_size=100
    )
    
    dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    rouge = ROUGEScore()
    bleu = BLEUScore(n_gram=2)
    
    results = []
    for batch in tqdm(dataloader):
        images = batch['images'].cuda()
        
        # Generate with anti-repetition
        generated = model.generate(
            images=images,
            max_length=256,
            num_beams=5,
            temperature=1.0,
            repetition_penalty=2.0,  # KEY: Prevents collapse
            no_repeat_ngram_size=3,
            length_penalty=1.0
        )
        
        predictions = model.tokenizer.batch_decode(generated, skip_special_tokens=True)
        references = model.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        
        for pred, ref in zip(predictions, references):
            results.append({
                'generated_report': pred,
                'ground_truth': ref
            })
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # Compute metrics
    all_preds = df['generated_report'].tolist()
    all_refs = df['ground_truth'].tolist()
    
    rouge_scores = rouge(all_preds, all_refs)
    bleu_score = bleu(all_preds, [[ref] for ref in all_refs])
    
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"ROUGE-1: {rouge_scores['rouge1_fmeasure']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2_fmeasure']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL_fmeasure']:.4f}")
    print(f"BLEU-2: {bleu_score:.4f}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    evaluate_model(
        model_path='/home/muhammedg/fvlm/checkpoints/improved_vlm/final_model',
        csv_file='/home/muhammedg/fvlm/image_first_dataset.csv',
        output_csv='/home/muhammedg/fvlm/evaluation_results_improved.csv'
    )
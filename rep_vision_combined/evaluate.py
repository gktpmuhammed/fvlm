#!/usr/bin/env python3
"""
Unified Evaluation Script for Medical VLM
Fixed: Weight Loading Logic for Q-Former/Combined Models
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from collections import Counter, defaultdict
import math

# Metrics
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import sentence_gleu
import nltk

try:
    nltk.data.find('wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

from train import MedicalReportDataset, build_transforms
from medical_vlm import MedicalVLM

# ------------------------------------------------------------------------------
# METRIC FUNCTIONS (Same as before)
# ------------------------------------------------------------------------------

def calculate_meteor(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        try: scores.append(meteor_score([ref.split()], pred.split()))
        except: scores.append(0.0)
    return np.mean(scores)

def calculate_bleu_scores(predictions, references):
    references_tok = [[ref.split()] for ref in references]
    hypotheses_tok = [pred.split() for pred in predictions]
    weights = [(1.0, 0, 0, 0), (0.5, 0.5, 0, 0), (1/3, 1/3, 1/3, 0), (0.25, 0.25, 0.25, 0.25)]
    scores = {}
    for i, w in enumerate(weights, start=1):
        try: scores[f'bleu{i}'] = corpus_bleu(references_tok, hypotheses_tok, weights=w)
        except: scores[f'bleu{i}'] = 0.0
    return scores

def calculate_greene(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        try: scores.append(sentence_gleu([ref.split()], pred.split()))
        except: scores.append(0.0)
    return np.mean(scores)

def calculate_accuracy(predictions, references):
    matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
    return matches / len(predictions) if len(predictions) > 0 else 0.0

def calculate_cider_approx(predictions, references, ngram=4):
    def ngrams(tokens, n):
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    docs = []
    for p in predictions:
        toks = p.split()
        ngs = []
        for k in range(1, ngram+1): ngs.extend(ngrams(toks, k))
        docs.append(ngs)
    for r in references:
        toks = r.split()
        ngs = []
        for k in range(1, ngram+1): ngs.extend(ngrams(toks, k))
        docs.append(ngs) 
    df = defaultdict(int)
    for doc in docs:
        for g in set(doc): df[g] += 1
    N = len(docs)
    def tf_idf(ng_list):
        tf = Counter(ng_list)
        vec = {}
        for k, v in tf.items():
            vec[k] = (v / sum(tf.values())) * math.log((N+1)/(1+df.get(k, 0)))
        return vec
    def cosine(v1, v2):
        dot = sum(v1.get(k,0)*v2.get(k,0) for k in v1)
        norm = math.sqrt(sum(v*v for v in v1.values())) * math.sqrt(sum(v*v for v in v2.values()))
        return dot/norm if norm else 0.0
    scores = []
    for p, r in zip(predictions, references):
        p_ng, r_ng = [], []
        for k in range(1, ngram+1):
            p_ng.extend(ngrams(p.split(), k))
            r_ng.extend(ngrams(r.split(), k))
        scores.append(cosine(tf_idf(p_ng), tf_idf(r_ng)))
    return np.mean(scores) * 10.0 if scores else 0.0

def create_metrics_table_plot(results_list, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    col_labels = ['Encoder', 'ACC', 'GREEN', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
    table_rows, csv_rows = [], []
    for label, res in zip(labels, results_list):
        row = [
            label,
            f"{res.get('accuracy', 0.0)*100:.1f}",
            f"{res.get('green', 0.0)*100:.1f}",
            f"{res.get('bleu1', 0.0)*100:.1f}",
            f"{res.get('bleu2', 0.0)*100:.1f}",
            f"{res.get('bleu3', 0.0)*100:.1f}",
            f"{res.get('bleu4', 0.0)*100:.1f}",
            f"{res.get('meteor', 0.0)*100:.1f}",
            f"{res.get('rougeL_fmeasure', 0.0)*100:.1f}",
            f"{res.get('cider', 0.0):.1f}",
        ]
        table_rows.append(row)
        csv_rows.append({k:v for k,v in zip(col_labels, row)})
    with open(os.path.join(output_dir, 'metrics_summary.csv'), 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=col_labels)
        writer.writeheader()
        writer.writerows(csv_rows)
    fig, ax = plt.subplots(figsize=(14, max(2, len(table_rows) * 0.8)))
    ax.axis('off')
    table = ax.table(cellText=table_rows, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#333333')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'metrics_table.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

class EvalDataset(MedicalReportDataset):
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        item = super().__getitem__(idx)
        if item:
            filename = os.path.basename(row['image_path'])
            item['patient_id'] = filename.split('.')[0]
        return item

def evaluate_model(args):
    print(f"Loading Unified Model Structure ({args.decoder_model})...")
    
    # 1. Initialize Outer Class
    full_model = MedicalVLM(
        vision_encoder_path=args.vision_encoder_path, 
        decoder_model_name=args.decoder_model,
        use_qformer=args.use_qformer,          
        num_query_tokens=args.num_query_tokens 
    )
    
    # 2. Load Weights (FIXED)
    print(f"Loading weights from {args.model_path}...")
    weights_path = os.path.join(args.model_path, "pytorch_model.bin")
    if not os.path.exists(weights_path): weights_path = args.model_path
    
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # FIX: Load into full_model.model (The VisionEncoderDecoderModel)
    # This aligns the keys: 'encoder...' matches 'encoder...' inside the inner model
    print("Applying state dict to inner model...")
    missing, unexpected = full_model.model.load_state_dict(state_dict, strict=False)
    
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    
    if len(missing) > 0:
        print("Example missing:", missing[:3])
    
    full_model.eval()
    full_model.cuda()

    transform = build_transforms()
    val_dataset = EvalDataset(args.csv_file, full_model.tokenizer, transform, args.max_length, args.subset_size, 'validation')
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    predictions, references, patient_ids = [], [], []
    
    print("Generating reports...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None: continue
            
            pixel_values = batch['pixel_values'].cuda()
            labels = batch['labels'].cuda()
            pids = batch['patient_id']
            
            generated_ids = full_model.generate(
                pixel_values,
                max_length=args.max_length,
                num_beams=args.num_beams,
                no_repeat_ngram_size=3,
                repetition_penalty=2.0,
                early_stopping=True
            )
            
            pred_batch = full_model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            labels_cpu = labels.cpu().clone()
            labels_cpu[labels_cpu == -100] = full_model.tokenizer.pad_token_id
            ref_batch = full_model.tokenizer.batch_decode(labels_cpu, skip_special_tokens=True)
            
            predictions.extend(pred_batch)
            references.extend(ref_batch)
            patient_ids.extend(pids)

    # Metrics
    valid_data = [(p, r) for p, r in zip(predictions, references) if len(str(p)) > 1]
    if not valid_data: return
    p_valid, r_valid = zip(*valid_data)
    
    print("\nComputing ROUGE...")
    rouge = ROUGEScore()(list(p_valid), list(r_valid))
    print("Computing BLEU...")
    bleu = calculate_bleu_scores(p_valid, r_valid)
    print("Computing METEOR...")
    meteor = calculate_meteor(p_valid, r_valid)
    print("Computing GREEN...")
    green = calculate_greene(p_valid, r_valid)
    print("Computing Accuracy...")
    acc = calculate_accuracy(p_valid, r_valid)
    print("Computing CIDEr...")
    cider = calculate_cider_approx(p_valid, r_valid)

    results = {
        'rougeL_fmeasure': rouge['rougeL_fmeasure'].item(),
        'bleu1': bleu['bleu1'], 'bleu2': bleu['bleu2'], 'bleu3': bleu['bleu3'], 'bleu4': bleu['bleu4'],
        'meteor': meteor, 'green': green, 'accuracy': acc, 'cider': cider
    }
    
    label = f"{args.decoder_model}_QFormer" if args.use_qformer else args.decoder_model
    create_metrics_table_plot([results], [label], args.output_dir)
    pd.DataFrame({'patient_id': patient_ids, 'prediction': predictions, 'reference': references}).to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
    
    print(f"\nResults (BLEU-4: {results['bleu4']:.4f}) saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--decoder_model', type=str, default='gpt2')
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--subset_size', type=int, default=None)
    parser.add_argument('--use_qformer', action='store_true')
    parser.add_argument('--num_query_tokens', type=int, default=32)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    evaluate_model(args)
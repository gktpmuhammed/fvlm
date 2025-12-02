#!/usr/bin/env python3
"""
Diagnostic Script: Check if Model is "Blind" (Posterior Collapse)
UPDATED: Uses exact same generation params and metrics as evaluate.py
"""

import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import argparse
from collections import Counter, defaultdict
import math

# Metrics Imports
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import sentence_gleu
import nltk

# Fix path to find local 'lavis'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)

from train import MedicalReportDataset, build_transforms
from medical_vlm import MedicalVLM

# ==============================================================================
# REUSE METRIC FUNCTIONS (Exact copies from evaluate.py)
# ==============================================================================

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

def compute_all_metrics(predictions, references):
    """Aggregates all metrics into a dictionary"""
    valid_data = [(p, r) for p, r in zip(predictions, references) if len(str(p)) > 1]
    if not valid_data: return {}
    p_valid, r_valid = zip(*valid_data)

    rouge = ROUGEScore()(list(p_valid), list(r_valid))
    bleu = calculate_bleu_scores(p_valid, r_valid)
    meteor = calculate_meteor(p_valid, r_valid)
    green = calculate_greene(p_valid, r_valid)
    acc = calculate_accuracy(p_valid, r_valid)
    cider = calculate_cider_approx(p_valid, r_valid)

    return {
        'ROUGE-L': rouge['rougeL_fmeasure'].item(),
        'BLEU-4': bleu['bleu4'],
        'METEOR': meteor,
        'GREEN': green,
        'CIDEr': cider,
        'Accuracy': acc
    }

# ==============================================================================
# MAIN TEST LOGIC
# ==============================================================================

def run_test(args):
    print(f"\nLOADING MODEL: {args.decoder_model}")
    # 1. Initialize Structure (Custom 3D ViT)
    full_model = MedicalVLM(
        vision_encoder_path='dummy', # Dummy path, weights loaded manually below
        decoder_model_name=args.decoder_model
    )
    
    # 2. Load Weights Manually
    print(f"Loading weights from {args.model_path}...")
    if os.path.isdir(args.model_path):
        weights_path = os.path.join(args.model_path, "pytorch_model.bin")
    else:
        weights_path = args.model_path

    state_dict = torch.load(weights_path, map_location='cpu')
    full_model.load_state_dict(state_dict, strict=False)
    
    full_model.eval()
    full_model.cuda()

    print(f"\nPREPARING DATA (Subset: {args.subset_size})...")
    transform = build_transforms()
    dataset = MedicalReportDataset(args.csv_file, full_model.tokenizer, transform, subset_size=args.subset_size, split='validation')
    
    # Collect all data into memory for shuffling
    all_pixels = []
    all_refs = []
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in tqdm(loader, desc="Loading tensors"):
        if batch is None: continue
        all_pixels.append(batch['pixel_values'].squeeze(0)) # Store as 4D tensors
        
        lbl = batch['labels'].clone()
        lbl[lbl == -100] = full_model.tokenizer.pad_token_id
        ref_text = full_model.tokenizer.decode(lbl[0], skip_special_tokens=True)
        all_refs.append(ref_text)

    # Convert to big tensor for batching
    all_pixels_tensor = torch.stack(all_pixels)
    
    def run_inference(pixel_tensor, ref_list, desc):
        preds = []
        # Process in batches
        for i in tqdm(range(0, len(pixel_tensor), args.batch_size), desc=desc):
            batch_px = pixel_tensor[i : i + args.batch_size].cuda()
            
            with torch.no_grad():
                # --- KEY UPDATE: EXACT MATCH with evaluate.py params ---
                gen_ids = full_model.model.generate(
                    batch_px,
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    no_repeat_ngram_size=3,       # Prevents loops
                    repetition_penalty=2.0,       # Prevents repetitions
                    early_stopping=True,
                    length_penalty=1.0,
                    pad_token_id=full_model.tokenizer.pad_token_id,
                    eos_token_id=full_model.tokenizer.eos_token_id,
                )
            batch_preds = full_model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            preds.extend(batch_preds)
        return preds

    # 1. CORRECT PAIRS
    print("\n>>> RUN 1: CORRECT IMAGE-TEXT PAIRS")
    preds_correct = run_inference(all_pixels_tensor, all_refs, "Generating (Correct)")
    metrics_correct = compute_all_metrics(preds_correct, all_refs)

    # 2. SHUFFLED PAIRS
    print("\n>>> RUN 2: SHUFFLED IMAGES (Randomized)")
    # Shuffle the pixel tensor along dimension 0
    idx = torch.randperm(all_pixels_tensor.size(0))
    shuffled_pixels = all_pixels_tensor[idx]
    
    preds_shuffled = run_inference(shuffled_pixels, all_refs, "Generating (Shuffled)")
    metrics_shuffled = compute_all_metrics(preds_shuffled, all_refs)

    # 3. PRINT COMPARISON
    print("\n" + "="*65)
    print(f"{'METRIC':<15} | {'CORRECT':<12} | {'SHUFFLED':<12} | {'DELTA':<12}")
    print("="*65)
    
    is_blind = True
    
    for k in metrics_correct.keys():
        val_c = metrics_correct[k]
        val_s = metrics_shuffled[k]
        delta = val_c - val_s
        
        print(f"{k:<15} | {val_c:<12.4f} | {val_s:<12.4f} | {delta:+.4f}")
        
        # We need a noticeable drop in metrics to prove vision usage
        if delta > 0.01: 
            is_blind = False

    print("="*65)
    print("\nCONCLUSION:")
    if is_blind:
        print("🔴 MODEL IS BLIND (Mode Collapse).")
        print("It ignores the image and generates the same generic text regardless of input.")
    else:
        print("🟢 MODEL IS ATTENTIVE.")
        print("The performance drops when images are shuffled, meaning the image content matters.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to checkpoint folder or bin file")
    parser.add_argument('--decoder_model', type=str, default='facebook/bart-base')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv')
    parser.add_argument('--subset_size', type=int, default=50, help="Number of samples to test")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument('--num_beams', type=int, default=4) # Matched default
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    run_test(args)
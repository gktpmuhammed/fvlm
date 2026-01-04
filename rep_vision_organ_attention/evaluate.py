#!/usr/bin/env python3
import sys
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import json
import nltk
import math
import csv
import matplotlib
matplotlib.use('Agg') # Fix for headless servers
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# NLTK Setup
try:
    nltk.data.find('wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

# Metrics Imports
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import sentence_gleu
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, Transposed, EnsureChannelFirstd

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)

from medical_vlm import MedicalVLM
from train import get_organ_ids_for_key, ALL_TARGET_KEYS, build_transforms

# --- METRIC FUNCTIONS ---

def calculate_meteor(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        try: 
            scores.append(meteor_score([nltk.word_tokenize(ref)], nltk.word_tokenize(pred)))
        except: scores.append(0.0)
    return np.mean(scores)

def calculate_bleu_scores(predictions, references):
    references_tok = [[nltk.word_tokenize(ref)] for ref in references]
    hypotheses_tok = [nltk.word_tokenize(pred) for pred in predictions]
    weights = [
        (1.0, 0, 0, 0),          # BLEU-1
        (0.5, 0.5, 0, 0),        # BLEU-2
        (1/3, 1/3, 1/3, 0),      # BLEU-3
        (0.25, 0.25, 0.25, 0.25) # BLEU-4
    ]
    scores = {}
    for i, w in enumerate(weights, start=1):
        try: scores[f'bleu{i}'] = corpus_bleu(references_tok, hypotheses_tok, weights=w)
        except: scores[f'bleu{i}'] = 0.0
    return scores

def calculate_greene(predictions, references):
    # Approximation using Sentence GLEU (Google-BLEU)
    scores = []
    for pred, ref in zip(predictions, references):
        try: scores.append(sentence_gleu([nltk.word_tokenize(ref)], nltk.word_tokenize(pred)))
        except: scores.append(0.0)
    return np.mean(scores)

def calculate_accuracy(predictions, references):
    # Exact match accuracy
    matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
    return matches / len(predictions) if len(predictions) > 0 else 0.0

def calculate_cider_approx(predictions, references, ngram=4):
    def ngrams(tokens, n):
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    docs_p, docs_r = [], []
    for p in predictions:
        toks = nltk.word_tokenize(p)
        ngs = []
        for k in range(1, ngram+1): ngs.extend(ngrams(toks, k))
        docs_p.append(ngs)
    
    for r in references:
        toks = nltk.word_tokenize(r)
        ngs = []
        for k in range(1, ngram+1): ngs.extend(ngrams(toks, k))
        docs_r.append(ngs) 
        
    # Document Frequency from References
    df = defaultdict(int)
    for doc in docs_r:
        for g in set(doc): df[g] += 1
    
    N = len(docs_r)
    
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
    for p_ng, r_ng in zip(docs_p, docs_r):
        scores.append(cosine(tf_idf(p_ng), tf_idf(r_ng)))
        
    return np.mean(scores) * 10.0 if scores else 0.0

def create_metrics_table_plot(results_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Define Column Order
    col_labels = ['Organ', 'N', 'ACC', 'GREEN', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
    
    table_rows, csv_rows = [], []
    
    for row_data in results_list:
        row = [
            str(row_data['Organ']).upper(),
            f"{row_data['N']}",
            f"{row_data['ACC']:.3f}",
            f"{row_data['GREEN']:.3f}",
            f"{row_data['BLEU-1']:.3f}",
            f"{row_data['BLEU-2']:.3f}",
            f"{row_data['BLEU-3']:.3f}",
            f"{row_data['BLEU-4']:.3f}",
            f"{row_data['METEOR']:.3f}",
            f"{row_data['ROUGE-L']:.3f}",
            f"{row_data['CIDEr']:.3f}",
        ]
        table_rows.append(row)
        csv_rows.append({k:v for k,v in zip(col_labels, row)})
    
    # Save CSV
    with open(os.path.join(output_dir, 'metrics_breakdown.csv'), 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=col_labels)
        writer.writeheader()
        writer.writerows(csv_rows)
        
    # Save Image
    fig, ax = plt.subplots(figsize=(14, len(table_rows) * 0.5 + 2))
    ax.axis('off')
    table = ax.table(cellText=table_rows, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#333333')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'metrics_table.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

# --- DATASET ---

class EvalDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform, subset_size=None):
        self.df = pd.read_csv(csv_file)
        # Assuming we evaluate on the validation set
        self.df = self.df[self.df['split'] == 'validation'].reset_index(drop=True)
        if subset_size:
            self.df = self.df.head(subset_size)
            print(f"Subset enabled: Evaluating on {len(self.df)} samples.")
            
        self.transform = transform
        self.tokenizer = tokenizer
        
    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            mask_path = row['image_path'].replace('images', 'masks')
            data = self.transform({'image': row['image_path'], 'mask': mask_path})
            
            img = data['image']
            if hasattr(img, 'as_tensor'): img = img.as_tensor().float()
            elif not isinstance(img, torch.Tensor): img = torch.from_numpy(img).float()
            
            mask = data['mask']
            if hasattr(mask, 'as_tensor'): mask = mask.as_tensor()
            elif not isinstance(mask, torch.Tensor): mask = torch.from_numpy(mask)
            
            return {
                'pixel_values': img,
                'full_mask': mask,
                'patient_id': os.path.basename(row['image_path']).split('.')[0]
            }
        except Exception as e:
            print(f"Error loading {row['image_path']}: {e}")
            return None

# --- EVALUATION LOGIC ---

def compute_all_metrics(preds, refs):
    """Helper to run all metric functions on a list of pairs."""
    if not preds: return {}
    
    bleu = calculate_bleu_scores(preds, refs)
    rouge = ROUGEScore()(preds, refs)
    meteor = calculate_meteor(preds, refs)
    green = calculate_greene(preds, refs)
    acc = calculate_accuracy(preds, refs)
    cider = calculate_cider_approx(preds, refs)
    
    return {
        'BLEU-1': bleu['bleu1'],
        'BLEU-2': bleu['bleu2'],
        'BLEU-3': bleu['bleu3'],
        'BLEU-4': bleu['bleu4'],
        'ROUGE-L': rouge['rougeL_fmeasure'].item(),
        'METEOR': meteor,
        'GREEN': green,
        'ACC': acc,
        'CIDEr': cider
    }

def evaluate(args):
    print(f"--- Starting Evaluation ---")
    print(f"Model: {args.decoder_model}")
    print(f"Dataset: {args.csv_file}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Model
    model = MedicalVLM(args.vision_encoder_path, args.decoder_model)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.model.config.pad_token_id = model.tokenizer.eos_token_id

    # 2. Load Weights
    print(f"Loading weights from {args.model_path}...")
    if os.path.isdir(args.model_path):
        weights_path = os.path.join(args.model_path, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location='cpu')
    else:
        state_dict = torch.load(args.model_path, map_location='cpu')
        
    model.model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()
    
    # 3. Setup Data
    transform = build_transforms()
    ds = EvalDataset(args.csv_file, model.tokenizer, transform, args.subset_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    
    # 4. Load Reference Text (JSON)
    # This is where the Ground Truth comes from
    with open(args.json_file, 'r') as f:
        ref_json = json.load(f)

    # Storage
    # Global: Concatenated strings per patient
    full_predictions = []
    full_references = []
    patient_ids = []
    
    # Organ-Specific: Lists of individual organ strings
    organ_specific_data = defaultdict(lambda: {'preds': [], 'refs': []})
    
    device = next(model.parameters()).device
    
    print("Generating Reports...")
    with torch.no_grad():
        for batch in tqdm(dl):
            if batch is None: continue
            
            pixel_values = batch['pixel_values'].to(device)
            full_mask = batch['full_mask'].to(device)
            pid = batch['patient_id'][0]
            
            # Prepare inputs
            mask_stack = []
            prompts = []
            
            for key in ALL_TARGET_KEYS:
                p_text = f"Describe {key}: "
                prompts.append(p_text)
                
                tids = get_organ_ids_for_key(key)
                if len(tids) > 0:
                    m = torch.zeros_like(full_mask)
                    for t in tids: m[full_mask == t] = 1.0
                else:
                    m = torch.zeros_like(full_mask)
                mask_stack.append(m)
            
            organ_masks = torch.stack(mask_stack, dim=1).float()
            
            prompt_inputs = model.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(device)
            
            # Generate
            try:
                outputs = model.generate(
                    pixel_values=pixel_values,
                    organ_masks=organ_masks,
                    input_ids=prompt_inputs.input_ids,
                    attention_mask=prompt_inputs.attention_mask,
                    max_length=120,
                    num_beams=4,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=3
                )
                
                decoded = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # --- DATA COLLECTION ---
                # Retrieve Reference for this specific patient ID
                base_id = pid.replace('.nii.gz', '').replace('.nii', '')
                p_ref_dict = {}
                
                # Try exact match or substring match for ID
                if base_id in ref_json: 
                    p_ref_dict = ref_json[base_id]
                elif '_' in base_id:
                    short = base_id.rsplit('_', 1)[0]
                    if short in ref_json: 
                        p_ref_dict = ref_json[short]
                
                patient_pred_concat = ""
                patient_ref_concat = ""
                
                for key, text, p_text in zip(ALL_TARGET_KEYS, decoded, prompts):
                    # Clean Prediction (Remove Prompt)
                    clean_pred = text.replace(p_text, "").strip()
                    
                    # Get specific organ reference from JSON
                    ref_sent = p_ref_dict.get(key, "")
                    if not ref_sent:
                        ref_sent = p_ref_dict.get(key.lower(), "")
                    
                    # Store if Valid Reference Exists
                    if ref_sent and len(ref_sent) > 2:
                        organ_specific_data[key]['preds'].append(clean_pred)
                        organ_specific_data[key]['refs'].append(ref_sent)
                        
                        # Build Concatenated Report
                        if clean_pred:
                            patient_pred_concat += f"{key.upper()}: {clean_pred}\n"
                        patient_ref_concat += f"{key.upper()}: {ref_sent}\n"

                # Store Full Report
                if patient_ref_concat.strip():
                    full_predictions.append(patient_pred_concat.strip())
                    full_references.append(patient_ref_concat.strip())
                    patient_ids.append(pid)
                
            except Exception as e:
                print(f"Skipping {pid} error: {e}")
                continue

    # # --- METRICS CALCULATION ---
    # print("\n" + "="*50)
    # print("   EVALUATION RESULTS")
    # print("="*50)
    
    # metrics_summary_list = []
    
    # # 1. Global (Concatenated)
    # if len(full_predictions) > 0:
    #     print("\nComputing Global Metrics (Concatenated)...")
    #     g_metrics = compute_all_metrics(full_predictions, full_references)
        
    #     # Add metadata for table
    #     g_metrics['Organ'] = 'GLOBAL_REPORT'
    #     g_metrics['N'] = len(full_predictions)
    #     metrics_summary_list.append(g_metrics)
        
    #     print(f"Global BLEU-4: {g_metrics['BLEU-4']:.4f} | ROUGE-L: {g_metrics['ROUGE-L']:.4f} | CIDEr: {g_metrics['CIDEr']:.4f}")
    
    # # 2. Per-Organ
    # print("\nComputing Per-Organ Metrics...")
    # for organ in ALL_TARGET_KEYS:
    #     data = organ_specific_data[organ]
    #     preds = data['preds']
    #     refs = data['refs']
        
    #     if len(refs) < 5:
    #         continue
            
    #     o_metrics = compute_all_metrics(preds, refs)
    #     o_metrics['Organ'] = organ
    #     o_metrics['N'] = len(refs)
    #     metrics_summary_list.append(o_metrics)
        
    #     print(f" > {organ.upper():<12}: BLEU-4 {o_metrics['BLEU-4']:.4f}")

    # # Save Results
    # create_metrics_table_plot(metrics_summary_list, args.output_dir)

    # Save Generated Text
    out_csv = os.path.join(args.output_dir, "generated_reports.csv")
    df = pd.DataFrame({
        'patient_id': patient_ids, 
        'prediction': full_predictions, 
        'reference': full_references
    })
    df.to_csv(out_csv, index=False)
    
    print("\n" + "-"*50)
    print(f"Full reports saved to: {out_csv}")
    print(f"Metrics table saved to: {os.path.join(args.output_dir, 'metrics_breakdown.csv')}")
    print("-"*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--decoder_model', type=str, default='gpt2')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv')
    parser.add_argument('--json_file', type=str, default='/home/muhammedg/fvlm/data/combined_desc_conc.json')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--subset_size', type=int, default=None)
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    evaluate(args)
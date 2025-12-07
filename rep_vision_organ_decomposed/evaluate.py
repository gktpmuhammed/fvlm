#!/usr/bin/env python3
import sys
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import argparse
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, Transposed, EnsureChannelFirstd
import numpy as np
import json
import nltk

# NLTK Setup
try:
    nltk.data.find('wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

# Metrics
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import sentence_gleu
from collections import Counter, defaultdict
import math
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from train import get_organ_ids_for_key, build_transforms

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from medical_vlm import MedicalVLM

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

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, tokenizer, transform, subset_size=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == 'validation']
        if subset_size:
            self.df = self.df.head(subset_size)
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Organs to evaluate
        self.target_organs = ["heart", "lung", "liver", "kidney", "aorta", 'esophagus', 'trachea']

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        mask_path = img_path.replace('images', 'masks')
        
        try:
            data = self.transform({'image': img_path, 'mask': mask_path})
            
            # --- FIX: Handle MONAI MetaTensor AND Dimensions ---
            img = data['image']
            if hasattr(img, 'as_tensor'): img_tensor = img.as_tensor().float()
            elif isinstance(img, torch.Tensor): img_tensor = img.float()
            else: img_tensor = torch.from_numpy(img).float()

            mask = data['mask']
            if hasattr(mask, 'as_tensor'): mask_tensor = mask.as_tensor()
            elif isinstance(mask, torch.Tensor): mask_tensor = mask
            else: mask_tensor = torch.from_numpy(mask)
            
            # CRITICAL FIX: Do NOT unsqueeze here. DataLoader adds batch dim.
            # Output shape should be (C, D, H, W) -> (1, 112, 256, 352)
            
            return {
                'pixel_values': img_tensor, 
                'full_mask': mask_tensor,
                'patient_id': os.path.basename(img_path)
            }
        except Exception as e:
            return None

def evaluate(args):
    print(f"Loading Model: {args.decoder_model} (Q-Former: {args.use_qformer})...")
    
    # 1. Initialize Model
    model = MedicalVLM(
        vision_encoder_path=args.vision_encoder_path, 
        decoder_model_name=args.decoder_model,
        use_qformer=args.use_qformer,
        num_query_tokens=args.num_query_tokens
    )
    
    # 1. Load Weights
    weights_path = os.path.join(args.model_path, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(args.model_path, "model.safetensors")
        if os.path.exists(weights_path):
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(args.model_path, map_location='cpu')
    else:
        state_dict = torch.load(weights_path, map_location='cpu')

    # 2. Checkpoint vs Model
    has_prefix = any(k.startswith("model.") for k in state_dict.keys())
    if has_prefix:
        print(" > Detected Trainer Checkpoint. Loading into wrapper.")
        model.load_state_dict(state_dict, strict=False)
    else:
        print(" > Detected SavedModel. Loading into inner model.")
        model.model.load_state_dict(state_dict, strict=False)

    model.model.eval().cuda()
    
    transform = build_transforms()
    ds = EvalDataset(args.csv_file, model.tokenizer, transform, args.subset_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    # Load Reference JSON
    with open(args.json_file, 'r') as f:
        ref_json = json.load(f)

    full_predictions = []
    full_references = []
    patient_ids = []
    
    print("Generating Organ Reports...")
    with torch.no_grad():
        for item in tqdm(dl):
            if item is None: continue
            
            # Data from DataLoader already has batch dim (1, C, D, H, W)
            pixel_values = item['pixel_values'].cuda()
            full_mask = item['full_mask'].cuda()
            pid = item['patient_id'][0]
            
            # Normalize PID for matching
            base_id = pid.replace('.nii.gz', '').replace('.nii', '')
            ref_data = {}
            if base_id in ref_json: ref_data = ref_json[base_id]
            elif '_' in base_id:
                short = base_id.rsplit('_', 1)[0]
                if short in ref_json: ref_data = ref_json[short]

            generated_report_parts = []
            reference_report_parts = []
            
            for organ in ds.target_organs:
                # A. Create Mask
                tids = get_organ_ids_for_key(organ)
                binary_mask = torch.zeros_like(full_mask)
                for tid in tids:
                    binary_mask[full_mask == tid] = 1.0
                
                # B. Prompt
                prompt_text = f"Describe the {organ}: "
                input_ids = model.tokenizer(prompt_text, return_tensors="pt").input_ids.cuda()
                
                # C. Generate
                # Note: 'pixel_mask' is used in the flattened MedicalVLM version
                output_ids = model.generate(
                    pixel_values=pixel_values,
                    pixel_mask=binary_mask, 
                    decoder_input_ids=input_ids, # Correct arg for Encoder-Decoder
                    max_length=100,
                    num_beams=4,
                    repetition_penalty=1.5
                )
                
                text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                text = text.replace(prompt_text, "").strip()
                
                generated_report_parts.append(f"{organ.upper()}: {text}")
                
                # Reference
                ref_text = ref_data.get(organ, "").strip()
                if ref_text:
                    reference_report_parts.append(f"{organ.upper()}: {ref_text}")

            full_pred = "\n".join(generated_report_parts)
            full_ref = "\n".join(reference_report_parts)
            
            full_predictions.append(full_pred)
            full_references.append(full_ref)
            patient_ids.append(base_id)
            
            # Debug Print (First sample)
            if len(patient_ids) == 1:
                print(f"\n--- {pid} ---\n{full_pred}\n")

    # Metrics
    print("\nComputing Metrics...")
    # Filter out empty references
    valid_data = [(p, r) for p, r in zip(full_predictions, full_references) if len(r) > 5]
    
    if len(valid_data) > 0:
        p_val, r_val = zip(*valid_data)
        
        print("\nComputing ROUGE...")
        rouge = ROUGEScore()(list(p_val), list(r_val))
        print("Computing BLEU...")
        bleu = calculate_bleu_scores(p_val, r_val)
        print("Computing METEOR...")
        meteor = calculate_meteor(p_val, r_val)
        print("Computing GREEN...")
        green = calculate_greene(p_val, r_val)
        print("Computing Accuracy...")
        acc = calculate_accuracy(p_val, r_val)
        print("Computing CIDEr...")
        cider = calculate_cider_approx(p_val, r_val)
        
        results = {
            'rougeL_fmeasure': rouge['rougeL_fmeasure'].item(),
            'bleu1': bleu['bleu1'], 'bleu2': bleu['bleu2'], 'bleu3': bleu['bleu3'], 'bleu4': bleu['bleu4'],
            'meteor': meteor, 'green': green, 'accuracy': acc, 'cider': cider
        }
        
        print(f"Results: BLEU-4: {results['bleu4']:.4f} | CIDEr: {results['cider']:.4f}")
        label = f"{args.decoder_model}_QFormer" if args.use_qformer else args.decoder_model
        create_metrics_table_plot([results], [label], args.output_dir)
    else:
        print("Warning: No valid references found for metrics.")

    # Save CSV
    df = pd.DataFrame({'patient_id': patient_ids, 'prediction': full_predictions, 'reference': full_references})
    df.to_csv(os.path.join(args.output_dir, "generated_reports.csv"), index=False)
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--decoder_model', type=str, default='gpt2')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv')
    parser.add_argument('--json_file', type=str, default='/home/muhammedg/fvlm/data/combined_desc_conc.json')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--subset_size', type=int, default=None)
    parser.add_argument('--use_qformer', action='store_true')
    parser.add_argument('--num_query_tokens', type=int, default=32)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    evaluate(args)
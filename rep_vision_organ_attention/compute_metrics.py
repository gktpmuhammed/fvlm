import os
import sys
import argparse
import pandas as pd
import numpy as np
import re
import csv
import gc
import torch
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# NLTK Setup
import nltk
try:
    nltk.data.find('wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from torchmetrics.text.rouge import ROUGEScore

# --- CONFIGURATION ---

ALL_ORGANS = [
    "LUNG", "HEART", "AORTA", "ESOPHAGUS", "TRACHEA", 
    "RIB", "LIVER", "GALLBLADDER", "STOMACH", "PANCREAS", 
    "SPLEEN", "KIDNEY"
]

# --- UTILS & MEMORY MANAGEMENT ---

def free_memory():
    """Forces garbage collection and empties CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def normalize_text(text):
    if not isinstance(text, str): return ""
    return " ".join(text.split()).strip()

def extract_organ_sections(text):
    """Parses 'LUNG: text... HEART: text...' into a dict."""
    text = normalize_text(text)
    sections = {k: "" for k in ALL_ORGANS}
    if not text: return sections
    
    pattern = r"(" + "|".join(ALL_ORGANS) + r"):"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    for i in range(1, len(parts)-1, 2):
        header = parts[i].upper().strip()
        content = parts[i+1].strip()
        if header in sections:
            sections[header] = content
    return sections

# --- METRIC FUNCTIONS ---
def calculate_cider_official(predictions, references):
    """
    Official COCO CIDEr implementation.
    Expects single-reference captions.
    """
    from pycocoevalcap.cider.cider import Cider

    # COCO format: {id: [caption]}
    gts = {}
    res = {}

    for i, (p, r) in enumerate(zip(predictions, references)):
        gts[i] = [r]
        res[i] = [p]

    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    return score

def run_nlp_metrics(predictions, references, device):
    """Runs Standard NLP Metrics: BLEU 1-4, ROUGE, METEOR, ACC, CIDEr"""
    if not predictions: return {}
    print("     > Tokenizing...")
    refs_tok = [[nltk.word_tokenize(r)] for r in references]
    hyps_tok = [nltk.word_tokenize(p) for p in predictions]
    
    # BLEU 1-4
    weights = [
        (1.0, 0, 0, 0),          # BLEU-1
        (0.5, 0.5, 0, 0),        # BLEU-2
        (0.33, 0.33, 0.33, 0),   # BLEU-3
        (0.25, 0.25, 0.25, 0.25) # BLEU-4
    ]
    print("     > Computing BLEU...")
    bleu_scores = {}
    smooth = SmoothingFunction().method1
    for i, w in enumerate(weights, 1):
        try: bleu_scores[f'BLEU-{i}'] = corpus_bleu(refs_tok, hyps_tok, weights=w, smoothing_function=smooth)
        except: bleu_scores[f'BLEU-{i}'] = 0.0
    
    # METEOR
    print("     > Computing METEOR...")
    met_scores = []
    for p, r in zip(predictions, references):
        try: met_scores.append(meteor_score([nltk.word_tokenize(r)], nltk.word_tokenize(p)))
        except: met_scores.append(0.0)
    meteor = np.mean(met_scores)
    
    # ROUGE
    print("     > Computing ROUGE...")
    rouge = ROUGEScore()(predictions, references)['rougeL_fmeasure'].item()
    
    # ACCURACY
    matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
    acc = matches / len(predictions)
    
    # CIDEr
    print("     > Computing CIDEr...")
    cider = calculate_cider_official(predictions, references)
    
    results = {
        'METEOR': meteor, 'ROUGE-L': rouge, 'ACC': acc, 'CIDEr': cider
    }
    results.update(bleu_scores)
    return results

def run_bertscore(predictions, references, device):
    print("   > Loading BERTScore...")
    try:
        from bert_score import score
        # Using distilroberta-base for speed and memory efficiency
        P, R, F1 = score(predictions, references, lang="en", verbose=True, 
                         model_type='roberta-base', device=device, batch_size=64, rescale_with_baseline=True)
        return F1.mean().item()
    except Exception as e:
        print(f"   ! Error in BERTScore: {e}"); return 0.0

def run_radgraph(predictions, references, device):
    print("   > Computing RadGraph F1...")
    try:
        from radgraph import F1RadGraph
        import torch

        scorer = F1RadGraph(reward_level="all")
        mean_reward, _, _, _ = scorer(hyps=predictions, refs=references)
        rg_e, rg_er, rg_bar_er = mean_reward
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return rg_er

    except Exception as e:
        print(f"   ! Error in RadGraph: {e}")
        return 0.0


def run_chexbert(predictions, references, device):
    print("   > Loading CheXbert...")
    try:
        from f1chexbert import F1CheXbert
        scorer = F1CheXbert(device=device)
        accuracy, accuracy_per_class, chexbert_all, chexbert_5 = scorer(predictions, references)
        del scorer
        return chexbert_all
    except Exception as e:
        print(f"   ! Error in CheXbert: {e}"); return 0.0

def run_radcliq(predictions, references, device):
    print("   > Running RadCliQ...")
    try:
        from radmetrics import RadCliQ
        scorer = RadCliQ()
        score = scorer(predictions, references)
        del scorer
        return score
    except Exception as e:
        print(f"   ! Error in RadCliQ: {e}"); return 0.0

def run_green(predictions, references, device, key_name="UNKNOWN"):
    print(f"   > Loading GREEN (Llama-7B) for {key_name}...")
    try:
        from green_score import GREEN
        
        # DYNAMIC BATCH SIZING to prevent OOM on Global reports
        if key_name == 'GLOBAL':
            current_bs = 2  # Very small for long text
        else:
            current_bs = 16 # Larger for short text
            
        print(f"     Batch Size: {current_bs}")
        
        model_name = "StanfordAIMI/GREEN-radllama2-7b"
        
        # Attempt 4-bit load if bitsandbytes is present (Major Memory Saver)
        try:
            import bitsandbytes
            # Note: The GREEN wrapper initializes the model immediately.
            # If the installed version of green_score supports quantization args, great.
            # If not, it loads default (16-bit).
            scorer = GREEN(model_name, output_dir="./green_cache")
        except:
            scorer = GREEN(model_name, output_dir="./green_cache")

        # Run Inference
        try:
            # Pass batch_size if supported
            mean, std, green_score_list, summary, result_df = scorer(references, predictions, batch_size=current_bs)
        except TypeError:
            # Fallback for older versions
            mean, std, green_score_list, summary, result_df = scorer(references, predictions)

        # Explicit Cleanup
        if hasattr(scorer, 'model'): del scorer.model
        if hasattr(scorer, 'tokenizer'): del scorer.tokenizer
        del scorer
        return mean
    except Exception as e:
        print(f"   ! Error in GREEN: {e}"); return 0.0

# --- MAIN EXECUTION ---

def evaluate_metrics_sequentially(args):
    # 1. Load Data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    df['pred'] = df['prediction'].fillna("").astype(str)
    df['ref'] = df['reference'].fillna("").astype(str)
    
    # Optional Subset for fast debugging
    if args.subset and len(df) > args.subset:
        print(f"⚠️  SUBSET MODE: Using random {args.subset} samples.")
        df = df.sample(n=args.subset, random_state=42).reset_index(drop=True)

    # 2. Organize Data
    data_buckets = defaultdict(lambda: {'preds': [], 'refs': []})
    
    print("Parsing reports...")
    for _, row in df.iterrows():
        p_text, r_text = row['pred'], row['ref']
        
        # Global Bucket
        if len(r_text) > 2:
            data_buckets['GLOBAL']['preds'].append(p_text)
            data_buckets['GLOBAL']['refs'].append(r_text)
            
        # Organ Buckets
        p_parts = extract_organ_sections(p_text)
        r_parts = extract_organ_sections(r_text)
        
        for organ in ALL_ORGANS:
            if len(r_parts[organ]) > 2: # Only evaluate if reference exists
                data_buckets[organ]['preds'].append(p_parts[organ])
                data_buckets[organ]['refs'].append(r_parts[organ])

    final_results = defaultdict(dict)
    
    # 3. Define Pipeline
    # (MetricName, Function, NeedsGPU)
    pipeline = []
    
    # Logic to select metrics
    run_all = 'all' in args.metrics
    
    if run_all or 'nlp' in args.metrics:
        pipeline.append(('NLP_Standard', run_nlp_metrics, False))
    if run_all or 'bertscore' in args.metrics:
        pipeline.append(('BERTScore', run_bertscore, True))
    if run_all or 'radgraph' in args.metrics:
        pipeline.append(('RadGraph', run_radgraph, False)) # RadGraph handles its own devices usually
    if run_all or 'chexbert' in args.metrics:
        pipeline.append(('CheXbert', run_chexbert, True))
    if run_all or 'radcliq' in args.metrics:
        pipeline.append(('RadCliQ', run_radcliq, False))
    if run_all or 'green' in args.metrics:
        pipeline.append(('GREEN', run_green, True))

    device = "cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu"
    print(f"Running metrics on device: {device}")

    # 4. Execute Sequentially
    for metric_name, metric_func, needs_gpu in pipeline:
        print(f"\n--- Running {metric_name} ---")
        free_memory()
        
        keys_to_process = ['GLOBAL'] + ALL_ORGANS
        
        for key in keys_to_process:
            preds = data_buckets[key]['preds']
            refs = data_buckets[key]['refs']
            
            if len(preds) == 0: continue
            
            print(f"   Processing {key} (N={len(preds)})...")
            
            try:
                # Special handling for GREEN to pass the key_name (for batch sizing)
                if metric_name == 'GREEN':
                    val = metric_func(preds, refs, device=device if needs_gpu else 'cpu', key_name=key)
                else:
                    val = metric_func(preds, refs, device=device if needs_gpu else 'cpu')
                
                # Store Results
                final_results[key]['N'] = len(preds)
                if isinstance(val, dict):
                    for sub_k, sub_v in val.items():
                        final_results[key][sub_k] = sub_v
                else:
                    final_results[key][metric_name] = val
                    
            except Exception as e:
                print(f"Error calculating {metric_name} for {key}: {e}")
            
            if needs_gpu:
                free_memory()

    save_results(final_results, args.output_dir)

def save_results(results_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Gather all metric keys
    metric_keys = set()
    for res in results_dict.values():
        metric_keys.update(res.keys())
    metric_keys.discard('N')
    
    # Define Column Order (Restored all columns)
    ordered_keys = [
        'Organ', 'N', 'ACC', 'GREEN', 
        'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 
        'METEOR', 'ROUGE-L', 'CIDEr',
        'BERTScore', 'RadGraph', 'CheXbert', 'RadCliQ'
    ]
    
    # Append any unexpected keys at the end
    remaining = [k for k in metric_keys if k not in ordered_keys]
    final_cols = ordered_keys + remaining
    
    table_rows = []
    csv_rows = []
    
    row_keys = ['GLOBAL'] + [k for k in ALL_ORGANS if k in results_dict]
    
    for key in row_keys:
        if key not in results_dict: continue
        data = results_dict[key]
        
        row_vals = [key, str(data.get('N', 0))]
        
        # For Table (Strings)
        for metric in final_cols[2:]:
            val = data.get(metric, 0.0)
            row_vals.append(f"{val:.3f}")
            
        table_rows.append(row_vals)
        
        # For CSV (Numeric)
        csv_row = {'Organ': key, 'N': data.get('N', 0)}
        for metric in final_cols[2:]:
            csv_row[metric] = data.get(metric, 0.0)
        csv_rows.append(csv_row)

    # Save CSV
    csv_path = os.path.join(output_dir, 'metrics_final.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_cols)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved CSV to: {csv_path}")

    # Save Image
    try:
        # Dynamic width based on number of columns
        fig_width = len(final_cols) * 1.0
        fig, ax = plt.subplots(figsize=(fig_width, len(table_rows)*0.5 + 2))
        ax.axis('off')
        tbl = ax.table(cellText=table_rows, colLabels=final_cols, cellLoc='center', loc='center')
        tbl.scale(1, 1.5)
        
        for (r, c), cell in tbl.get_celld().items():
            if r == 0: 
                cell.set_facecolor('#333333')
                cell.set_text_props(color='white', weight='bold')
        
        plt.tight_layout()
        img_path = os.path.join(output_dir, 'metrics_table.png')
        fig.savefig(img_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved Image to: {img_path}")
    except Exception as e:
        print(f"Could not generate summary image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Medical VLM Metrics Sequentially")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to generated_reports.csv")
    parser.add_argument('--output_dir', type=str, default='./results_metrics', help="Folder to save results")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    # Flags
    parser.add_argument('--metrics', nargs='+', default=['all'], 
                        help="List of metrics: all, nlp, green, bertscore, radgraph, chexbert, radcliq")
    parser.add_argument('--subset', type=int, default=None, help="Debug: Evaluate only N random samples")
    
    args = parser.parse_args()
    
    if os.path.exists(args.input_csv):
        evaluate_metrics_sequentially(args)
    else:
        print(f"Error: Input file {args.input_csv} not found.")
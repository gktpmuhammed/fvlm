import os
import sys
import argparse
import pandas as pd
import numpy as np
import re
import csv
import gc
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# NLTK Setup for fallback metrics
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

# Import RadEval
from RadEval import RadEval

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

# --- FALLBACK METRICS (Not in RadEval) ---

def calculate_cider_official(predictions, references):
    """Official COCO CIDEr implementation with Tokenization Fix."""
    try:
        from pycocoevalcap.cider.cider import Cider
        import nltk
        
        # FIX: Tokenize and rejoin with spaces so "normal." becomes "normal ."
        # This ensures 'normal' matches 'normal' regardless of punctuation.
        preds_tok = [" ".join(nltk.word_tokenize(p)) for p in predictions]
        refs_tok  = [" ".join(nltk.word_tokenize(r)) for r in references]
        
        gts = {i: [r] for i, r in enumerate(refs_tok)}
        res = {i: [p] for i, p in enumerate(preds_tok)}
        
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts, res)
        return score
    except Exception as e:
        print(f"   ! CIDEr fallback error: {e}")
        return 0.0

def calculate_meteor_fallback(predictions, references):
    """METEOR calculation - FALLBACK."""
    try:
        met_scores = []
        for p, r in zip(predictions, references):
            try:
                met_scores.append(meteor_score([nltk.word_tokenize(r)], nltk.word_tokenize(p)))
            except:
                met_scores.append(0.0)
        return np.mean(met_scores)
    except Exception as e:
        print(f"   ! METEOR fallback error: {e}")
        return 0.0

def calculate_accuracy_fallback(predictions, references):
    """Exact match accuracy - FALLBACK."""
    matches = sum(1 for p, r in zip(predictions, references) 
                  if p.strip().lower() == r.strip().lower())
    return matches / len(predictions) if len(predictions) > 0 else 0.0

# --- RADEVAL INTEGRATION ---

def run_radeval_metrics(predictions, references, metrics_config):
    """
    Run RadEval with specified metrics.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        metrics_config: Dict of metric flags (e.g., {'do_bleu': True, 'do_rouge': True})
    
    Returns:
        Dictionary of metric results
    """
    if not predictions:
        return {}
    
    print(f"     > Running RadEval with {sum(metrics_config.values())} metrics...")
    
    try:
        # Initialize RadEval with selected metrics
        evaluator = RadEval(**metrics_config)
        
        # RadEval expects (references, predictions) order
        results = evaluator(references, predictions)
        
        # Clean up evaluator
        del evaluator
        free_memory()
        
        return results
        
    except Exception as e:
        print(f"   ! RadEval error: {e}")
        return {}

def extract_radeval_scores(radeval_results):
    """
    Extract scalar scores from RadEval output using the keys confirmed by debug.
    """
    extracted = {}
    
    key_mapping = {
        # NLP Metrics
        'bleu': 'BLEU-4',
        'rougeL': 'ROUGE-L',
        'bertscore': 'BERTScore',
        
        # Clinical / Specialized Metrics
        'green': 'GREEN',
        'radcliq-v1': 'RadCliQ',
        
        # RadGraph: 'simple' usually corresponds to the standard F1 score in benchmarks
        'radgraph_simple': 'RadGraph',
        
        # CheXbert: Micro-F1 over all 14 observations is the standard single-number summary
        'chexbert-all_micro avg_f1-score': 'CheXbert',
        
        # Extra Metrics (Optional)
        'srr_bert_weighted_f1': 'SRR-BERT'
    }
    
    for radeval_key, your_key in key_mapping.items():
        if radeval_key in radeval_results:
            value = radeval_results[radeval_key]
            
            # Extract float value if it's nested (though based on your logs, they look like direct floats)
            if isinstance(value, dict):
                # Fallback for nested dicts
                if 'f1' in value: extracted[your_key] = value['f1']
                elif 'score' in value: extracted[your_key] = value['score']
                else: extracted[your_key] = list(value.values())[0]
            else:
                extracted[your_key] = float(value)
                
    return extracted

# --- MAIN EXECUTION ---

def evaluate_metrics_sequentially(args):
    # 1. Load Data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    df['pred'] = df['prediction'].fillna("").astype(str)
    df['ref'] = df['reference'].fillna("").astype(str)
    
    # Optional Subset
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
            if len(r_parts[organ]) > 2:
                data_buckets[organ]['preds'].append(p_parts[organ])
                data_buckets[organ]['refs'].append(r_parts[organ])

    final_results = defaultdict(dict)
    
    # 3. Configure RadEval Metrics
    run_all = 'all' in args.metrics
    
    radeval_config = {
        'do_bleu': run_all or 'nlp' in args.metrics or 'bleu' in args.metrics,
        'do_rouge': run_all or 'nlp' in args.metrics or 'rouge' in args.metrics,
        'do_bertscore': run_all or 'bertscore' in args.metrics,
        'do_radgraph': run_all or 'radgraph' in args.metrics,
        'do_chexbert': run_all or 'chexbert' in args.metrics,
        'do_radcliq': run_all or 'radcliq' in args.metrics,
        'do_green': run_all or 'green' in args.metrics,
        'do_srr_bert': run_all or 'srr_bert' in args.metrics,
        'do_ratescore': 'ratescore' in args.metrics,
        'do_temporal': 'temporal' in args.metrics,
        # 'do_radeval_bertscore': run_all or 'radeval_bertscore' in args.metrics,
        # 'do_detail': True,  # Always get detailed outputs
    }
    
    # Fallback metrics (not in RadEval)
    run_meteor = run_all or 'nlp' in args.metrics or 'meteor' in args.metrics
    run_cider = run_all or 'nlp' in args.metrics or 'cider' in args.metrics
    run_accuracy = run_all or 'nlp' in args.metrics or 'accuracy' in args.metrics

    print(f"\n{'='*60}")
    print("METRIC CONFIGURATION")
    print(f"{'='*60}")
    print("RadEval Metrics:", {k: v for k, v in radeval_config.items() if v})
    print("Fallback Metrics:", 
          [m for m, flag in [('METEOR', run_meteor), ('CIDEr', run_cider), ('ACC', run_accuracy)] if flag])
    print(f"{'='*60}\n")

    # 4. Execute Evaluation
    keys_to_process = ['GLOBAL'] + ALL_ORGANS
    
    for key in keys_to_process:
        preds = data_buckets[key]['preds']
        refs = data_buckets[key]['refs']
        
        if len(preds) == 0:
            continue
        
        print(f"\n{'─'*60}")
        print(f"Processing: {key} (N={len(preds)})")
        print(f"{'─'*60}")
        
        final_results[key]['N'] = len(preds)
        
        # --- RUN RADEVAL ---
        if any(radeval_config.values()):
            print("\n[1/2] Running RadEval Suite...")
            radeval_results = run_radeval_metrics(preds, refs, radeval_config)
            
            extracted_scores = extract_radeval_scores(radeval_results)
            final_results[key].update(extracted_scores)
            
            # Print what we got
            if extracted_scores:
                print(f"     ✓ Extracted {len(extracted_scores)} metrics from RadEval")
        
        # --- RUN FALLBACK METRICS ---
        print("\n[2/2] Running Fallback Metrics...")
        
        if run_meteor:
            print("     > METEOR...")
            meteor = calculate_meteor_fallback(preds, refs)
            final_results[key]['METEOR'] = meteor
            print(f"     ✓ METEOR: {meteor:.4f}")
        
        if run_cider:
            print("     > CIDEr...")
            cider = calculate_cider_official(preds, refs)
            final_results[key]['CIDEr'] = cider
            print(f"     ✓ CIDEr: {cider:.4f}")
        
        if run_accuracy:
            acc = calculate_accuracy_fallback(preds, refs)
            final_results[key]['ACC'] = acc
            print(f"     ✓ ACC: {acc:.4f}")
        
        free_memory()

    save_results(final_results, args.output_dir)

def save_results(results_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Gather all metric keys
    metric_keys = set()
    for res in results_dict.values():
        metric_keys.update(res.keys())
    metric_keys.discard('N')
    
    # Define Column Order
    ordered_keys = [
        'Organ', 'N', 'ACC', 'GREEN', 
        'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 
        'METEOR', 'ROUGE-L', 'CIDEr',
        'BERTScore', 'RadGraph', 'CheXbert', 'RadCliQ',
        'SRR-BERT'
    ]
    
    # Append unexpected keys
    remaining = [k for k in sorted(metric_keys) if k not in ordered_keys]
    final_cols = ordered_keys + remaining
    
    table_rows = []
    csv_rows = []
    
    row_keys = ['GLOBAL'] + [k for k in ALL_ORGANS if k in results_dict]
    
    for key in row_keys:
        if key not in results_dict:
            continue
        data = results_dict[key]
        
        row_vals = [key, str(data.get('N', 0))]
        
        for metric in final_cols[2:]:
            val = data.get(metric, 0.0)
            row_vals.append(f"{val:.3f}")
            
        table_rows.append(row_vals)
        
        # CSV
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
    print(f"\n{'='*60}")
    print(f"✓ Saved CSV to: {csv_path}")

    # Save Image
    try:
        fig_width = max(len(final_cols) * 1.0, 15)
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
        print(f"✓ Saved Image to: {img_path}")
    except Exception as e:
        print(f"Could not generate summary image: {e}")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Report Evaluation with RadEval Integration")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to generated_reports.csv")
    parser.add_argument('--output_dir', type=str, default='./results_metrics', help="Folder to save results")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    parser.add_argument('--metrics', nargs='+', default=['all'], 
                        help="List: all, nlp, bleu, rouge, meteor, cider, accuracy, "
                             "bertscore, radgraph, chexbert, radcliq, green, srr_bert, ratescore")
    parser.add_argument('--subset', type=int, default=None, help="Debug: Evaluate only N samples")
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    if os.path.exists(args.input_csv):
        evaluate_metrics_sequentially(args)
    else:
        print(f"Error: Input file {args.input_csv} not found.")

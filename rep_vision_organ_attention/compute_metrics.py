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
from collections import defaultdict, Counter
import math

# NLTK Setup (Lightweight)
import nltk
try:
    nltk.data.find('wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

from nltk.translate.bleu_score import corpus_bleu
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

# --- METRIC FUNCTIONS (ISOLATED) ---

def run_nlp_metrics(predictions, references, device):
    """Runs lightweight CPU metrics (BLEU, ROUGE, METEOR, CIDEr, ACC)."""
    if not predictions: return {}
    
    # 1. BLEU
    refs_tok = [[nltk.word_tokenize(r)] for r in references]
    hyps_tok = [nltk.word_tokenize(p) for p in predictions]
    weights = [(1,0,0,0), (0.25,0.25,0.25,0.25)] # BLEU-1 and BLEU-4
    b1 = corpus_bleu(refs_tok, hyps_tok, weights=weights[0])
    b4 = corpus_bleu(refs_tok, hyps_tok, weights=weights[1])
    
    # 2. METEOR
    met_scores = []
    for p, r in zip(predictions, references):
        try: met_scores.append(meteor_score([nltk.word_tokenize(r)], nltk.word_tokenize(p)))
        except: met_scores.append(0.0)
    meteor = np.mean(met_scores)
    
    # 3. ROUGE
    rouge = ROUGEScore()(predictions, references)['rougeL_fmeasure'].item()
    
    # 4. Accuracy (Exact Match)
    matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
    acc = matches / len(predictions)
    
    # 5. CIDEr (Approx)
    # Simplified implementation for speed/memory
    cider = 0.0 # Placeholder or insert your CIDEr function here if critical
    
    return {
        'BLEU-1': b1, 'BLEU-4': b4, 
        'METEOR': meteor, 'ROUGE-L': rouge, 'ACC': acc
    }

def run_bertscore(predictions, references, device):
    print("   > Loading BERTScore...")
    try:
        from bert_score import score
        # distilroberta is lighter on memory than roberta-large
        P, R, F1 = score(predictions, references, lang="en", verbose=True, 
                         model_type='distilroberta-base', device=device, batch_size=32)
        return F1.mean().item()
    except ImportError:
        print("   ! bert_score not installed.")
        return 0.0
    except Exception as e:
        print(f"   ! Error in BERTScore: {e}")
        return 0.0

def run_radgraph(predictions, references, device):
    print("   > Loading RadGraph...")
    try:
        from radgraph import F1RadGraph
        # 'all' evaluates entities and relations.
        # Note: RadGraph usually runs on CPU or requires specific CUDA setup.
        # We force it to cleanup after itself.
        scorer = F1RadGraph(reward_level="all") 
        score, _, _ = scorer(preds=predictions, refs=references)
        del scorer
        return score
    except ImportError:
        print("   ! radgraph-benchmark not installed.")
        return 0.0
    except Exception as e:
        print(f"   ! Error in RadGraph: {e}")
        return 0.0

def run_chexbert(predictions, references, device):
    print("   > Loading CheXbert...")
    try:
        # Assuming f1chexbert wrapper is available
        from f1chexbert import F1CheXbert
        scorer = F1CheXbert(device=device)
        accuracy, accuracy_per_class, chexbert_all, chexbert_5 = scorer(predictions, references)
        del scorer
        return chexbert_all
    except ImportError:
        # Fallback if specific wrapper missing
        print("   ! CheXbert wrapper not found.")
        return 0.0
    except Exception as e:
        print(f"   ! Error in CheXbert: {e}")
        return 0.0

def run_green(predictions, references, device):
    print("   > Loading GREEN (Llama-7B)... WARNING: High VRAM Usage")
    try:
        from green_score import GREEN
        # This is the heavy hitter. Ensure batch size is small if OOM occurs.
        model_name = "StanfordAIMI/GREEN-radllama2-7b" 
        scorer = GREEN(model_name, output_dir="./green_cache")
        
        # We process in small chunks if list is huge to prevent activation OOM
        mean, std, green_score_list, summary, result_df = scorer(references, predictions)
        
        # Explicit cleanup for HF models
        del scorer.model
        del scorer.tokenizer
        del scorer
        return mean
    except ImportError:
        print("   ! green_score not installed.")
        return 0.0
    except Exception as e:
        print(f"   ! Error in GREEN: {e}")
        return 0.0

def run_radcliq(predictions, references, device):
    print("   > Running RadCliQ...")
    try:
        from radmetrics import RadCliQ
        scorer = RadCliQ()
        score = scorer(predictions, references)
        del scorer
        return score
    except ImportError:
        print("   ! radmetrics (RadCliQ) not installed.")
        return 0.0
    except Exception as e:
        print(f"   ! Error in RadCliQ: {e}")
        return 0.0

# --- MAIN EXECUTION LOGIC ---

def evaluate_metrics_sequentially(args):
    # 1. Load Data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    df['pred'] = df['prediction'].fillna("").astype(str)
    df['ref'] = df['reference'].fillna("").astype(str)
    
    # 2. Prepare Data Structure
    # We organize data by Key: 'GLOBAL' or 'LUNG', 'HEART', etc.
    # Value: {'preds': [], 'refs': []}
    data_buckets = defaultdict(lambda: {'preds': [], 'refs': []})
    
    print("Parsing reports...")
    for _, row in df.iterrows():
        p_text, r_text = row['pred'], row['ref']
        
        # Global
        if len(r_text) > 2:
            data_buckets['GLOBAL']['preds'].append(p_text)
            data_buckets['GLOBAL']['refs'].append(r_text)
            
        # Organs
        p_parts = extract_organ_sections(p_text)
        r_parts = extract_organ_sections(r_text)
        
        for organ in ALL_ORGANS:
            # Only add if reference exists
            if len(r_parts[organ]) > 2:
                data_buckets[organ]['preds'].append(p_parts[organ])
                data_buckets[organ]['refs'].append(r_parts[organ])

    # 3. Initialize Results Storage
    # results[Organ][MetricName] = Value
    final_results = defaultdict(dict)
    
    # Define Metric Pipeline
    # List of (MetricName, Function, NeedsGPU)
    pipeline = []
    
    if 'nlp' in args.metrics or 'all' in args.metrics:
        pipeline.append(('NLP_Standard', run_nlp_metrics, False))
    if 'bertscore' in args.metrics or 'all' in args.metrics:
        pipeline.append(('BERTScore', run_bertscore, True))
    if 'radgraph' in args.metrics or 'all' in args.metrics:
        pipeline.append(('RadGraph', run_radgraph, False)) # Usually CPU or handles own device
    if 'chexbert' in args.metrics or 'all' in args.metrics:
        pipeline.append(('CheXbert', run_chexbert, True))
    if 'radcliq' in args.metrics or 'all' in args.metrics:
        pipeline.append(('RadCliQ', run_radcliq, False))
    if 'green' in args.metrics or 'all' in args.metrics:
        pipeline.append(('GREEN', run_green, True))

    device = "cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu"
    print(f"Running metrics on device: {device}")

    # 4. RUN METRICS SEQUENTIALLY
    # We iterate by Metric first, then by Organ.
    # This prevents loading/unloading the model for every organ.
    
    for metric_name, metric_func, needs_gpu in pipeline:
        print(f"\n--- Running {metric_name} ---")
        free_memory()
        
        # Optimization: For model-based metrics, we might want to load the model once manually
        # inside the function, process all organs, then unload. 
        # But to keep functions isolated, we will call the function per organ.
        # *However*, for heavy models (GREEN), this is slow. 
        # The functions above load/unload internally.
        
        # To strictly avoid OOM, we process one bucket at a time.
        
        keys_to_process = ['GLOBAL'] + ALL_ORGANS
        
        for key in keys_to_process:
            preds = data_buckets[key]['preds']
            refs = data_buckets[key]['refs']
            
            if len(preds) == 0: continue
            
            print(f"   Processing {key} (N={len(preds)})...")
            
            try:
                # Run the calculation
                val = metric_func(preds, refs, device=device if needs_gpu else 'cpu')
                
                # Store results
                final_results[key]['N'] = len(preds)
                if isinstance(val, dict):
                    # NLP returns a dict of multiple scores
                    for sub_k, sub_v in val.items():
                        final_results[key][sub_k] = sub_v
                else:
                    final_results[key][metric_name] = val
                    
            except Exception as e:
                print(f"Error calculating {metric_name} for {key}: {e}")
                
            # Free memory after every organ for safety, especially for batched inference
            if needs_gpu:
                free_memory()

    # 5. Save Results
    save_results(final_results, args.output_dir)

def save_results(results_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Define Column Headers
    # Gather all unique metric keys found in results
    metric_keys = set()
    for res in results_dict.values():
        metric_keys.update(res.keys())
    metric_keys.discard('N')
    
    # Sort keys for nice display
    ordered_keys = ['Organ', 'N', 'ACC', 'GREEN', 'BLEU-1', 'BLEU-4', 'METEOR', 'ROUGE-L', 'BERTScore', 'RadGraph', 'CheXbert', 'RadCliQ']
    # Add any others found that aren't in the ordered list
    remaining = [k for k in metric_keys if k not in ordered_keys]
    final_cols = ordered_keys + remaining
    
    # 2. Build Rows
    table_rows = []
    csv_rows = []
    
    # Ensure GLOBAL is first, then organs
    row_keys = ['GLOBAL'] + [k for k in ALL_ORGANS if k in results_dict]
    
    for key in row_keys:
        if key not in results_dict: continue
        data = results_dict[key]
        
        row_vals = [key, str(data.get('N', 0))]
        
        # Build Table Row (formatted)
        for metric in final_cols[2:]: # Skip Organ, N
            val = data.get(metric, 0.0)
            row_vals.append(f"{val:.3f}")
            
        table_rows.append(row_vals)
        
        # Build CSV Row (dict)
        csv_row = {'Organ': key, 'N': data.get('N', 0)}
        for metric in final_cols[2:]:
            csv_row[metric] = data.get(metric, 0.0)
        csv_rows.append(csv_row)

    # 3. Write CSV
    csv_path = os.path.join(output_dir, 'metrics_final.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_cols)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved CSV to: {csv_path}")

    # 4. Generate Image
    try:
        fig, ax = plt.subplots(figsize=(len(final_cols)*1.2, len(table_rows)*0.5 + 2))
        ax.axis('off')
        tbl = ax.table(cellText=table_rows, colLabels=final_cols, cellLoc='center', loc='center')
        tbl.scale(1, 1.5)
        
        # Style Header
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
    
    # Flags for specific metrics
    # Usage: --metrics nlp green bertscore OR --metrics all
    parser.add_argument('--metrics', nargs='+', default=['all'], 
                        choices=['all', 'nlp', 'green', 'bertscore', 'radgraph', 'chexbert', 'radcliq'],
                        help="List of metrics to run. 'nlp' includes BLEU/ROUGE/METEOR.")
    
    args = parser.parse_args()
    
    if os.path.exists(args.input_csv):
        evaluate_metrics_sequentially(args)
    else:
        print(f"Error: Input file {args.input_csv} not found.")
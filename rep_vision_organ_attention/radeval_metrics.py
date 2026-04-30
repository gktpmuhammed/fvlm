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
import json
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_GT_JSON = PROJECT_ROOT / "data_sym" / "combined_desc_conc_v2.json"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

# --- MONKEYPATCH FIXED FOR RADEVAL GREEN METRIC ---
# The GREEN metric's clustering utils can fail when n_samples is small (e.g., <= 5)
# because of edge cases in silhouette_score (requires 2 <= n_labels <= n_samples - 1).
# Using batch_size=32 triggers this often as we cluster per-batch error sentences.
# We fully reimplement the binary search to be robust.

try:
    import RadEval.factual.green_score.utils as green_utils
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import warnings

    def _robust_binary_search_optimal_kmeans(data, min_k, max_k):
        # 1. Cap max_k based on data size
        n_samples = len(data)
        if max_k >= n_samples:
            max_k = n_samples - 1
            
        best_k = min_k
        best_score = -1.0
        # Default fallback: 1 cluster
        best_kmeans = KMeans(n_clusters=1, random_state=42).fit(data)

        while min_k <= max_k:
            mid_k = (min_k + max_k) // 2
            
            # Needs at least 2 clusters for silhouette_score
            if mid_k < 2:
                # If we are forced below 2, we can't find a better 'score', stop.
                break

            try:
                # Suppress convergence warnings for small data
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kmeans = KMeans(n_clusters=mid_k, random_state=42).fit(data)
                
                labels = kmeans.labels_
                
                # Check effectively used labels
                n_labels = len(set(labels))
                if n_labels < 2 or n_labels >= n_samples:
                    # Invalid for silhouette_score
                    # Try searching for fewer clusters
                    max_k = mid_k - 1
                    continue

                score = silhouette_score(data, labels)

                if score > best_score:
                    best_score = score
                    best_k = mid_k
                    best_kmeans = kmeans
                    min_k = mid_k + 1
                else:
                    max_k = mid_k - 1
                    
            except Exception:
                # If anything goes wrong (e.g. ValueError), treat as invalid k
                max_k = mid_k - 1

        return best_kmeans

    print("   ! Applying Robust Monkeypatch to RadEval.factual.green_score.utils.binary_search_optimal_kmeans")
    green_utils.binary_search_optimal_kmeans = _robust_binary_search_optimal_kmeans

except ImportError:
    print("   ! Warning: Could not import RadEval items for patching.")
except Exception as e:
    print(f"   ! Warning: Failed to apply RadEval monkeypatch: {e}")

# --------------------------------------------------

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
    
    BATCH_SIZE = 32
    num_samples = len(predictions)
    print(f"     > Running RadEval with {sum(metrics_config.values())} metrics on {num_samples} samples (Batch Size: {BATCH_SIZE})...")
    
    try:
        # Initialize RadEval with selected metrics
        evaluator = RadEval(**metrics_config)
        
        aggregated_scores = defaultdict(float)
        total_counts = defaultdict(int) 
        
        # Helper to extract scalar from result value (which might be dict)
        def get_scalar(val):
            if isinstance(val, (int, float)): return float(val)
            if isinstance(val, dict):
                # Prioritize standard keys
                for k in ['f1', 'score', 'micro avg_f1-score']:
                    if k in val: return float(val[k])
                # Fallback: first value
                return float(list(val.values())[0])
            return 0.0

        for i in range(0, num_samples, BATCH_SIZE):
            batch_preds = predictions[i : i + BATCH_SIZE]
            batch_refs = references[i : i + BATCH_SIZE]
            current_batch_size = len(batch_preds)
            
            print(f"       Batch {i//BATCH_SIZE + 1}/{(num_samples + BATCH_SIZE - 1)//BATCH_SIZE}...")
            
            # RadEval expects (references, predictions) order
            batch_results = evaluator(batch_refs, batch_preds)
            
            # extract_radeval_scores usually normalizes logic, but we do it here for aggregation
            # We must aggregate the raw keys returned by RadEval
            for key, val in batch_results.items():
                scalar = get_scalar(val)
                # Weighted sum
                aggregated_scores[key] += scalar * current_batch_size
                total_counts[key] += current_batch_size
            
            free_memory()

        # Compute averages
        final_results = {}
        for key, total_score in aggregated_scores.items():
            count = total_counts[key]
            if count > 0:
                final_results[key] = total_score / count
            else:
                final_results[key] = 0.0
        
        # Clean up evaluator
        del evaluator
        free_memory()
        
        return final_results
        
    except Exception as e:
        print(f"   ! RadEval error: {e}")
        import traceback
        traceback.print_exc()
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
    print(f"Loading generated reports from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    df['pred'] = df['prediction'].fillna("").astype(str)
    # df['ref'] is no longer the primary source of truth, we use the JSON.
    
    # Load Ground Truth JSON
    print(f"Loading ground truth from {args.ground_truth_json}...")
    with open(args.ground_truth_json, 'r') as f:
        ground_truth_data = json.load(f)

    # Optional Subset
    if args.subset and len(df) > args.subset:
        print(f"⚠️  SUBSET MODE: Using random {args.subset} samples.")
        df = df.sample(n=args.subset, random_state=42).reset_index(drop=True)

    # 2. Organize Data
    data_buckets = defaultdict(lambda: {'preds': [], 'refs': []})
    
    print("Parsing reports and matching with ground truth...")
    for _, row in df.iterrows():
        p_text = row['pred']
        patient_id = str(row['patient_id'])
        
        # --- Resolve Patient ID ---
        # Try exact match first
        gt_key = None
        if patient_id in ground_truth_data:
            gt_key = patient_id
        else:
            # Try stripping suffix like _1, _2
            # e.g. valid_730_a_1 -> valid_730_a
            base_id = re.sub(r'_\d+$', '', patient_id)
            if base_id in ground_truth_data:
                gt_key = base_id
        
        if not gt_key:
            print(f"   ! Warning: No ground truth found for ID {patient_id} (tried {base_id if 'base_id' in locals() else 'N/A'})")
            continue
            
        gt_record = ground_truth_data[gt_key]
        
        # --- Construct Reference Text from JSON & Filter Predictions ---
        # Strategy: Iterate ALL_ORGANS. 
        # If in JSON -> Add to global list, add to organ bucket.
        # If NOT in JSON -> SKIP entirely (avoids metric inflation).
        
        global_ref_parts = []
        global_pred_parts = []
        
        p_parts = extract_organ_sections(p_text)
        
        for organ in ALL_ORGANS:
            organ_key = organ.lower()
            
            # --- CRITICAL CHANGE: Only process if present in Ground Truth ---
            if organ_key in gt_record:
                # 1. Get Reference
                r_content = gt_record[organ_key]
                r_content = normalize_text(r_content)
                
                # 2. Get Prediction (for this specific organ)
                p_content = p_parts.get(organ, "")
                p_content = normalize_text(p_content)
                
                # 3. Add to Organ Bucket
                data_buckets[organ]['preds'].append(p_content)
                data_buckets[organ]['refs'].append(r_content)
                
                # 4. Add to Global Construction
                # We rebuild the global strings to only contain the relevant organs
                if r_content:
                    global_ref_parts.append(f"{organ}: {r_content}")
                if p_content:
                    global_pred_parts.append(f"{organ}: {p_content}")

        # Join global parts
        # If no organs were found for this patient, we might end up with empty strings.
        # That's acceptable, they will be empty in both or have minor mismatches.
        if global_ref_parts:
            r_text_constructed = " ".join(global_ref_parts)
            # For prediction, we can either use the original full text OR the filtered constructed text.
            # Using filtered text is fairer if we are strictly ignoring other organs.
            # Let's use the filtered prediction to match the filtered reference.
            p_text_constructed = " ".join(global_pred_parts)
            
            data_buckets['GLOBAL']['preds'].append(p_text_constructed)
            data_buckets['GLOBAL']['refs'].append(r_text_constructed)

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
          [m for m, flag in [('METEOR', run_meteor), ('CIDEr', run_cider)] if flag])
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
            print(radeval_results)
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
        'Organ', 'N', 'GREEN', 
        'BLEU-4', 
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
    parser.add_argument('--ground_truth_json', type=str, default=str(DEFAULT_GT_JSON), 
                        help="Path to ground truth JSON file (default: combined_desc_conc.json)")
    parser.add_argument('--output_dir', type=str, default='./results_metrics', help="Folder to save results")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    parser.add_argument('--metrics', nargs='+', default=['all'], 
                        help="List: all, nlp, bleu, rouge, meteor, cider, accuracy, "
                             "bertscore, radgraph, chexbert, radcliq, green, srr_bert, ratescore")
    parser.add_argument('--subset', type=int, default=None, help="Debug: Evaluate only N samples")
    
    args = parser.parse_args()
    
    if os.path.exists(args.input_csv):
        evaluate_metrics_sequentially(args)
    else:
        print(f"Error: Input file {args.input_csv} not found.")

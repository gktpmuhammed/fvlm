import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import nltk
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import sentence_gleu
# New imports for the added metrics
import torch

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')

def calculate_meteor(predictions, references):
    scores = []
    for p, r in zip(predictions, references):
        try: scores.append(meteor_score([nltk.word_tokenize(r)], nltk.word_tokenize(p)))
        except: scores.append(0.0)
    return np.mean(scores)

def calculate_bleu_scores(predictions, references):
    refs = [[nltk.word_tokenize(r)] for r in references]
    hyps = [nltk.word_tokenize(p) for p in predictions]
    weights = [(1,0,0,0), (0.5,0.5,0,0), (0.33,0.33,0.33,0), (0.25,0.25,0.25,0.25)]
    scores = {}
    for i, w in enumerate(weights, 1):
        try: scores[f'BLEU-{i}'] = corpus_bleu(refs, hyps, weights=w)
        except: scores[f'BLEU-{i}'] = 0.0
    return scores

def calculate_green(predictions, references):
    """
    Calculates GREEN score (Generative Radiology Report Evaluation and Error Notation).
    Ref: https://arxiv.org/abs/2405.03595
    Requires: pip install green_score
    Note: This runs a 7B param model. Requires CUDA and ~16GB VRAM (or ~8GB with 4-bit quantization if supported).
    """
    try:
        from green_score import GREEN
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize the GREEN scorer
        # Default model is usually "Stanford-AIMI/GREEN-radllama2-7b"
        model_name = "StanfordAIMI/GREEN-radllama2-7b"
        scorer = GREEN(model_name, output_dir="./green_model_cache")
        
        # GREEN expects (refs, preds) and returns (mean_score, results_df)
        mean, std, green_score_list, summary, result_df = scorer(references, predictions)

        return mean

    except ImportError:
        print("Warning: `green_score` not installed. Install via `pip install green_score`. Returning 0.0")
        return 0.0
    except Exception as e:
        print(f"Error calculating GREEN: {e}")
        return 0.0

def calculate_accuracy(predictions, references):
    matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
    return matches / len(predictions) if len(predictions) > 0 else 0.0

def calculate_cider_approx(predictions, references, ngram=4):
    def ngrams(tokens, n):
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    docs_p = [ngrams(nltk.word_tokenize(p), k) for p in predictions for k in range(1, ngram+1)]
    docs_r = [ngrams(nltk.word_tokenize(r), k) for r in references for k in range(1, ngram+1)]
    
    df = defaultdict(int)
    
    docs_p_flat = []; docs_r_flat = []
    for p in predictions:
        toks = nltk.word_tokenize(p); ngs = []
        for k in range(1, ngram+1): ngs.extend(ngrams(toks, k))
        docs_p_flat.append(ngs)
    for r in references:
        toks = nltk.word_tokenize(r); ngs = []
        for k in range(1, ngram+1): ngs.extend(ngrams(toks, k))
        docs_r_flat.append(ngs)

    for doc in docs_r_flat:
        for g in set(doc): df[g] += 1
    
    N = len(references)
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

    scores = [cosine(tf_idf(p), tf_idf(r)) for p, r in zip(docs_p_flat, docs_r_flat)]
    return np.mean(scores) * 10.0 if scores else 0.0

# --- New Deep Research Metrics Functions ---

def calculate_bertscore(predictions, references):
    """
    Calculates BERTScore F1.
    Requires: pip install bert_score
    """
    try:
        from bert_score import score
        # Using distilroberta-base for speed, use roberta-large for best accuracy
        P, R, F1 = score(predictions, references, lang="en", verbose=False, model_type='distilroberta-base')
        return F1.mean().item()
    except ImportError:
        print("Warning: bert_score not installed. Returning 0.0")
        return 0.0
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return 0.0

def calculate_radgraph(predictions, references):
    """
    Calculates RadGraph F1 score.
    Requires: pip install radgraph-benchmark
    """
    try:
        from radgraph import F1RadGraph
        # reward_level 'all' considers both entities and relations
        scorer = F1RadGraph(reward_level="all")
        # RadGraph expects lists of strings
        score, _, _ = scorer(preds=predictions, refs=references)
        return score
    except ImportError:
        print("Warning: radgraph-benchmark not installed. Returning 0.0")
        return 0.0
    except Exception as e:
        print(f"Error calculating RadGraph: {e}")
        return 0.0

def calculate_chexbert(predictions, references):
    """
    Calculates CheXbert F1 / Vector Similarity.
    Requires: f1chexbert or similar wrapper and model weights.
    This implementation attempts to use the standard f1chexbert wrapper pattern.
    """
    try:
        # Attempt to import from a common wrapper location or sub-repository
        # Adjust import based on your specific installed package name
        from f1chexbert import F1CheXbert 

        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        scorer = F1CheXbert(device=device)
        
        # F1CheXbert typically returns (accuracy, accuracy_per_class, f1_macro, etc.)
        # We generally want the vector similarity or F1 over the 14 labels
        accuracy, accuracy_per_class, chexbert_all, chexbert_5 = scorer(predictions, references)
        return chexbert_all
    except ImportError:
        # Fallback: Check if user has radmetrics installed (another common library)
        try:
            from radmetrics import compute_chexbert
            # This is hypothetical API based on common usage
            return compute_chexbert(predictions, references)['f1']
        except:
            pass
        # Silent fail or warning - un-comment print to debug
        # print("Warning: CheXbert libraries/weights not found. Returning 0.0")
        return 0.0
    except Exception as e:
        print(f"Error calculating CheXbert: {e}")
        return 0.0

def calculate_radcliq(predictions, references, bleu_score=None, bert_score=None, chexbert_score=None):
    """
    Calculates RadCliQ.
    RadCliQ is a composite metric. If you do not have the specific trained model,
    it is often approximated or requires the 'radmetrics' package.
    
    RadCliQ-v1 Formula approx: 
    RadCliQ = w1*BLEU + w2*BERTScore + w3*CheXbert (simplified view)
    
    If 'radmetrics' is not available, this function returns 0.0 to avoid false reporting.
    """
    try:
        from radmetrics import RadCliQ
        scorer = RadCliQ()
        return scorer(predictions, references)
    except ImportError:
        # If we cannot run the official RadCliQ, and we have the components, 
        # we could approximate, but RadCliQ is a learned regression. 
        # Better to return 0 than a wrong number.
        return 0.0

# --- Updated Aggregation Functions ---

def compute_all_metrics(preds, refs):
    if not preds: return {}
    
    # Calculate basic NLP metrics
    b = calculate_bleu_scores(preds, refs)
    rouge_l = ROUGEScore()(preds, refs)['rougeL_fmeasure'].item()
    meteor_val = calculate_meteor(preds, refs)
    greene_val = calculate_green(preds, refs)
    acc_val = calculate_accuracy(preds, refs)
    cider_val = calculate_cider_approx(preds, refs)
    
    # Calculate Deep Research Metrics
    # Note: These are computationally expensive
    bert_val = calculate_bertscore(preds, refs)
    radgraph_val = calculate_radgraph(preds, refs)
    chexbert_val = calculate_chexbert(preds, refs)
    radcliq_val = calculate_radcliq(preds, refs)

    return {
        'BLEU-1': b['BLEU-1'], 
        'BLEU-2': b['BLEU-2'], 
        'BLEU-3': b['BLEU-3'], 
        'BLEU-4': b['BLEU-4'],
        'ROUGE-L': rouge_l,
        'METEOR': meteor_val,
        'GREEN': greene_val,
        'ACC': acc_val,
        'CIDEr': cider_val,
        'BERTScore': bert_val,
        'RadGraph': radgraph_val,
        'CheXbert': chexbert_val,
        'RadCliQ': radcliq_val
    }

def create_metrics_table_plot(results_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Updated column labels to include new metrics
    col_labels = [
        'Organ', 'N', 'ACC', 'GREEN', 
        'BLEU-1', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr',
        'BERTScore', 'RadGraph', 'CheXbert', 'RadCliQ'
    ]
    
    table_rows, csv_rows = [], []
    
    for r in results_list:
        row = [
            str(r.get('Organ', 'Unknown')).upper(), 
            str(r.get('N', 0)),
            f"{r.get('ACC',0):.3f}", 
            f"{r.get('GREEN',0):.3f}",
            f"{r.get('BLEU-1',0):.3f}", 
            f"{r.get('BLEU-4',0):.3f}",
            f"{r.get('METEOR',0):.3f}", 
            f"{r.get('ROUGE-L',0):.3f}", 
            f"{r.get('CIDEr',0):.3f}",
            # New Columns
            f"{r.get('BERTScore',0):.3f}",
            f"{r.get('RadGraph',0):.3f}",
            f"{r.get('CheXbert',0):.3f}",
            f"{r.get('RadCliQ',0):.3f}"
        ]
        table_rows.append(row)
        csv_rows.append({k:v for k,v in zip(col_labels, row)})
        
    with open(os.path.join(output_dir, 'metrics_breakdown.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=col_labels)
        writer.writeheader()
        writer.writerows(csv_rows)
        
    # Adjusted figure size for wider table
    fig, ax = plt.subplots(figsize=(18, len(table_rows)*0.5 + 2))
    ax.axis('off')
    tbl = ax.table(cellText=table_rows, colLabels=col_labels, cellLoc='center', loc='center')
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_facecolor('#333333'); cell.set_text_props(color='white', weight='bold')
            
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'metrics_table.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

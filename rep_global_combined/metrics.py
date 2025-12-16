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

def calculate_greene(predictions, references):
    scores = []
    for p, r in zip(predictions, references):
        try: scores.append(sentence_gleu([nltk.word_tokenize(r)], nltk.word_tokenize(p)))
        except: scores.append(0.0)
    return np.mean(scores)

def calculate_accuracy(predictions, references):
    matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
    return matches / len(predictions) if len(predictions) > 0 else 0.0

def calculate_cider_approx(predictions, references, ngram=4):
    def ngrams(tokens, n):
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    docs_p = [ngrams(nltk.word_tokenize(p), k) for p in predictions for k in range(1, ngram+1)]
    docs_r = [ngrams(nltk.word_tokenize(r), k) for r in references for k in range(1, ngram+1)]
    
    # Flatten structure for TF-IDF calc is complex, doing simple cosine avg approx here
    # A true CIDEr implementation is much more complex (like pycocoevalcap), 
    # but this aligns with the provided script's logic.
    
    # Re-implementing logic from provided script exactly:
    df = defaultdict(int)
    # This logic seems to treat the list of ngrams as a single document? 
    # Let's match the provided evaluate.py logic strictly.
    
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

def compute_all_metrics(preds, refs):
    if not preds: return {}
    b = calculate_bleu_scores(preds, refs)
    return {
        'BLEU-1': b['BLEU-1'], 'BLEU-2': b['BLEU-2'], 'BLEU-3': b['BLEU-3'], 'BLEU-4': b['BLEU-4'],
        'ROUGE-L': ROUGEScore()(preds, refs)['rougeL_fmeasure'].item(),
        'METEOR': calculate_meteor(preds, refs),
        'GREEN': calculate_greene(preds, refs),
        'ACC': calculate_accuracy(preds, refs),
        'CIDEr': calculate_cider_approx(preds, refs)
    }

def create_metrics_table_plot(results_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    col_labels = ['Organ', 'N', 'ACC', 'GREEN', 'BLEU-1', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
    table_rows, csv_rows = [], []
    
    for r in results_list:
        row = [
            str(r.get('Organ', 'Unknown')).upper(), str(r.get('N', 0)),
            f"{r.get('ACC',0):.3f}", f"{r.get('GREEN',0):.3f}",
            f"{r.get('BLEU-1',0):.3f}", f"{r.get('BLEU-4',0):.3f}",
            f"{r.get('METEOR',0):.3f}", f"{r.get('ROUGE-L',0):.3f}", f"{r.get('CIDEr',0):.3f}"
        ]
        table_rows.append(row)
        csv_rows.append({k:v for k,v in zip(col_labels, row)})
        
    with open(os.path.join(output_dir, 'metrics_breakdown.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=col_labels)
        writer.writeheader(); writer.writerows(csv_rows)
        
    fig, ax = plt.subplots(figsize=(14, len(table_rows)*0.5 + 2))
    ax.axis('off')
    tbl = ax.table(cellText=table_rows, colLabels=col_labels, cellLoc='center', loc='center')
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_facecolor('#333333'); cell.set_text_props(color='white', weight='bold')
            
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'metrics_table.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
# following CT-CLIP's evaluation protocol

import math
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score

def find_threshold(probabilities, true_labels):
    """
    Finds the optimal threshold for binary classification based on ROC curve.

    Args:
        probabilities (numpy.ndarray): Predicted probabilities.
        true_labels (numpy.ndarray): True labels.

    Returns:
        float: Optimal threshold.
    """
    best_threshold = 0
    best_roc = 10000

    # Iterate over potential thresholds
    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        predictions = (probabilities > threshold).astype(int)
        confusion = confusion_matrix(true_labels, predictions)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        TP_r = TP / (TP + FN)
        FP_r = FP / (FP + TN)
        current_roc = math.sqrt(((1 - TP_r) ** 2) + (FP_r ** 2))
        if current_roc <= best_roc:
            best_roc = current_roc
            best_threshold = threshold

    return best_threshold

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', type=str, default='')
parser.add_argument('--out_dir', type=str, default='', help='Directory to save metrics CSV. Defaults to zero_shot_logs/$SLURM_JOB_ID if set, else alongside logs.')
args = parser.parse_args()

label_csv = pd.read_csv('data/multi_abnormality_labels/valid_predicted_labels.csv')

result = pd.read_csv(args.csv_file)

columns = list(result.columns[1:])

auc_scores = {}
spec_scores = {}
sens_scores = {}
acc_scores = {}
f1_scores = {}
prec_scores = {}

for column in columns:
    labels = []
    probs = []
    
    abnormality = column.split('_')[1]

    for file_name, prob in zip(result['file_name'], result[column]):
        if np.isnan(prob): # very few cases
            prob = 0
        probs.append(prob)
        labels.append(label_csv[label_csv['VolumeName'] == file_name][abnormality].values[0])

    auc_scores[abnormality] = roc_auc_score(labels, probs)

    threshold = find_threshold(np.array(probs), np.array(labels))
    
    pd_labels = (np.array(probs) > threshold).astype(int)
    gt_labels = np.array(labels)

    tp = ((pd_labels == 1) & (gt_labels == 1)).sum().item()
    fp = ((pd_labels == 1) & (gt_labels == 0)).sum().item()
    tn = ((pd_labels == 0) & (gt_labels == 0)).sum().item()
    fn = ((pd_labels == 0) & (gt_labels == 1)).sum().item()

    if tn + fp != 0:
        spec = tn / (tn + fp)
    else:
        spec = np.nan

    if tp + fn != 0:
        sens = tp / (tp + fn)
    else:
        sens = np.nan

    spec_scores[abnormality] = spec
    sens_scores[abnormality] = sens

    f1 = f1_score(gt_labels, pd_labels, average="weighted")
    prec = precision_score(gt_labels, pd_labels)

    f1_scores[abnormality] = f1
    prec_scores[abnormality] = prec

    acc = (tp + tn) / (tp + tn + fp + fn)
    acc_scores[abnormality] = acc

print('\n\n')

print('Average AUC of Ours:', round(np.mean(list(auc_scores.values())), 3))
print('Average ACC of Ours:', round(np.mean(list(acc_scores.values())), 3))
print('Average Spec of Ours:', round(np.mean(list(spec_scores.values())), 3))
print('Average Sens of Ours:', round(np.mean(list(sens_scores.values())), 3))
print('Average F1 of Ours:', round(np.mean(list(f1_scores.values())), 3))
print('Average Prec of Ours:', round(np.mean(list(prec_scores.values())), 3))

# =============================
# Save metrics to CSV files
# =============================

# Determine output directory: prefer the Slurm log folder
job_id = os.environ.get('SLURM_JOB_ID')
default_out_dir = f"zero_shot_logs/{job_id}" if job_id else ''
out_dir = args.out_dir or default_out_dir

if not out_dir:
    # Fallback to current working directory if no slurm job id and no explicit out_dir
    out_dir = os.getcwd()

os.makedirs(out_dir, exist_ok=True)

# Build per-abnormality metrics table
abnormalities = list(auc_scores.keys())
records = []
for ab in abnormalities:
    records.append({
        'abnormality': ab,
        'auc': auc_scores[ab],
        'acc': acc_scores[ab],
        'spec': spec_scores[ab],
        'sens': sens_scores[ab],
        'f1': f1_scores[ab],
        'prec': prec_scores[ab],
    })

per_class_df = pd.DataFrame.from_records(records)

# Summary row
summary_df = pd.DataFrame([{ 
    'avg_auc': round(np.mean(list(auc_scores.values())), 6),
    'avg_acc': round(np.mean(list(acc_scores.values())), 6),
    'avg_spec': round(np.mean(list(spec_scores.values())), 6),
    'avg_sens': round(np.mean(list(sens_scores.values())), 6),
    'avg_f1': round(np.mean(list(f1_scores.values())), 6),
    'avg_prec': round(np.mean(list(prec_scores.values())), 6),
}])

# Compose output filenames
base_name = os.path.splitext(os.path.basename(args.csv_file))[0]
per_class_path = os.path.join(out_dir, f"metrics_per_class_{base_name}.csv")
summary_path = os.path.join(out_dir, f"metrics_summary_{base_name}.csv")

per_class_df.to_csv(per_class_path, index=False)
summary_df.to_csv(summary_path, index=False)

print(f"Saved per-class metrics to: {per_class_path}")
print(f"Saved summary metrics to: {summary_path}")

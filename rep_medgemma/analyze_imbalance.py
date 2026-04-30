import json
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

# Path to ground truth JSON
JSON_FILE = str(PROJECT_ROOT / 'data_sym/combined_desc_conc.json')
CSV_FILE = str(PROJECT_ROOT / 'data_sym/image_first_dataset.csv')

ALL_TARGET_KEYS = [
    'lung', 'heart', 'esophagus', 
    'liver', 'gallbladder', 'stomach', 'pancreas', 'spleen', 'kidney',
    'aorta', 'trachea', 'rib'
]

def analyze_imbalance():
    print(f"Loading JSON from {JSON_FILE}...")
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    # Note: The dataset in train.py filters by CSV file first. 
    # We should mimic that to match the actual training set.
    print(f"Loading CSV from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    df = df[df['split'] == 'training'].reset_index(drop=True)
    
    valid_ids = []
    for _, row in df.iterrows():
        fname = os.path.basename(row['image_path'])
        base_id = fname.replace('.nii.gz', '').replace('.nii', '')
        if base_id in data:
            valid_ids.append(base_id)
        elif len(base_id.split('_')) > 1 and base_id.rsplit('_', 1)[0] in data:
            valid_ids.append(base_id.rsplit('_', 1)[0])
            
    print(f"Found {len(valid_ids)} valid training samples.")
    
    counts = defaultdict(lambda: {'explicit': 0, 'default': 0})
    
    for pid in valid_ids:
        patient_data = data.get(pid, {})
        for organ in ALL_TARGET_KEYS:
            text = patient_data.get(organ, "").strip()
            # Logic from OnePassOrganDataset
            if len(text) < 3:
                counts[organ]['default'] += 1
            else:
                counts[organ]['explicit'] += 1
                
    # Format results
    print("\nImbalance Analysis:")
    print(f"{'Organ':<15} | {'Explicit':<10} | {'Default':<10} | {'Total':<10} | {'% Explicit':<10}")
    print("-" * 65)
    
    results = {}
    for organ in ALL_TARGET_KEYS:
        # Sort explicitly if needed, here preserving order
        c = counts[organ]
        total = c['explicit'] + c['default']
        pct_explicit = (c['explicit'] / total * 100) if total > 0 else 0
        print(f"{organ:<15} | {c['explicit']:<10} | {c['default']:<10} | {total:<10} | {pct_explicit:<9.1f}%")
        
        
        
        # Calculate Weights with Softening (Power Scaling)
        # alpha = 1.0 -> Inverse Frequency (Aggressive)
        # alpha = 0.5 -> Square Root Inverse (Moderate)
        # alpha = 0.0 -> No Weighting (1.0 each)
        
        # We'll calculate both for comparison
        n_exp = max(c['explicit'], 1)
        n_def = max(c['default'], 1)
        
        # 1. Pure Inverse (alpha=1.0)
        inv_exp = 1.0 / n_exp
        inv_def = 1.0 / n_def
        tot_inv = inv_exp + inv_def
        w_exp_pure = inv_exp / tot_inv
        w_def_pure = inv_def / tot_inv
        
        # 2. Softened (alpha=0.5)
        soft_exp = 1.0 / (n_exp ** 0.5)
        soft_def = 1.0 / (n_def ** 0.5)
        tot_soft = soft_exp + soft_def
        w_exp_soft = soft_exp / tot_soft
        w_def_soft = soft_def / tot_soft
        
        print(f"{organ:<15} | exp:{c['explicit']:<6} def:{c['default']:<6} | Pure [Exp:{w_exp_pure:.3f} Def:{w_def_pure:.3f}] | Soft [Exp:{w_exp_soft:.3f} Def:{w_def_soft:.3f}]")
        
        results[organ] = {
            'explicit_count': c['explicit'],
            'default_count': c['default'],
            'weight_explicit': w_exp_soft, # Defaulting to soft for safety, but user can choose
            'weight_default': w_def_soft,
            'weight_explicit_pure': w_exp_pure,
            'weight_default_pure': w_def_pure
        }

    # Save to JSON
    output_path = os.path.join(os.path.dirname(CSV_FILE), 'organ_loss_weights.json')
    print(f"\nSaving weights to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    analyze_imbalance()

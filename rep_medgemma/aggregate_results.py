import os
from pathlib import Path
import pandas as pd
import glob

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

RESULTS_DIR = str(PROJECT_ROOT / 'rep_medgemma/results_retrain')
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'global_model_comparison.csv')
TABLE_IMAGE_FILE = os.path.join(RESULTS_DIR, 'global_model_comparison_table.png')

def save_table_image(df, filename):
    fig, ax = plt.figure(figsize=(20, 8)), plt.gca()
    ax.axis('off')
    
    # Round float columns for better display
    display_df = df.copy()
    for col in display_df.select_dtypes(include=['float']).columns:
        display_df[col] = display_df[col].round(4)
        
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully created table image: {filename}")

def main():
    all_results = []
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        return

    # Iterate through all subdirectories in results folder
    for model_name in os.listdir(RESULTS_DIR):
        model_path = os.path.join(RESULTS_DIR, model_name)
        if not os.path.isdir(model_path):
            continue
            
        metrics_file = os.path.join(model_path, 'metrics_final.csv')
        if not os.path.exists(metrics_file):
            # Try to look recursively if needed, or just skip
            # specific request was "iterate all results folde ... if metrics csv file present"
            print(f"Skipping {model_name}: metrics_final.csv not found")
            continue
            
        try:
            df = pd.read_csv(metrics_file)
            
            # Check if required columns exist to avoid errors
            if 'Organ' not in df.columns:
                print(f"Skipping {model_name}: 'Organ' column missing in CSV")
                continue

            # Find the GLOBAL row
            global_row = df[df['Organ'] == 'GLOBAL'].copy()
            
            if not global_row.empty:
                # Add Model identifier
                global_row.insert(0, 'Model', model_name)
                all_results.append(global_row)
            else:
                print(f"Skipping {model_name}: GLOBAL row not found")
                
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        # Sort by Model name
        final_df = final_df.sort_values(by='Model')
        
        # Save to CSV
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully created comparison CSV: {OUTPUT_FILE}")
        
        # Save as image
        try:
            save_table_image(final_df, TABLE_IMAGE_FILE)
        except Exception as e:
            print(f"Error creating table image: {e}")
        
        # Print table
        print("\nGlobal Model Comparison:")
        print(final_df.to_string(index=False))
    else:
        print("No results found to aggregate.")

if __name__ == '__main__':
    main()

import json
import os
import argparse
from pathlib import Path

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def merge_datasets(findings, impressions):
    merged = {}
    all_ids = set(findings.keys()) | set(impressions.keys())
    
    for uid in all_ids:
        merged[uid] = {}
        p_findings = findings.get(uid, {})
        p_impressions = impressions.get(uid, {})
        
        all_organs = set(p_findings.keys()) | set(p_impressions.keys())
        
        for organ in all_organs:
            f_text = p_findings.get(organ, "").strip()
            i_text = p_impressions.get(organ, "").strip()
            
            combined_text = f_text
            if i_text:
                if combined_text:
                     combined_text += " " + i_text
                else:
                     combined_text = i_text
            
            merged[uid][organ] = combined_text
    return merged

def main():
    project_root = Path(
        os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1])
    ).resolve()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default=str(project_root / "decomposed_data"),
        help="Directory containing train/val findings and impressions JSON files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional output path for combined JSON (defaults to <base_dir>/combined_train_val.json).",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    
    paths = {
        "train_findings": os.path.join(base_dir, "train_findings.json"),
        "train_impressions": os.path.join(base_dir, "train_impressions.json"),
        "val_findings": os.path.join(base_dir, "val_findings.json"),
        "val_impressions": os.path.join(base_dir, "val_impressions.json"),
    }
    
    print("Loading files...")
    train_findings = load_json(paths["train_findings"])
    train_impressions = load_json(paths["train_impressions"])
    val_findings = load_json(paths["val_findings"])
    val_impressions = load_json(paths["val_impressions"])
    
    print("Merging train data...")
    train_merged = merge_datasets(train_findings, train_impressions)
    print(f"Merged {len(train_merged)} train records.")
    
    print("Merging val data...")
    val_merged = merge_datasets(val_findings, val_impressions)
    print(f"Merged {len(val_merged)} val records.")
    
    # Combine both
    final_data = {**train_merged, **val_merged}
    print(f"Total records: {len(final_data)}")
    
    output_path = args.output_file or os.path.join(base_dir, "combined_train_val.json")
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()

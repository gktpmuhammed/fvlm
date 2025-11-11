import pandas as pd
import os

def apply_voxel_threshold(csv_path, threshold=70, output_dir='output/organ_voxel_pct_threshold70'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    organ_columns = [col.replace('_retained_pct', '') for col in df.columns if '_retained_pct' in col]
    results = []

    # Analyze retained/eliminated per organ
    summary = []
    for organ in organ_columns:
        col = f"{organ}_retained_pct"
        # Some scans may not have the organ; filter out nans if present
        valid = df[col].notna()
        retained = (df.loc[valid, col] >= threshold).sum()
        eliminated = (df.loc[valid, col] < threshold).sum()
        total = valid.sum()
        summary.append({
            'organ': organ,
            'retained_scans': retained,
            'eliminated_scans': eliminated,
            'total_scans': total,
            'retained_pct': retained / total * 100 if total > 0 else 0,
            'eliminated_pct': eliminated / total * 100 if total > 0 else 0,
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, f'organ_retention_summary_{threshold}pct.csv'), index=False)

    # Print general overview
    print("Retention summary at {}% threshold:".format(threshold))
    print(summary_df[['organ','retained_scans','eliminated_scans','total_scans','retained_pct','eliminated_pct']])

    return summary_df

if __name__ == "__main__":
    # Point to your per_scan_organ_voxel_percentages.csv file
    csv_file = 'output/organ_voxel_pct_analysis_debug/per_scan_organ_voxel_percentages.csv'
    summary = apply_voxel_threshold(csv_file, threshold=70)

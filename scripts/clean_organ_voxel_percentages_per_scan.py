import pandas as pd
import os

df = pd.read_csv('../output/organ_voxel_pct_analysis_debug/per_scan_organ_voxel_percentages.csv')
output_dir = '../output/organ_voxel_pct_analysis_debug/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# keep only columns with 'retained' in their name together with 'patient_id'
df = df[['patient_id'] + [col for col in df.columns if 'retained' in col]]
print(df.head())

# save to new csv
df.to_csv(os.path.join(output_dir, 'cleaned_per_scan_organ_voxel_percentages.csv'), index=False)
print(f"Cleaned data saved to {os.path.join(output_dir, 'cleaned_per_scan_organ_voxel_percentages.csv')}")
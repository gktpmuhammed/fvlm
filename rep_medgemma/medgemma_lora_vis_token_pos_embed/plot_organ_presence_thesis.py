import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_grouped_bar_chart(csv_path, output_path):
    # Overall Visibility vs. Reporting Percentage
    df = pd.read_csv(csv_path)
    df = df[df['split'] == 'training'] # Focus on training set for general statistics
    
    # Sort by report_pct descending
    df = df.sort_values('report_pct', ascending=False)
    
    organs = df['organ'].tolist()
    report_pcts = df['report_pct'].tolist()
    mask_pcts = df['visible_pct'].tolist()
    
    x = np.arange(len(organs))
    width = 0.35
    
    # Use seaborn pastel colors
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("pastel")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, mask_pcts, width, label='Visible in Image (Mask)', color=colors[0])
    rects2 = ax.bar(x + width/2, report_pcts, width, label='Mentioned in Report', color=colors[3])
    
    ax.set_ylabel('Percentage of Scans (%)', fontsize=12)
    ax.set_title('Organ Presence: Image Visibility vs. Report Mentions (Training Set)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([o.capitalize() for o in organs], rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=11)
    
    sns.despine()
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_stacked_bar_chart(csv_path, output_path):
    df = pd.read_csv(csv_path)
    df = df[df['split'] == 'training']
    
    # Keep consistent organ order
    organs = ['lung', 'heart', 'esophagus', 'liver', 'gallbladder', 'stomach', 'pancreas', 'spleen', 'kidney', 'aorta', 'trachea', 'rib']
    
    data = []
    
    for organ in organs:
        mask_col = f'mask_{organ}'
        report_col = f'report_{organ}'
        
        # Categories
        both = len(df[(df[mask_col] == 1) & (df[report_col] == 1)])
        mask_only = len(df[(df[mask_col] == 1) & (df[report_col] == 0)])
        report_only = len(df[(df[mask_col] == 0) & (df[report_col] == 1)])
        neither = len(df[(df[mask_col] == 0) & (df[report_col] == 0)])
        
        total = len(df)
        
        data.append({
            'Organ': organ.capitalize(),
            'Visible & Mentioned': both / total * 100,
            'Image Only (Unreported)': mask_only / total * 100,
            'Mentioned Only': report_only / total * 100,
            'Neither': neither / total * 100
        })
        
    df_stacked = pd.DataFrame(data)
    # Sort by 'Visible & Mentioned' percentage descending for better visualization, OR keep consistent order
    # User requested: "keep organ order consistent with the rest of the thesis"
    # Wait, the rest of the thesis usually reports organs in a specific order (e.g. Lung, Heart, Esophagus, Trachea, Aorta, Liver, Kidney, Gallbladder, Spleen, Rib, Pancreas, Stomach as in Table 04_experiments.tex tab:organ_coverage_eval). Let's use THAT order.
    
    thesis_order = ['lung', 'heart', 'esophagus', 'trachea', 'aorta', 'liver', 'kidney', 'gallbladder', 'spleen', 'rib', 'pancreas', 'stomach']
    df_stacked['Organ_lower'] = df_stacked['Organ'].str.lower()
    df_stacked['Order'] = df_stacked['Organ_lower'].map({o: i for i, o in enumerate(thesis_order)})
    df_stacked = df_stacked.sort_values('Order').drop(['Organ_lower', 'Order'], axis=1)
    
    sns.set_theme(style="whitegrid")
    # Soft pastel palette: green, yellow/orange, red/pink, grey
    # Let's use specific hex codes for a nice pastel look
    colors = ['#A8D5BA', '#FBD490', '#F4A261', '#D3D3D3'] 
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot stacked
    df_stacked.set_index('Organ').plot(
        kind='bar', stacked=True, ax=ax,
        color=colors,
        edgecolor='white',
        width=0.8
    )
    
    ax.set_ylabel('Percentage of Scans (%)', fontsize=12)
    ax.set_title('Alignment Types of Organ Information (Training Set)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    ax.legend(title='Presence Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title_fontsize=12)
    ax.set_ylim(0, 100)
    
    sns.despine(left=True, bottom=False)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    counts_csv = '/home/muhammedg/fvlm/rep_medgemma/medgemma_lora_vis_token_pos_embed/analysis_outputs/organ_coverage_both_20260311_065235/organ_coverage_counts.csv'
    wide_csv = '/home/muhammedg/fvlm/rep_medgemma/medgemma_lora_vis_token_pos_embed/analysis_outputs/patient_organ_presence_20260311_070119/patient_organ_presence_wide.csv'
    
    out_dir = '/home/muhammedg/fvlm/rep_medgemma/medgemma_lora_vis_token_pos_embed/analysis_outputs/thesis_figures'
    os.makedirs(out_dir, exist_ok=True)
    
    create_grouped_bar_chart(counts_csv, os.path.join(out_dir, 'grouped_bar_presence.png'))
    create_stacked_bar_chart(wide_csv, os.path.join(out_dir, 'stacked_bar_alignment.png'))

#!/usr/bin/env python3
"""
Comprehensive Encoder Diversity Analysis
Analyzes ALL samples in training and validation sets
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_encoder_diversity(encoder, dataloader, split_name, device, save_embeddings=False):
    """
    Analyze encoder diversity across entire dataset

    Args:
        encoder: Vision encoder model
        dataloader: DataLoader for the split
        split_name: 'training' or 'validation'
        device: torch device
        save_embeddings: Whether to save embeddings to disk

    Returns:
        results: Dict with diversity statistics
        embeddings: All encoder outputs (if save_embeddings=True)
    """

    print(f"\n{'='*80}")
    print(f"ENCODER DIVERSITY ANALYSIS: {split_name.upper()} SET")
    print("="*80)

    encoder.eval()

    # Collect all embeddings
    all_embeddings = []
    all_image_paths = []

    print(f"\nEncoding {len(dataloader.dataset)} samples...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding {split_name}"):
            images = batch['pixel_values'].to(device)

            # Encode (ViT returns tuple: (features, intermediate_outs))
            encoder_output = encoder(images)
            if isinstance(encoder_output, tuple):
                features = encoder_output[0]  # [B, num_patches, hidden_dim]
            else:
                features = encoder_output
            
            # Mean pool to get single vector per image
            pooled = features.mean(dim=1)  # [B, hidden_dim]

            all_embeddings.append(pooled.cpu())

            if 'image_path' in batch:
                all_image_paths.extend(batch['image_path'])

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, hidden_dim]
    N, D = all_embeddings.shape

    print(f"\nCollected {N} embeddings of dimension {D}")

    # ========================================================================
    # 1. GLOBAL STATISTICS
    # ========================================================================

    mean = all_embeddings.mean().item()
    std = all_embeddings.std().item()
    min_val = all_embeddings.min().item()
    max_val = all_embeddings.max().item()

    print(f"\n{'='*80}")
    print("1. GLOBAL EMBEDDING STATISTICS")
    print("="*80)
    print(f"Mean:  {mean:.6f}")
    print(f"Std:   {std:.6f}")
    print(f"Min:   {min_val:.6f}")
    print(f"Max:   {max_val:.6f}")
    print(f"Range: {max_val - min_val:.6f}")

    # ========================================================================
    # 2. PAIRWISE SIMILARITY ANALYSIS
    # ========================================================================

    print(f"\n{'='*80}")
    print("2. PAIRWISE SIMILARITY ANALYSIS")
    print("="*80)

    # Normalize embeddings for cosine similarity
    normalized = all_embeddings / (all_embeddings.norm(dim=1, keepdim=True) + 1e-8)

    # Compute pairwise cosine similarities (memory efficient for large N)
    print(f"\nComputing {N}x{N} similarity matrix...")

    if N < 5000:
        # For smaller datasets, compute full matrix
        similarity_matrix = normalized @ normalized.t()
        similarity_matrix = similarity_matrix.numpy()
    else:
        # For large datasets, sample pairs
        print("  (Using sampling for large dataset)")
        n_samples = 10000
        indices = np.random.choice(N, size=min(n_samples, N), replace=False)
        sampled = normalized[indices]
        similarity_matrix = sampled @ sampled.t()
        similarity_matrix = similarity_matrix.numpy()

    # Extract off-diagonal (between different samples)
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    off_diagonal_sims = similarity_matrix[mask]

    # Statistics
    mean_sim = off_diagonal_sims.mean()
    std_sim = off_diagonal_sims.std()
    min_sim = off_diagonal_sims.min()
    max_sim = off_diagonal_sims.max()
    median_sim = np.median(off_diagonal_sims)

    # Percentiles
    p25 = np.percentile(off_diagonal_sims, 25)
    p75 = np.percentile(off_diagonal_sims, 75)

    print(f"\nPairwise Cosine Similarity Statistics:")
    print(f"  Mean:     {mean_sim:.4f}")
    print(f"  Median:   {median_sim:.4f}")
    print(f"  Std:      {std_sim:.4f}")
    print(f"  Min:      {min_sim:.4f}")
    print(f"  Max:      {max_sim:.4f}")
    print(f"  25th %ile: {p25:.4f}")
    print(f"  75th %ile: {p75:.4f}")

    # Interpretation
    print(f"\nINTERPRETATION:")
    if mean_sim > 0.95:
        print(f"  VERY HIGH similarity ({mean_sim:.3f}) - Encoder outputs are nearly identical!")
        print(f"     This suggests the encoder may be collapsed or not discriminative.")
    elif mean_sim > 0.85:
        print(f"  HIGH similarity ({mean_sim:.3f}) - Encoder outputs are quite similar.")
        print(f"     This is somewhat expected for similar data (all chest CTs) but could be better.")
    elif mean_sim > 0.70:
        print(f"  MODERATE similarity ({mean_sim:.3f}) - Reasonable diversity.")
        print(f"     Encoder captures both shared anatomy and individual differences.")
    else:
        print(f"  LOW similarity ({mean_sim:.3f}) - Good diversity!")
        print(f"     Encoder produces distinct representations for different samples.")

    # ========================================================================
    # 3. DIMENSIONALITY ANALYSIS
    # ========================================================================

    print(f"\n{'='*80}")
    print("3. DIMENSIONALITY ANALYSIS")
    print("="*80)

    # Compute PCA to see how many dimensions capture variance
    from sklearn.decomposition import PCA

    print(f"\nComputing PCA on {N} samples...")
    pca = PCA(n_components=min(50, D, N-1))
    pca.fit(all_embeddings.numpy())

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # How many components for 90%, 95%, 99% variance?
    n_90 = np.argmax(cumulative_var >= 0.90) + 1
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    n_99 = np.argmax(cumulative_var >= 0.99) + 1

    print(f"\nPCA Analysis:")
    print(f"  Dimensions for 90% variance: {n_90} / {D}")
    print(f"  Dimensions for 95% variance: {n_95} / {D}")
    print(f"  Dimensions for 99% variance: {n_99} / {D}")
    print(f"  Top 10 components explain: {cumulative_var[9]:.2%} of variance")

    if n_90 < D * 0.1:
        print(f"\n  Only {n_90} dims needed for 90% variance (<<{D} total)")
        print(f"     Embeddings may be redundant or low-rank.")
    else:
        print(f"\n  Embeddings use a good portion of available dimensions.")

    # ========================================================================
    # 4. DEAD NEURONS
    # ========================================================================

    print(f"\n{'='*80}")
    print("4. NEURON ACTIVITY ANALYSIS")
    print("="*80)

    # Check per-dimension variance
    dim_stds = all_embeddings.std(dim=0)  # [hidden_dim]
    dead_threshold = 1e-6
    dead_neurons = (dim_stds < dead_threshold).sum().item()

    print(f"\nNeuron Activity:")
    print(f"  Total dimensions: {D}")
    print(f"  Dead neurons (std < {dead_threshold}): {dead_neurons} ({100*dead_neurons/D:.2f}%)")
    print(f"  Active neurons: {D - dead_neurons} ({100*(D-dead_neurons)/D:.2f}%)")

    if dead_neurons > D * 0.1:
        print(f"\n  >10% dead neurons - encoder may not be fully utilized")
    else:
        print(f"\n  Most neurons are active")

    # ========================================================================
    # 5. MOST/LEAST SIMILAR PAIRS
    # ========================================================================

    print(f"\n{'='*80}")
    print("5. EXTREME SIMILARITY EXAMPLES")
    print("="*80)

    if N < 5000:
        # Find most and least similar pairs
        np.fill_diagonal(similarity_matrix, -1)  # Ignore self-similarity

        # Most similar
        max_idx = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
        max_sim_val = similarity_matrix[max_idx]

        # Least similar
        min_idx = np.unravel_index(similarity_matrix.argmin(), similarity_matrix.shape)
        min_sim_val = similarity_matrix[min_idx]

        print(f"\nMost similar pair:")
        print(f"  Samples {max_idx[0]} and {max_idx[1]}: {max_sim_val:.4f} similarity")

        print(f"\nLeast similar pair:")
        print(f"  Samples {min_idx[0]} and {min_idx[1]}: {min_sim_val:.4f} similarity")

    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================

    results = {
        'split': split_name,
        'n_samples': N,
        'embedding_dim': D,
        'mean': mean,
        'std': std,
        'similarity_mean': mean_sim,
        'similarity_std': std_sim,
        'similarity_median': median_sim,
        'similarity_min': min_sim,
        'similarity_max': max_sim,
        'similarity_p25': p25,
        'similarity_p75': p75,
        'pca_90': n_90,
        'pca_95': n_95,
        'pca_99': n_99,
        'dead_neurons': dead_neurons,
        'dead_neuron_pct': 100*dead_neurons/D,
    }

    embeddings_to_return = all_embeddings if save_embeddings else None

    return results, embeddings_to_return, similarity_matrix


def plot_similarity_distribution(train_sims, val_sims, output_dir):
    """Plot similarity distributions for train and val"""

    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training set
    axes[0].hist(train_sims, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(train_sims.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {train_sims.mean():.3f}')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Training Set - Pairwise Similarities')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Validation set
    axes[1].hist(val_sims, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1].axvline(val_sims.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {val_sims.mean():.3f}')
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Validation Set - Pairwise Similarities')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'encoder_diversity_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {plot_path}")
    plt.close()


def main(args):
    print("="*80)
    print("COMPREHENSIVE ENCODER DIVERSITY ANALYSIS")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load encoder
    print(f"\nLoading encoder from: {args.vision_encoder_path}")

    # Add parent directory to path to import lavis
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from lavis.models.blip_models.vit import ViT

    encoder = ViT(
        in_channels=1,
        img_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        num_classes=0,
    )

    checkpoint = torch.load(args.vision_encoder_path, map_location='cpu', weights_only=False)
    vision_state = {}

    for source in ['state_dict', 'model']:
        if source in checkpoint:
            for k, v in checkpoint[source].items():
                if k.startswith('visual_encoder.'):
                    vision_state[k.replace('visual_encoder.', '')] = v

    if not vision_state:
        for k, v in checkpoint.items():
            if k.startswith('visual_encoder.'):
                vision_state[k.replace('visual_encoder.', '')] = v
            else:
                vision_state[k] = v

    encoder.load_state_dict(vision_state, strict=False)
    encoder = encoder.to(device)
    encoder.eval()
    print("Encoder loaded")

    # Load datasets
    from monai.transforms import (
        Compose,
        LoadImaged,
        ScaleIntensityRanged,
        SpatialPadd,
        CenterSpatialCropd,
        Transposed,
        EnsureChannelFirstd,
    )
    from torch.utils.data import Dataset, DataLoader
    import SimpleITK as sitk

    class SimpleDataset(Dataset):
        def __init__(self, csv_file, split, transform):
            df = pd.read_csv(csv_file)
            self.data = df[df['split'] == split].reset_index(drop=True)
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            image_dict = self.transform({'image': row['image_path']})
            image = image_dict['image']
            if isinstance(image, sitk.Image):
                image = sitk.GetArrayFromImage(image)
            image = torch.from_numpy(np.array(image)).float()
            if image.dim() == 3:
                image = image.unsqueeze(0)

            return {
                'pixel_values': image,
                'image_path': row['image_path']
            }

    transform = Compose([
        LoadImaged(keys=['image'], reader='ITKReader', image_only=True),
        EnsureChannelFirstd(keys=['image']),
        Transposed(keys=['image'], indices=(0, 3, 2, 1)),
        ScaleIntensityRanged(keys=['image'], a_min=-1150, a_max=350, b_min=0, b_max=1, clip=True),
        SpatialPadd(keys=['image'], spatial_size=(112, 256, 352)),
        CenterSpatialCropd(keys=['image'], roi_size=(112, 256, 352)),
    ])

    print(f"\nLoading datasets from: {args.csv_file}")

    train_dataset = SimpleDataset(args.csv_file, 'training', transform)
    val_dataset = SimpleDataset(args.csv_file, 'validation', transform)

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Analyze training set
    train_results, _, train_sim_matrix = analyze_encoder_diversity(
        encoder, train_loader, 'training', device, save_embeddings=False
    )

    # Analyze validation set
    val_results, _, val_sim_matrix = analyze_encoder_diversity(
        encoder, val_loader, 'validation', device, save_embeddings=False
    )

    # Compare train vs val
    print(f"\n\n{'='*80}")
    print("TRAIN VS VALIDATION COMPARISON")
    print("="*80)

    print(f"\nSimilarity Statistics:")
    print(f"{'Metric':<30} {'Training':<15} {'Validation':<15}")
    print("-"*60)
    print(f"{'Mean similarity':<30} {train_results['similarity_mean']:.4f}          {val_results['similarity_mean']:.4f}")
    print(f"{'Std similarity':<30} {train_results['similarity_std']:.4f}          {val_results['similarity_std']:.4f}")
    print(f"{'Median similarity':<30} {train_results['similarity_median']:.4f}          {val_results['similarity_median']:.4f}")
    print(f"{'Min similarity':<30} {train_results['similarity_min']:.4f}          {val_results['similarity_min']:.4f}")
    print(f"{'Max similarity':<30} {train_results['similarity_max']:.4f}          {val_results['similarity_max']:.4f}")

    print(f"\nDimensionality:")
    print(f"{'PCA dims for 90% var':<30} {train_results['pca_90']:<15} {val_results['pca_90']:<15}")
    print(f"{'PCA dims for 95% var':<30} {train_results['pca_95']:<15} {val_results['pca_95']:<15}")
    print(f"{'Dead neurons %':<30} {train_results['dead_neuron_pct']:.2f}%          {val_results['dead_neuron_pct']:.2f}%")

    # Plot
    if args.plot:
        os.makedirs(args.output_dir, exist_ok=True)

        # Extract off-diagonal similarities
        mask_train = ~np.eye(train_sim_matrix.shape[0], dtype=bool)
        mask_val = ~np.eye(val_sim_matrix.shape[0], dtype=bool)

        train_sims = train_sim_matrix[mask_train]
        val_sims = val_sim_matrix[mask_val]

        plot_similarity_distribution(train_sims, val_sims, args.output_dir)

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        results_df = pd.DataFrame([train_results, val_results])
        results_path = os.path.join(args.output_dir, 'diversity_analysis_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved results to: {results_path}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_encoder_path', default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--csv_file', default='/home/muhammedg/fvlm/data/image_first_dataset.csv')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', default='./diversity_analysis')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()
    main(args)

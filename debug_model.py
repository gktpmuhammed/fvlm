#!/usr/bin/env python3

"""
Debug Medical BLIP-2 with BioGPT
Checks:
1. Cross-attention usage in BioGPT
2. Q-Former effectiveness
3. Encoder output diversity across different patients
4. Attention flow from vision to text
"""

import torch
import numpy as np
import pandas as pd
from medical_blip2_biogpt import MedicalBLIP2BioGPT
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Transposed,
    ScaleIntensityRanged,
    SpatialPadd,
    CenterSpatialCropd,
)
import SimpleITK as sitk
from scipy.spatial.distance import cosine
import argparse


def build_transforms():
    """Transform pipeline for 3D medical images"""
    return Compose([
        LoadImaged(keys=['image'], reader='ITKReader', image_only=True),
        EnsureChannelFirstd(keys=['image']),
        Transposed(keys=['image'], indices=(0, 3, 2, 1)),
        ScaleIntensityRanged(
            keys=['image'],
            a_min=-1150,
            a_max=350,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        SpatialPadd(keys=['image'], spatial_size=(112, 256, 352), mode='constant', constant_values=0),
        CenterSpatialCropd(keys=['image'], roi_size=(112, 256, 352)),
    ])


def load_sample_images(csv_file, num_samples=5):
    """Load a few sample images for debugging"""
    df = pd.read_csv(csv_file)
    df = df[df['split'] == 'validation'].reset_index(drop=True)

    # Get diverse samples
    samples = df.sample(n=min(num_samples, len(df)), random_state=42)

    transform = build_transforms()
    images = []
    paths = []
    references = []

    for _, row in samples.iterrows():
        image_dict = transform({'image': row['image_path']})
        image = image_dict['image']

        if isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)
        image = torch.from_numpy(np.array(image)).float()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        images.append(image)
        paths.append(row['image_path'])
        references.append(f"{row['findings']} {row['impressions']}")

    return torch.stack(images), paths, references


def check_cross_attention_usage(model, sample_input):
    """
    Check if cross-attention is being used in BioGPT decoder
    """
    print("="*80)
    print("TEST 1: Cross-Attention Usage in BioGPT")
    print("="*80)

    model.eval()
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)

    # Hook to capture cross-attention weights
    cross_attn_weights = []

    def hook_fn(module, input, output):
        # BioGPT cross-attention outputs
        if hasattr(output, 'cross_attentions') and output.cross_attentions is not None:
            cross_attn_weights.append(output.cross_attentions)

    # Register hooks on decoder layers
    hooks = []
    for i, layer in enumerate(model.decoder.biogpt.layers):
        if hasattr(layer, 'encoder_attn'):
            hook = layer.encoder_attn.register_forward_hook(hook_fn)
            hooks.append(hook)

    # Forward pass with dummy text
    dummy_text = "The patient"
    encoding = model.tokenizer(dummy_text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(
            pixel_values=sample_input,
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analysis
    print(f"\nCross-Attention Analysis:")
    print(f"  Number of decoder layers: {len(model.decoder.biogpt.layers)}")

    # Check if cross-attention was called
    has_cross_attn = False
    for i, layer in enumerate(model.decoder.biogpt.layers):
        if hasattr(layer, 'encoder_attn'):
            has_cross_attn = True
            print(f"  Layer {i}: Has cross-attention module ")

    if has_cross_attn:
        print(f"\n Cross-attention modules ARE present in BioGPT")
    else:
        print(f"\n Cross-attention modules NOT found in BioGPT")

    # Check if cross_attentions were captured
    if len(cross_attn_weights) > 0:
        print(f"\n Cross-attention IS being used during forward pass")
        print(f"  Captured {len(cross_attn_weights)} cross-attention weight tensors")
    else:
        print(f"\n  Cross-attention weights not captured (may not be enabled)")

    print("\n" + "-"*80)
    print("VERDICT:")
    if has_cross_attn:
        print(" Cross-attention architecture is present")
        print(" Vision features can attend to text generation")
    else:
        print(" Cross-attention NOT properly configured")

    return has_cross_attn


def check_qformer_effectiveness(model, sample_inputs):
    """
    Check if Q-Former is compressing vision features effectively
    """
    print("\n" + "="*80)
    print("TEST 2: Q-Former Effectiveness")
    print("="*80)

    model.eval()
    device = next(model.parameters()).device
    sample_inputs = sample_inputs.to(device)

    if not model.use_qformer:
        print("\n  Q-Former is NOT being used (direct connection)")
        return

    with torch.no_grad():
        # Get vision features
        encoder_outputs = model.encoder(sample_inputs, return_dict=True)
        vision_features = encoder_outputs.last_hidden_state

        print(f"\nVision Features (before Q-Former):")
        print(f"  Shape: {vision_features.shape}")
        print(f"  Mean: {vision_features.mean().item():.4f}")
        print(f"  Std: {vision_features.std().item():.4f}")

        # Q-Former compression
        batch_size = sample_inputs.shape[0]
        query_tokens = model.query_tokens.expand(batch_size, -1, -1)
        compressed_features = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=vision_features,
        )

        print(f"\nCompressed Features (after Q-Former):")
        print(f"  Shape: {compressed_features.shape}")
        print(f"  Mean: {compressed_features.mean().item():.4f}")
        print(f"  Std: {compressed_features.std().item():.4f}")

        # Compute compression ratio
        input_tokens = vision_features.shape[1]
        output_tokens = compressed_features.shape[1]
        compression_ratio = input_tokens / output_tokens

        print(f"\nCompression Analysis:")
        print(f"  Input tokens: {input_tokens}")
        print(f"  Output tokens: {output_tokens}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")

        # Check if Q-Former is learning (not just identity)
        cosine_sim = torch.nn.functional.cosine_similarity(
            vision_features[:, :output_tokens, :].reshape(batch_size, -1),
            compressed_features.reshape(batch_size, -1),
            dim=1
        ).mean().item()

        print(f"  Cosine similarity with input: {cosine_sim:.4f}")

        print("\n" + "-"*80)
        print("VERDICT:")
        if compression_ratio > 1.5:
            print(f" Q-Former IS compressing: {compression_ratio:.1f}x reduction")
        else:
            print(f"  Q-Former compression minimal: {compression_ratio:.1f}x")

        if 0.3 < cosine_sim < 0.8:
            print(f" Q-Former IS transforming features (sim: {cosine_sim:.2f})")
        elif cosine_sim > 0.9:
            print(f"  Q-Former may be learning identity (sim: {cosine_sim:.2f})")
        else:
            print(f" Q-Former heavily transforms features (sim: {cosine_sim:.2f})")


def check_encoder_diversity(model, sample_inputs, paths):
    """
    Check if encoder generates diverse outputs for different patients
    """
    print("\n" + "="*80)
    print("TEST 3: Encoder Output Diversity")
    print("="*80)

    model.eval()
    device = next(model.parameters()).device
    sample_inputs = sample_inputs.to(device)

    with torch.no_grad():
        # Get encoder outputs for all samples
        encoder_outputs = model.encoder(sample_inputs, return_dict=True)
        vision_features = encoder_outputs.last_hidden_state  # [B, num_patches, hidden_dim]

        print(f"\nEncoder Outputs:")
        print(f"  Shape: {vision_features.shape}")
        print(f"  Batch size: {vision_features.shape[0]}")
        print(f"  Sequence length: {vision_features.shape[1]}")
        print(f"  Hidden dim: {vision_features.shape[2]}")

        # Compute pairwise similarities
        print(f"\nPairwise Cosine Similarities:")
        print("-"*80)

        # Flatten spatial dimensions for comparison
        flat_features = vision_features.mean(dim=1)  # [B, hidden_dim]

        similarities = []
        for i in range(len(flat_features)):
            for j in range(i+1, len(flat_features)):
                sim = torch.nn.functional.cosine_similarity(
                    flat_features[i].unsqueeze(0),
                    flat_features[j].unsqueeze(0)
                ).item()
                similarities.append(sim)

                print(f"  Sample {i} vs Sample {j}: {sim:.4f}")

        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)

        print(f"\nStatistics:")
        print(f"  Average similarity: {avg_similarity:.4f}")
        print(f"  Std deviation: {std_similarity:.4f}")
        print(f"  Min similarity: {min(similarities):.4f}")
        print(f"  Max similarity: {max(similarities):.4f}")

        # Per-sample statistics
        print(f"\nPer-Sample Statistics:")
        print("-"*80)
        for i in range(len(vision_features)):
            sample_mean = vision_features[i].mean().item()
            sample_std = vision_features[i].std().item()
            sample_norm = vision_features[i].norm().item()

            print(f"  Sample {i}:")
            print(f"    Mean: {sample_mean:.4f}")
            print(f"    Std:  {sample_std:.4f}")
            print(f"    Norm: {sample_norm:.2f}")

        print("\n" + "-"*80)
        print("VERDICT:")

        if avg_similarity > 0.95:
            print(f" PROBLEM: Encoder outputs are TOO similar ({avg_similarity:.4f})")
            print("   → Encoder may be collapsed or frozen improperly")
            print("   → All images producing nearly identical features")
        elif avg_similarity > 0.85:
            print(f"  WARNING: Encoder outputs somewhat similar ({avg_similarity:.4f})")
            print("   → May indicate limited discrimination")
            print("   → Check if training data is too homogeneous")
        elif avg_similarity < 0.5:
            print(f" EXCELLENT: High diversity ({avg_similarity:.4f})")
            print("   → Encoder discriminates well between patients")
        else:
            print(f" GOOD: Reasonable diversity ({avg_similarity:.4f})")
            print("   → Encoder produces patient-specific features")

        return avg_similarity, similarities


def check_generation_diversity(model, sample_inputs, references):
    """
    Check if different inputs produce different outputs
    """
    print("\n" + "="*80)
    print("TEST 4: Generation Diversity")
    print("="*80)

    model.eval()
    device = next(model.parameters()).device
    sample_inputs = sample_inputs.to(device)

    print(f"\nGenerating predictions for {len(sample_inputs)} samples...")
    predictions = []

    with torch.no_grad():
        for i in range(len(sample_inputs)):
            single_input = sample_inputs[i:i+1]

            # Generate
            generated_ids = model.generate(
                pixel_values=single_input,
                max_length=150,  # Shorter to avoid repetition for debugging
                num_beams=1,
            )

            prediction = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            predictions.append(prediction)

            print(f"\n  Sample {i}:")
            print(f"    First 100 chars: {prediction[:100]}...")

    # Compute prediction similarities
    print(f"\nPrediction Similarity Analysis:")
    print("-"*80)

    # Simple word overlap
    from collections import Counter

    similarities = []
    for i in range(len(predictions)):
        for j in range(i+1, len(predictions)):
            words_i = set(predictions[i].lower().split())
            words_j = set(predictions[j].lower().split())

            overlap = len(words_i & words_j)
            union = len(words_i | words_j)
            jaccard = overlap / union if union > 0 else 0

            similarities.append(jaccard)
            print(f"  Prediction {i} vs {j}: Jaccard similarity = {jaccard:.4f}")

    avg_pred_sim = np.mean(similarities)

    print(f"\nPrediction Statistics:")
    print(f"  Average Jaccard similarity: {avg_pred_sim:.4f}")
    print(f"  Std deviation: {np.std(similarities):.4f}")

    # Compare with references
    print(f"\nReference vs Prediction Similarity:")
    print("-"*80)
    ref_similarities = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        words_pred = set(pred.lower().split())
        words_ref = set(ref.lower().split())

        overlap = len(words_pred & words_ref)
        union = len(words_pred | words_ref)
        jaccard = overlap / union if union > 0 else 0

        ref_similarities.append(jaccard)
        print(f"  Sample {i}: {jaccard:.4f}")

    avg_ref_sim = np.mean(ref_similarities)
    print(f"\nAverage reference similarity: {avg_ref_sim:.4f}")

    print("\n" + "-"*80)
    print("VERDICT:")

    if avg_pred_sim > 0.8:
        print(f" PROBLEM: Predictions TOO similar ({avg_pred_sim:.4f})")
        print("   → Model may be generating templates regardless of input")
        print("   → Vision features not influencing generation enough")
    elif avg_pred_sim > 0.6:
        print(f"  WARNING: Predictions somewhat similar ({avg_pred_sim:.4f})")
        print("   → Some variation but may be too templated")
    else:
        print(f" GOOD: Predictions are diverse ({avg_pred_sim:.4f})")
        print("   → Model adapts to different inputs")

    if avg_ref_sim > 0.3:
        print(f" GOOD: Predictions match references ({avg_ref_sim:.4f})")
    else:
        print(f"  Predictions diverge from references ({avg_ref_sim:.4f})")

    return predictions


def main(args):
    print("="*80)
    print("MEDICAL BLIP-2 BIOGPT DEBUGGING SUITE")
    print("="*80)
    print("\nThis script will check:")
    print("1. Cross-attention usage in BioGPT")
    print("2. Q-Former effectiveness")
    print("3. Encoder output diversity")
    print("4. Generation diversity")
    print("="*80)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    image_size = tuple(map(int, args.image_size.split(',')))
    patch_size = tuple(map(int, args.patch_size.split(',')))

    model = MedicalBLIP2BioGPT(
        vision_encoder_path=args.vision_encoder_path,
        decoder_model_name="microsoft/biogpt",
        image_size=image_size,
        patch_size=patch_size,
        num_query_tokens=args.num_query_tokens,
        use_qformer=args.use_qformer,
    )

    # Load trained weights
    if os.path.exists(os.path.join(args.model_path, 'encoder.pt')):
        print("Loading trained weights...")
        model.encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder.pt')))
        model.projection.load_state_dict(torch.load(os.path.join(args.model_path, 'projection.pt')))
        if model.use_qformer:
            model.qformer.load_state_dict(torch.load(os.path.join(args.model_path, 'qformer.pt')))
            model.query_tokens = torch.load(os.path.join(args.model_path, 'query_tokens.pt'))

    model = model.to(device)
    model.eval()

    print(" Model loaded successfully")

    # Load sample images
    print(f"\nLoading {args.num_samples} sample images...")
    sample_images, paths, references = load_sample_images(args.csv_file, args.num_samples)
    print(f" Loaded {len(sample_images)} samples")

    # Run tests
    print("\n" + "="*80)
    print("STARTING DIAGNOSTIC TESTS")
    print("="*80)

    # Test 1: Cross-attention
    has_cross_attn = check_cross_attention_usage(model, sample_images[:1])

    # Test 2: Q-Former
    check_qformer_effectiveness(model, sample_images)

    # Test 3: Encoder diversity
    avg_sim, similarities = check_encoder_diversity(model, sample_images, paths)

    # Test 4: Generation diversity
    predictions = check_generation_diversity(model, sample_images, references)

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)

    print(f"\n Cross-Attention: {'PRESENT' if has_cross_attn else 'MISSING'}")
    print(f" Q-Former: {'ACTIVE' if model.use_qformer else 'DISABLED'}")
    print(f"{'' if avg_sim < 0.85 else ' '} Encoder Diversity: {avg_sim:.4f} avg similarity")
    print(f"\nAll tests completed!")
    print("="*80)


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="Debug Medical BLIP-2 with BioGPT")

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--vision_encoder_path', type=str,
                       default='/home/muhammedg/fvlm/checkpoints/model.pth',
                       help='Path to vision encoder')
    parser.add_argument('--csv_file', type=str, required=True,
                       help='Path to CSV file')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to test')
    parser.add_argument('--image_size', type=str, default='112,256,352',
                       help='Image size')
    parser.add_argument('--patch_size', type=str, default='16,16,32',
                       help='Patch size')
    parser.add_argument('--num_query_tokens', type=int, default=32,
                       help='Number of query tokens')
    parser.add_argument('--use_qformer', action='store_true', default=True,
                       help='Use Q-Former')

    args = parser.parse_args()
    main(args)

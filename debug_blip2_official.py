#!/usr/bin/env python3

"""
Debug Medical BLIP-2 Official Model
Comprehensive diagnostic tests:
1. Cross-attention in BERT Q-Former
2. Q-Former effectiveness
3. Encoder output diversity
4. Generation diversity
"""

import torch
import numpy as np
import pandas as pd
from medical_blip2_official import MedicalBLIP2Official
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
import argparse


def build_transforms():
    """Transform pipeline"""
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
    """Load sample images"""
    df = pd.read_csv(csv_file)
    df = df[df['split'] == 'validation'].reset_index(drop=True)

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


def check_bert_cross_attention(model, sample_input):
    """
    Test 1: Check if BERT Q-Former has cross-attention enabled
    """
    print("="*80)
    print("TEST 1: BERT Q-Former Cross-Attention")
    print("="*80)

    model.eval()
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)

    # Check Q-Former configuration
    config = model.Qformer.config

    print(f"\nQ-Former Configuration:")
    print(f"  Model type: {config.model_type}")
    print(f"  Is decoder: {config.is_decoder}")
    print(f"  Add cross-attention: {config.add_cross_attention}")
    print(f"  Number of layers: {config.num_hidden_layers}")

    # Check for cross-attention modules
    has_cross_attn = False
    cross_attn_layers = []

    for i, layer in enumerate(model.Qformer.encoder.layer):
        if hasattr(layer, 'crossattention'):
            has_cross_attn = True
            cross_attn_layers.append(i)

    print(f"\nCross-Attention Modules:")
    if has_cross_attn:
        print(f"   FOUND in {len(cross_attn_layers)} layers")
        print(f"  Layers with cross-attention: {cross_attn_layers}")
    else:
        print(f"   NOT FOUND")

    # Test forward pass with cross-attention
    print(f"\nTesting Forward Pass with Cross-Attention:")

    with torch.no_grad():
        # Get vision features
        image_output = model.visual_encoder(sample_input[:1])
        if isinstance(image_output, tuple):
            image_embeds = image_output[0]
        else:
            image_embeds = image_output

        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        # Q-Former forward
        query_tokens = model.query_tokens.expand(1, -1, -1)

        try:
            query_output = model.Qformer(
                inputs_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                output_attentions=True,  # Request attention weights
            )

            print(f"   Forward pass successful")
            print(f"  Output shape: {query_output.last_hidden_state.shape}")

            # Check if cross-attention was used
            if hasattr(query_output, 'cross_attentions') and query_output.cross_attentions is not None:
                print(f"   Cross-attentions returned: {len(query_output.cross_attentions)} layers")
                print(f"  Cross-attention shape: {query_output.cross_attentions[0].shape}")
            else:
                print(f"    Cross-attentions not in output (may still be working)")

        except Exception as e:
            print(f"   Forward pass failed: {e}")

    print("\n" + "-"*80)
    print("VERDICT:")
    if has_cross_attn and config.add_cross_attention:
        print(" Cross-attention IS properly configured")
        print(" Q-Former can attend to vision features")
        print(" This is CORRECT BLIP-2 architecture")
    else:
        print(" Cross-attention NOT properly configured")

    return has_cross_attn


def check_qformer_effectiveness(model, sample_inputs):
    """
    Test 2: Check Q-Former compression and transformation
    """
    print("\n" + "="*80)
    print("TEST 2: Q-Former Effectiveness")
    print("="*80)

    model.eval()
    device = next(model.parameters()).device
    sample_inputs = sample_inputs.to(device)

    with torch.no_grad():
        # Get vision features
        image_output = model.visual_encoder(sample_inputs)
        if isinstance(image_output, tuple):
            vision_features = image_output[0]
        else:
            vision_features = image_output

        vision_features = vision_features.float()

        print(f"\nVision Features (before Q-Former):")
        print(f"  Shape: {vision_features.shape}")
        print(f"  Mean: {vision_features.mean().item():.4f}")
        print(f"  Std: {vision_features.std().item():.4f}")

        # Q-Former compression
        batch_size = sample_inputs.shape[0]
        image_atts = torch.ones(vision_features.size()[:-1], dtype=torch.long).to(device)
        query_tokens = model.query_tokens.expand(batch_size, -1, -1)

        query_output = model.Qformer(
            inputs_embeds=query_tokens,
            encoder_hidden_states=vision_features,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        compressed_features = query_output.last_hidden_state

        print(f"\nCompressed Features (after Q-Former):")
        print(f"  Shape: {compressed_features.shape}")
        print(f"  Mean: {compressed_features.mean().item():.4f}")
        print(f"  Std: {compressed_features.std().item():.4f}")

        # Compression analysis
        input_tokens = vision_features.shape[1]
        output_tokens = compressed_features.shape[1]
        compression_ratio = input_tokens / output_tokens

        print(f"\nCompression Analysis:")
        print(f"  Input tokens: {input_tokens}")
        print(f"  Output tokens: {output_tokens}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")

        # Check transformation (not just identity)
        # Sample first few tokens for comparison
        sample_size = min(output_tokens, input_tokens)
        cosine_sim = torch.nn.functional.cosine_similarity(
            vision_features[:, :sample_size, :].reshape(batch_size, -1),
            compressed_features[:, :sample_size, :].reshape(batch_size, -1),
            dim=1
        ).mean().item()

        print(f"  Cosine similarity with input: {cosine_sim:.4f}")

        print("\n" + "-"*80)
        print("VERDICT:")
        if compression_ratio > 1.5:
            print(f" Q-Former IS compressing: {compression_ratio:.1f}x reduction")
        else:
            print(f"  Q-Former compression minimal: {compression_ratio:.1f}x")

        if 0.2 < cosine_sim < 0.8:
            print(f" Q-Former IS transforming features (sim: {cosine_sim:.2f})")
        elif cosine_sim > 0.9:
            print(f"  Q-Former may be learning identity (sim: {cosine_sim:.2f})")
        else:
            print(f" Q-Former heavily transforms features (sim: {cosine_sim:.2f})")

    return compression_ratio, cosine_sim


def check_encoder_diversity(model, sample_inputs, paths):
    """
    Test 3: Encoder output diversity
    """
    print("\n" + "="*80)
    print("TEST 3: Encoder Output Diversity")
    print("="*80)

    model.eval()
    device = next(model.parameters()).device
    sample_inputs = sample_inputs.to(device)

    with torch.no_grad():
        # Get encoder outputs
        image_output = model.visual_encoder(sample_inputs)
        if isinstance(image_output, tuple):
            vision_features = image_output[0]
        else:
            vision_features = image_output

        vision_features = vision_features.float()

        print(f"\nEncoder Outputs:")
        print(f"  Shape: {vision_features.shape}")
        print(f"  Batch size: {vision_features.shape[0]}")
        print(f"  Sequence length: {vision_features.shape[1]}")
        print(f"  Hidden dim: {vision_features.shape[2]}")

        # Pairwise similarities
        print(f"\nPairwise Cosine Similarities:")
        print("-"*80)

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

        if len(similarities) > 0:
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

        if len(similarities) > 0:
            if avg_similarity > 0.95:
                print(f" PROBLEM: Encoder outputs TOO similar ({avg_similarity:.4f})")
                print("   → Encoder may be collapsed")
            elif avg_similarity > 0.85:
                print(f"  WARNING: Encoder outputs somewhat similar ({avg_similarity:.4f})")
                print("   → Limited discrimination")
            elif avg_similarity < 0.5:
                print(f" EXCELLENT: High diversity ({avg_similarity:.4f})")
                print("   → Strong patient-specific features")
            else:
                print(f" GOOD: Reasonable diversity ({avg_similarity:.4f})")
                print("   → Patient-specific features present")

            return avg_similarity
        else:
            print("  Not enough samples to compute diversity")
            return None


def check_generation_diversity(model, sample_inputs, references):
    """
    Test 4: Generation diversity
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
            generated = model.generate(
                image=single_input,
                max_length=150,
                num_beams=3,
                repetition_penalty=1.5,
            )

            prediction = generated[0]
            predictions.append(prediction)

            print(f"\n  Sample {i}:")
            print(f"    Generated ({len(prediction)} chars): {prediction[:100]}...")

    # Prediction similarity
    print(f"\nPrediction Similarity Analysis:")
    print("-"*80)

    from collections import Counter

    similarities = []
    for i in range(len(predictions)):
        for j in range(i+1, len(predictions)):
            words_i = set(predictions[i].lower().split())
            words_j = set(predictions[j].lower().split())

            if len(words_i | words_j) > 0:
                overlap = len(words_i & words_j)
                union = len(words_i | words_j)
                jaccard = overlap / union
                similarities.append(jaccard)
                print(f"  Prediction {i} vs {j}: Jaccard = {jaccard:.4f}")

    if len(similarities) > 0:
        avg_pred_sim = np.mean(similarities)

        print(f"\nPrediction Statistics:")
        print(f"  Average Jaccard similarity: {avg_pred_sim:.4f}")
        print(f"  Std deviation: {np.std(similarities):.4f}")

    # Reference similarity
    print(f"\nReference vs Prediction Similarity:")
    print("-"*80)
    ref_similarities = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        words_pred = set(pred.lower().split())
        words_ref = set(ref.lower().split())

        if len(words_pred | words_ref) > 0:
            overlap = len(words_pred & words_ref)
            union = len(words_pred | words_ref)
            jaccard = overlap / union
            ref_similarities.append(jaccard)
            print(f"  Sample {i}: {jaccard:.4f}")

    if len(ref_similarities) > 0:
        avg_ref_sim = np.mean(ref_similarities)
        print(f"\nAverage reference similarity: {avg_ref_sim:.4f}")

    print("\n" + "-"*80)
    print("VERDICT:")

    if len(similarities) > 0:
        if avg_pred_sim > 0.8:
            print(f" PROBLEM: Predictions TOO similar ({avg_pred_sim:.4f})")
            print("   → Model generating templates")
        elif avg_pred_sim > 0.6:
            print(f"  WARNING: Predictions somewhat similar ({avg_pred_sim:.4f})")
        else:
            print(f" GOOD: Predictions are diverse ({avg_pred_sim:.4f})")

    if len(ref_similarities) > 0:
        if avg_ref_sim > 0.3:
            print(f" GOOD: Predictions match references ({avg_ref_sim:.4f})")
        else:
            print(f"  Predictions diverge from references ({avg_ref_sim:.4f})")

    return predictions


def main(args):
    print("="*80)
    print("MEDICAL BLIP-2 OFFICIAL - DEBUGGING SUITE")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model
    print(f"\nLoading model from: {args.model_path}")

    model = MedicalBLIP2Official.from_pretrained(
        args.model_path,
        vision_encoder_path=args.vision_encoder_path,
    )

    model = model.to(device)
    model.eval()
    print(" Model loaded")

    # Load samples
    print(f"\nLoading {args.num_samples} sample images...")
    sample_images, paths, references = load_sample_images(args.csv_file, args.num_samples)
    print(f" Loaded {len(sample_images)} samples")

    # Run tests
    print("\n" + "="*80)
    print("STARTING DIAGNOSTIC TESTS")
    print("="*80)

    # Test 1: Cross-attention
    has_cross_attn = check_bert_cross_attention(model, sample_images[:1])

    # Test 2: Q-Former
    compression_ratio, cosine_sim = check_qformer_effectiveness(model, sample_images)

    # Test 3: Encoder diversity
    encoder_sim = check_encoder_diversity(model, sample_images, paths)

    # Test 4: Generation diversity
    predictions = check_generation_diversity(model, sample_images, references)

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)

    print(f"\n{'' if has_cross_attn else ''} Cross-Attention: {'ENABLED' if has_cross_attn else 'MISSING'}")
    print(f" Q-Former: {compression_ratio:.1f}x compression, {cosine_sim:.2f} similarity")
    if encoder_sim:
        status = "" if encoder_sim < 0.85 else " "
        print(f"{status} Encoder Diversity: {encoder_sim:.4f} avg similarity")

    print("\nAll tests completed!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Medical BLIP-2 Official")

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--vision_encoder_path', type=str,
                       default='/home/muhammedg/fvlm/checkpoints/model.pth',
                       help='Path to vision encoder')
    parser.add_argument('--csv_file', type=str, required=True,
                       help='Path to CSV file')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to test')

    args = parser.parse_args()
    main(args)

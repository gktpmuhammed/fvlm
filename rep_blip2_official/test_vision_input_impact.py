#!/usr/bin/env python3
"""
Test Vision Input Impact on BLIP-2 Generation
Amplifies encoder embeddings 10-20x to verify vision affects output
"""

import torch
import torch.nn as nn
import numpy as np
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
import pandas as pd
import argparse
import os


def build_transforms():
    return Compose([
        LoadImaged(keys=['image'], reader='ITKReader', image_only=True),
        EnsureChannelFirstd(keys=['image']),
        Transposed(keys=['image'], indices=(0, 3, 2, 1)),
        ScaleIntensityRanged(keys=['image'], a_min=-1150, a_max=350, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=['image'], spatial_size=(112, 256, 352), mode='constant'),
        CenterSpatialCropd(keys=['image'], roi_size=(112, 256, 352)),
    ])


@torch.no_grad()
def generate_with_amplified_embeddings(
    model,
    image,
    prompt,
    amplification_factor=1.0,
    max_length=256,
    num_beams=1,
    repetition_penalty=1.5,
):
    """
    Modified generation that amplifies encoder embeddings

    Args:
        model: BLIP-2 model
        image: Input image tensor
        prompt: Text prompt
        amplification_factor: How much to amplify encoder embeddings (1.0 = normal, 10.0 = 10x)
        max_length: Max generation length
        num_beams: Number of beams
        repetition_penalty: Repetition penalty

    Returns:
        generated_text: Generated text
        vision_stats: Statistics about vision embeddings
    """
    batch_size = image.shape[0]
    device = image.device

    # 1. Get vision features (SAME AS ORIGINAL)
    image_output = model.visual_encoder(image)
    if isinstance(image_output, tuple):
        image_embeds = image_output[0]
    else:
        image_embeds = image_output
    image_embeds = image_embeds.float()

    # Calculate statistics BEFORE amplification
    original_mean = image_embeds.mean().item()
    original_std = image_embeds.std().item()
    original_norm = image_embeds.norm().item()

    # AMPLIFY EMBEDDINGS HERE
    if amplification_factor != 1.0:
        print(f"  Amplifying vision embeddings by {amplification_factor}x...")
        image_embeds = image_embeds * amplification_factor

    # Calculate statistics AFTER amplification
    amplified_mean = image_embeds.mean().item()
    amplified_std = image_embeds.std().item()
    amplified_norm = image_embeds.norm().item()

    vision_stats = {
        'original_mean': original_mean,
        'original_std': original_std,
        'original_norm': original_norm,
        'amplified_mean': amplified_mean,
        'amplified_std': amplified_std,
        'amplified_norm': amplified_norm,
        'amplification_factor': amplification_factor,
    }

    # 2. Q-Former (processes amplified vision features)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
    query_tokens = model.query_tokens.expand(batch_size, -1, -1)

    query_output = model.Qformer(
        inputs_embeds=query_tokens,
        encoder_hidden_states=image_embeds,  # Amplified embeddings used here
        encoder_attention_mask=image_atts,
        return_dict=True,
    )

    # 3. Project to OPT space
    inputs_opt = model.opt_proj(query_output.last_hidden_state)
    atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(device)

    # 4. Prepare prompt
    if isinstance(prompt, str):
        prompt = [prompt] * batch_size

    prompt_tokens = model.opt_tokenizer(
        prompt,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    ).to(device)

    prompt_embeds = model.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
    prompt_atts = prompt_tokens.attention_mask

    # 5. Concatenate
    prefix_embeds = torch.cat([inputs_opt, prompt_embeds], dim=1)
    prefix_atts = torch.cat([atts_opt, prompt_atts], dim=1)

    # 6. Generate
    generated_ids = model._simple_greedy_generate(
        prefix_embeds=prefix_embeds,
        prefix_atts=prefix_atts,
        max_length=max_length,
        min_length=10,
        eos_token_id=model.opt_tokenizer.eos_token_id,
        pad_token_id=model.opt_tokenizer.pad_token_id,
        repetition_penalty=repetition_penalty,
    )

    # 7. Decode
    vocab_size = model.opt_tokenizer.vocab_size
    generated_ids = torch.clamp(generated_ids, max=vocab_size - 1)
    generated_text = model.opt_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_text[0], vision_stats


def test_amplification_effect(model, image_path, prompt, device, amplification_factors=[1.0, 5.0, 10.0, 20.0]):
    """
    Test generation with different amplification factors

    Args:
        model: BLIP-2 model
        image_path: Path to CT scan
        prompt: Generation prompt
        device: Device to use
        amplification_factors: List of amplification factors to test

    Returns:
        results: Dict with results for each factor
    """
    print("\n" + "="*80)
    print("VISION INPUT IMPACT TEST")
    print("="*80)

    # Load image
    print(f"\nLoading image: {image_path}")
    transform = build_transforms()
    image_dict = transform({'image': image_path})
    image = image_dict['image']

    if isinstance(image, sitk.Image):
        image = sitk.GetArrayFromImage(image)
    image = torch.from_numpy(np.array(image)).float()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.unsqueeze(0).to(device)  # Add batch dimension

    print(f"Image shape: {image.shape}")
    print(f"Prompt: '{prompt}'")

    # Test with different amplification factors
    results = {}

    for factor in amplification_factors:
        print(f"\n{'='*80}")
        print(f"Amplification Factor: {factor}x")
        print("="*80)

        generated_text, vision_stats = generate_with_amplified_embeddings(
            model=model,
            image=image,
            prompt=prompt,
            amplification_factor=factor,
            max_length=256,
            repetition_penalty=1.5,
        )

        results[factor] = {
            'text': generated_text,
            'stats': vision_stats,
        }

        print(f"\nVision Embedding Statistics:")
        print(f"  Original - Mean: {vision_stats['original_mean']:.4f}, Std: {vision_stats['original_std']:.4f}, Norm: {vision_stats['original_norm']:.2f}")
        print(f"  Amplified - Mean: {vision_stats['amplified_mean']:.4f}, Std: {vision_stats['amplified_std']:.4f}, Norm: {vision_stats['amplified_norm']:.2f}")

        print(f"\nGenerated Text:")
        print(f"  {generated_text}")

    return results


def compare_results(results):
    """Compare and analyze results"""
    print("\n\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)

    factors = sorted(results.keys())

    # Check if texts are different
    texts = [results[f]['text'] for f in factors]
    unique_texts = len(set(texts))

    print(f"\nSummary:")
    print(f"  Tested amplification factors: {factors}")
    print(f"  Unique generated texts: {unique_texts} / {len(factors)}")

    if unique_texts == 1:
        print("\n  WARNING: All outputs are IDENTICAL!")
        print("  This suggests vision input has NO effect on generation.")
        print("  The model may be ignoring visual features entirely.")
    elif unique_texts == len(factors):
        print("\n  GOOD: All outputs are DIFFERENT!")
        print("  Vision input clearly affects generation.")
        print("  Amplifying embeddings changes the output.")
    else:
        print(f"\n  PARTIAL: {unique_texts} different outputs from {len(factors)} tests")
        print("  Some amplification factors produce the same output.")

    # Show text comparison
    print(f"\nText Comparison:")
    print("-"*80)
    for factor in factors:
        text = results[factor]['text']
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"  {factor}x: {preview}")

    # Show embedding norm changes
    print(f"\nEmbedding Norm Changes:")
    print("-"*80)
    baseline_norm = results[factors[0]]['stats']['original_norm']
    for factor in factors:
        amplified_norm = results[factor]['stats']['amplified_norm']
        norm_ratio = amplified_norm / baseline_norm
        print(f"  {factor}x amplification → Norm increased by {norm_ratio:.1f}x")

    return unique_texts, texts


def main(args):
    print("="*80)
    print("BLIP-2 Vision Input Impact Test")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = MedicalBLIP2Official.from_pretrained(
        args.model_path,
        vision_encoder_path=args.vision_encoder_path,
    )
    model = model.to(device)
    model.eval()
    print("  Model loaded")

    # Get test image
    if args.test_image:
        image_path = args.test_image
    else:
        # Load from CSV
        df = pd.read_csv(args.csv_file)
        test_df = df[df['split'] == 'validation'].head(1)
        image_path = test_df.iloc[0]['image_path']

    print(f"\nUsing test image: {image_path}")

    # Run test
    results = test_amplification_effect(
        model=model,
        image_path=image_path,
        prompt=args.prompt,
        device=device,
        amplification_factors=args.amplification_factors,
    )

    # Compare results
    unique_count, texts = compare_results(results)

    # Save results
    if args.output_file:
        print(f"\n\nSaving results to: {args.output_file}")
        with open(args.output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BLIP-2 Vision Input Impact Test Results\n")
            f.write("="*80 + "\n\n")

            f.write(f"Model: {args.model_path}\n")
            f.write(f"Test Image: {image_path}\n")
            f.write(f"Prompt: {args.prompt}\n\n")

            f.write("Results:\n")
            f.write("-"*80 + "\n\n")

            for factor in sorted(results.keys()):
                f.write(f"Amplification: {factor}x\n")
                f.write(f"Generated: {results[factor]['text']}\n")
                f.write(f"Vision Stats: {results[factor]['stats']}\n")
                f.write("\n")

            f.write("="*80 + "\n")
            f.write(f"Summary: {unique_count} unique texts from {len(results)} tests\n")
            f.write("="*80 + "\n")

    # Final verdict
    print("\n\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    if unique_count == 1:
        print("\nVISION INPUT HAS NO EFFECT")
        print("   All outputs are identical regardless of amplification.")
        print("   The model is likely ignoring visual features.")
        print("   Possible causes:")
        print("   - Q-Former not learning")
        print("   - Vision embeddings not propagating")
        print("   - OPT decoder ignoring vision prefix")
    elif unique_count == len(results):
        print("\nVISION INPUT AFFECTS GENERATION")
        print("   Amplifying embeddings changes output significantly.")
        print("   The model IS using visual features.")
    else:
        print("\nPARTIAL VISION INPUT EFFECT")
        print("   Some amplification factors change output, others don't.")
        print("   Model may be using vision features weakly.")

    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained BLIP-2 model')
    parser.add_argument('--vision_encoder_path', type=str,
                       default='/home/muhammedg/fvlm/checkpoints/model.pth',
                       help='Path to vision encoder checkpoint')
    parser.add_argument('--csv_file', type=str,
                       default='/home/muhammedg/fvlm/data/image_first_dataset.csv',
                       help='CSV file with image paths')
    parser.add_argument('--test_image', type=str, default=None,
                       help='Specific test image (if not provided, uses first validation image)')
    parser.add_argument('--prompt', type=str,
                       default="A medical report describing this CT scan: ",
                       help='Generation prompt')
    parser.add_argument('--amplification_factors', type=float, nargs='+',
                       default=[1.0, 5.0, 10.0, 20.0],
                       help='List of amplification factors to test')
    parser.add_argument('--output_file', type=str,
                       default='vision_impact_test_results.txt',
                       help='Output file for results')

    args = parser.parse_args()
    main(args)

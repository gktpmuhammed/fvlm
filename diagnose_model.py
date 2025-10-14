#!/usr/bin/env python3
"""
DIAGNOSTIC SCRIPT: Prove Vision-Language Learning
Tests if encoder outputs differ for different images and influence generation
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from train_vision_gpt2 import MedicalReportDataset, build_transforms
from medical_vision_gpt2 import MedicalVisionGPT2


def compute_encoder_similarity(encoder_outputs_list):
    """Compute cosine similarity between encoder outputs"""
    n = len(encoder_outputs_list)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Average pool encoder outputs to get single vector
            vec_i = encoder_outputs_list[i].mean(dim=1).squeeze()  # [hidden_dim]
            vec_j = encoder_outputs_list[j].mean(dim=1).squeeze()

            # Cosine similarity
            sim = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0))
            similarity_matrix[i, j] = sim.item()

    return similarity_matrix


def compute_cross_attention_stats(model, encoder_outputs, decoder_input_ids):
    """Extract cross-attention weights to see if decoder attends to encoder"""
    with torch.no_grad():
        # Get decoder outputs with cross-attention
        decoder_outputs = model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            output_attentions=True,
            return_dict=True
        )

        # Cross-attention weights from last layer
        # Shape: [batch, num_heads, seq_len, encoder_seq_len]
        cross_attentions = decoder_outputs.cross_attentions[-1]

        # Average over heads and batch
        attn_weights = cross_attentions.mean(dim=(0, 1))  # [seq_len, encoder_seq_len]

        return attn_weights.cpu().numpy()


def analyze_generation_with_swapped_encodings(model, tokenizer, samples):
    """
    CRITICAL TEST: Swap encoder outputs between images
    If model learned properly, different encoder outputs → different generations
    """
    results = []

    print("\n" + "="*80)
    print("ENCODER SWAP TEST:")
    print("="*80)
    print("Testing if different encoder outputs produce different generations...")
    print("If model learned: Swapping encodings should change output significantly")
    print("If model didn't learn: Output stays similar (decoder ignores encoder)")
    print()

    for i in range(0, len(samples)-1, 2):
        sample_a = samples[i]
        sample_b = samples[i+1]

        # Get encoder outputs for both images
        with torch.no_grad():
            enc_a = model.encoder(sample_a['image'].unsqueeze(0).cuda(), return_dict=True)
            enc_b = model.encoder(sample_b['image'].unsqueeze(0).cuda(), return_dict=True)

            # Generate with correct encoder
            gen_a_correct = model.decoder.generate(
                encoder_hidden_states=enc_a.last_hidden_state,
                max_length=150,
                do_sample=False,  # Deterministic
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Generate with SWAPPED encoder (image A with encoder from image B)
            gen_a_swapped = model.decoder.generate(
                encoder_hidden_states=enc_b.last_hidden_state,  # Wrong encoder!
                max_length=150,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            text_correct = tokenizer.decode(gen_a_correct[0], skip_special_tokens=True)
            text_swapped = tokenizer.decode(gen_a_swapped[0], skip_special_tokens=True)

            # Measure difference
            words_correct = set(text_correct.lower().split())
            words_swapped = set(text_swapped.lower().split())

            overlap = len(words_correct & words_swapped) / len(words_correct | words_swapped)
            difference = 1 - overlap

            results.append({
                'pair': i//2 + 1,
                'text_with_correct_encoder': text_correct,
                'text_with_swapped_encoder': text_swapped,
                'word_difference': difference,
            })

    avg_difference = np.mean([r['word_difference'] for r in results])

    print(f"\nAverage word-level difference when swapping encoders: {avg_difference:.2%}")
    print()

    if avg_difference > 0.4:
        print(" HIGH DIFFERENCE (>40%): Encoder strongly influences generation!")
        print("   → Model LEARNED vision-language alignment")
    elif avg_difference > 0.2:
        print("  MODERATE DIFFERENCE (20-40%): Encoder has some influence")
        print("   → Model learned partial alignment")
    else:
        print(" LOW DIFFERENCE (<20%): Encoder barely influences generation")
        print("   → Model did NOT learn properly (decoder ignores encoder)")

    print("="*80)

    return results, avg_difference


def diagnostic_analysis(args):
    print("="*80)
    print("VISION-LANGUAGE MODEL DIAGNOSTIC ANALYSIS")
    print("="*80)

    # Load model
    print("\nLoading model...")
    vision_encoder_path = '/home/muhammedg/fvlm/checkpoints/model.pth'

    full_model = MedicalVisionGPT2(
        vision_encoder_path=vision_encoder_path,
        decoder_model_name="gpt2",
        freeze_encoder=False,
        freeze_decoder_base=False
    )

    print(f"Loading weights from {args.model_path}...")
    from transformers import VisionEncoderDecoderModel
    trained_model = VisionEncoderDecoderModel.from_pretrained(args.model_path)

    full_model.model.decoder = trained_model.decoder
    full_model.model.eval()
    full_model.model.cuda()

    # Load data
    transform = build_transforms()
    val_dataset = MedicalReportDataset(
        csv_file=args.csv_file,
        tokenizer=full_model.tokenizer,
        transform=transform,
        max_length=512,
        split='validation',
        subset_size=20  # Use 20 diverse samples
    )

    print(f"Loaded {len(val_dataset)} validation samples")

    # Collect samples
    print("\nExtracting encoder outputs...")
    samples = []
    encoder_outputs_list = []

    with torch.no_grad():
        for idx in tqdm(range(min(20, len(val_dataset)))):
            data = val_dataset[idx]
            image = data['pixel_values'].cuda()

            # Get encoder output
            encoder_out = full_model.model.encoder(image.unsqueeze(0), return_dict=True)

            samples.append({
                'image': image,
                'encoder_output': encoder_out.last_hidden_state,
                'labels': data['labels']
            })
            encoder_outputs_list.append(encoder_out.last_hidden_state)

    # TEST 1: Encoder Output Diversity
    print("\n" + "="*80)
    print("TEST 1: ENCODER OUTPUT DIVERSITY")
    print("="*80)
    print("Measuring if different images produce different encoder representations...")

    similarity_matrix = compute_encoder_similarity(encoder_outputs_list)

    # Remove diagonal (self-similarity = 1.0)
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    off_diagonal_sims = similarity_matrix[mask]

    avg_similarity = np.mean(off_diagonal_sims)
    std_similarity = np.std(off_diagonal_sims)
    min_similarity = np.min(off_diagonal_sims)
    max_similarity = np.max(off_diagonal_sims)

    print(f"\nEncoder Output Similarity Statistics:")
    print(f"  Average similarity:  {avg_similarity:.4f}")
    print(f"  Std deviation:       {std_similarity:.4f}")
    print(f"  Min similarity:      {min_similarity:.4f}")
    print(f"  Max similarity:      {max_similarity:.4f}")
    print()

    if avg_similarity < 0.7:
        print(" LOW SIMILARITY (<0.7): Encoder produces DIVERSE outputs!")
        print("   → Different images → Different representations")
    elif avg_similarity < 0.85:
        print("  MODERATE SIMILARITY (0.7-0.85): Some differentiation")
        print("   → Encoder partially discriminates between images")
    else:
        print(" HIGH SIMILARITY (>0.85): Encoder outputs are TOO similar!")
        print("   → Encoder may not be discriminating between images")

    print("="*80)

    # Save similarity heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=False, cmap='RdYlGn_r', 
                vmin=0.5, vmax=1.0, square=True)
    plt.title('Encoder Output Similarity Matrix\n(Lower = More Diverse)')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    plt.savefig('encoder_similarity_heatmap.png', dpi=150)
    print("\nSaved: encoder_similarity_heatmap.png")

    # TEST 2: Cross-Attention Analysis
    print("\n" + "="*80)
    print("TEST 2: CROSS-ATTENTION ANALYSIS")
    print("="*80)
    print("Checking if decoder actually attends to encoder outputs...")

    # Get cross-attention for first few samples
    attn_entropies = []

    with torch.no_grad():
        for i in range(min(5, len(samples))):
            sample = samples[i]

            # Create simple decoder input (just BOS token)
            decoder_input = torch.tensor([[full_model.tokenizer.bos_token_id]]).cuda()

            decoder_outputs = full_model.model.decoder(
                input_ids=decoder_input,
                encoder_hidden_states=sample['encoder_output'],
                output_attentions=True,
                return_dict=True
            )

            # Get cross-attention from last layer
            cross_attn = decoder_outputs.cross_attentions[-1]  # [1, num_heads, 1, seq_len]

            # Average over heads
            attn_weights = cross_attn.mean(dim=1).squeeze()  # [seq_len]

            # Compute entropy (higher = more uniform attention)
            attn_probs = F.softmax(attn_weights, dim=0)
            entropy = -(attn_probs * torch.log(attn_probs + 1e-9)).sum().item()
            attn_entropies.append(entropy)

    avg_entropy = np.mean(attn_entropies)
    max_entropy = np.log(sample['encoder_output'].shape[1])  # Maximum possible entropy

    print(f"\nCross-Attention Entropy:")
    print(f"  Average:       {avg_entropy:.4f}")
    print(f"  Max possible:  {max_entropy:.4f}")
    print(f"  Normalized:    {avg_entropy/max_entropy:.2%}")
    print()

    if avg_entropy / max_entropy < 0.3:
        print(" LOW ENTROPY (<30%): Decoder focuses on SPECIFIC encoder positions!")
        print("   → Model learned selective attention to visual features")
    elif avg_entropy / max_entropy < 0.6:
        print("  MODERATE ENTROPY (30-60%): Some selective attention")
        print("   → Model uses encoder outputs but not very selectively")
    else:
        print(" HIGH ENTROPY (>60%): Attention is too uniform!")
        print("   → Decoder may be ignoring encoder (treating all positions equally)")

    print("="*80)

    # TEST 3: Encoder Swap Test (MOST IMPORTANT!)
    swap_results, swap_difference = analyze_generation_with_swapped_encodings(
        full_model.model, full_model.tokenizer, samples
    )

    # Save detailed results
    df_swap = pd.DataFrame(swap_results)
    df_swap.to_csv('encoder_swap_test_results.csv', index=False)
    print("\nSaved: encoder_swap_test_results.csv")

    # TEST 4: Generate multiple samples with same image
    print("\n" + "="*80)
    print("TEST 4: SAMPLING CONSISTENCY CHECK")
    print("="*80)
    print("Generate 5 times for SAME image with sampling...")

    sample = samples[0]
    generations = []

    with torch.no_grad():
        for i in range(5):
            gen = full_model.model.decoder.generate(
                encoder_hidden_states=sample['encoder_output'],
                max_length=150,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=full_model.tokenizer.pad_token_id,
                eos_token_id=full_model.tokenizer.eos_token_id,
            )
            text = full_model.tokenizer.decode(gen[0], skip_special_tokens=True)
            generations.append(text)
            print(f"\nGen {i+1}: {text[:150]}...")

    # Measure diversity
    unique_words_per_gen = [set(g.lower().split()) for g in generations]
    all_words = set.union(*unique_words_per_gen)
    overlap_ratios = []

    for i in range(len(generations)-1):
        overlap = len(unique_words_per_gen[i] & unique_words_per_gen[i+1])
        union = len(unique_words_per_gen[i] | unique_words_per_gen[i+1])
        overlap_ratios.append(overlap / union)

    avg_overlap = np.mean(overlap_ratios)
    print(f"\nAverage word overlap between consecutive generations: {avg_overlap:.2%}")

    if avg_overlap < 0.5:
        print(" LOW OVERLAP (<50%): High diversity from sampling")
    else:
        print("  HIGH OVERLAP (>50%): Limited diversity even with sampling")

    print("="*80)

    # FINAL VERDICT
    print("\n" + "="*80)
    print("FINAL DIAGNOSTIC VERDICT")
    print("="*80)

    scores = {
        'encoder_diversity': 1 if avg_similarity < 0.7 else (0.5 if avg_similarity < 0.85 else 0),
        'cross_attention': 1 if avg_entropy/max_entropy < 0.3 else (0.5 if avg_entropy/max_entropy < 0.6 else 0),
        'encoder_influence': 1 if swap_difference > 0.4 else (0.5 if swap_difference > 0.2 else 0),
    }

    total_score = sum(scores.values()) / len(scores)

    print(f"\nDiagnostic Scores:")
    print(f"  Encoder Diversity:    {scores['encoder_diversity']:.1f} / 1.0")
    print(f"  Cross-Attention:      {scores['cross_attention']:.1f} / 1.0")
    print(f"  Encoder Influence:    {scores['encoder_influence']:.1f} / 1.0")
    print(f"  ")
    print(f"  TOTAL SCORE:          {total_score:.2f} / 1.0")
    print()

    if total_score >= 0.8:
        print(" EXCELLENT: Model learned strong vision-language alignment!")
        print("   → Encoder produces diverse representations")
        print("   → Decoder attends to encoder selectively")
        print("   → Different images → Different outputs")
    elif total_score >= 0.5:
        print("  MODERATE: Model learned partial alignment")
        print("   → Some vision-language connection exists")
        print("   → May need more training or better architecture")
    else:
        print(" POOR: Model did NOT learn vision-language alignment")
        print("   → Encoder/decoder not properly connected")
        print("   → Need different training approach (BLIP-2, contrastive loss, etc.)")

    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                       default='./checkpoints/gpt2_full_unfrozen/checkpoint-4900')
    parser.add_argument('--csv_file', type=str,
                       default='/home/muhammedg/fvlm/image_first_dataset.csv')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    diagnostic_analysis(args)

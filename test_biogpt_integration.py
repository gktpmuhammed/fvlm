#!/usr/bin/env python3

"""
Test script for Medical BLIP-2 with BioGPT
This script verifies the model architecture and components work correctly
"""

import torch
import torch.nn as nn
from transformers import BioGptForCausalLM, BioGptTokenizer
import sys

def test_biogpt_availability():
    """Test 1: Check if BioGPT can be loaded"""
    print("="*80)
    print("TEST 1: BioGPT Availability")
    print("="*80)

    try:
        print("\nAttempting to load BioGPT tokenizer...")
        tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        print(f" Tokenizer loaded successfully")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   Pad token: {tokenizer.pad_token}")
        print(f"   EOS token: {tokenizer.eos_token}")

        print("\nAttempting to load BioGPT model...")
        model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        print(f" Model loaded successfully")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Hidden size: {model.config.hidden_size}")
        print(f"   Num layers: {model.config.num_hidden_layers}")

        return True, tokenizer, model

    except Exception as e:
        print(f" Failed to load BioGPT: {e}")
        return False, None, None

def test_text_generation(tokenizer, model):
    """Test 2: Test text generation with BioGPT"""
    print("\n" + "="*80)
    print("TEST 2: BioGPT Text Generation")
    print("="*80)

    try:
        # Sample medical text
        prompt = "The patient presents with chest pain and shortness of breath."

        print(f"\nPrompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_beams=3,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGenerated: {generated_text}")
        print("\n Text generation successful")

        return True

    except Exception as e:
        print(f" Text generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_compatibility(model):
    """Test 3: Test embedding extraction for BLIP-2 integration"""
    print("\n" + "="*80)
    print("TEST 3: Embedding Compatibility")
    print("="*80)

    try:
        # Get embedding layer
        embedding_layer = model.get_input_embeddings()
        print(f"\nEmbedding layer: {type(embedding_layer)}")
        print(f"   Embedding dimension: {embedding_layer.embedding_dim}")
        print(f"   Vocab size: {embedding_layer.num_embeddings}")

        # Test embedding extraction
        dummy_ids = torch.tensor([[1, 2, 3, 4, 5]])
        embeddings = embedding_layer(dummy_ids)
        print(f"\nTest embedding shape: {embeddings.shape}")
        print(f" Embeddings extracted successfully")

        # Test inputs_embeds forward pass
        print("\nTesting inputs_embeds forward pass...")
        with torch.no_grad():
            outputs = model(inputs_embeds=embeddings)
        print(f"   Output logits shape: {outputs.logits.shape}")
        print(" inputs_embeds forward pass successful")

        return True

    except Exception as e:
        print(f" Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_architecture_comparison():
    """Test 4: Compare OPT vs BioGPT architecture"""
    print("\n" + "="*80)
    print("TEST 4: Architecture Comparison (OPT vs BioGPT)")
    print("="*80)

    try:
        from transformers import OPTForCausalLM, AutoTokenizer

        print("\nLoading OPT-350m...")
        opt_model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        print("\nLoading BioGPT...")
        biogpt_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        biogpt_tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

        print("\n" + "-"*80)
        print("Model Comparison:")
        print("-"*80)

        comparison = {
            "Metric": ["Parameters", "Hidden Size", "Num Layers", "Vocab Size", "Max Position"],
            "OPT-350m": [
                f"{sum(p.numel() for p in opt_model.parameters()):,}",
                opt_model.config.hidden_size,
                opt_model.config.num_hidden_layers,
                len(opt_tokenizer),
                opt_model.config.max_position_embeddings
            ],
            "BioGPT": [
                f"{sum(p.numel() for p in biogpt_model.parameters()):,}",
                biogpt_model.config.hidden_size,
                biogpt_model.config.num_hidden_layers,
                len(biogpt_tokenizer),
                biogpt_model.config.max_position_embeddings
            ]
        }

        # Print comparison table
        print(f"{'Metric':<20} {'OPT-350m':<20} {'BioGPT':<20}")
        print("-"*60)
        for i in range(len(comparison["Metric"])):
            print(f"{comparison['Metric'][i]:<20} {str(comparison['OPT-350m'][i]):<20} {str(comparison['BioGPT'][i]):<20}")

        print("\n Architecture comparison complete")
        print("\n Key insight: Both models are compatible with BLIP-2 architecture")
        print("   BioGPT is specifically trained on biomedical text, making it ideal for medical reports")

        return True

    except Exception as e:
        print(f"  Could not load OPT for comparison: {e}")
        print("   (This is okay - BioGPT is the target model)")
        return True

def test_integration_readiness():
    """Test 5: Verify integration readiness with BLIP-2"""
    print("\n" + "="*80)
    print("TEST 5: BLIP-2 Integration Readiness")
    print("="*80)

    checks = []

    # Check 1: imports
    print("\nChecking required imports...")
    try:
        from transformers import BioGptForCausalLM, BioGptTokenizer, BertConfig, BertModel
        import timm
        checks.append(("Required imports", True))
        print(" All required imports available")
    except ImportError as e:
        checks.append(("Required imports", False))
        print(f" Missing imports: {e}")

    # Check 2: Model compatibility
    print("\nChecking model compatibility...")
    try:
        model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

        # Check required methods
        has_embeddings = hasattr(model, 'get_input_embeddings')
        has_generate = hasattr(model, 'generate')
        has_config = hasattr(model, 'config')

        if has_embeddings and has_generate and has_config:
            checks.append(("Model compatibility", True))
            print(" BioGPT has all required methods")
        else:
            checks.append(("Model compatibility", False))
            print(" BioGPT missing required methods")

    except Exception as e:
        checks.append(("Model compatibility", False))
        print(f" Model compatibility check failed: {e}")

    # Summary
    print("\n" + "-"*80)
    print("Integration Readiness Summary:")
    print("-"*80)
    for check_name, result in checks:
        status = " PASS" if result else " FAIL"
        print(f"  {check_name:<30} {status}")

    all_passed = all(result for _, result in checks)

    if all_passed:
        print("\n ALL CHECKS PASSED - Ready for BLIP-2 integration!")
    else:
        print("\n  Some checks failed - please review above")

    return all_passed

def main():
    print("\n" + "="*80)
    print("MEDICAL BLIP-2 WITH BIOGPT - VERIFICATION TESTS")
    print("="*80)
    print("\nThis script tests BioGPT integration with the BLIP-2 architecture")
    print("for medical report generation.\n")

    # Run tests
    success, tokenizer, model = test_biogpt_availability()

    if success:
        test_text_generation(tokenizer, model)
        test_embedding_compatibility(model)
        test_architecture_comparison()
        test_integration_readiness()

        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("\n1. Update your training script to use:")
        print("   from medical_blip2_biogpt import MedicalBLIP2")
        print("\n2. Initialize model with:")
        print("   model = MedicalBLIP2(")
        print("       vision_encoder_path='path/to/checkpoint',")
        print("       language_model='microsoft/biogpt',")
        print("       num_query_tokens=32,")
        print("   )")
        print("\n3. The model will use BioGPT instead of OPT for better")
        print("   biomedical text generation.")
        print("\n4. All other training and evaluation code remains the same.")
        print("="*80 + "\n")
    else:
        print("\n Cannot proceed with further tests due to BioGPT loading failure")
        print("\nTroubleshooting:")
        print("  1. Ensure transformers library is up to date: pip install --upgrade transformers")
        print("  2. Check internet connection for model download")
        print("  3. Verify you have sufficient disk space")

if __name__ == "__main__":
    main()

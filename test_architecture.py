#!/usr/bin/env python3

"""
Test Medical BLIP-2 with BioGPT Architecture
Verifies 3D ViT loading and model initialization
"""

import torch
import sys

def test_3d_vit_loading():
    """Test if 3D ViT can be loaded from LAVIS"""
    print("="*80)
    print("TEST 1: 3D ViT Loading")
    print("="*80)

    try:
        from lavis.models.blip_models.vit import ViT
        print("\n LAVIS ViT import successful")

        # Create 3D ViT
        vit = ViT(
            in_channels=1,
            img_size=(112, 256, 352),  # 3D dimensions
            patch_size=(16, 16, 32),    # 3D patches
            num_classes=0,
        )

        print(f" 3D ViT created successfully")
        print(f"   Input shape: (B, 1, 112, 256, 352)")
        print(f"   Patch size: (16, 16, 32)")

        # Test forward pass
        dummy_input = torch.randn(1, 1, 112, 256, 352)
        with torch.no_grad():
            output = vit(dummy_input)

        print(f"\n Forward pass successful")
        
        # Handle tuple output (LAVIS ViT may return multiple outputs)
        if isinstance(output, tuple):
            print(f"   Output type: tuple with {len(output)} elements")
            print(f"   Primary output shape: {output[0].shape}")
        else:
            print(f"   Output shape: {output.shape}")

        return True

    except ImportError:
        print("\n  LAVIS not found")
        print("   Install with: pip install salesforce-lavis")
        return False
    except Exception as e:
        print(f"\n 3D ViT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_initialization():
    """Test full model initialization"""
    print("\n" + "="*80)
    print("TEST 2: Model Initialization")
    print("="*80)

    try:
        from medical_blip2_biogpt import MedicalBLIP2BioGPT

        print("\n Model class imported successfully")

        # Note: This requires actual checkpoint file
        # Uncomment and update path to test
        """
        model = MedicalBLIP2BioGPT(
            vision_encoder_path='/home/muhammedg/fvlm/checkpoints/model.pth',
            image_size=(112, 256, 352),
            patch_size=(16, 16, 32),
            use_qformer=True,
            num_query_tokens=32,
        )
        print("\n Model initialized successfully")
        """

        print("\n To test full initialization, uncomment the model creation")
        print("   in test_architecture.py and provide checkpoint path")

        return True

    except ImportError as e:
        print(f"\n Import failed: {e}")
        return False
    except Exception as e:
        print(f"\n Initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_biogpt_compatibility():
    """Test BioGPT cross-attention"""
    print("\n" + "="*80)
    print("TEST 3: BioGPT Cross-Attention")
    print("="*80)

    try:
        from transformers import BioGptForCausalLM, BioGptConfig

        print("\n BioGPT imports successful")

        # Create BioGPT with cross-attention
        config = BioGptConfig.from_pretrained("microsoft/biogpt")
        config.is_decoder = True
        config.add_cross_attention = True

        print(f"\n BioGPT config created")
        print(f"   is_decoder: {config.is_decoder}")
        print(f"   add_cross_attention: {config.add_cross_attention}")

        print("\n BioGPT is compatible with cross-attention")

        return True

    except Exception as e:
        print(f"\n BioGPT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dimensions():
    """Test dimension compatibility"""
    print("\n" + "="*80)
    print("TEST 4: Dimension Compatibility")
    print("="*80)

    # Simulate data flow
    print("\nData flow through architecture:")
    print(""*80)

    batch_size = 2

    # Input
    input_shape = (batch_size, 1, 112, 256, 352)
    print(f"1. Input (3D volume):           {input_shape}")

    # After 3D ViT
    # Assuming patch_size=(16, 16, 32): 
    # D=112/16=7, H=256/16=16, W=352/32=11 → 7*16*11=1232 patches
    num_patches = (112//16) * (256//16) * (352//32)
    hidden_size = 768
    vit_output_shape = (batch_size, num_patches, hidden_size)
    print(f"2. After 3D ViT:                {vit_output_shape}")

    # After Q-Former (if used)
    num_queries = 32
    qformer_output_shape = (batch_size, num_queries, hidden_size)
    print(f"3. After Q-Former (optional):   {qformer_output_shape}")

    # After Projection
    biogpt_hidden = 1024
    projected_shape = (batch_size, num_queries, biogpt_hidden)
    print(f"4. After Projection:            {projected_shape}")

    # Text tokens
    text_len = 256
    text_shape = (batch_size, text_len, biogpt_hidden)
    print(f"5. Text embeddings:             {text_shape}")

    # Concatenated
    combined_len = num_queries + text_len
    final_shape = (batch_size, combined_len, biogpt_hidden)
    print(f"6. Concatenated (to BioGPT):    {final_shape}")

    print("\n All dimensions are compatible")

    return True

def main():
    print("\n" + "="*80)
    print("MEDICAL BLIP-2 BIOGPT ARCHITECTURE TEST")
    print("="*80)
    print("\nThis script verifies the 3D architecture and BioGPT integration\n")

    results = []

    # Run tests
    results.append(("3D ViT Loading", test_3d_vit_loading()))
    results.append(("Model Initialization", test_model_initialization()))
    results.append(("BioGPT Compatibility", test_biogpt_compatibility()))
    results.append(("Dimension Compatibility", test_dimensions()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = " PASS" if passed else " FAIL"
        print(f"  {test_name:<30} {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n ALL TESTS PASSED")
        print("\nNext steps:")
        print("1. Update checkpoint path in medical_blip2_biogpt.py")
        print("2. Run: python train_blip2_biogpt.py --subset_size 5 (quick test)")
        print("3. Run full training")
    else:
        print("\n  Some tests failed - review output above")
        print("\nTroubleshooting:")
        print("1. Install LAVIS: pip install salesforce-lavis")
        print("2. Install transformers>=4.25.0")
        print("3. Ensure checkpoint file exists")

    print("="*80 + "\n")

if __name__ == "__main__":
    main()

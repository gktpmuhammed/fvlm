#!/usr/bin/env python3
"""
Test script for report generation using a trained checkpoint
"""
import os
import sys
import torch
import pandas as pd
from PIL import Image
import numpy as np

# Add the project root to Python path
sys.path.append('/home/muhammedg/fvlm')

from lavis.models import load_model_and_preprocess
from lavis.common.config import Config
from lavis.common.registry import registry
from lavis.models.blip_models.blip_pretrain import BlipPretrain
from lavis.processors import load_processor

def load_trained_model(checkpoint_path, config_path):
    """Load the trained model from checkpoint"""
    print(f"🔄 Loading model from checkpoint: {checkpoint_path}")
    
    # Set up command line arguments for config parsing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, required=True)
    parser.add_argument("--options", nargs="+", default=[])
    args = parser.parse_args(["--cfg-path", config_path])
    
    # Load config
    cfg = Config(args)
    
    # Build model
    model_cls = registry.get_model_class(cfg.config.model.arch)
    model = model_cls.from_config(cfg.config.model)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Debug tokenizer
    print(f"🔍 Tokenizer info:")
    print(f"  Vocab size: {model.tokenizer.vocab_size}")
    print(f"  UNK token: '{model.tokenizer.unk_token}' (ID: {model.tokenizer.unk_token_id})")
    print(f"  PAD token: '{model.tokenizer.pad_token}' (ID: {model.tokenizer.pad_token_id})")
    print(f"  CLS token: '{model.tokenizer.cls_token}' (ID: {model.tokenizer.cls_token_id})")
    print(f"  SEP token: '{model.tokenizer.sep_token}' (ID: {model.tokenizer.sep_token_id})")
    
    # Test tokenizer
    test_text = "The chest X-ray shows normal findings."
    tokens = model.tokenizer.encode(test_text)
    decoded = model.tokenizer.decode(tokens)
    print(f"  Test encoding/decoding: '{test_text}' -> {tokens} -> '{decoded}'")
    
    print(f"✅ Model loaded successfully on {device}")
    return model, device

def load_test_images(data_root, num_samples=3):
    """Load some test images and their ground truth reports"""
    print(f"📂 Loading test images from: {data_root}")
    
    # Load validation reports
    val_reports_path = os.path.join(data_root, "dataset/radiology_text_reports/validation_reports.csv")
    df = pd.read_csv(val_reports_path)
    df = df.dropna(subset=['Impressions_EN'])
    
    # Take first few samples
    test_samples = []
    for idx, row in df.head(num_samples).iterrows():
        volume_name = row["VolumeName"]
        ground_truth = row["Impressions_EN"]
        
        # Construct image path (same logic as dataset)
        base_name = volume_name.replace('.nii.gz', '')
        parts = base_name.split('_')
        if len(parts) >= 3:
            nested_path = f"{parts[0]}_{parts[1]}/{parts[0]}_{parts[1]}_{parts[2]}/{volume_name}"
            image_path = os.path.join(data_root, "valid/images/valid", nested_path)
        else:
            image_path = os.path.join(data_root, "valid/images/valid", volume_name)
        
        if os.path.exists(image_path):
            test_samples.append({
                'image_path': image_path,
                'ground_truth': ground_truth,
                'volume_name': volume_name
            })
            print(f"  📄 {volume_name}")
        else:
            print(f"  ❌ Image not found: {image_path}")
    
    print(f"✅ Loaded {len(test_samples)} test samples")
    return test_samples

def preprocess_3d_image(image_path):
    """Preprocess 3D medical image for the model"""
    import nibabel as nib
    from monai import transforms
    
    # Load the 3D image
    nii_img = nib.load(image_path)
    image_data = nii_img.get_fdata()
    
    # Add channel dimension and convert to tensor
    image_data = np.expand_dims(image_data, axis=0)  # Add channel dim
    image_tensor = torch.from_numpy(image_data).float()
    
    # Apply the same transforms as in training
    transform = transforms.Compose([
        transforms.Transposed(keys=["image"], indices=(0, 3, 2, 1)),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-1150, a_max=350,
            b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.SpatialPadd(
            keys=["image"],
            spatial_size=(112, 256, 352),
            mode="constant",
            constant_values=0
        ),
        transforms.CenterSpatialCropd(
            keys=["image"],
            roi_size=(112, 256, 352)
        ),
    ])
    
    # Apply transforms
    data_dict = {"image": image_tensor}
    transformed = transform(data_dict)
    
    return transformed["image"]

def generate_report(model, image_tensor, device, max_length=100, num_beams=3):
    """Generate a report for the given image"""
    
    # Prepare input
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Create input samples dict
    samples = {
        "image": image_tensor,
    }
    
    # Generate report
    with torch.no_grad():
        try:
            # Use the model's generate method with different parameters
            print(f"  🔧 Generation params: max_length={max_length}, num_beams={num_beams}")
            
            # Try simpler generation first
            print("  🔄 Trying basic generation...")
            generated = model.generate(
                samples, 
                max_length=50, 
                num_beams=1, 
                min_length=5,
                use_nucleus_sampling=False,
                repetition_penalty=1.0
            )
            
            # If that doesn't work, try nucleus sampling
            if isinstance(generated, list) and len(generated) > 0 and generated[0].strip() in ['', '_' * len(generated[0].strip())]:
                print("  🔄 Basic generation failed, trying nucleus sampling...")
                generated = model.generate(
                    samples, 
                    max_length=30, 
                    num_beams=1, 
                    min_length=5,
                    use_nucleus_sampling=True,
                    top_p=0.9,
                    repetition_penalty=1.5
                )
            
            if isinstance(generated, list) and len(generated) > 0:
                result = generated[0]
                print(f"  📊 Generated text length: {len(result)} characters")
                print(f"  🔍 First 100 chars: {repr(result[:100])}")
                
                # Debug: check what tokens were generated
                if result:
                    # Try to encode the result back to see token IDs
                    try:
                        token_ids = model.tokenizer.encode(result, add_special_tokens=False)
                        unique_tokens = set(token_ids)
                        print(f"  🔢 Unique token IDs in result: {unique_tokens}")
                        print(f"  📝 Token ID -> text mapping:")
                        for token_id in unique_tokens:
                            token_text = model.tokenizer.decode([token_id])
                            print(f"    {token_id}: '{token_text}'")
                    except Exception as e:
                        print(f"  ⚠️  Could not analyze tokens: {e}")
                
                return result
            else:
                return "No report generated"
        except Exception as e:
            print(f"❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            return f"Generation failed: {str(e)}"

def main():
    """Main function to test report generation"""
    print("🚀 Starting report generation test...")
    
    # Paths
    data_root = "/home/muhammedg/fvlm/data"
    checkpoint_path = "./lavis/output/BLIP/Report_Generation_Test/20250927011/checkpoint_9.pth"
    config_path = "lavis/projects/blip/train/test_report_generation.yaml"
    
    try:
        # Load trained model
        model, device = load_trained_model(checkpoint_path, config_path)
        
        # Load test images
        test_samples = load_test_images(data_root, num_samples=2)
        
        if not test_samples:
            print("❌ No test samples found!")
            return
        
        # Generate reports for each test image
        print(f"\n🔤 Generating reports...")
        for i, sample in enumerate(test_samples):
            print(f"\n{'='*60}")
            print(f"📋 Sample {i+1}: {sample['volume_name']}")
            print(f"{'='*60}")
            
            try:
                # Preprocess image
                print("🔄 Preprocessing image...")
                image_tensor = preprocess_3d_image(sample['image_path'])
                print(f"  Image shape: {image_tensor.shape}")
                
                # Generate report
                print("🤖 Generating report...")
                generated_report = generate_report(model, image_tensor, device)
                
                # Display results
                print(f"\n📝 **Generated Report:**")
                print(f"{generated_report}")
                
                print(f"\n📄 **Ground Truth:**")
                print(f"{sample['ground_truth']}")
                
            except Exception as e:
                print(f"❌ Error processing sample {i+1}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ Report generation test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

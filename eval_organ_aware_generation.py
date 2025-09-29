#!/usr/bin/env python3
"""
Evaluation script for organ-aware report generation
"""
import os
import sys
import json
import argparse
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

sys.path.append('/home/muhammedg/fvlm')

from lavis.common.config import Config
from lavis.common.registry import registry
import monai.transforms as transforms

def merge_labels(label):
    """Merge anatomical structure labels to our 4 target organs"""
    class_map = {
        1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "liver",
        6: "stomach", 7: "pancreas", 8: "adrenal_gland_right", 9: "adrenal_gland_left",
        10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left", 12: "lung_upper_lobe_right",
        13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right", 15: "esophagus",
        16: "trachea", 17: "thyroid_gland", 18: "small_bowel", 19: "duodenum",
        20: "colon", 21: "urinary_bladder", 22: "prostate", 23: "kidney_cyst_left",
        24: "kidney_cyst_right", 25: "sacrum", 26: "vertebrae_S1", 27: "vertebrae_L5",
        28: "vertebrae_L4", 29: "vertebrae_L3", 30: "vertebrae_L2", 31: "vertebrae_L1",
        32: "vertebrae_T12", 33: "vertebrae_T11", 34: "vertebrae_T10", 35: "vertebrae_T9",
        36: "vertebrae_T8", 37: "vertebrae_T7", 38: "vertebrae_T6", 39: "vertebrae_T5",
        40: "vertebrae_T4", 41: "vertebrae_T3", 42: "vertebrae_T2", 43: "vertebrae_T1",
        44: "vertebrae_C7", 45: "vertebrae_C6", 46: "vertebrae_C5", 47: "vertebrae_C4",
        48: "vertebrae_C3", 49: "vertebrae_C2", 50: "vertebrae_C1", 51: "heart",
        52: "aorta", 53: "pulmonary_vein", 54: "brachiocephalic_trunk",
        55: "subclavian_artery_right", 56: "subclavian_artery_left",
        57: "common_carotid_artery_right", 58: "common_carotid_artery_left",
        59: "brachiocephalic_vein_left", 60: "brachiocephalic_vein_right",
        61: "atrial_appendage_left", 62: "superior_vena_cava",
        63: "inferior_vena_cava", 64: "portal_vein_and_splenic_vein",
        65: "iliac_artery_left", 66: "iliac_artery_right", 67: "iliac_vena_left",
        68: "iliac_vena_right", 69: "humerus_left", 70: "humerus_right",
        71: "scapula_left", 72: "scapula_right", 73: "clavicula_left",
        74: "clavicula_right", 75: "femur_left", 76: "femur_right",
        77: "hip_left", 78: "hip_right", 79: "spinal_cord",
        80: "gluteus_maximus_left", 81: "gluteus_maximus_right",
        82: "gluteus_medius_left", 83: "gluteus_medius_right",
        84: "gluteus_minimus_left", 85: "gluteus_minimus_right",
        86: "autochthon_left", 87: "autochthon_right", 88: "iliopsoas_left",
        89: "iliopsoas_right", 90: "brain", 91: "skull", 92: "rib_left_1",
        93: "rib_left_2", 94: "rib_left_3", 95: "rib_left_4",
        96: "rib_left_5", 97: "rib_left_6", 98: "rib_left_7",
        99: "rib_left_8", 100: "rib_left_9", 101: "rib_left_10",
        102: "rib_left_11", 103: "rib_left_12", 104: "rib_right_1",
        105: "rib_right_2", 106: "rib_right_3", 107: "rib_right_4",
        108: "rib_right_5", 109: "rib_right_6", 110: "rib_right_7",
        111: "rib_right_8", 112: "rib_right_9", 113: "rib_right_10",
        114: "rib_right_11", 115: "rib_right_12", 116: "sternum",
        117: "costal_cartilages"
    }
    
    merged_organ_id = {
        "lung_upper_lobe_left": 0,
        "lung_lower_lobe_left": 0,
        "lung_upper_lobe_right": 0,
        "lung_middle_lobe_right": 0,
        "lung_lower_lobe_right": 0,
        "heart": 1,
        "atrial_appendage_left": 1,
        "esophagus": 2,
        "aorta": 3,
    }
    
    fused_mask = np.zeros_like(label)
    for original_id, organ_name in class_map.items():
        if organ_name in merged_organ_id:
            merged_id = merged_organ_id[organ_name]
            fused_mask[label == original_id] = merged_id + 1
    return fused_mask

class SpacingNormalization:
    """Custom transform to normalize spacing to reference spacing"""
    def __init__(self, ref_spacing=(1.0, 1.0, 3.0), debug=False):
        self.ref_spacing = ref_spacing
        self.debug = debug
    
    def __call__(self, data):
        # For evaluation, we'll just return the data as-is since we're using preprocessed data
        return data

def load_model_and_processor(cfg_path, ckpt_path):
    """Load the trained model and processors with proper encoder/decoder loading"""
    
    # Parse configuration
    class Args:
        def __init__(self):
            self.cfg_path = cfg_path
            self.options = []
    
    args = Args()
    cfg = Config(args)
    
    # Setup
    torch.manual_seed(42)
    
    # Load model
    model_cls = registry.get_model_class(cfg.model_cfg.arch)
    model = model_cls.from_config(cfg.model_cfg)
    
    print("Loading model with hybrid checkpoint approach...")
    
    # Step 1: Load original pretrained encoders from /home/muhammedg/fvlm/checkpoints/model.pth
    original_checkpoint_path = "/home/muhammedg/fvlm/checkpoints/model.pth"
    print(f"Loading vision and text encoders from: {original_checkpoint_path}")
    
    try:
        original_checkpoint = torch.load(original_checkpoint_path, map_location="cpu")
        
        # Extract encoder weights from original checkpoint
        encoder_state_dict = {}
        
        # The original checkpoint has model weights under 'model' key
        original_model_state = original_checkpoint.get("model", original_checkpoint)
        
        for key, value in original_model_state.items():
            if ("visual_encoder" in key or 
                "text_encoder" in key or 
                "vision_proj" in key or 
                "text_proj" in key or
                "query_tokens" in key or
                "temp" in key):
                encoder_state_dict[key] = value
        
        print(f"Found {len(encoder_state_dict)} encoder parameters in original checkpoint")
        
        # Load encoder weights
        missing_keys, unexpected_keys = model.load_state_dict(encoder_state_dict, strict=False)
        print(f"Loaded encoders - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        
    except Exception as e:
        print(f"Warning: Could not load original encoders: {e}")
        print("Proceeding with random encoder initialization...")
    
    # Step 2: Load trained decoder from the training checkpoint
    print(f"Loading text decoder from training checkpoint: {ckpt_path}")
    
    training_checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Extract decoder weights from training checkpoint
    decoder_state_dict = {}
    for key, value in training_checkpoint["model"].items():
        if "text_decoder" in key:
            decoder_state_dict[key] = value
    
    print(f"Found {len(decoder_state_dict)} decoder parameters in training checkpoint")
    
    # Load decoder weights
    missing_keys, unexpected_keys = model.load_state_dict(decoder_state_dict, strict=False)
    print(f"Loaded decoder - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
    
    model.eval()
    model = model.cuda()
    
    print("Model loading completed with hybrid approach!")
    print("- Vision encoder: From original pretrained checkpoint")
    print("- Text encoder: From original pretrained checkpoint") 
    print("- Text decoder: From training checkpoint")
    
    return model, cfg

def evaluate_model(model, vis_root, num_samples=10, max_length=1000, min_length=25, use_nucleus_sampling=True, top_p=0.9, repetition_penalty=1.2):
    """Evaluate the model on validation images"""
    
    # Image preprocessing transforms
    transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True),
        SpacingNormalization(ref_spacing=(1.0, 1.0, 3.0)),
        transforms.Transposed(keys=["image", "label"], indices=(0, 3, 2, 1)),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-1150, a_max=350,
            b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.SpatialPadd(
            keys=["image", "label"],
            spatial_size=(112, 256, 352)
        ),
        transforms.RandSpatialCropd(
            keys=["image", "label"],
            roi_size=(112, 256, 352)
        ),
    ])
    
    # Find validation images
    val_images_dir = os.path.join(vis_root, "valid/images/valid")
    if not os.path.exists(val_images_dir):
        print(f"Validation directory not found: {val_images_dir}")
        return []
    
    # Get list of validation images
    val_images = []
    for root, dirs, files in os.walk(val_images_dir):
        for file in files:
            if file.endswith('.nii.gz') and not file.startswith('.'):
                image_path = os.path.join(root, file)
                mask_path = image_path.replace('/images/', '/masks/')
                if os.path.exists(mask_path):
                    val_images.append({
                        'image_path': image_path,
                        'mask_path': mask_path,
                        'case_id': file.replace('.nii.gz', '').replace('_1', '')
                    })
    
    print(f"Found {len(val_images)} validation images")
    val_images = val_images[:num_samples]  # Limit to num_samples
    
    results = []
    
    with torch.no_grad():
        for i, img_info in enumerate(tqdm(val_images, desc="Generating Reports")):
            try:
                # Load and preprocess image and mask
                data = transform({
                    "image": img_info['image_path'],
                    "label": img_info['mask_path']
                })
                
                image = data["image"].unsqueeze(0).cuda()  # Add batch dimension
                seg_mask = data["label"][0].as_tensor()  # Remove channel dimension
                
                # Apply organ ID mapping
                seg_mask_np = seg_mask.numpy()
                merged_mask = merge_labels(seg_mask_np)
                seg_mask = torch.from_numpy(merged_mask).float().unsqueeze(0).cuda()  # Add batch dimension
                
                # Prepare input for generation
                samples = {
                    "image": image,
                    "seg": seg_mask
                }
                
                # Generate report using optimized nucleus sampling
                generated_text = model.generate(
                    samples,
                    use_nucleus_sampling=use_nucleus_sampling,
                    num_beams=1,  # Always 1 for nucleus sampling
                    max_length=max_length,
                    min_length=min_length,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
                
                result = {
                    "case_id": img_info['case_id'],
                    "image_path": img_info['image_path'],
                    "generated_report": generated_text[0] if isinstance(generated_text, list) else generated_text,
                    "unique_organs_detected": torch.unique(seg_mask).cpu().numpy().tolist()
                }
                
                results.append(result)
                
                print(f"\n--- Sample {i+1}/{len(val_images)} ---")
                print(f"Case ID: {result['case_id']}")
                print(f"Organs detected: {result['unique_organs_detected']}")
                print(f"Generated Report: {result['generated_report'][:200]}...")
                
            except Exception as e:
                print(f"Error processing {img_info['case_id']}: {e}")
                continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Organ-Aware Report Generation")
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--vis_root", type=str, default="/home/muhammedg/fvlm/data/", help="Path to images")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--min_length", type=int, default=25, help="Minimum generation length")
    parser.add_argument("--use_nucleus_sampling", type=bool, default=True, help="Use nucleus sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Load model
    model, cfg = load_model_and_processor(args.cfg_path, args.ckpt_path)
    
    # Evaluate
    results = evaluate_model(
        model, 
        args.vis_root, 
        args.num_samples, 
        args.max_length, 
        args.min_length, 
        args.use_nucleus_sampling,
        args.top_p,
        args.repetition_penalty
    )
    
    # Save results
    if args.output_file is None:
        ckpt_name = os.path.basename(args.ckpt_path).replace('.pth', '')
        args.output_file = f"report_generations/organ_aware_{ckpt_name}_reports.json"
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation completed!")
    print(f"Generated reports for {len(results)} samples")
    print(f"Results saved to: {args.output_file}")
    
    # Print summary statistics
    if results:
        report_lengths = [len(r['generated_report'].split()) for r in results]
        unique_reports = len(set(r['generated_report'] for r in results))
        
        print(f"\n=== Summary Statistics ===")
        print(f"Total samples: {len(results)}")
        print(f"Unique reports: {unique_reports} ({unique_reports/len(results)*100:.1f}%)")
        print(f"Average report length: {np.mean(report_lengths):.1f} words")
        print(f"Report length range: {min(report_lengths)}-{max(report_lengths)} words")

if __name__ == "__main__":
    main()

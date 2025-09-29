#!/usr/bin/env python3
"""
Comprehensive Debug Suite for Organ-Aware Report Generation
===========================================================

This script consolidates all debugging functionality developed during the 
organ-aware report generation project. It includes tests for:

1. Dataset loading and preprocessing
2. Model architecture and configuration
3. Vision encoder functionality
4. Cross-attention mechanism
5. Text generation and diversity
6. Training checkpoint analysis
7. Segmentation mask processing

Usage:
    python comprehensive_debug_suite.py --test <test_name>
    python comprehensive_debug_suite.py --all

Available tests:
    - dataset: Test dataset loading and sample processing
    - vision: Test vision encoder outputs and diversity
    - cross_attention: Test cross-attention mechanism
    - generation: Test text generation diversity
    - checkpoints: Analyze training checkpoints
    - segmentation: Test segmentation mask processing
    - config: Test model configuration
    - all: Run all tests
"""

import torch
import torch.nn.functional as F
import sys
import os
import argparse
import numpy as np
from collections import Counter
from pathlib import Path

sys.path.append('/home/muhammedg/fvlm')

from lavis.common.config import Config
from lavis.common.registry import registry
from lavis.datasets.datasets.organ_aware_report_datasets import OrganAwareReportDataset

class ComprehensiveDebugSuite:
    def __init__(self):
        self.config_path = "/home/muhammedg/fvlm/lavis/projects/blip/train/organ_aware_report_generation.yaml"
        self.data_root = "/home/muhammedg/fvlm/data/"
        self.conc_info_path = "/home/muhammedg/test/fvlm/data/conc_info.json"
        self.desc_info_path = "/home/muhammedg/test/fvlm/data/desc_info.json"
        self.pretrained_checkpoint = "/home/muhammedg/fvlm/checkpoints/model.pth"
        
        # Find latest training checkpoint
        self.latest_checkpoint = self._find_latest_checkpoint()
        
    def _find_latest_checkpoint(self):
        """Find the most recent training checkpoint"""
        output_dirs = [
            "/home/muhammedg/fvlm/lavis/output/BLIP/Organ_Aware_Report_Generation_Cross_Attention_1K",
            "/home/muhammedg/fvlm/lavis/output/BLIP/Organ_Aware_Report_Generation_Cross_Attention_Test",
            "/home/muhammedg/fvlm/lavis/output/BLIP/Organ_Aware_Report_Generation_Full_Dataset"
        ]
        
        latest_checkpoint = None
        latest_time = 0
        
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                for subdir in os.listdir(output_dir):
                    subdir_path = os.path.join(output_dir, subdir)
                    if os.path.isdir(subdir_path):
                        for file in os.listdir(subdir_path):
                            if file.startswith("checkpoint_") and file.endswith(".pth"):
                                file_path = os.path.join(subdir_path, file)
                                file_time = os.path.getmtime(file_path)
                                if file_time > latest_time:
                                    latest_time = file_time
                                    latest_checkpoint = file_path
        
        return latest_checkpoint
    
    def _load_model(self, checkpoint_path=None):
        """Load model with proper configuration"""
        class Args:
            def __init__(self, config_path):
                self.cfg_path = config_path
                self.options = []
        
        args = Args(self.config_path)
        cfg = Config(args)
        
        model_cls = registry.get_model_class(cfg.model_cfg.arch)
        model = model_cls.from_config(cfg.model_cfg)
        
        if checkpoint_path:
            if "model.pth" in checkpoint_path:
                # Original pretrained checkpoint - hybrid loading
                original_checkpoint = torch.load(checkpoint_path, map_location="cpu")
                original_model_state = original_checkpoint.get("model", original_checkpoint)
                
                encoder_state_dict = {}
                for key, value in original_model_state.items():
                    if ("visual_encoder" in key or "text_encoder" in key or 
                        "vision_proj" in key or "text_proj" in key or
                        "query_tokens" in key or "temp" in key):
                        encoder_state_dict[key] = value
                
                model.load_state_dict(encoder_state_dict, strict=False)
                print(f"Loaded encoders from: {checkpoint_path}")
            else:
                # Training checkpoint
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(checkpoint["model"], strict=False)
                print(f"Loaded training checkpoint: {checkpoint_path}")
        
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model
    
    def test_dataset(self, max_samples=10):
        """Test dataset loading and sample processing"""
        print("=" * 80)
        print("TESTING DATASET LOADING AND PROCESSING")
        print("=" * 80)
        
        try:
            dataset = OrganAwareReportDataset(
                vis_processor=None,
                text_processor=None,
                vis_root=self.data_root,
                conc_info_path=self.conc_info_path,
                desc_info_path=self.desc_info_path,
                max_samples=max_samples
            )
            
            print(f"Dataset loaded successfully")
            print(f"   Total samples: {len(dataset)}")
            
            # Test sample loading
            sample = dataset[0]
            print(f"   Sample keys: {list(sample.keys())}")
            print(f"   Case ID: {sample['case_id']}")
            print(f"   Image shape: {sample['image'].shape}")
            
            if 'seg' in sample:
                seg_mask = sample['seg']
                print(f"   Segmentation shape: {seg_mask.shape}")
                unique_labels = torch.unique(seg_mask)
                print(f"   Unique segmentation labels: {unique_labels.tolist()}")
            
            if 'organ_reports' in sample:
                organ_reports = sample['organ_reports']
                print(f"   Available organ reports: {list(organ_reports.keys())}")
                for organ, report in organ_reports.items():
                    print(f"     {organ}: {report[:100]}..." if len(report) > 100 else f"     {organ}: {report}")
            
            print(f"   Mode: {sample.get('mode', 'Not set')}")
            
        except Exception as e:
            print(f"Dataset test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def test_vision_encoder(self, num_samples=3):
        """Test vision encoder outputs and diversity"""
        print("=" * 80)
        print("TESTING VISION ENCODER DIVERSITY")
        print("=" * 80)
        
        try:
            model = self._load_model(self.pretrained_checkpoint)
            dataset = OrganAwareReportDataset(
                vis_processor=None,
                text_processor=None,
                vis_root=self.data_root,
                conc_info_path=self.conc_info_path,
                desc_info_path=self.desc_info_path,
                max_samples=100
            )
            
            vision_features = []
            case_ids = []
            
            for i in range(num_samples):
                sample = dataset[i * 30]  # Sample different patients
                case_ids.append(sample['case_id'])
                
                image = sample["image"].unsqueeze(0)
                if torch.cuda.is_available():
                    image = image.cuda()
                
                with torch.no_grad():
                    image_embeds, _ = model.visual_encoder(image)
                    vision_features.append(image_embeds.cpu())
                
                print(f"Patient {i+1} ({sample['case_id']}):")
                print(f"  Vision features shape: {image_embeds.shape}")
                print(f"  Mean: {image_embeds.mean().item():.6f}")
                print(f"  Std: {image_embeds.std().item():.6f}")
            
            # Compare diversity
            print(f"\n--- DIVERSITY ANALYSIS ---")
            for i in range(len(vision_features)):
                for j in range(i+1, len(vision_features)):
                    cosine_sim = F.cosine_similarity(
                        vision_features[i].flatten(), 
                        vision_features[j].flatten(), 
                        dim=0
                    ).item()
                    mse = F.mse_loss(vision_features[i], vision_features[j]).item()
                    
                    print(f"Patients {case_ids[i]} vs {case_ids[j]}:")
                    print(f"  Cosine similarity: {cosine_sim:.6f}")
                    print(f"  MSE: {mse:.6f}")
                    
                    if cosine_sim > 0.99:
                        print("  IDENTICAL: Vision features are nearly identical!")
                    elif cosine_sim > 0.95:
                        print("  SIMILAR: Vision features are very similar")
                    else:
                        print("  DIVERSE: Vision features are different")
            
        except Exception as e:
            print(f"Vision encoder test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def test_cross_attention(self, num_samples=2):
        """Test cross-attention mechanism"""
        print("=" * 80)
        print("TESTING CROSS-ATTENTION MECHANISM")
        print("=" * 80)
        
        try:
            model = self._load_model(self.latest_checkpoint)
            dataset = OrganAwareReportDataset(
                vis_processor=None,
                text_processor=None,
                vis_root=self.data_root,
                conc_info_path=self.conc_info_path,
                desc_info_path=self.desc_info_path,
                max_samples=100
            )
            
            print(f"Using checkpoint: {self.latest_checkpoint}")
            print(f"Decoder config:")
            print(f"  is_decoder: {model.text_decoder.config.is_decoder}")
            print(f"  add_cross_attention: {model.text_decoder.config.add_cross_attention}")
            print(f"  encoder_width: {getattr(model.text_decoder.config, 'encoder_width', 'Not set')}")
            
            cross_attentions = []
            
            for i in range(num_samples):
                sample = dataset[i * 40]
                print(f"\n--- Testing sample {i+1}: {sample['case_id']} ---")
                
                image = sample["image"].unsqueeze(0)
                if torch.cuda.is_available():
                    image = image.cuda()
                
                with torch.no_grad():
                    image_embeds, _ = model.visual_encoder(image)
                
                # Test cross-attention
                prompt = "The chest CT scan shows"
                text_input = model.tokenizer(prompt, return_tensors="pt", max_length=50)
                if torch.cuda.is_available():
                    text_input = {k: v.cuda() for k, v in text_input.items()}
                
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)
                if torch.cuda.is_available():
                    image_atts = image_atts.cuda()
                
                with torch.no_grad():
                    decoder_output = model.text_decoder(
                        input_ids=text_input["input_ids"],
                        attention_mask=text_input["attention_mask"],
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                        output_attentions=True
                    )
                
                print(f"  Vision features: {image_embeds.shape}")
                print(f"  Text input: {text_input['input_ids'].shape}")
                
                # Check cross-attention
                if hasattr(decoder_output, 'cross_attentions') and decoder_output.cross_attentions is not None:
                    if len(decoder_output.cross_attentions) > 0:
                        cross_attn = decoder_output.cross_attentions[-1]
                        print(f"  Cross-attention working!")
                        print(f"     Shape: {cross_attn.shape}")
                        
                        # Analyze attention focus
                        avg_attn = cross_attn.mean(dim=(0, 1))
                        attn_std = avg_attn.std(dim=-1).mean()
                        print(f"     Attention focus (std): {attn_std.item():.6f}")
                        
                        if attn_std.item() > 0.01:
                            print("     Attention is focused")
                        else:
                            print("     Attention is uniform - needs more training")
                        
                        cross_attentions.append(avg_attn.cpu())
                    else:
                        print("  Cross-attention list empty")
                else:
                    print("  No cross-attention found")
                
                # Test next token predictions
                next_token_logits = decoder_output.logits[0, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                top_probs, top_indices = torch.topk(next_token_probs, 5)
                
                print("  Top predictions:")
                for prob, idx in zip(top_probs, top_indices):
                    token = model.tokenizer.decode([idx])
                    print(f"    '{token}' ({prob.item():.3f})")
            
            # Compare cross-attention patterns
            if len(cross_attentions) >= 2:
                print(f"\n--- CROSS-ATTENTION COMPARISON ---")
                attn_cosine = F.cosine_similarity(
                    cross_attentions[0].flatten(), 
                    cross_attentions[1].flatten(), 
                    dim=0
                ).item()
                
                print(f"Cross-attention cosine similarity: {attn_cosine:.6f}")
                
                if attn_cosine > 0.99:
                    print("IDENTICAL: Cross-attention patterns are nearly identical!")
                elif attn_cosine > 0.9:
                    print("SIMILAR: Cross-attention patterns are very similar")
                else:
                    print("DIVERSE: Cross-attention patterns are different")
            
        except Exception as e:
            print(f"Cross-attention test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def test_generation_diversity(self, num_samples=4):
        """Test text generation diversity"""
        print("=" * 80)
        print("TESTING TEXT GENERATION DIVERSITY")
        print("=" * 80)
        
        try:
            model = self._load_model(self.latest_checkpoint)
            dataset = OrganAwareReportDataset(
                vis_processor=None,
                text_processor=None,
                vis_root=self.data_root,
                conc_info_path=self.conc_info_path,
                desc_info_path=self.desc_info_path,
                max_samples=100
            )
            
            print(f"Using checkpoint: {self.latest_checkpoint}")
            
            generated_reports = []
            case_ids = []
            
            for i in range(num_samples):
                sample = dataset[i * 25]
                case_ids.append(sample['case_id'])
                
                print(f"\n--- Patient {i+1}: {sample['case_id']} ---")
                
                image = sample["image"].unsqueeze(0)
                if torch.cuda.is_available():
                    image = image.cuda()
                
                with torch.no_grad():
                    generated = model.generate(
                        {"image": image, "prompt": "The chest CT scan shows"},
                        use_nucleus_sampling=False,
                        num_beams=3,
                        max_length=200,
                        min_length=50
                    )
                    
                    if isinstance(generated, list):
                        report = generated[0]
                    else:
                        report = generated
                    
                    print(f"Generated: {report}")
                    generated_reports.append(report)
            
            # Analyze diversity
            print(f"\n" + "="*60)
            print("DIVERSITY ANALYSIS")
            print("="*60)
            
            unique_reports = set(generated_reports)
            print(f"Unique reports: {len(unique_reports)}/{len(generated_reports)}")
            
            if len(unique_reports) == 1:
                print("IDENTICAL REPORTS: All reports are the same!")
            elif len(unique_reports) == len(generated_reports):
                print("FULLY DIVERSE: All reports are different!")
            else:
                print(f"PARTIALLY DIVERSE: {len(unique_reports)} unique reports")
            
            # Word analysis
            all_words = []
            for report in generated_reports:
                words = report.lower().split()
                all_words.extend(words)
            
            word_counts = Counter(all_words)
            common_words = word_counts.most_common(10)
            
            print(f"\nMost common words:")
            for word, count in common_words:
                print(f"  '{word}': {count} times")
            
        except Exception as e:
            print(f"Generation test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def test_checkpoints(self):
        """Analyze training checkpoints"""
        print("=" * 80)
        print("ANALYZING TRAINING CHECKPOINTS")
        print("=" * 80)
        
        try:
            # Find all checkpoints
            checkpoints = []
            output_dirs = [
                "/home/muhammedg/fvlm/lavis/output/BLIP/Organ_Aware_Report_Generation_Cross_Attention_1K",
                "/home/muhammedg/fvlm/lavis/output/BLIP/Organ_Aware_Report_Generation_Cross_Attention_Test",
                "/home/muhammedg/fvlm/lavis/output/BLIP/Organ_Aware_Report_Generation_Full_Dataset"
            ]
            
            for output_dir in output_dirs:
                if os.path.exists(output_dir):
                    print(f"\nFound training directory: {output_dir}")
                    for subdir in os.listdir(output_dir):
                        subdir_path = os.path.join(output_dir, subdir)
                        if os.path.isdir(subdir_path):
                            print(f"  Run: {subdir}")
                            
                            # Check log file
                            log_file = os.path.join(subdir_path, "log.txt")
                            if os.path.exists(log_file):
                                with open(log_file, 'r') as f:
                                    lines = f.readlines()
                                    if lines:
                                        print(f"    Log entries: {len(lines)}")
                                        # Show last few training stats
                                        for line in lines[-3:]:
                                            if "epoch" in line and "loss" in line:
                                                print(f"    {line.strip()}")
                            
                            # List checkpoints
                            checkpoint_files = [f for f in os.listdir(subdir_path) 
                                              if f.startswith("checkpoint_") and f.endswith(".pth")]
                            if checkpoint_files:
                                checkpoint_files.sort()
                                print(f"    Checkpoints: {', '.join(checkpoint_files)}")
                                
                                # Analyze latest checkpoint
                                latest_checkpoint = os.path.join(subdir_path, checkpoint_files[-1])
                                checkpoint = torch.load(latest_checkpoint, map_location="cpu")
                                
                                print(f"    Latest checkpoint info:")
                                if "epoch" in checkpoint:
                                    print(f"      Epoch: {checkpoint['epoch']}")
                                if "model" in checkpoint:
                                    model_keys = list(checkpoint["model"].keys())
                                    print(f"      Model parameters: {len(model_keys)}")
                                    
                                    # Check for cross-attention parameters
                                    cross_attn_keys = [k for k in model_keys if "crossattention" in k]
                                    if cross_attn_keys:
                                        print(f"      Cross-attention parameters: {len(cross_attn_keys)}")
                                    else:
                                        print(f"      No cross-attention parameters found")
            
            print(f"\nLatest checkpoint being used: {self.latest_checkpoint}")
            
        except Exception as e:
            print(f"Checkpoint analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    def test_segmentation(self, num_samples=3):
        """Test segmentation mask processing"""
        print("=" * 80)
        print("TESTING SEGMENTATION MASK PROCESSING")
        print("=" * 80)
        
        try:
            dataset = OrganAwareReportDataset(
                vis_processor=None,
                text_processor=None,
                vis_root=self.data_root,
                conc_info_path=self.conc_info_path,
                desc_info_path=self.desc_info_path,
                max_samples=100
            )
            
            for i in range(num_samples):
                sample = dataset[i * 30]
                print(f"\n--- Sample {i+1}: {sample['case_id']} ---")
                
                if 'seg' in sample:
                    seg_mask = sample['seg']
                    print(f"  Segmentation shape: {seg_mask.shape}")
                    
                    unique_labels = torch.unique(seg_mask)
                    print(f"  Unique labels: {unique_labels.tolist()}")
                    
                    # Count pixels per organ
                    for label in unique_labels:
                        if label > 0:  # Skip background
                            count = (seg_mask == label).sum().item()
                            organ_name = dataset.organs[int(label) - 1] if int(label) <= len(dataset.organs) else f"Unknown_{int(label)}"
                            print(f"    {organ_name} (label {int(label)}): {count} pixels")
                else:
                    print("  No segmentation mask found")
            
        except Exception as e:
            print(f"Segmentation test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def test_config(self):
        """Test model configuration"""
        print("=" * 80)
        print("TESTING MODEL CONFIGURATION")
        print("=" * 80)
        
        try:
            class Args:
                def __init__(self, config_path):
                    self.cfg_path = config_path
                    self.options = []
            
            args = Args(self.config_path)
            cfg = Config(args)
            
            print(f"Config file: {self.config_path}")
            print(f"Model architecture: {cfg.model_cfg.arch}")
            
            # Load model to check configuration
            model_cls = registry.get_model_class(cfg.model_cfg.arch)
            model = model_cls.from_config(cfg.model_cfg)
            
            print(f"\nModel components:")
            print(f"  Visual encoder: {type(model.visual_encoder).__name__}")
            print(f"  Text encoder: {type(model.text_encoder).__name__}")
            print(f"  Text decoder: {type(model.text_decoder).__name__}")
            
            print(f"\nText decoder config:")
            print(f"  is_decoder: {model.text_decoder.config.is_decoder}")
            print(f"  add_cross_attention: {model.text_decoder.config.add_cross_attention}")
            print(f"  encoder_width: {getattr(model.text_decoder.config, 'encoder_width', 'Not set')}")
            print(f"  hidden_size: {model.text_decoder.config.hidden_size}")
            print(f"  num_hidden_layers: {model.text_decoder.config.num_hidden_layers}")
            print(f"  num_attention_heads: {model.text_decoder.config.num_attention_heads}")
            
            # Check if layers have cross-attention
            cross_attn_layers = 0
            for i, layer in enumerate(model.text_decoder.bert.encoder.layer):
                if hasattr(layer, 'crossattention'):
                    cross_attn_layers += 1
            
            print(f"  Layers with cross-attention: {cross_attn_layers}/{len(model.text_decoder.bert.encoder.layer)}")
            
            # Parameter counts
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\nParameter counts:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
            print(f"  Frozen parameters: {total_params-trainable_params:,} ({100*(total_params-trainable_params)/total_params:.1f}%)")
            
        except Exception as e:
            print(f"Config test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_all_tests(self):
        """Run all available tests"""
        print("RUNNING COMPREHENSIVE DEBUG SUITE")
        print("=" * 80)
        
        tests = [
            ("Dataset", self.test_dataset),
            ("Vision Encoder", self.test_vision_encoder),
            ("Cross-Attention", self.test_cross_attention),
            ("Generation Diversity", self.test_generation_diversity),
            ("Checkpoints", self.test_checkpoints),
            ("Segmentation", self.test_segmentation),
            ("Configuration", self.test_config)
        ]
        
        for test_name, test_func in tests:
            print(f"\nRunning {test_name} test...")
            try:
                test_func()
                print(f"{test_name} test completed")
            except Exception as e:
                print(f"{test_name} test failed: {e}")
            
            print("\n" + "-" * 80)
        
        print("\nAll tests completed!")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Debug Suite for Organ-Aware Report Generation")
    parser.add_argument("--test", choices=[
        "dataset", "vision", "cross_attention", "generation", 
        "checkpoints", "segmentation", "config", "all"
    ], help="Specific test to run")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    debug_suite = ComprehensiveDebugSuite()
    
    if args.all or args.test == "all":
        debug_suite.run_all_tests()
    elif args.test == "dataset":
        debug_suite.test_dataset()
    elif args.test == "vision":
        debug_suite.test_vision_encoder()
    elif args.test == "cross_attention":
        debug_suite.test_cross_attention()
    elif args.test == "generation":
        debug_suite.test_generation_diversity()
    elif args.test == "checkpoints":
        debug_suite.test_checkpoints()
    elif args.test == "segmentation":
        debug_suite.test_segmentation()
    elif args.test == "config":
        debug_suite.test_config()
    else:
        print("Please specify a test to run. Use --help for options.")

if __name__ == "__main__":
    main()

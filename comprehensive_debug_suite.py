#!/usr/bin/env python3
"""
Comprehensive Debug Suite for Simplified Report Generation
==========================================================

This script consolidates all debugging functionality for the simplified 
report generation model. It includes tests for:

1. Dataset loading and preprocessing
2. Model architecture and configuration  
3. Vision encoder functionality
4. Cross-attention mechanism
5. Text generation and diversity
6. Training checkpoint analysis
7. Model training and evaluation

Usage:
    python comprehensive_debug_suite.py --test <test_name>
    python comprehensive_debug_suite.py --all

Available tests:
    - dataset: Test dataset loading and sample processing
    - vision: Test vision encoder outputs and diversity
    - cross_attention: Test cross-attention mechanism
    - generation: Test text generation diversity
    - checkpoints: Analyze training checkpoints
    - config: Test model configuration
    - training: Test training loop functionality
    - evaluation: Test evaluation and inference
    - all: Run all tests
"""

import torch
import torch.nn.functional as F
import sys
import os
import argparse
import numpy as np
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm

sys.path.append('/home/muhammedg/fvlm')

from lavis.common.config import Config
import argparse as argparse_module
from lavis.common.registry import registry
from lavis.datasets.datasets.report_generation_datasets import ReportGenerationDataset

class ComprehensiveDebugSuite:
    def __init__(self):
        self.config_path = "/home/muhammedg/fvlm/lavis/projects/blip/train/report_generation.yaml"
        self.data_root = "/home/muhammedg/fvlm/data/"
        self.train_csv = "/home/muhammedg/fvlm/data/dataset/radiology_text_reports/train_reports.csv"
        self.val_csv = "/home/muhammedg/fvlm/data/dataset/radiology_text_reports/validation_reports.csv"
        self.pretrained_checkpoint = "/home/muhammedg/fvlm/checkpoints/model.pth"
        
        # Find latest training checkpoint
        self.latest_checkpoint = self._find_latest_checkpoint()
        
    def _load_config(self):
        """Load configuration with proper args object"""
        mock_args = argparse_module.Namespace()
        mock_args.cfg_path = self.config_path
        mock_args.options = []
        return Config(mock_args)
    
    def _find_latest_checkpoint(self):
        """Find the most recent training checkpoint"""
        output_dirs = [
            "/home/muhammedg/fvlm/outputs/BLIP/Report_Generation",
            "/home/muhammedg/fvlm/outputs/BLIP/Report_Generation_Simple",
            "/home/muhammedg/fvlm/outputs/BLIP/Report_Generation_Simplified",
            "/home/muhammedg/fvlm/outputs/BLIP/Report_Generation_Test"
        ]
        
        latest_checkpoint = None
        latest_time = 0
        
        for output_dir in output_dirs:
            if not os.path.exists(output_dir):
                continue
                
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

    def test_dataset(self):
        """Test dataset loading and preprocessing"""
        print("\n" + "="*60)
        print("🗂️  TESTING DATASET LOADING AND PREPROCESSING")
        print("="*60)
        
        try:
            # Test dataset creation
            print("📊 Creating dataset...")
            
            # Create dataset configuration
            dataset_config = {
                'data_type': 'images',
                'build_info': {
                    'annotations': {
                        'train': {'storage': self.train_csv},
                        'val': {'storage': self.val_csv}
                    },
                    'images': {'storage': self.data_root}
                },
                'text_processor': {
                    'train': {'name': 'blip_caption'},
                    'eval': {'name': 'blip_caption'}
                }
            }
            
            # Test with limited samples
            os.environ['FVLM_TEST_MODE'] = 'true'
            
            dataset = ReportGenerationDataset(
                vis_processor=None,
                text_processor=None,
                vis_root=self.data_root,
                ann_paths=[self.train_csv]
            )
            
            print(f"✅ Dataset created successfully")
            print(f"📈 Dataset size: {len(dataset)} samples")
            
            # Test sample loading
            print("\n🔍 Testing sample loading...")
            sample = dataset[0]
            
            print(f"✅ Sample keys: {list(sample.keys())}")
            print(f"📝 Text input preview: {sample['text_input'][:100]}...")
            print(f"🖼️  Image shape: {sample['image'].shape}")
            print(f"🆔 Image ID: {sample['image_id']}")
            
            # Test multiple samples
            print("\n📊 Testing multiple samples...")
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"Sample {i}: Image {sample['image_id']}, Text length: {len(sample['text_input'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Dataset test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Clean up
            if 'FVLM_TEST_MODE' in os.environ:
                del os.environ['FVLM_TEST_MODE']

    def test_model_config(self):
        """Test model configuration and loading"""
        print("\n" + "="*60)
        print("⚙️  TESTING MODEL CONFIGURATION")
        print("="*60)
        
        try:
            # Load configuration
            print("📋 Loading configuration...")
            cfg = self._load_config()
            
            print(f"✅ Config loaded successfully")
            print(f"🏗️  Model architecture: {cfg.model_cfg.arch}")
            print(f"📏 Max text length: {cfg.model_cfg.max_txt_len}")
            print(f"🔧 Model type: {cfg.model_cfg.model_type}")
            
            # Test model creation
            print("\n🏗️  Creating model...")
            model_cls = registry.get_model_class(cfg.model_cfg.arch)
            model = model_cls.from_config(cfg.model_cfg)
            
            print(f"✅ Model created successfully")
            print(f"🧠 Model class: {type(model).__name__}")
            
            # Test model components
            print(f"🔍 Vision encoder: {type(model.visual_encoder).__name__}")
            print(f"📝 Text decoder: {type(model.text_decoder).__name__}")
            print(f"🔤 Tokenizer: {type(model.tokenizer).__name__}")
            print(f"📊 Vocab size: {len(model.tokenizer)}")
            
            # Test model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"📈 Total parameters: {total_params:,}")
            print(f"🎯 Trainable parameters: {trainable_params:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model config test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_vision_encoder(self):
        """Test vision encoder functionality"""
        print("\n" + "="*60)
        print("👁️  TESTING VISION ENCODER")
        print("="*60)
        
        try:
            # Load model
            cfg = self._load_config()
            model_cls = registry.get_model_class(cfg.model_cfg.arch)
            model = model_cls.from_config(cfg.model_cfg)
            model.eval()
            
            print("✅ Model loaded for vision testing")
            
            # Create dummy 3D medical images
            batch_size = 2
            dummy_images = torch.randn(batch_size, 1, 112, 256, 352)  # Medical CT format
            
            print(f"🖼️  Testing with dummy images: {dummy_images.shape}")
            
            # Test vision encoder
            with torch.no_grad():
                image_embeds, hidden_embeds = model.visual_encoder(dummy_images)
            
            print(f"✅ Vision encoder forward pass successful")
            print(f"📊 Image embeddings shape: {image_embeds.shape}")
            print(f"🔍 Hidden embeddings count: {len(hidden_embeds) if hidden_embeds else 0}")
            
            # Test embedding diversity
            print("\n🎲 Testing embedding diversity...")
            embed1 = image_embeds[0].flatten()
            embed2 = image_embeds[1].flatten()
            
            cosine_sim = F.cosine_similarity(embed1.unsqueeze(0), embed2.unsqueeze(0))
            print(f"📏 Cosine similarity between embeddings: {cosine_sim.item():.4f}")
            
            if cosine_sim.item() < 0.99:
                print("✅ Embeddings show good diversity")
            else:
                print("⚠️  Embeddings are very similar (might indicate issues)")
            
            return True
            
        except Exception as e:
            print(f"❌ Vision encoder test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_cross_attention(self):
        """Test cross-attention mechanism"""
        print("\n" + "="*60)
        print("🔗 TESTING CROSS-ATTENTION MECHANISM")
        print("="*60)
        
        try:
            # Load model
            cfg = self._load_config()
            model_cls = registry.get_model_class(cfg.model_cfg.arch)
            model = model_cls.from_config(cfg.model_cfg)
            model.eval()
            
            print("✅ Model loaded for cross-attention testing")
            
            # Create test data
            batch_size = 1
            dummy_images = torch.randn(batch_size, 1, 112, 256, 352)
            test_text = ["The chest radiograph shows normal findings."]
            
            # Get image embeddings
            with torch.no_grad():
                image_embeds, _ = model.visual_encoder(dummy_images)
            
            print(f"🖼️  Image embeddings: {image_embeds.shape}")
            
            # Test text tokenization
            text_tokens = model.tokenizer(
                test_text,
                padding="longest",
                truncation=True,
                max_length=100,
                return_tensors="pt"
            )
            
            print(f"📝 Text tokens: {text_tokens.input_ids.shape}")
            
            # Test cross-attention in decoder
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)
            
            with torch.no_grad():
                decoder_output = model.text_decoder(
                    text_tokens.input_ids,
                    attention_mask=text_tokens.attention_mask,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True
                )
            
            print(f"✅ Cross-attention forward pass successful")
            print(f"📊 Decoder output shape: {decoder_output.last_hidden_state.shape}")
            
            # Test attention weights if available
            if hasattr(decoder_output, 'cross_attentions') and decoder_output.cross_attentions:
                print(f"🔍 Cross-attention weights available: {len(decoder_output.cross_attentions)} layers")
                attention_weights = decoder_output.cross_attentions[0]
                print(f"📏 Attention shape: {attention_weights.shape}")
            else:
                print("ℹ️  Cross-attention weights not returned (normal for some configurations)")
            
            return True
            
        except Exception as e:
            print(f"❌ Cross-attention test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_text_generation(self):
        """Test text generation functionality"""
        print("\n" + "="*60)
        print("📝 TESTING TEXT GENERATION")
        print("="*60)
        
        try:
            # Load model
            cfg = self._load_config()
            model_cls = registry.get_model_class(cfg.model_cfg.arch)
            model = model_cls.from_config(cfg.model_cfg)
            model.eval()
            
            print("✅ Model loaded for generation testing")
            
            # Load checkpoint if available
            if self.latest_checkpoint and os.path.exists(self.latest_checkpoint):
                print(f"📂 Loading checkpoint: {self.latest_checkpoint}")
                checkpoint = torch.load(self.latest_checkpoint, map_location='cpu')
                model.load_state_dict(checkpoint['model'], strict=False)
                print("✅ Checkpoint loaded")
            else:
                print("⚠️  No checkpoint found, using pretrained weights")
            
            # Create test samples
            batch_size = 2
            dummy_images = torch.randn(batch_size, 1, 112, 256, 352)
            
            samples = {
                "image": dummy_images,
                "image_id": torch.tensor([1, 2])
            }
            
            print(f"🖼️  Testing generation with {batch_size} images")
            
            # Test different generation strategies
            generation_configs = [
                {"use_nucleus_sampling": True, "num_beams": 1, "top_p": 0.9, "repetition_penalty": 1.2},
                {"use_nucleus_sampling": False, "num_beams": 3, "repetition_penalty": 1.1},
                {"use_nucleus_sampling": True, "num_beams": 1, "top_p": 0.7, "repetition_penalty": 1.5}
            ]
            
            for i, gen_config in enumerate(generation_configs):
                print(f"\n🎯 Generation strategy {i+1}: {gen_config}")
                
                with torch.no_grad():
                    generated_texts = model.generate(
                        samples,
                        max_length=100,
                        min_length=10,
                        **gen_config
                    )
                
                print(f"✅ Generated {len(generated_texts)} texts")
                for j, text in enumerate(generated_texts):
                    print(f"  Sample {j+1}: {text[:80]}...")
                
                # Analyze diversity
                if len(generated_texts) > 1:
                    unique_texts = len(set(generated_texts))
                    print(f"🎲 Unique texts: {unique_texts}/{len(generated_texts)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Text generation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_training_functionality(self):
        """Test training loop functionality"""
        print("\n" + "="*60)
        print("🎓 TESTING TRAINING FUNCTIONALITY")
        print("="*60)
        
        try:
            # Load model and config
            cfg = self._load_config()
            model_cls = registry.get_model_class(cfg.model_cfg.arch)
            model = model_cls.from_config(cfg.model_cfg)
            model.train()
            
            print("✅ Model loaded for training testing")
            
            # Create training sample
            batch_size = 2
            dummy_images = torch.randn(batch_size, 1, 112, 256, 352)
            dummy_texts = [
                "The chest radiograph shows normal cardiac and pulmonary findings.",
                "There is evidence of pneumonia in the right lower lobe."
            ]
            
            samples = {
                "image": dummy_images,
                "text_input": dummy_texts,
                "mode": "generation"
            }
            
            print(f"🎯 Testing forward pass with batch size {batch_size}")
            
            # Test forward pass
            with torch.no_grad():  # Just testing, not actually training
                output = model(samples)
            
            print(f"✅ Forward pass successful")
            print(f"📊 Output keys: {list(output.keys())}")
            print(f"📉 Loss: {output['loss'].item():.4f}")
            
            # Test that loss is reasonable
            loss_value = output['loss'].item()
            if 0.1 < loss_value < 20.0:
                print("✅ Loss value is in reasonable range")
            else:
                print(f"⚠️  Loss value might be unusual: {loss_value}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training functionality test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_checkpoints(self):
        """Analyze training checkpoints"""
        print("\n" + "="*60)
        print("💾 TESTING CHECKPOINT ANALYSIS")
        print("="*60)
        
        try:
            # Find all checkpoints
            checkpoints = []
            output_dirs = [
                "/home/muhammedg/fvlm/outputs/BLIP/Report_Generation",
                "/home/muhammedg/fvlm/outputs/BLIP/Report_Generation_Simple",
                "/home/muhammedg/fvlm/outputs/BLIP/Report_Generation_Simplified",
                "/home/muhammedg/fvlm/outputs/BLIP/Report_Generation_Test"
            ]
            
            for output_dir in output_dirs:
                if os.path.exists(output_dir):
                    print(f"\n📁 Checking directory: {output_dir}")
                    
                    for subdir in os.listdir(output_dir):
                        subdir_path = os.path.join(output_dir, subdir)
                        if os.path.isdir(subdir_path):
                            # List checkpoints
                            checkpoint_files = [f for f in os.listdir(subdir_path) 
                                              if f.startswith("checkpoint_") and f.endswith(".pth")]
                            if checkpoint_files:
                                checkpoint_files.sort()
                                print(f"  📂 {subdir}: {len(checkpoint_files)} checkpoints")
                                for ckpt in checkpoint_files:
                                    ckpt_path = os.path.join(subdir_path, ckpt)
                                    size_mb = os.path.getsize(ckpt_path) / (1024*1024)
                                    checkpoints.append((ckpt_path, size_mb))
                                    print(f"    💾 {ckpt}: {size_mb:.1f} MB")
                            
                            # Check for log files
                            log_file = os.path.join(subdir_path, "log.txt")
                            if os.path.exists(log_file):
                                print(f"  📋 Log file found: {os.path.getsize(log_file)} bytes")
            
            if not checkpoints:
                print("⚠️  No checkpoints found")
                return False
            
            print(f"\n✅ Found {len(checkpoints)} total checkpoints")
            
            # Test loading latest checkpoint
            if self.latest_checkpoint:
                print(f"\n🔍 Testing latest checkpoint: {self.latest_checkpoint}")
                
                try:
                    checkpoint = torch.load(self.latest_checkpoint, map_location='cpu')
                    print(f"✅ Checkpoint loaded successfully")
                    print(f"📊 Checkpoint keys: {list(checkpoint.keys())}")
                    
                    if 'model' in checkpoint:
                        model_state = checkpoint['model']
                        print(f"🧠 Model state dict keys: {len(model_state)} parameters")
                        
                        # Check for key components
                        vision_keys = [k for k in model_state.keys() if 'visual_encoder' in k]
                        decoder_keys = [k for k in model_state.keys() if 'text_decoder' in k]
                        
                        print(f"👁️  Vision encoder parameters: {len(vision_keys)}")
                        print(f"📝 Text decoder parameters: {len(decoder_keys)}")
                    
                except Exception as e:
                    print(f"❌ Failed to load checkpoint: {e}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Checkpoint analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_evaluation(self):
        """Test evaluation functionality"""
        print("\n" + "="*60)
        print("📊 TESTING EVALUATION FUNCTIONALITY")
        print("="*60)
        
        try:
            # Test evaluation script components
            from lavis.tasks.report_generation import ReportGenerationTask
            
            print("✅ Report generation task imported")
            
            # Create task
            task_config = {
                'num_beams': 3,
                'max_length': 100,
                'min_length': 10,
                'evaluate': False,
                'cuda_enabled': False
            }
            
            task = ReportGenerationTask(**task_config)
            print("✅ Task created successfully")
            
            # Test model loading
            cfg = self._load_config()
            model = task.build_model(cfg)
            model.eval()
            
            print("✅ Model built through task")
            
            # Create validation samples
            batch_size = 2
            samples = {
                "image": torch.randn(batch_size, 1, 112, 256, 352),
                "image_id": torch.tensor([1, 2])
            }
            
            # Test validation step
            results = task.valid_step(model, samples)
            
            print(f"✅ Validation step successful")
            print(f"📊 Generated {len(results)} results")
            
            for i, result in enumerate(results):
                print(f"  Result {i+1}: {result['caption'][:50]}...")
            
            # Test after_evaluation
            val_result = results
            eval_metrics = task.after_evaluation(val_result, "val", epoch=1)
            
            print(f"✅ After evaluation successful")
            print(f"📈 Metrics: {eval_metrics}")
            
            return True
            
        except Exception as e:
            print(f"❌ Evaluation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_tests(self):
        """Run all available tests"""
        print("🚀 COMPREHENSIVE DEBUG SUITE - SIMPLIFIED REPORT GENERATION")
        print("=" * 80)
        
        tests = [
            ("Dataset Loading", self.test_dataset),
            ("Model Configuration", self.test_model_config),
            ("Vision Encoder", self.test_vision_encoder),
            ("Cross-Attention", self.test_cross_attention),
            ("Text Generation", self.test_text_generation),
            ("Training Functionality", self.test_training_functionality),
            ("Checkpoint Analysis", self.test_checkpoints),
            ("Evaluation", self.test_evaluation)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                results[test_name] = False
        
        # Summary
        print("\n" + "="*80)
        print("📋 TEST SUMMARY")
        print("="*80)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! The system is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the output above for details.")
        
        return passed == total

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Debug Suite for Simplified Report Generation")
    parser.add_argument("--test", type=str, choices=[
        "dataset", "config", "vision", "cross_attention", "generation", 
        "training", "checkpoints", "evaluation", "all"
    ], default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    suite = ComprehensiveDebugSuite()
    
    if args.test == "all":
        success = suite.run_all_tests()
    else:
        test_methods = {
            "dataset": suite.test_dataset,
            "config": suite.test_model_config,
            "vision": suite.test_vision_encoder,
            "cross_attention": suite.test_cross_attention,
            "generation": suite.test_text_generation,
            "training": suite.test_training_functionality,
            "checkpoints": suite.test_checkpoints,
            "evaluation": suite.test_evaluation
        }
        
        success = test_methods[args.test]()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
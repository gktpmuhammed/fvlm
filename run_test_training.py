#!/usr/bin/env python3
"""
Simple test script to run training with limited samples
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Monkey patch the dataset to enable test mode
def patch_dataset_for_testing():
    from lavis.datasets.datasets.report_generation_datasets import ReportGenerationDataset
    
    # Store original init
    original_init = ReportGenerationDataset.__init__
    
    def patched_init(self, vis_processor, text_processor, vis_root, ann_paths):
        # Call original init
        original_init(self, vis_processor, text_processor, vis_root, ann_paths)
        # Enable test mode
        self.test_mode = True
        print(f"🧪 Test mode enabled - limiting dataset size")
    
    # Apply patch
    ReportGenerationDataset.__init__ = patched_init

if __name__ == "__main__":
    print("🚀 Starting test training with limited samples...")
    
    # Set environment variable to enable test mode
    os.environ['FVLM_TEST_MODE'] = 'true'
    
    # Apply dataset patch
    patch_dataset_for_testing()
    
    # Import and run training
    from train import main
    
    # Set config path as command line argument
    sys.argv = ["train.py", "--cfg-path", "lavis/projects/blip/train/test_report_generation.yaml"]
    
    try:
        main()
        print("✅ Test training completed!")
    except Exception as e:
        print(f"❌ Test training failed: {e}")
        raise

#!/usr/bin/env python3
"""
Test script for checkpoint saving functionality
"""
import os
import sys
import torch

def test_checkpoint_saving():
    """Test training with minimal data to verify checkpoint saving works"""
    
    # Set environment variable to enable test mode
    os.environ['FVLM_TEST_MODE'] = 'true'
    
    # Set up command line arguments for the training script
    sys.argv = ["train.py", "--cfg-path", "lavis/projects/blip/train/test_report_generation.yaml"]
    
    # Import and run training
    from train import main
    
    print("🚀 Starting checkpoint saving test with limited samples...")
    print("📝 Configuration: 10 epochs, save every 2 iterations")
    
    # Run training
    try:
        main()
        print("✅ Test completed successfully!")
        
        # Check if checkpoints were saved
        output_dir = "output/BLIP/Report_Generation_Test"
        print(f"🔍 Checking for checkpoints in: {output_dir}")
        
        if os.path.exists(output_dir):
            # Look for checkpoint files recursively
            checkpoint_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.pth'):
                        checkpoint_files.append(os.path.join(root, file))
            
            if checkpoint_files:
                print(f"✅ Checkpoints saved: {len(checkpoint_files)} files")
                for cp in checkpoint_files:
                    print(f"  📁 {cp}")
            else:
                print("⚠️  No checkpoint files found")
                # List all files in output directory
                print("📂 Files in output directory:")
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        print(f"  📄 {os.path.join(root, file)}")
        else:
            print(f"❌ Output directory does not exist: {output_dir}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    test_checkpoint_saving()

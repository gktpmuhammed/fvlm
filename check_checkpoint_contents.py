#!/usr/bin/env python3
"""
Script to check the contents of a saved checkpoint
"""
import torch
import os

def check_checkpoint_contents(checkpoint_path):
    """Check what's inside a checkpoint file"""
    print(f"🔍 Examining checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"📦 Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        print(f"\n🧠 Model state_dict contains {len(model_state)} parameters")
        
        # Check for different components
        visual_encoder_keys = [k for k in model_state.keys() if k.startswith('visual_encoder')]
        text_encoder_keys = [k for k in model_state.keys() if k.startswith('text_encoder')]
        text_decoder_keys = [k for k in model_state.keys() if k.startswith('text_decoder')]
        
        print(f"\n📊 Component breakdown:")
        print(f"  🖼️  Visual encoder parameters: {len(visual_encoder_keys)}")
        print(f"  📝 Text encoder parameters: {len(text_encoder_keys)}")
        print(f"  🔤 Text decoder parameters: {len(text_decoder_keys)}")
        
        # Show some example keys for each component
        if visual_encoder_keys:
            print(f"\n🖼️  Visual encoder examples:")
            for key in visual_encoder_keys[:3]:
                print(f"    {key}: {model_state[key].shape}")
        
        if text_encoder_keys:
            print(f"\n📝 Text encoder examples:")
            for key in text_encoder_keys[:3]:
                print(f"    {key}: {model_state[key].shape}")
        
        if text_decoder_keys:
            print(f"\n🔤 Text decoder examples:")
            for key in text_decoder_keys[:5]:
                print(f"    {key}: {model_state[key].shape}")
        else:
            print(f"\n❌ No text decoder parameters found!")
            
        # Check for specific decoder components
        decoder_components = {
            'embeddings': [k for k in text_decoder_keys if 'embeddings' in k],
            'encoder_layers': [k for k in text_decoder_keys if 'encoder.layer' in k],
            'cls_predictions': [k for k in text_decoder_keys if 'cls.predictions' in k]
        }
        
        print(f"\n🔍 Text decoder component breakdown:")
        for component, keys in decoder_components.items():
            print(f"  {component}: {len(keys)} parameters")
    
    # Check other checkpoint info
    if 'epoch' in checkpoint:
        print(f"\n📅 Epoch: {checkpoint['epoch']}")
    if 'optimizer' in checkpoint:
        print(f"🎯 Optimizer state saved: Yes")
    if 'lr_scheduler' in checkpoint:
        print(f"📈 LR scheduler state saved: Yes")

if __name__ == "__main__":
    # Check the latest checkpoint
    checkpoint_path = "./lavis/output/BLIP/Report_Generation_Test/20250927011/checkpoint_9.pth"
    check_checkpoint_contents(checkpoint_path)

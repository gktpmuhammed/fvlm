print("Starting script...")

import sys
import os
import torch
import pandas as pd
from unittest.mock import MagicMock
from transformers import AutoTokenizer

# Mock monai before importing train
sys.modules['monai'] = MagicMock()
sys.modules['monai.transforms'] = MagicMock()
sys.modules['wandb'] = MagicMock()
sys.modules['medical_vlm'] = MagicMock()

# Now import train
from train import OnePassOrganDataset
from disease_classifier import DISEASE_CONFIG

# Valid paths
csv_file = '/home/muhammedg/fvlm/data_sym/image_first_dataset.csv'
json_file = '/home/muhammedg/fvlm/data_sym/combined_desc_conc.json'

def verify():
    print("Mocking dependencies and initializing Dataset...")
    
    # Mocking tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    except:
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.return_value = {'input_ids': torch.tensor([[1,2]]), 'attention_mask': torch.tensor([[1,1]])}

    # Mock transform
    transform = MagicMock()
    transform.return_value = {
        'image': torch.zeros((1, 10, 10)), 
        'mask': torch.zeros((1, 10, 10))
    }
    
    ds = OnePassOrganDataset(
        csv_file=csv_file,
        json_file=json_file,
        tokenizer=tokenizer,
        transform=transform,
        split='training',
        subset_size=10
    )
    
    print(f"Dataset length: {len(ds)}")
    
    # Check disease_lookup
    print("Checking disease_lookup...")
    if ds.disease_lookup is None or len(ds.disease_lookup) == 0:
        print("ERROR: disease_lookup is empty!")
    else:
        print(f"disease_lookup loaded. Size: {len(ds.disease_lookup)}")
        print("Sample Index:", ds.disease_lookup.index[0])
        print("Sample Columns:", ds.disease_lookup.columns.tolist()[:5])

    
    print("Simulating __getitem__ call...")
    try:
        item = ds[0]
        print("Success! Item 0 loaded.")
        print("Keys:", item.keys())
        if 'disease_labels' in item:
            print("Disease Labels found.")
            for org, tens in item['disease_labels'].items():
                print(f"  {org}: shape {tens.shape}, values {tens.tolist()}")
        else:
            print("ERROR: disease_labels missing from item.")
            
    except Exception as e:
        print(f"ERROR calling __getitem__: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()

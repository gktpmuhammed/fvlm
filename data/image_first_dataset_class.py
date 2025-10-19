
"""
Example Dataset class using image-first approach
"""

import torch
from torch.utils.data import Dataset
import json
import pandas as pd
from monai import transforms
import os

class ImageFirstMedicalDataset(Dataset):
    def __init__(self, json_path, split='train', transform=None):
        """
        Dataset that loads images first, then maps to reports
        
        Args:
            json_path: Path to image_first_dataset_split.json
            split: 'train' or 'validation'
            transform: Image transforms
        """
        
        # Load the image-first dataset
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.samples = data[split]
        self.transform = transform or self._default_transforms()
        
        print(f"Loaded {len(self.samples)} {split} samples")
        
        # Count patients and scans
        patients = set(item['patient_id'] for item in self.samples)
        print(f"  - {len(patients)} unique patients")
        print(f"  - {len(self.samples)} total scans")
    
    def _default_transforms(self):
        return transforms.Compose([
            transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
            transforms.Transposed(keys=["image"], indices=(0, 3, 2, 1)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1150, a_max=350,
                b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.SpatialPadd(
                keys=["image"], spatial_size=(112, 256, 352),
                mode="constant", constant_values=0
            ),
            transforms.CenterSpatialCropd(
                keys=["image"], roi_size=(112, 256, 352)
            ),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        try:
            data = self.transform({"image": sample['image_path']})
            pixel_values = data["image"]
        except Exception as e:
            print(f"Error loading {sample['image_path']}: {e}")
            # Return dummy image if loading fails
            pixel_values = torch.randn(1, 112, 256, 352)
        
        return {
            'pixel_values': pixel_values,
            'text': sample['combined_report'],
            'patient_id': sample['patient_id'],
            'scan_id': sample['scan_id'],
            'image_path': sample['image_path']
        }

# Usage example:
if __name__ == "__main__":
    # Create datasets
    train_dataset = ImageFirstMedicalDataset(
        json_path="/home/muhammedg/fvlm/image_first_dataset_split.json",
        split='train'
    )
    
    val_dataset = ImageFirstMedicalDataset(
        json_path="/home/muhammedg/fvlm/image_first_dataset_split.json", 
        split='validation'
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Test loading a sample
    sample = train_dataset[0]
    print(f"Sample image shape: {sample['pixel_values'].shape}")
    print(f"Sample text: {sample['text'][:100]}...")

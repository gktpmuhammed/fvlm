import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, Transposed, EnsureChannelFirstd

# --- UPDATED CONFIGURATION ---
ALL_TARGET_KEYS = [
    'lung', 'heart', 'aorta', 'esophagus', 'trachea', 'rib',
    'liver', 'gallbladder', 'stomach', 'pancreas', 'spleen', 'kidney'
]

def get_organ_ids_for_key(report_key):
    key = report_key.lower().strip()
    
    # Thorax / Vessels / Airway
    if "lung" in key: return [10, 11, 12, 13, 14] 
    if "heart" in key: return [51, 61] 
    if "aorta" in key: return [52]
    if "esophagus" in key: return [15]
    if "trachea" in key: return [16]
    if "rib" in key: return list(range(92, 116)) 

    # Abdomen
    if "liver" in key: return [5]
    if "gallbladder" in key: return [4]
    if "stomach" in key: return [6]
    if "pancreas" in key: return [7]
    if "spleen" in key: return [1]
    if "kidney" in key: return [2, 3] 

    return []

def build_transforms():
    return Compose([
        LoadImaged(keys=['image', 'mask'], reader='ITKReader', allow_missing_keys=True),
        EnsureChannelFirstd(keys=['image', 'mask'], allow_missing_keys=True),
        Transposed(keys=['image', 'mask'], indices=(0, 3, 2, 1), allow_missing_keys=True),
        ScaleIntensityRanged(keys=['image'], a_min=-1150, a_max=350, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=['image', 'mask'], spatial_size=(112, 256, 352), mode='constant'),
        CenterSpatialCropd(keys=['image', 'mask'], roi_size=(112, 256, 352)),
    ])

class UnifiedMedicalDataset(Dataset):
    def __init__(self, csv_file, json_file, tokenizer, transform, mode='global', split='training', max_length=150, subset_size=None):
        self.mode = mode
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.target_keys = ALL_TARGET_KEYS
        
        # Load CSV
        df = pd.read_csv(csv_file)
        self.df = df[df['split'] == split].reset_index(drop=True)
        if subset_size: self.df = self.df.head(subset_size)
        
        # Load JSON
        self.reports_json = {}
        if json_file and os.path.exists(json_file):
            with open(json_file, 'r') as f: self.reports_json = json.load(f)

        # Prepare Samples
        self.samples = []
        
        if self.mode == 'masked_single':
            # Flatten: Patient -> 12 Organs
            for _, row in self.df.iterrows():
                pid = self._get_pid(row['image_path'])
                p_data = self._get_patient_data(pid)
                
                # Iterate ONLY the 12 target keys
                for key in self.target_keys:
                    text = p_data.get(key, "").strip()
                    if len(text) > 3:
                        self.samples.append({
                            'image_path': row['image_path'],
                            'mask_path': row['image_path'].replace('images', 'masks'),
                            'organ_key': key,
                            'text': text
                        })
        else:
            # 1 Row = 1 Sample (Global or Parallel)
            for _, row in self.df.iterrows():
                self.samples.append({
                    'image_path': row['image_path'],
                    'mask_path': row['image_path'].replace('images', 'masks'),
                    'findings': row.get('findings', ''),
                    'impressions': row.get('impressions', '')
                })

    def _get_pid(self, path):
        fname = os.path.basename(path)
        base = fname.replace('.nii.gz', '').replace('.nii', '')
        return base.rsplit('_', 1)[0] if '_' in base else base

    def _get_patient_data(self, pid):
        if pid in self.reports_json: return self.reports_json[pid]
        # Fallback for ID variations
        if '_' in pid:
            short = pid.rsplit('_', 1)[0]
            if short in self.reports_json: return self.reports_json[short]
        return {}

    def _load_tensors(self, img_path, mask_path):
        data = self.transform({'image': img_path, 'mask': mask_path})
        
        img = data['image']
        img = img.as_tensor().float() if hasattr(img, 'as_tensor') else torch.from_numpy(img).float()
        
        mask = None
        if 'mask' in data:
            m = data['mask']
            mask = m.as_tensor() if hasattr(m, 'as_tensor') else torch.from_numpy(m)
        return img, mask

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            img, full_mask = self._load_tensors(item['image_path'], item.get('mask_path'))

            # --- GLOBAL MODE ---
            if self.mode == 'global':
                text = f"{item['findings']} {item['impressions']}"
                labels = self._tokenize(text)
                return {'pixel_values': img, 'labels': labels}

            # --- MASKED SINGLE MODE ---
            elif self.mode == 'masked_single':
                target_ids = get_organ_ids_for_key(item['organ_key'])
                binary_mask = torch.zeros_like(full_mask)
                for tid in target_ids: binary_mask[full_mask == tid] = 1.0
                
                text = f"Describe {item['organ_key']}: {item['text']}"
                labels = self._tokenize(text)
                return {'pixel_values': img, 'pixel_mask': binary_mask.float(), 'labels': labels}

            # --- PARALLEL MODE (ROI / ATTENTION) ---
            elif self.mode == 'parallel':
                pid = self._get_pid(item['image_path'])
                p_data = self._get_patient_data(pid)
                
                mask_stack = []
                label_stack = []
                
                for key in self.target_keys:
                    # Mask Logic
                    target_ids = get_organ_ids_for_key(key)
                    m = torch.zeros_like(full_mask)
                    if len(target_ids) > 0:
                        for tid in target_ids: m[full_mask == tid] = 1.0
                    mask_stack.append(m)
                    
                    # Text Logic
                    text = p_data.get(key, "").strip()
                    prompt = f"Describe {key}: "
                    
                    if len(text) > 3:
                        label_stack.append(self._tokenize(prompt + text))
                    else:
                        # Pad with -100 if no text for this organ
                        label_stack.append(torch.full((self.max_length,), -100, dtype=torch.long))
                
                return {
                    'pixel_values': img,
                    'organ_masks': torch.stack(mask_stack).float(),
                    'labels': torch.stack(label_stack)
                }
                
        except Exception as e:
            return None

    def _tokenize(self, text):
        enc = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        ids = enc['input_ids'].squeeze(0)
        ids[ids == self.tokenizer.pad_token_id] = -100
        return ids

@dataclass
class ModularCollator:
    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features: raise ValueError("Empty batch")
        
        batch = {}
        batch['pixel_values'] = torch.stack([f['pixel_values'] for f in features])
        batch['labels'] = torch.stack([f['labels'] for f in features])
        
        if 'pixel_mask' in features[0]:
            batch['pixel_mask'] = torch.stack([f['pixel_mask'] for f in features])
        if 'organ_masks' in features[0]:
            batch['organ_masks'] = torch.stack([f['organ_masks'] for f in features])
            
        return batch
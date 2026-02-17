#!/usr/bin/env python3
import sys
import os
import logging
import json
import pandas as pd
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, Transposed, EnsureChannelFirstd
import traceback

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, os.path.join(parent_dir, "../"))
    sys.path.append(parent_dir)

from medical_vlm import MedicalVLM
import wandb

logger = logging.getLogger(__name__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Moved to main()
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "thesis")
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ALL_TARGET_KEYS = [
    'lung', 'heart', 'esophagus', 
    'liver', 'gallbladder', 'stomach', 'pancreas', 'spleen', 'kidney',
    'aorta', 'trachea', 'rib'
]

def get_organ_ids_for_key(report_key):
    # Same masking logic as before
    key = report_key.lower().strip()
    if "lung" in key: return [10, 11, 12, 13, 14] 
    if "heart" in key: return [51, 61] 
    if "aorta" in key: return [52]
    if "esophagus" in key: return [15]
    if "trachea" in key: return [16]
    if "rib" in key: return list(range(92, 116)) 
    if "liver" in key: return [5]
    if "gallbladder" in key: return [4]
    if "stomach" in key: return [6]
    if "pancreas" in key: return [7]
    if "spleen" in key: return [1]
    if "kidney" in key: return [2, 3] 
    return []

def build_transforms():
    return Compose([
        LoadImaged(keys=['image', 'mask'], reader='ITKReader', image_only=True),
        EnsureChannelFirstd(keys=['image', 'mask']),
        Transposed(keys=['image', 'mask'], indices=(0, 3, 2, 1)),
        ScaleIntensityRanged(keys=['image'], a_min=-1150, a_max=350, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=['image', 'mask'], spatial_size=(112, 256, 352), mode='constant', constant_values=0),
        CenterSpatialCropd(keys=['image', 'mask'], roi_size=(112, 256, 352)),
    ])

@dataclass
class OrganCollator:
    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features: raise ValueError("Empty batch")
        
        pixel_values = torch.stack([f['pixel_values'] for f in features])
        organ_masks = torch.stack([f['organ_masks'] for f in features])
        
        # New: Stack input_ids and attention_mask
        input_ids = torch.stack([f['input_ids'] for f in features])
        attention_mask = torch.stack([f['attention_mask'] for f in features])
        labels = torch.stack([f['labels'] for f in features])
        
        return {
            'pixel_values': pixel_values, 
            'organ_masks': organ_masks,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class OnePassOrganDataset(Dataset):
    def __init__(self, csv_file, json_file, tokenizer, transform, max_length=128, subset_size=None, split='training'):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        if subset_size: self.df = self.df.head(subset_size)
        
        with open(json_file, 'r') as f: 
            self.reports_json = json.load(f)

        self.tokenizer = tokenizer
        # Enforce right padding for correct masking logic
        self.tokenizer.padding_side = 'right'
        self.transform = transform
        self.max_length = max_length
        self.target_keys = ALL_TARGET_KEYS
        
        # Filter patients
        self.valid_patients = []
        for _, row in self.df.iterrows():
            fname = os.path.basename(row['image_path'])
            base_id = fname.replace('.nii.gz', '').replace('.nii', '')
            if base_id in self.reports_json or (len(base_id.split('_')) > 1 and base_id.rsplit('_', 1)[0] in self.reports_json):
                self.valid_patients.append(row)

    def __len__(self): return len(self.valid_patients)

    def apply_chat_template(self, organ_name, findings):
        """
        Formats prompt for Gemma Instruct.
        Format: <start_of_turn>user\nPROMPT<end_of_turn>\n<start_of_turn>model\nRESPONSE
        """
        # The visual token is implicitly prepended in the model's forward pass.
        # So the text starts after the image.
        prompt = f"<start_of_turn>user\nAnalyze the specific image feature. Describe the findings for the {organ_name}.<end_of_turn>\n<start_of_turn>model\n"
        full_text = prompt + findings + "<eos>"
        return full_text, prompt

    @staticmethod
    def _find_subsequence(sequence, subsequence):
        max_start = len(sequence) - len(subsequence)
        for i in range(max_start + 1):
            if sequence[i:i+len(subsequence)] == subsequence:
                return i
        return None

    def __getitem__(self, idx):
        row = self.valid_patients[idx]
        try:
            # Load Images
            # Fix potential path path issue
            image_path = row['image_path'].replace('/data_sym_sym/', '/data_sym/')
            mask_path = image_path.replace('images', 'masks')
            data = self.transform({'image': image_path, 'mask': mask_path})
            
            img_tensor = data['image'].as_tensor().float() if hasattr(data['image'], 'as_tensor') else torch.tensor(data['image']).float()
            mask_tensor = data['mask'].as_tensor() if hasattr(data['mask'], 'as_tensor') else torch.tensor(data['mask'])
            
            # Identify Patient ID
            fname = os.path.basename(row['image_path'])
            base_id = fname.replace('.nii.gz', '').replace('.nii', '')
            pid = base_id if base_id in self.reports_json else base_id.rsplit('_', 1)[0]
            patient_data = self.reports_json.get(pid, {})

            mask_stack, input_ids_stack, att_stack, label_stack = [], [], [], []

            for key in self.target_keys:
                # 1. Mask
                tids = get_organ_ids_for_key(key)
                m = torch.zeros_like(mask_tensor)
                for t in tids: m[mask_tensor == t] = 1.0
                mask_stack.append(m)
                
                # 2. Text
                text = patient_data.get(key, "").strip()
                if len(text) < 3: 
                    # If empty, teach model to say "No findings." or mask loss
                    text = "No significant findings." 
                
                # 3. Tokenize with Chat Template
                full_text, prompt_text = self.apply_chat_template(key, text)
                
                tokenized = self.tokenizer(
                    full_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = tokenized['input_ids'].squeeze(0)
                att_mask = tokenized['attention_mask'].squeeze(0)
                
                # 4. Create Labels (Mask User Prompt)
                # We only want to calculate loss on the "Response" part
                labels = input_ids.clone()
                
                pad_id = self.tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = self.tokenizer.eos_token_id

                # Mask padding tokens
                labels[labels == pad_id] = -100
                labels[labels == 0] = -100 # Explicitly mask <pad> (id 0) just in case

                # Tokenize prompt ONLY (no padding)
                # This ensures we find the exact prompt structure within the full sequence
                prompt_tokens = self.tokenizer(
                    prompt_text,
                    add_special_tokens=True,
                    padding=False,
                    truncation=False
                )['input_ids']

                prompt_len = len(prompt_tokens)
                input_ids_list = input_ids.tolist()

                # Find prompt location inside padded sequence
                start_idx = self._find_subsequence(input_ids_list, prompt_tokens)

                if start_idx is None:
                    # Fallback: only if something VERY strange happens (should never trigger)
                    # print("WARNING: prompt tokens not found in input_ids. Falling back to prefix masking.")
                    start_idx = 0

                # Mask prompt region
                labels[start_idx : start_idx + prompt_len] = -100

                input_ids_stack.append(input_ids)
                att_stack.append(att_mask)
                label_stack.append(labels)

            return {
                'pixel_values': img_tensor,
                'organ_masks': torch.stack(mask_stack).float(),
                'input_ids': torch.stack(input_ids_stack),
                'attention_mask': torch.stack(att_stack),
                'labels': torch.stack(label_stack)
            }

        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            traceback.print_exc()
            return None

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder_model', type=str, default='google/medgemma-4b-it')
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data_sym/image_first_dataset.csv')
    parser.add_argument('--json_file', type=str, default='/home/muhammedg/fvlm/data_sym/combined_desc_conc.json')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/medgemma_vlm')
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--subset_size', type=int, default=None, help='Train on a small subset for debugging')
    args = parser.parse_args()
    
    # Init Model
    model = MedicalVLM(args.vision_encoder_path, args.decoder_model)

    # Init Data
    transform = build_transforms()
    train_ds = OnePassOrganDataset(args.csv_file, args.json_file, model.tokenizer, transform, split='training', subset_size=args.subset_size)
    val_ds = OnePassOrganDataset(args.csv_file, args.json_file, model.tokenizer, transform, split='validation', subset_size=args.subset_size)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, # Fix OOM: Match train batch size
        gradient_accumulation_steps=8,
        learning_rate=1e-4, # Higher LR for Vision+Projector since LLM is frozen
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=200,
        eval_steps=200,
        save_total_limit=3, # Keep only the last 3 checkpoints to save space
        eval_accumulation_steps=1, # Fix OOM: Offload predictions to CPU immediately
        gradient_checkpointing=True, # Fix OOM: Save memory during training
        bf16=True, # Use BF16 for stability
        fp16=False,
        dataloader_num_workers=4,
        remove_unused_columns=False, # Essential for custom forward pass
        report_to="wandb"
    )

    class MedicalTrainer(Trainer):
        def save_model(self, output_dir=None, _internal_call=False):
            """
            Override to explicitly save only the trainable parts using our custom method.
            This avoids the 'safetensors' shared memory error with tied weights in Gemma.
            """
            if output_dir is None:
                output_dir = self.args.output_dir
            self.model.save_pretrained(output_dir)

    trainer = MedicalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=OrganCollator()
    )

    trainer.train()
    model.save_pretrained(f"{args.output_dir}/final")

if __name__ == '__main__':
    import argparse
    main()
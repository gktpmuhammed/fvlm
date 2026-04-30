#!/usr/bin/env python3
import sys
import os
import logging
import json
import pandas as pd
import torch
import numpy as np
import argparse
import SimpleITK as sitk
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, Transposed, Resized, EnsureTyped, EnsureChannelFirstd
import random

# Fix path for lavis
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from medical_vlm import MedicalVLM
from paths import checkpoints_root, data_sym_root
import wandb

logger = logging.getLogger(__name__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Allow shell override
os.environ["WANDB_PROJECT"] = "thesis_retrain_v3"
os.environ["WANDB_ENTITY"] = "gktp-thesis"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# --- CONFIGURATION ---

# The refined list of targets based on high representation (>900 text reports)
# Removed: Conclusion, Brain, Face, Colon, Bones, Muscles, Reproductive organs
ALL_TARGET_KEYS = [
    'lung', 'heart', 'esophagus', 
    'liver', 'gallbladder', 'stomach', 'pancreas', 'spleen', 'kidney',
    'aorta', 'trachea', 'rib'
]

NO_FINDING_TEMPLATES = [
    "No significant findings in the {organ}.",
    "The {organ} is unremarkable.",
    "No abnormalities detected in the {organ}.",
    "Normal limits for the {organ}.",
    "No pathology in the {organ}.",
    "No acute findings in the {organ}.",
    "The {organ} appears normal.",
    "Clear {organ}."
]

# --- MAPPING LOGIC ---
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

@dataclass
class OrganCollator:
    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features: raise ValueError("Empty batch")
        
        pixel_values = torch.stack([f['pixel_values'] for f in features])
        organ_masks = torch.stack([f['organ_masks'] for f in features])
        input_ids = torch.stack([f['input_ids'] for f in features])
        labels = torch.stack([f['labels'] for f in features])
        sample_weights = torch.stack([f['sample_weights'] for f in features])
        
        return {
            'pixel_values': pixel_values, 
            'organ_masks': organ_masks,
            'input_ids': input_ids,
            'labels': labels,
            'sample_weights': sample_weights
        }

def build_transforms():
    return Compose([
        LoadImaged(keys=['image', 'mask'], reader='ITKReader', image_only=True),
        EnsureChannelFirstd(keys=['image', 'mask']),
        Transposed(keys=['image', 'mask'], indices=(0, 3, 2, 1)),
        ScaleIntensityRanged(keys=['image'], a_min=-1150, a_max=350, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=['image', 'mask'], spatial_size=(112, 256, 352), mode='constant', constant_values=0),
        Resized(keys=['image', 'mask'], spatial_size=(112, 256, 352), mode=['trilinear', 'nearest']),
        EnsureTyped(keys=['image', 'mask'])
    ])

class OnePassOrganDataset(Dataset):
    def __init__(self, csv_file, json_file, tokenizer, transform, max_length=128, subset_size=None, split='training'):
        print(f"--- Loading One-Pass Dataset ({split}) ---")
        self.split = split
        
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        if subset_size: self.df = self.df.head(subset_size)

        # Load Single JSON
        with open(json_file, 'r') as f: 
            self.reports_json = json.load(f)

        # Load Sampling Probabilities (Only for training)
        self.sampling_probs = None
        if split == 'training':
            probs_path = os.path.join(os.path.dirname(csv_file), 'organ_sampling_probs.json')
            if os.path.exists(probs_path):
                with open(probs_path, 'r') as f:
                    self.sampling_probs = json.load(f)
                print(f" Loaded organ sampling probabilities from {probs_path}")
            else:
                print("WARNING: No organ sampling probabilities found for training. Using keep_prob=1.0.")

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.target_keys = ALL_TARGET_KEYS
        
        # Filter patients that exist in JSON
        self.valid_patients = []
        for _, row in self.df.iterrows():
            fname = os.path.basename(row['image_path'])
            base_id = fname.replace('.nii.gz', '').replace('.nii', '')
            
            # ID Matching
            target_pid = None
            if base_id in self.reports_json: 
                target_pid = base_id
            elif '_' in base_id:
                short_id = base_id.rsplit('_', 1)[0]
                if short_id in self.reports_json: 
                    target_pid = short_id
            
            if target_pid:
                self.valid_patients.append({
                    'image_path': row['image_path'],
                    'mask_path': row['image_path'].replace('images', 'masks'),
                    'pid': target_pid
                })

        print(f" Found {len(self.valid_patients)} valid patients for training.")

    def __len__(self): return len(self.valid_patients)

    def __getitem__(self, idx):
        item = self.valid_patients[idx]
        try:
            if not os.path.exists(item['mask_path']): return None

            # 1. Load Image & Mask (ONCE per patient)
            data = self.transform({'image': item['image_path'], 'mask': item['mask_path']})
            
            # Tensor Conversion
            img_data = data['image']
            if hasattr(img_data, 'as_tensor'): image_tensor = img_data.as_tensor().float()
            elif isinstance(img_data, torch.Tensor): image_tensor = img_data.float()
            else: image_tensor = torch.from_numpy(img_data).float()

            mask_data = data['mask']
            if hasattr(mask_data, 'as_tensor'): full_mask_tensor = mask_data.as_tensor()
            elif isinstance(mask_data, torch.Tensor): full_mask_tensor = mask_data
            else: full_mask_tensor = torch.from_numpy(mask_data)
            
            # 2. Iterate ALL Target Organs
            mask_stack = []
            label_stack = []
            input_id_stack = []
            weights_stack = []
            
            patient_data = self.reports_json.get(item['pid'], {})
            
            for key in self.target_keys:
                target_ids = get_organ_ids_for_key(key)
                
                # A. Prepare MASK
                if len(target_ids) > 0:
                    binary_mask = torch.zeros_like(full_mask_tensor)
                    for tid in target_ids:
                        binary_mask[full_mask_tensor == tid] = 1.0
                else:
                    binary_mask = torch.zeros_like(full_mask_tensor)
                
                mask_stack.append(binary_mask)
                
                # B. Prepare TEXT & Weighting
                text = patient_data.get(key, "").strip()
                is_default = False
                
                if len(text) < 3: 
                    if self.split == 'training':
                        tmpl = random.choice(NO_FINDING_TEMPLATES)
                        text = tmpl.format(organ=key)
                    else:
                        text = NO_FINDING_TEMPLATES[0].format(organ=key)
                    is_default = True

                # Balanced Masking Logic (matches MedGemma V3)
                weight = 1.0
                if self.sampling_probs:
                    if is_default:
                        prob = self.sampling_probs.get(key, 1.0)
                        if random.random() > prob:
                            weight = 0.0
                    # Explicit findings always kept (weight=1.0)
                
                weights_stack.append(weight)

                # C. Tokenize
                prompt = f"Describe {key}: "
                full_input = prompt + text
                
                tokens = self.tokenizer(
                    full_input, 
                    max_length=self.max_length, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt'
                )['input_ids'].squeeze(0)
                
                # --- FIX: Proper label masking ---
                content_ids = self.tokenizer(
                    full_input, add_special_tokens=True, truncation=True,
                    max_length=self.max_length
                )['input_ids']
                
                content_len = len(content_ids)  # includes BOS/EOS if tokenizer adds them
                
                # 2. Find prompt boundary in the JOINT token sequence.
                #    BPE tokenizers merge subwords across the prompt-content boundary
                #    (e.g., "Describe heart: Posterior" → [.., " Per", "ior", ..])
                #    so we can't count tokens from separate prompt tokenization.
                #    Instead, decode token-by-token until we've covered the prompt text.
                prompt_boundary = 0
                decoded_so_far = ""
                for idx, tid in enumerate(content_ids):
                    decoded_so_far = self.tokenizer.decode(content_ids[:idx+1], skip_special_tokens=True)
                    # Check if we've decoded at least up to the end of the prompt
                    if len(decoded_so_far.rstrip()) >= len(prompt.rstrip()):
                        prompt_boundary = idx + 1
                        break
                
                # 3. Build label mask:
                #    - Tokens 0..prompt_boundary-1: mask (-100) [BOS + prompt]
                #    - Tokens prompt_boundary..content_len-1: keep [content + EOS]
                #    - Tokens content_len..max_length-1: mask (-100) [padding]
                labels = tokens.clone()
                labels[:prompt_boundary] = -100
                if content_len < self.max_length:
                    labels[content_len:] = -100
                
                label_stack.append(labels)
                input_id_stack.append(tokens)

            # Stack tensors
            return {
                'pixel_values': image_tensor,
                'organ_masks': torch.stack(mask_stack).float(),
                'input_ids': torch.stack(input_id_stack),
                'labels': torch.stack(label_stack),
                'sample_weights': torch.tensor(weights_stack, dtype=torch.float32)
            }

        except Exception as e:
            logger.error(f"Error {e}")
            return None

def main(args):
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    
    model = MedicalVLM(
        vision_encoder_path=args.vision_encoder_path,
        decoder_model_name=args.decoder_model,
        queries_per_organ=args.queries_per_organ,
        align_loss_weight=args.align_loss_weight
    )

    transform = build_transforms()
    
    train_dataset = OnePassOrganDataset(
        args.csv_file, args.json_file, model.tokenizer, transform, 
        args.max_length, args.subset_size, 'training'
    )
    val_dataset = OnePassOrganDataset(
        args.csv_file, args.json_file, model.tokenizer, transform, 
        args.max_length, args.subset_size, 'validation'
    )

    run_name = f"{args.decoder_model.split('/')[-1]}_refined_organs_batch"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,
        report_to="wandb",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8, 
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=False,
        bf16=True,
        fp16=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=True,
    )

    # Custom Trainer to handle tied weight saving (BERT also has shared embeddings/lm_head)
    class MedicalTrainer(Seq2SeqTrainer):
        def save_model(self, output_dir=None, _internal_call=False):
            if output_dir is None:
                output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            # Use torch.save to avoid safetensors tied-weight error
            torch.save(
                self.model.state_dict(), 
                os.path.join(output_dir, "pytorch_model.bin")
            )
            if hasattr(self.model, 'tokenizer'):
                self.model.tokenizer.save_pretrained(output_dir)

    trainer = MedicalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=OrganCollator(),
    )

    trainer.train()
    
    # Save final model
    final_dir = f"{args.output_dir}/final_model"
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "pytorch_model.bin"))
    model.tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")
    
    wandb.finish()

if __name__ == '__main__':
    ckpt_default = checkpoints_root() / "model.pth"
    csv_default = data_sym_root() / "image_first_dataset.csv"
    json_default = data_sym_root() / "combined_desc_conc_v2.json"

    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder_model', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--vision_encoder_path', type=str, default=str(ckpt_default))
    parser.add_argument('--csv_file', type=str, default=str(csv_default))
    parser.add_argument('--json_file', type=str, default=str(json_default))
    parser.add_argument('--output_dir', type=str, default='./checkpoints/medical_vlm')
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--subset_size', type=int, default=None)
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--queries_per_organ', type=int, default=8, help='Number of visual tokens per organ')
    parser.add_argument('--align_loss_weight', type=float, default=10.0, help='Weight for alignment loss (higher forces more visual grounding)')
    
    args = parser.parse_args()
    main(args)

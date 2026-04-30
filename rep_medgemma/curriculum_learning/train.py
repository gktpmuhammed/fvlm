#!/usr/bin/env python3
# Variant: Curriculum Learning — train on pathological cases first, then all data
import sys
import os
from pathlib import Path
import logging
import json
import pandas as pd
import torch
import random

from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, Transposed, Resized, EnsureTyped, EnsureChannelFirstd
import traceback


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

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

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    # sys.path.insert(0, parent_dir) # REMOVED
    sys.path.append(parent_dir)
    sys.path.insert(0, os.path.join(parent_dir, "../"))

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
        # V3 Update: Use Resizing instead of Cropping to preserve all organs
        Resized(keys=['image', 'mask'], spatial_size=(112, 256, 352), mode=['trilinear', 'nearest']),
        EnsureTyped(keys=['image', 'mask']),
    ])

@dataclass
class OrganCollator:
    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features: raise ValueError("Empty batch")
        
        pixel_values = torch.stack([f['pixel_values'] for f in features])
        organ_masks = torch.stack([f['organ_masks'] for f in features])
        
        input_ids = torch.stack([f['input_ids'] for f in features])
        attention_mask = torch.stack([f['attention_mask'] for f in features])
        labels = torch.stack([f['labels'] for f in features])
        sample_weights = torch.stack([f['sample_weights'] for f in features])
        
        return {
            'pixel_values': pixel_values, 
            'organ_masks': organ_masks,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sample_weights': sample_weights
        }

class OnePassOrganDataset(Dataset):
    def __init__(self, csv_file, json_file, tokenizer, transform, max_length=128, subset_size=None, split='training', pathology_only=False):
        self.split = split
        self.pathology_only = pathology_only
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        if subset_size: self.df = self.df.head(subset_size)
        
        with open(json_file, 'r') as f: 
            self.reports_json = json.load(f)

        # Load Sampling Probabilities (Only for training)
        self.sampling_probs = None
        if split == 'training':
            probs_path = os.path.join(os.path.dirname(csv_file), 'organ_sampling_probs.json')
            if os.path.exists(probs_path):
                with open(probs_path, 'r') as f:
                    self.sampling_probs = json.load(f)
                print(f"Loaded organ sampling probabilities from {probs_path}")
            else:
                print("WARNING: No organ sampling probabilities found for training. Using keep_prob=1.0.")

        self.tokenizer = tokenizer
        # Enforce right padding for correct masking logic
        self.tokenizer.padding_side = 'right'
        self.transform = transform
        self.max_length = max_length
        self.target_keys = ALL_TARGET_KEYS
        
        # Filter patients
        self._all_valid_patients = []
        for _, row in self.df.iterrows():
            fname = os.path.basename(row['image_path'])
            base_id = fname.replace('.nii.gz', '').replace('.nii', '')
            if base_id in self.reports_json or (len(base_id.split('_')) > 1 and base_id.rsplit('_', 1)[0] in self.reports_json):
                self._all_valid_patients.append(row)
        
        self.valid_patients = self._all_valid_patients
        
        # Curriculum: organ-level pathology filtering
        # Phase 1: zero out weight for organs with only normal/generic text
        # Phase 2: restore normal weighting
        self.curriculum_phase = 1 if pathology_only else 2
        
        # Phrases that indicate "normal" (not actual pathology)
        self.normal_phrases = [
            'normal', 'unremarkable', 'no significant', 'no acute', 'no abnormal',
            'within normal', 'no evidence', 'no finding', 'no patholog', 'no obstructive',
            'are open', 'caliber is normal', 'size are normal', 'no mass',
            'no consolidation', 'no infiltration', 'no effusion', 'no calcification'
        ]
        
        if pathology_only:
            # Count how many organ-samples will be active in Phase 1
            active = 0
            total = 0
            for row in self.valid_patients:
                fname = os.path.basename(row['image_path'])
                base_id = fname.replace('.nii.gz', '').replace('.nii', '')
                pid = base_id if base_id in self.reports_json else base_id.rsplit('_', 1)[0]
                patient_data = self.reports_json.get(pid, {})
                for key in self.target_keys:
                    total += 1
                    text = patient_data.get(key, "").strip()
                    if len(text) >= 3 and self._is_pathological(text):
                        active += 1
            print(f"  [Curriculum] Phase 1: {active}/{total} organ-samples are pathological ({active/total*100:.1f}%)")
    
    def _is_pathological(self, text):
        """Check if text describes actual pathology vs normal finding."""
        text_lower = text.lower()
        return not any(phrase in text_lower for phrase in self.normal_phrases)
    
    def switch_to_phase2(self):
        """Called by CurriculumCallback to switch from pathology-only to full training."""
        self.curriculum_phase = 2
        print(f"  [Curriculum] Phase 2: All organ-samples now receive normal weight")

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
            weights_stack = []

            for key in self.target_keys:
                # 1. Mask
                tids = get_organ_ids_for_key(key)
                m = torch.zeros_like(mask_tensor)
                for t in tids: m[mask_tensor == t] = 1.0
                mask_stack.append(m)
                
                # 2. Text & Weighting
                text = patient_data.get(key, "").strip()
                is_default = False
                
                if len(text) < 3: 
                    # If empty, teach model to say "No findings." or a synonym
                    if self.split == 'training':
                        tmpl = random.choice(NO_FINDING_TEMPLATES)
                        text = tmpl.format(organ=key)
                    else:
                        # Use a consistent template for validation to avoid artificial loss spikes
                        # while still matching the training distribution format.
                        text = NO_FINDING_TEMPLATES[0].format(organ=key)
                    is_default = True
                
                # Balanced Masking Logic
                weight = 1.0
                if self.sampling_probs:
                    # If Default, we might mask it out (weight=0)
                    if is_default:
                        prob = self.sampling_probs.get(key, 1.0)
                        if random.random() > prob:
                            weight = 0.0
                    # Explicit findings always kept (weight=1.0)
                
                # Curriculum Learning: Phase 1 zeroes out normal/generic organ-samples
                if self.curriculum_phase == 1 and not is_default:
                    # Has text, but is it actually pathological?
                    if not self._is_pathological(text):
                        weight = 0.0  # Skip normal descriptions in Phase 1
                
                weights_stack.append(weight)

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
                'labels': torch.stack(label_stack),
                'sample_weights': torch.tensor(weights_stack, dtype=torch.float32)
            }

        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            traceback.print_exc()
            return None

class CurriculumCallback(TrainerCallback):
    """
    Switches training from organ-level pathology-only to full dataset at a given step ratio.
    Phase 1 (first curriculum_ratio of training): Only pathological organ-samples get weight > 0.
    Phase 2 (remaining): All organ-samples get normal weights.
    """
    def __init__(self, train_dataset, curriculum_ratio=0.3):
        self.train_dataset = train_dataset
        self.curriculum_ratio = curriculum_ratio
        self.switched = False
    
    def on_step_begin(self, args, state, control, **kwargs):
        if self.switched:
            return
        # Switch at curriculum_ratio of total steps
        if state.max_steps > 0 and state.global_step >= int(state.max_steps * self.curriculum_ratio):
            print(f"\n{'='*60}")
            print(f"  [Curriculum] Switching at step {state.global_step}/{state.max_steps}")
            print(f"{'='*60}")
            self.train_dataset.switch_to_phase2()
            self.switched = True


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Allow shell override
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder_model', type=str, default='google/medgemma-4b-it')
    parser.add_argument('--vision_encoder_path', type=str, default=str(PROJECT_ROOT / 'checkpoints' / 'model.pth'))
    parser.add_argument('--csv_file', type=str, default=str(PROJECT_ROOT / 'data_sym' / 'image_first_dataset.csv'))
    parser.add_argument('--json_file', type=str, default='../../data_sym/combined_desc_conc_v2.json')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/retrain_v2')
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--subset_size', type=int, default=None, help='Train on a small subset for debugging')
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--queries_per_organ', type=int, default=8, help='Number of visual tokens per organ')
    parser.add_argument('--curriculum_ratio', type=float, default=0.3, help='Fraction of training for pathology-only phase')
    args = parser.parse_args()
    
    # Init Model
    model = MedicalVLM(args.vision_encoder_path, args.decoder_model, queries_per_organ=args.queries_per_organ)

    # Init Data
    transform = build_transforms()
    # Phase 1 starts with pathology-only data
    print(f"\n[Curriculum Learning] Phase 1: Pathology-only (first {int(args.curriculum_ratio*100)}% of training)")
    train_ds = OnePassOrganDataset(args.csv_file, args.json_file, model.tokenizer, transform, split='training', subset_size=args.subset_size, pathology_only=True)
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
        logging_steps=args.logging_steps,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=200,
        eval_steps=args.eval_steps,
        save_total_limit=3, # Keep only the last 3 checkpoints to save space
        eval_accumulation_steps=None, # Fix: Disable accumulation to ensure correct scalar loss reporting
        gradient_checkpointing=True, # Fix OOM: Save memory during training
        gradient_checkpointing_kwargs={'use_reentrant': False}, # Fix DDP: Prevent "marked ready twice" error
        bf16=True, # Use BF16 for stability
        fp16=False,
        dataloader_num_workers=4,
        remove_unused_columns=False, # Essential for custom forward pass
        ddp_find_unused_parameters=True, # Fixed: Required to avoid DDP error with unused params
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
            
            # Helper to unwrap model in DDP (Distributed Data Parallel)
            model_to_save = self.model
            while hasattr(model_to_save, 'module'):
                model_to_save = model_to_save.module
                
            model_to_save.save_pretrained(output_dir)

        def _load_from_checkpoint(self, resume_from_checkpoint, fit_model=True):
            """
            Override to handle our custom checkpoint format.
            """
            if resume_from_checkpoint is None:
                return

            print(f"Loading custom checkpoint from {resume_from_checkpoint}")
            
            # 1. Load Vision Encoder
            vision_path = os.path.join(resume_from_checkpoint, "vision_encoder.bin")
            if os.path.exists(vision_path):
                self.model.vision_encoder.load_state_dict(torch.load(vision_path, map_location="cpu"))
                print("  - Vision Encoder loaded.")
            
            # 2. Load Projector
            proj_path = os.path.join(resume_from_checkpoint, "projector.bin")
            if os.path.exists(proj_path):
                self.model.visual_projection.load_state_dict(torch.load(proj_path, map_location="cpu"))
                print("  - Projector loaded.")
                
            # 3. Load Projector LayerNorm
            ln_path = os.path.join(resume_from_checkpoint, "projector_layernorm.bin")
            if os.path.exists(ln_path):
                self.model.projector_layernorm.load_state_dict(torch.load(ln_path, map_location="cpu"))
                print("  - Projector LayerNorm loaded.")

            # 4. Load Visual Pos Embed
            pos_path = os.path.join(resume_from_checkpoint, "visual_pos_embed.bin")
            if os.path.exists(pos_path):
                # It's saved as a tensor/param, so load it directly
                # If it was saved as state_dict it would be different, but save_pretrained does torch.save(param)
                # Let's check medical_vlm.py: 
                # torch.save(self.visual_pos_embed, ...) -> saves the Tensor/Parameter object
                self.model.visual_pos_embed.data = torch.load(pos_path, map_location="cpu").data
                print("  - Visual Pos Embed loaded.")
            
            # 5. Load LoRA Adapters (Managed by PEFT/Transformers usually)
            # The 'adapter_model.safetensors' is present, so we should load it.
            # Using standard PEFT loader if possible, or manual load if we must.
            # Since self.model.decoder is a PeftModel, we can use load_peft_weights
            from peft import PeftModel
            if isinstance(self.model.decoder, PeftModel):
                self.model.decoder.load_adapter(resume_from_checkpoint, adapter_name="default")
                print("  - LoRA Adapters loaded.")

            # 6. Load Optimizer/Scheduler/Step (Handled by Trainer.train logic usually, but we need to ensure state is loaded)
            # calling super()._load_from_checkpoint might fail because it tries to load model weights too.
            # However, super() logic is complex. 
            # Actually, standard Trainer._load_from_checkpoint primarily loads the model weights.
            # The optimizer/scheduler loading happens in `train()` method logic explicitly via `_load_optimizer_and_scheduler`.
            # So we just need to ensure the MODEL weights are correct here.
            
            return

    curriculum_cb = CurriculumCallback(train_ds, curriculum_ratio=args.curriculum_ratio)
    
    trainer = MedicalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=OrganCollator(),
        callbacks=[curriculum_cb]
    )

    trainer.train()
    model.save_pretrained(f"{args.output_dir}/final")

if __name__ == '__main__':
    import argparse
    main()
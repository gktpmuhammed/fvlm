#!/usr/bin/env python3
import sys
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
from tqdm import tqdm
import argparse
from peft import PeftModel

# Add parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir in sys.path: sys.path.remove(parent_dir)
if current_dir in sys.path: sys.path.remove(current_dir)

sys.path.insert(0, os.path.join(parent_dir, "../"))
sys.path.insert(0, current_dir)

from medical_vlm import MedicalVLM
from train import get_organ_ids_for_key, ALL_TARGET_KEYS, build_transforms

class EvalDataset(Dataset):
    def __init__(self, csv_file, transform, subset_size=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == 'validation'].reset_index(drop=True)
        if subset_size: self.df = self.df.head(subset_size)
        self.transform = transform
        
    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            mask_path = row['image_path'].replace('images', 'masks')
            data = self.transform({'image': row['image_path'], 'mask': mask_path})
            
            img = data['image'].as_tensor().float() if hasattr(data['image'], 'as_tensor') else torch.tensor(data['image']).float()
            mask = data['mask'].as_tensor() if hasattr(data['mask'], 'as_tensor') else torch.tensor(data['mask'])
            
            return {
                'pixel_values': img,
                'full_mask': mask,
                'patient_id': os.path.basename(row['image_path']).split('.')[0]
            }
        except:
            return None


def _load_lora_if_available(model, checkpoint_dir):
    adapter_bin = os.path.join(checkpoint_dir, "adapter_model.bin")
    adapter_safe = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    adapter_cfg = os.path.join(checkpoint_dir, "adapter_config.json")

    if os.path.exists(adapter_cfg) and (os.path.exists(adapter_bin) or os.path.exists(adapter_safe)):
        print(f"Loading LoRA adapters from {checkpoint_dir}...")
        model.decoder = PeftModel.from_pretrained(model.decoder, checkpoint_dir, is_trainable=False)
    else:
        print("No LoRA adapter files found; using base MedGemma weights.")

def evaluate(args):
    print(f"Loading Model: {args.decoder_model}")
    model = MedicalVLM(
        args.vision_encoder_path,
        args.decoder_model,
        organ_chunk_size=args.organ_chunk_size,
        apply_lora=False,
        use_4bit=args.use_4bit,
        device_map=args.device_map,
        local_files_only=(args.local_files_only or not args.allow_online_model_fetch),
    )
    _load_lora_if_available(model, args.checkpoint_dir)

    if args.device_map is None and torch.cuda.is_available():
        model.cuda()
    model.eval()

    transform = build_transforms()
    ds = EvalDataset(args.csv_file, transform, args.subset_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = []
    print("Generating...")
    with torch.no_grad():
        for batch in tqdm(dl):
            if batch is None:
                continue

            pixel_values = batch["pixel_values"]
            full_mask = batch["full_mask"]
            pids = batch["patient_id"]
            bsz = pixel_values.shape[0]

            # Build organ masks: (B, N, 1, D, H, W)
            organ_masks_list = []
            for key in ALL_TARGET_KEYS:
                tids = get_organ_ids_for_key(key)
                organ_mask = torch.zeros_like(full_mask, dtype=torch.bool)
                for t in tids:
                    organ_mask = organ_mask | (full_mask == t)
                organ_masks_list.append(organ_mask.float())
            organ_masks = torch.stack(organ_masks_list, dim=1)

            # Build prompts as (B, N)
            prompts_per_batch = []
            for _ in range(bsz):
                prompts = []
                for key in ALL_TARGET_KEYS:
                    prompts.append(
                        f"<start_of_turn>user\nAnalyze the specific image feature. "
                        f"Describe the findings for the {key}.<end_of_turn>\n<start_of_turn>model\n"
                    )
                prompts_per_batch.append(prompts)

            flat_prompts = [prompt for prompts in prompts_per_batch for prompt in prompts]
            inputs = model.tokenizer(flat_prompts, return_tensors="pt", padding=True)
            seq_len = inputs.input_ids.shape[1]
            num_organs = len(ALL_TARGET_KEYS)
            input_ids = inputs.input_ids.view(bsz, num_organs, seq_len)
            attention_mask = inputs.attention_mask.view(bsz, num_organs, seq_len)

            outputs = model.generate(
                pixel_values=pixel_values,
                organ_masks=organ_masks,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,
                num_beams=3,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
            decoded = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for i, pid in enumerate(pids):
                start = i * num_organs
                end = (i + 1) * num_organs
                organ_texts = decoded[start:end]

                report = ""
                for key, text in zip(ALL_TARGET_KEYS, organ_texts):
                    clean_text = text.split("model\n")[-1].strip() if "model\n" in text else text.strip()
                    report += f"{key.upper()}: {clean_text}\n"

                results.append({"patient_id": pid, "prediction": report})

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "generated_reports_gemma.csv"), index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--decoder_model', type=str, default='google/medgemma-4b-it')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data_sym/image_first_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='./results/retrain_v2')
    parser.add_argument('--subset_size', type=int, default=None)
    parser.add_argument('--queries_per_organ', type=int, default=8)
    parser.add_argument('--organ_chunk_size', type=int, default=1, help='How many organ prompts to run per generation chunk')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--use_4bit', action='store_true', help='Load MedGemma in 4-bit quantization')
    parser.add_argument('--device_map', type=str, default=None, help='Optional HF device_map (e.g. auto)')
    parser.add_argument('--local_files_only', action='store_true', help='Force use of local HF cache only')
    parser.add_argument('--allow_online_model_fetch', action='store_true', help='Allow remote model download')
    args = parser.parse_args()
    evaluate(args)

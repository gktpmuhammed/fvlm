import sys
import os
import argparse
import torch
import pandas as pd
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict, OrderedDict

# Add local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.insert(0, current_dir)

from model import MedicalVLM
from data import UnifiedMedicalDataset, build_transforms, ALL_TARGET_KEYS
import metrics

def get_pid(path):
    fname = os.path.basename(path)
    base = fname.replace('.nii.gz', '').replace('.nii', '')
    return base.rsplit('_', 1)[0] if '_' in base else base

def clean_state_dict(state_dict):
    """
    Normalizes weight keys to match VisionEncoderDecoderModel expectation:
    1. Removes 'model.' prefix (from Trainer checkpoints).
    2. Maps 'adapter.' to 'encoder.' (if saved from custom wrapper).
    3. Ensures keys start with 'encoder.' or 'decoder.'.
    """
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        new_key = k
        
        # 1. Strip 'model.' prefix (Common in Trainer intermediate checkpoints)
        if new_key.startswith("model."):
            new_key = new_key[6:] # Remove "model."
            
        # 2. Map 'adapter.' to 'encoder.'
        # In your model.py, self.adapter IS self.model.encoder.
        # HF VisionEncoderDecoderModel expects 'encoder.xxx'.
        if new_key.startswith("adapter."):
            new_key = new_key.replace("adapter.", "encoder.", 1)
            
        # 3. Filter irrelevant keys (optional, prevents strict warnings)
        # We only care about encoder/decoder keys for the inner model
        if not (new_key.startswith("encoder.") or new_key.startswith("decoder.") or new_key.startswith("enc_to_dec_proj.")):
            continue
            
        new_state_dict[new_key] = v
        
    return new_state_dict

def evaluate(args):
    print(f"\n--- Starting Evaluation: {args.strategy.upper()} ---")
    
    # 1. SETUP MODEL
    model_wrapper = MedicalVLM(
        vision_encoder_path=args.vision_encoder_path,
        decoder_model_name=args.decoder_model,
        strategy=args.strategy,
        use_qformer=args.use_qformer
    )
    
    # --- ROBUST WEIGHT LOADING ---
    print(f"Loading weights from {args.model_path}...")
    
    # A. Resolve Path
    weights_path = args.model_path
    if os.path.isdir(args.model_path):
        bin_path = os.path.join(args.model_path, "pytorch_model.bin")
        safe_path = os.path.join(args.model_path, "model.safetensors")
        if os.path.exists(bin_path): weights_path = bin_path
        elif os.path.exists(safe_path): weights_path = safe_path
        else: raise FileNotFoundError(f"No weights found in {args.model_path}")

    # B. Load Raw File
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        raw_state = load_file(weights_path)
    else:
        raw_state = torch.load(weights_path, map_location='cpu')

    # C. Handle Nesting (e.g., {'model': ...})
    if 'model' in raw_state and isinstance(raw_state['model'], dict):
        print(" -> Detected nested 'model' key in checkpoint.")
        raw_state = raw_state['model']

    # D. Normalize Keys
    final_state = clean_state_dict(raw_state)

    # E. Apply to Inner Model (VisionEncoderDecoderModel)
    # We load into model_wrapper.model because that is the standard HF class
    print("Applying weights to internal VisionEncoderDecoderModel...")
    msg = model_wrapper.model.load_state_dict(final_state, strict=False)
    
    print(f"Weights Result:")
    print(f"  Missing: {len(msg.missing_keys)}")
    print(f"  Unexpected: {len(msg.unexpected_keys)}")
    
    # Validation: Check critical layers
    decoder_missing = [k for k in msg.missing_keys if "decoder" in k]
    encoder_missing = [k for k in msg.missing_keys if "encoder" in k]
    
    if len(decoder_missing) > 0:
        print("\nCRITICAL WARNING: Decoder weights are MISSING. Generation will be garbage.")
        print(f"Sample missing: {decoder_missing[:3]}")
    elif len(encoder_missing) > 0:
        print("\nWARNING: Some Encoder weights are missing (might be okay if partial freezing used).")
    else:
        print("SUCCESS: Encoder and Decoder weights aligned and loaded.")

    model_wrapper.cuda().eval()

    # 2. DATA SETUP
    eval_mode = 'global' if args.strategy == 'global' else 'parallel'
    
    transform = build_transforms()
    ds = UnifiedMedicalDataset(
        args.csv_file, args.json_file, model_wrapper.tokenizer, transform, 
        mode=eval_mode, split='validation', subset_size=args.subset_size
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    
    with open(args.json_file, 'r') as f: ref_json = json.load(f)

    # Storage
    all_preds_concat, all_refs_concat, patient_ids = [], [], []
    organ_breakdown = defaultdict(lambda: {'preds': [], 'refs': []})

    print("Generating Reports...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dl)):
            if batch is None: continue
            
            pixel_values = batch['pixel_values'].cuda()
            
            # --- GLOBAL ---
            if args.strategy == 'global':
                outputs = model_wrapper.generate(pixel_values, max_length=args.max_length, num_beams=4)
                decoded_text = model_wrapper.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                
                lbl = batch['labels'][0].cpu()
                lbl[lbl==-100] = model_wrapper.tokenizer.pad_token_id
                ref_text = model_wrapper.tokenizer.decode(lbl, skip_special_tokens=True)
                
                all_preds_concat.append(decoded_text)
                all_refs_concat.append(ref_text)
            
            # --- ORGAN-BASED ---
            else:
                organ_masks = batch['organ_masks'].cuda()
                prompts = [f"Describe {k}: " for k in ALL_TARGET_KEYS]
                
                # Get Inputs AND Masks for Decoder
                tokenized_prompts = model_wrapper.tokenizer(prompts, return_tensors='pt', padding=True)
                prompt_ids = tokenized_prompts.input_ids.cuda()
                prompt_mask = tokenized_prompts.attention_mask.cuda()

                gen_texts = []
                
                if args.strategy in ['roi', 'attention', 'attention_qformer']:
                    outputs = model_wrapper.generate(
                        pixel_values=pixel_values,
                        organ_masks=organ_masks,
                        decoder_input_ids=prompt_ids,
                        attention_mask=prompt_mask, # Important for correct generation
                        max_length=150,
                        min_length=5,
                        num_beams=4,
                        repetition_penalty=2.0,
                        no_repeat_ngram_size=3
                    )
                    gen_texts = model_wrapper.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                elif args.strategy == 'masked_single':
                    masks_sq = organ_masks.squeeze(0)
                    for j in range(len(ALL_TARGET_KEYS)):
                        s_mask = masks_sq[j].unsqueeze(0).unsqueeze(0)
                        s_prompt = prompt_ids[j].unsqueeze(0)
                        out = model_wrapper.generate(
                            pixel_values=pixel_values, pixel_mask=s_mask,
                            decoder_input_ids=s_prompt, max_length=120, num_beams=4
                        )
                        gen_texts.append(model_wrapper.tokenizer.decode(out[0], skip_special_tokens=True))
                
                # Reconstruct & Store
                path = dl.dataset.samples[i]['image_path']
                pid = get_pid(path)
                p_ref = ref_json.get(pid, {})
                if not p_ref and '_' in pid: p_ref = ref_json.get(pid.rsplit('_',1)[0], {})

                p_concat_pred = ""
                p_concat_ref = ""

                for k, pred, prompt_txt in zip(ALL_TARGET_KEYS, gen_texts, prompts):
                    clean_pred = pred.replace(prompt_txt, "").strip()
                    ref = p_ref.get(k, "")
                    if not ref: ref = p_ref.get(k.lower(), "")
                    
                    if ref and len(ref) > 2:
                        organ_breakdown[k]['preds'].append(clean_pred)
                        organ_breakdown[k]['refs'].append(ref)
                        
                        if clean_pred: p_concat_pred += f"{k.upper()}: {clean_pred}\n"
                        p_concat_ref += f"{k.upper()}: {ref}\n"

                if p_concat_ref.strip():
                    all_preds_concat.append(p_concat_pred.strip())
                    all_refs_concat.append(p_concat_ref.strip())
                    patient_ids.append(pid)

    # # 3. COMPUTE METRICS
    # summary = []
    
    # if all_preds_concat:
    #     print("\nComputing Global Metrics...")
    #     g_stats = metrics.compute_all_metrics(all_preds_concat, all_refs_concat)
    #     g_stats['Organ'] = 'GLOBAL'
    #     g_stats['N'] = len(all_preds_concat)
    #     summary.append(g_stats)
    #     print(f"Global BLEU-4: {g_stats['BLEU-4']:.4f}")

    # if args.strategy != 'global':
    #     print("Computing Per-Organ Metrics...")
    #     for organ in ALL_TARGET_KEYS:
    #         data = organ_breakdown[organ]
    #         if len(data['refs']) > 5:
    #             o_stats = metrics.compute_all_metrics(data['preds'], data['refs'])
    #             o_stats['Organ'] = organ
    #             o_stats['N'] = len(data['refs'])
    #             summary.append(o_stats)

    # metrics.create_metrics_table_plot(summary, args.output_dir)
    
    pd.DataFrame({'patient_id': patient_ids, 'pred': all_preds_concat, 'ref': all_refs_concat})\
      .to_csv(os.path.join(args.output_dir, "generated_reports.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, required=True, choices=['global', 'masked_single', 'roi', 'attention', 'attention_qformer'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--decoder_model', type=str, default='gpt2')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv')
    parser.add_argument('--json_file', type=str, default='/home/muhammedg/fvlm/data/combined_desc_conc.json')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--subset_size', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--use_qformer', action='store_true')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    evaluate(args)
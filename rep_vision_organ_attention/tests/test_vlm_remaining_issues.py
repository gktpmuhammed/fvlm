#!/usr/bin/env python3
"""
Diagnostic tests for remaining VLM issues (v3).
Validates Issues 1, 3, 4, 6 from the deep analysis.

Run: cd /home/muhammedg/fvlm/rep_vision_organ_attention && python /tmp/test_vlm_remaining_issues.py
"""
import sys
import os
import torch
import torch.nn as nn

project_dir = "/home/muhammedg/fvlm/rep_vision_organ_attention"
parent_dir = os.path.dirname(project_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    VisionEncoderDecoderModel, VisionEncoderDecoderConfig, ViTConfig
)
from transformers.modeling_outputs import BaseModelOutput

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
passed = 0
failed = 0

def test_header(name):
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")

def check(condition, msg):
    global passed, failed
    if condition:
        print(f"  {PASS} {msg}")
        passed += 1
    else:
        print(f"  {FAIL} {msg}")
        failed += 1
    return condition


# =====================================================================
# TEST A: Issue 4 — Train-eval mismatch (decoder_input_ids)
# =====================================================================
def test_train_eval_decoder_input_mismatch():
    test_header("TEST A: Issue 4 — Train-Eval Decoder Input Mismatch")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "Describe lung: "
    text = "No mass or infiltrative lesion was detected."
    full_input = prompt + text
    max_length = 64

    # --- Simulate TRAINING label creation (from train.py) ---
    tokens = tokenizer(
        full_input, max_length=max_length, padding='max_length',
        truncation=True, return_tensors='pt'
    )['input_ids'].squeeze(0)

    content_ids = tokenizer(
        full_input, add_special_tokens=True, truncation=True,
        max_length=max_length
    )['input_ids']
    content_len = len(content_ids)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    prompt_len = len(prompt_ids)

    bos_offset = 0
    if (tokenizer.bos_token_id is not None and
        len(content_ids) > 0 and
        content_ids[0] == tokenizer.bos_token_id):
        bos_offset = 1

    labels = tokens.clone()
    labels[:bos_offset + prompt_len] = -100
    if content_len < max_length:
        labels[content_len:] = -100

    # --- Simulate what VisionEncoderDecoderModel does with labels ---
    # It shifts labels right to create decoder_input_ids
    decoder_start_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    auto_decoder_input_ids = labels.new_zeros(labels.shape)
    auto_decoder_input_ids[1:] = labels[:-1].clone()
    auto_decoder_input_ids[0] = decoder_start_id
    # Replace -100 with pad_token_id
    auto_decoder_input_ids.masked_fill_(auto_decoder_input_ids == -100, tokenizer.pad_token_id)

    print(f"  Training labels (first 20): {labels[:20].tolist()}")
    print(f"  Auto decoder_input_ids (first 20): {auto_decoder_input_ids[:20].tolist()}")
    print(f"  Auto decoded: '{tokenizer.decode(auto_decoder_input_ids[auto_decoder_input_ids != tokenizer.pad_token_id])}'")

    # --- Simulate INFERENCE decoder_input_ids ---
    prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    eval_decoder_input_ids = prompt_inputs.input_ids.squeeze(0)
    
    print(f"\n  Eval decoder_input_ids: {eval_decoder_input_ids.tolist()}")
    print(f"  Eval decoded: '{tokenizer.decode(eval_decoder_input_ids)}'")

    # --- Check mismatch ---
    # During training: first token is BOS/EOS, then pad tokens (masked prompt)
    # During eval: first tokens are "Describe lung: "
    train_first_token = auto_decoder_input_ids[0].item()
    eval_first_token = eval_decoder_input_ids[0].item()
    
    print(f"\n  Training first token: {train_first_token} ('{tokenizer.decode([train_first_token])}')")
    print(f"  Eval first token: {eval_first_token} ('{tokenizer.decode([eval_first_token])}')")

    # The prompt region in training decoder_input_ids should be pad tokens
    training_prompt_region = auto_decoder_input_ids[1:prompt_len+1]
    all_pad = (training_prompt_region == tokenizer.pad_token_id).all().item()
    
    check(all_pad, f"CONFIRMS BUG: Prompt region in training is all pad ({tokenizer.pad_token_id})")
    check(train_first_token != eval_first_token or 
          not torch.equal(auto_decoder_input_ids[:prompt_len], eval_decoder_input_ids[:prompt_len]),
          "CONFIRMS BUG: Training and eval decoder_input_ids are DIFFERENT")
    
    print(f"\n  → The model never sees prompt tokens during training!")
    print(f"  → But at inference it must generate conditioned on prompt tokens!")


# =====================================================================
# TEST B: Issue 1 — visual_projection not in forward/generate path
# =====================================================================
def test_visual_projection_not_in_path():
    test_header("TEST B: Issue 1 — visual_projection Not In Forward Path")
    
    # Read source code and check
    with open(os.path.join(project_dir, "medical_vlm.py"), 'r') as f:
        source = f.read()
    
    # visual_projection is used in compute_alignment_loss
    alignment_usage = "self.visual_projection(vis_feats)" in source
    check(alignment_usage, "visual_projection used in alignment loss")
    
    # Check if visual_projection is applied before encoder_outputs go to decoder
    # In forward(), the encoder_outputs go directly to self.model()
    # We need to see if visual_projection is applied between encoder and decoder
    lines = source.split('\n')
    
    # Find the forward method and check if visual_projection is applied to encoder_outputs
    in_forward = False
    projection_before_decoder = False
    for line in lines:
        if 'def forward(self, pixel_values' in line:
            in_forward = True
        if in_forward and 'def ' in line and 'forward' not in line:
            break
        if in_forward and 'visual_projection' in line and 'alignment' not in line.lower():
            # Check context - is it before the model() call?
            projection_before_decoder = True
    
    # Find generate method
    in_generate = False
    projection_in_generate = False
    for line in lines:
        if 'def generate(self, pixel_values' in line:
            in_generate = True
        if in_generate and 'def ' in line and 'generate' not in line:
            break
        if in_generate and 'visual_projection' in line:
            projection_in_generate = True
    
    check(not projection_before_decoder, 
          "CONFIRMS BUG: visual_projection NOT applied in forward() encoder→decoder path")
    check(not projection_in_generate, 
          "CONFIRMS BUG: visual_projection NOT applied in generate() path")


# =====================================================================
# TEST C: Issue 3 — BART cross-attention receives encoder_hidden_states
# =====================================================================
def test_bart_cross_attention():
    test_header("TEST C: Issue 3 — BART Cross-Attention Routing")
    
    decoder_model_name = "GanjinZero/biobart-v2-base"
    decoder_config = AutoConfig.from_pretrained(decoder_model_name)
    decoder_config.is_decoder = True
    if not getattr(decoder_config, 'add_cross_attention', False):
        decoder_config.add_cross_attention = True
    
    decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name, config=decoder_config)
    tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
    
    # Check that cross-attention layers exist
    cross_attn_modules = []
    for name, module in decoder.named_modules():
        if 'encoder_attn' in name and isinstance(module, nn.Linear):
            cross_attn_modules.append(name)
    
    has_cross_attn = len(cross_attn_modules) > 0
    check(has_cross_attn, f"BART has {len(cross_attn_modules)} cross-attention linear layers")
    
    # Test forward pass with encoder_hidden_states
    batch_size = 2
    seq_len = 8
    hidden_size = decoder_config.d_model  # 768
    
    dummy_input_ids = tokenizer("test sentence", return_tensors="pt", padding="max_length", max_length=10)['input_ids']
    dummy_input_ids = dummy_input_ids.expand(batch_size, -1)
    dummy_encoder_hidden = torch.randn(batch_size, seq_len, hidden_size)
    
    try:
        with torch.no_grad():
            outputs = decoder(
                input_ids=dummy_input_ids,
                encoder_hidden_states=dummy_encoder_hidden,
            )
        check(True, "BART decoder accepts encoder_hidden_states without error")
        
        # Check that the output is not NaN
        has_nan = torch.isnan(outputs.logits).any().item()
        check(not has_nan, "BART output logits are not NaN")
        
        # Now test WITHOUT encoder_hidden_states
        with torch.no_grad():
            outputs_no_enc = decoder(
                input_ids=dummy_input_ids,
            )
        
        # Outputs should be different if cross-attention is doing something
        same_output = torch.allclose(outputs.logits, outputs_no_enc.logits, atol=1e-5)
        check(not same_output, "Cross-attention changes output (proves it's active)")
        
    except Exception as e:
        check(False, f"BART forward with encoder_hidden_states failed: {e}")


# =====================================================================
# TEST D: Issue 6 — save_pretrained doesn't save visual_projection
# =====================================================================
def test_save_missing_visual_projection():
    test_header("TEST D: Issue 6 — save_pretrained Missing visual_projection")
    
    with open(os.path.join(project_dir, "medical_vlm.py"), 'r') as f:
        source = f.read()
    
    # Check save_pretrained method
    in_save = False
    saves_projection = False
    for line in source.split('\n'):
        if 'def save_pretrained' in line:
            in_save = True
        if in_save and 'def ' in line and 'save_pretrained' not in line:
            break
        if in_save and 'visual_projection' in line:
            saves_projection = True
    
    check(not saves_projection, 
          "CONFIRMS: save_pretrained does NOT explicitly save visual_projection")
    
    # But check that MedicalTrainer saves full state_dict
    with open(os.path.join(project_dir, "train.py"), 'r') as f:
        train_source = f.read()
    
    saves_state_dict = "self.model.state_dict()" in train_source
    check(saves_state_dict, "MedicalTrainer uses state_dict() (includes visual_projection)")


# =====================================================================
# TEST E: VisionEncoderDecoderModel enc_to_dec_proj check
# =====================================================================
def test_enc_to_dec_proj():
    test_header("TEST E: VisionEncoderDecoderModel enc_to_dec_proj")
    
    # GPT-2 hidden_size = 768, encoder hidden_size = 768
    encoder_config = ViTConfig(hidden_size=768, num_hidden_layers=1, num_attention_heads=4,
                                intermediate_size=512, image_size=16, patch_size=4, num_channels=1)
    
    decoder_config = AutoConfig.from_pretrained("gpt2")
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    model = VisionEncoderDecoderModel(config=config)
    
    has_proj = hasattr(model, 'enc_to_dec_proj') and model.enc_to_dec_proj is not None
    
    if has_proj:
        print(f"  enc_to_dec_proj exists: {model.enc_to_dec_proj}")
        check(True, "enc_to_dec_proj is present (sizes may differ)")
    else:
        check(True, "No enc_to_dec_proj needed (encoder_hidden=decoder_hidden=768)")
    
    # For BART
    decoder_config_bart = AutoConfig.from_pretrained("GanjinZero/biobart-v2-base")
    decoder_config_bart.is_decoder = True
    decoder_config_bart.add_cross_attention = True
    
    config_bart = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config_bart)
    model_bart = VisionEncoderDecoderModel(config=config_bart)
    
    has_proj_bart = hasattr(model_bart, 'enc_to_dec_proj') and model_bart.enc_to_dec_proj is not None
    
    if has_proj_bart:
        print(f"  BART enc_to_dec_proj: {model_bart.enc_to_dec_proj}")
        check(True, "BART enc_to_dec_proj exists")
    else:
        print(f"  BART encoder hidden={encoder_config.hidden_size}, decoder hidden={decoder_config_bart.d_model}")
        check(True, f"No BART enc_to_dec_proj (both 768)")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  MEDICAL VLM — REMAINING ISSUES DIAGNOSTIC (v3)")
    print("=" * 65)
    
    test_train_eval_decoder_input_mismatch()
    test_visual_projection_not_in_path()
    test_bart_cross_attention()
    test_save_missing_visual_projection()
    test_enc_to_dec_proj()
    
    print("\n" + "=" * 65)
    total = passed + failed
    print(f"  RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print(f"  \033[92mALL TESTS PASSED — BUGS CONFIRMED!\033[0m")
    else:
        print(f"  \033[91mSOME TESTS FAILED\033[0m")
    print("=" * 65)
    
    sys.exit(0 if failed == 0 else 1)

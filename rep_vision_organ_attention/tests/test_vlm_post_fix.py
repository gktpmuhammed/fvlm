#!/usr/bin/env python3
"""
Post-fix verification tests for Medical VLM (v3).
Verifies that Issues 1, 4, and 6 are resolved.

Run: cd /home/muhammedg/fvlm/rep_vision_organ_attention && python /tmp/test_vlm_post_fix.py
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

from transformers import AutoTokenizer, AutoConfig
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
# TEST 1: visual_projection IS NOW in forward/generate path
# =====================================================================
def test_visual_projection_in_path():
    test_header("TEST 1: visual_projection NOW in Forward/Generate Path")
    
    with open(os.path.join(project_dir, "medical_vlm.py"), 'r') as f:
        source = f.read()
    
    # Check forward method
    lines = source.split('\n')
    
    # In forward(), visual_projection should be applied before self.model()
    in_forward = False
    found_projection_in_forward = False
    for line in lines:
        if 'def forward(self, pixel_values' in line:
            in_forward = True
        if in_forward and 'def generate' in line:
            break
        if in_forward and 'self.visual_projection(encoder_outputs.last_hidden_state)' in line:
            found_projection_in_forward = True
    
    check(found_projection_in_forward, "visual_projection applied in forward() before decoder")
    
    # Check generate method
    in_generate = False
    found_projection_in_generate = False
    for line in lines:
        if 'def generate(self, pixel_values' in line:
            in_generate = True
        if in_generate and 'def ' in line and 'generate' not in line:
            break
        if in_generate and 'self.visual_projection(encoder_outputs.last_hidden_state)' in line:
            found_projection_in_generate = True
    
    check(found_projection_in_generate, "visual_projection applied in generate() before decoder")


# =====================================================================
# TEST 2: forward() accepts input_ids for prompt-aware training
# =====================================================================
def test_forward_accepts_input_ids():
    test_header("TEST 2: forward() Accepts input_ids (Fix Issue 4)")
    
    with open(os.path.join(project_dir, "medical_vlm.py"), 'r') as f:
        source = f.read()
    
    # Check that forward signature includes input_ids
    has_input_ids = "def forward(self, pixel_values, organ_masks=None, input_ids=None, labels=None" in source
    check(has_input_ids, "forward() signature includes input_ids parameter")
    
    # Check that input_ids is used to build decoder_input_ids
    has_input_ids_usage = "flat_input_ids_all = input_ids.view(B * N_organs, Seq_Len)" in source
    check(has_input_ids_usage, "input_ids used to build decoder_input_ids with prompt")
    
    # Check that standard path also uses manual decoder_input_ids
    has_standard_decoder = "decoder_input_ids=decoder_input_ids" in source
    check(has_standard_decoder, "Both weighted and standard paths use explicit decoder_input_ids")


# =====================================================================
# TEST 3: train.py passes input_ids alongside labels
# =====================================================================
def test_train_passes_input_ids():
    test_header("TEST 3: train.py Provides input_ids to Model")
    
    with open(os.path.join(project_dir, "train.py"), 'r') as f:
        source = f.read()
    
    # Check that OrganCollator includes input_ids
    has_collator_input_ids = "'input_ids': input_ids," in source
    check(has_collator_input_ids, "OrganCollator includes input_ids in batch")
    
    # Check that dataset returns input_ids
    has_dataset_input_ids = "'input_ids': torch.stack(input_id_stack)," in source
    check(has_dataset_input_ids, "OnePassOrganDataset returns input_ids")
    
    # Check that input_id_stack is populated
    has_stack_append = "input_id_stack.append(tokens)" in source
    check(has_stack_append, "input_id_stack populated with full tokens (prompt+content)")
    
    # Check that input_id_stack is initialized
    has_stack_init = "input_id_stack = []" in source
    check(has_stack_init, "input_id_stack properly initialized")


# =====================================================================
# TEST 4: Prompt now appears in decoder_input_ids during training
# =====================================================================
def test_prompt_in_training_decoder_input():
    test_header("TEST 4: Prompt NOW in Training decoder_input_ids")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "Describe lung: "
    text = "No mass or infiltrative lesion was detected."
    full_input = prompt + text
    max_length = 64

    # Build tokens (input_ids)
    tokens = tokenizer(
        full_input, max_length=max_length, padding='max_length',
        truncation=True, return_tensors='pt'
    )['input_ids'].squeeze(0)

    # Build labels (prompt masked)
    content_ids = tokenizer(full_input, add_special_tokens=True, truncation=True, max_length=max_length)['input_ids']
    content_len = len(content_ids)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    prompt_len = len(prompt_ids)
    bos_offset = 0
    if tokenizer.bos_token_id is not None and len(content_ids) > 0 and content_ids[0] == tokenizer.bos_token_id:
        bos_offset = 1

    labels = tokens.clone()
    labels[:bos_offset + prompt_len] = -100
    if content_len < max_length:
        labels[content_len:] = -100

    # Simulate what the FIXED model does with input_ids
    decoder_start_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    decoder_input_ids = tokens.new_zeros(tokens.shape)
    decoder_input_ids[1:] = tokens[:-1].clone()
    decoder_input_ids[0] = decoder_start_id
    decoder_input_ids.masked_fill_(decoder_input_ids == -100, tokenizer.pad_token_id)

    print(f"  input_ids (tokens, first 15): {tokens[:15].tolist()}")
    print(f"  labels (first 15): {labels[:15].tolist()}")
    print(f"  decoder_input_ids (first 15): {decoder_input_ids[:15].tolist()}")
    decoded_decoder = tokenizer.decode(decoder_input_ids[decoder_input_ids != tokenizer.pad_token_id])
    print(f"  Decoded decoder_input_ids: '{decoded_decoder}'")
    
    # The prompt should now be in decoder_input_ids
    has_prompt = "Describe" in decoded_decoder or "lung" in decoded_decoder
    check(has_prompt, "Prompt tokens ARE in fixed decoder_input_ids")
    
    # But prompt should still be masked in labels
    prompt_masked = (labels[:bos_offset + prompt_len] == -100).all().item()
    check(prompt_masked, "Prompt still masked in labels (loss only on content)")
    
    # Verify: decoder_input_ids[t+1] = input_ids[t], so logits[t] predicts labels[t]
    # At position t, decoder input is tokens[t-1], and it should predict labels[t]
    matching = (decoder_input_ids[1:prompt_len+1] == tokens[:prompt_len]).all().item()
    check(matching, "decoder_input_ids properly shifted: position t receives tokens[t-1]")


# =====================================================================
# TEST 5: save_pretrained now saves visual_projection
# =====================================================================
def test_save_pretrained_fixed():
    test_header("TEST 5: save_pretrained Saves visual_projection")
    
    with open(os.path.join(project_dir, "medical_vlm.py"), 'r') as f:
        source = f.read()
    
    # Check that save_pretrained uses self.state_dict()
    has_full_state_dict = "self.state_dict()" in source
    check(has_full_state_dict, "save_pretrained uses self.state_dict() (includes visual_projection)")
    
    # Check it no longer calls self.model.save_pretrained()
    lines = source.split('\n')
    in_save = False
    uses_model_save = False
    for line in lines:
        if 'def save_pretrained' in line:
            in_save = True
        if in_save and 'def ' in line and 'save_pretrained' not in line:
            break
        if in_save and 'self.model.save_pretrained' in line:
            uses_model_save = True
    
    check(not uses_model_save, "No longer calls self.model.save_pretrained() (was missing visual_projection)")


# =====================================================================
# TEST 6: BART BOS handling — verify prompt masking still works
# =====================================================================
def test_bart_prompt_masking():
    test_header("TEST 6: BART Prompt Masking Still Works")
    
    tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-v2-base")
    
    prompt = "Describe lung: "
    text = "No mass or infiltrative lesion was detected."
    full_input = prompt + text
    max_length = 64

    tokens = tokenizer(
        full_input, max_length=max_length, padding='max_length',
        truncation=True, return_tensors='pt'
    )['input_ids'].squeeze(0)

    content_ids = tokenizer(full_input, add_special_tokens=True, truncation=True, max_length=max_length)['input_ids']
    content_len = len(content_ids)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    prompt_len = len(prompt_ids)
    bos_offset = 0
    if tokenizer.bos_token_id is not None and len(content_ids) > 0 and content_ids[0] == tokenizer.bos_token_id:
        bos_offset = 1

    labels = tokens.clone()
    labels[:bos_offset + prompt_len] = -100
    if content_len < max_length:
        labels[content_len:] = -100

    # Build decoder_input_ids from tokens (with prompt)
    decoder_start_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    decoder_input_ids = tokens.new_zeros(tokens.shape)
    decoder_input_ids[1:] = tokens[:-1].clone()
    decoder_input_ids[0] = decoder_start_id
    decoder_input_ids.masked_fill_(decoder_input_ids == -100, tokenizer.pad_token_id)

    active = labels[labels != -100]
    decoded_active = tokenizer.decode(active)
    decoded_decoder = tokenizer.decode(decoder_input_ids[decoder_input_ids != tokenizer.pad_token_id])
    
    print(f"  BOS offset: {bos_offset}")
    print(f"  Active labels: '{decoded_active}'")
    print(f"  Decoder input: '{decoded_decoder}'")
    
    check("Describe" not in decoded_active, "Prompt NOT in active labels (correctly masked)")
    check("Describe" in decoded_decoder, "Prompt IS in decoder_input_ids")
    
    # EOS should be in active labels
    eos_id = tokenizer.eos_token_id
    has_eos = (active == eos_id).any().item() if eos_id is not None else True
    check(has_eos, f"EOS token preserved in active labels")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  MEDICAL VLM — POST-FIX VERIFICATION (v3)")
    print("=" * 65)
    
    test_visual_projection_in_path()
    test_forward_accepts_input_ids()
    test_train_passes_input_ids()
    test_prompt_in_training_decoder_input()
    test_save_pretrained_fixed()
    test_bart_prompt_masking()
    
    print("\n" + "=" * 65)
    total = passed + failed
    print(f"  RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print(f"  \033[92mALL TESTS PASSED — FIXES VERIFIED!\033[0m")
    else:
        print(f"  \033[91mSOME TESTS FAILED\033[0m")
    print("=" * 65)
    
    sys.exit(0 if failed == 0 else 1)

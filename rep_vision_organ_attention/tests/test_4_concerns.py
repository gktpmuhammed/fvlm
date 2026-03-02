#!/usr/bin/env python3
"""
Targeted tests for 4 user-raised hardening concerns.
Run: cd /home/muhammedg/fvlm/rep_vision_organ_attention && python /tmp/test_4_concerns.py
"""
import sys, os, torch
project_dir = "/home/muhammedg/fvlm/rep_vision_organ_attention"
parent_dir = os.path.dirname(project_dir)
for p in [parent_dir, project_dir]:
    if p not in sys.path: sys.path.insert(0, p)

from transformers import AutoTokenizer

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"
results = []

def header(name):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

def check(condition, msg, is_warn=False):
    if condition:
        print(f"  {PASS} {msg}")
        results.append(('pass', msg))
    elif is_warn:
        print(f"  {WARN} {msg}")
        results.append(('warn', msg))
    else:
        print(f"  {FAIL} {msg}")
        results.append(('fail', msg))
    return condition


# =====================================================================
# CONCERN 1: GPT-2 doesn't auto-append EOS
# =====================================================================
def test_gpt2_eos_appending():
    header("CONCERN 1: GPT-2 EOS Auto-Appending")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text = "Describe lung: No mass detected."
    
    # Method 1: Default tokenization (add_special_tokens=True)
    ids_default = tokenizer(text, add_special_tokens=True)['input_ids']
    
    # Method 2: With padding to max_length (as in training code)
    ids_padded = tokenizer(text, max_length=30, padding='max_length', truncation=True)['input_ids']
    
    eos_id = tokenizer.eos_token_id  # 50256
    
    print(f"  eos_token_id: {eos_id}")
    print(f"  Default tokenization: {ids_default}")
    print(f"  Last real token: {ids_default[-1]} ({'= EOS ✓' if ids_default[-1] == eos_id else '≠ EOS ✗'})")
    
    has_eos_at_end = ids_default[-1] == eos_id
    
    if not has_eos_at_end:
        print(f"\n  GPT-2 tokenizer does NOT auto-append EOS!")
        print(f"  The content tokens end at '{tokenizer.decode([ids_default[-1]])}'")
        print(f"  Without explicit EOS, the model cannot learn to stop generating.")
        
        # Check: in training labels, what's at the last content position?
        content_len = len(ids_default)
        labels = torch.tensor(ids_padded)
        # After masking padding:
        if content_len < 30:
            labels[content_len:] = -100
        
        last_active_pos = content_len - 1
        last_active_token = labels[last_active_pos].item()
        
        print(f"\n  Last active label token: {last_active_token} = '{tokenizer.decode([last_active_token])}'")
        print(f"  The model's last prediction target is '{tokenizer.decode([last_active_token])}', not EOS")
        
        check(False, "GPT-2 does NOT append EOS — model can't learn to stop", is_warn=True)
        
        # Now check: does padding with pad_token=eos accidentally serve as EOS?
        # After content, all positions are pad_token_id=50256=eos
        # But we mask those positions with -100, so the model never sees them as targets
        print(f"\n  Padding tokens after content are masked (-100) → model never")
        print(f"  sees EOS as a target. It will generate until max_length.")
        
        # Show what BART does for comparison
        bart_tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-v2-base")
        bart_ids = bart_tokenizer(text, add_special_tokens=True)['input_ids']
        bart_eos = bart_tokenizer.eos_token_id
        
        print(f"\n  BART comparison:")
        print(f"  BART eos_token_id: {bart_eos}")
        print(f"  BART last token: {bart_ids[-1]} ({'= EOS ✓' if bart_ids[-1] == bart_eos else '≠ EOS ✗'})")
        check(bart_ids[-1] == bart_eos, "BART auto-appends EOS (no issue for BART)")
        
    else:
        check(True, "GPT-2 auto-appends EOS (concern not applicable)")


# =====================================================================
# CONCERN 2: Eval prompt stripping is brittle
# =====================================================================
def test_eval_prompt_stripping():
    header("CONCERN 2: Eval Prompt Stripping — text.replace() is Brittle")
    
    # Current code (evaluate.py line 199):
    # clean_pred = text.replace(p_text, "").strip()
    
    # This removes ALL occurrences, not just the prefix
    
    # Case 1: Prompt appears in generated medical text (unlikely but possible)
    p_text = "Describe lung: "
    text = "Describe lung: The CT Describe lung: findings show consolidation."
    
    result_replace = text.replace(p_text, "").strip()
    result_prefix = text.removeprefix(p_text).strip() if text.startswith(p_text) else text.strip()
    
    print(f"  Input: '{text}'")
    print(f"  replace():      '{result_replace}'")
    print(f"  removeprefix():  '{result_prefix}'")
    
    check(result_replace != result_prefix, 
          f"replace() removes {text.count(p_text)} occurrences vs removeprefix() removes 1")
    
    # Case 2: Model generates text that doesn't start with prompt
    text2 = "No findings. Describe lung: Normal."
    result_replace2 = text2.replace(p_text, "").strip()
    result_prefix2 = text2.removeprefix(p_text).strip() if text2.startswith(p_text) else text2.strip()
    
    print(f"\n  Input: '{text2}'")
    print(f"  replace():      '{result_replace2}'")
    print(f"  removeprefix():  '{result_prefix2}'")
    
    check(result_replace2 != result_prefix2,
          "replace() strips from middle; removeprefix() preserves original")
    
    # Case 3: Normal case (prompt at start only) — both methods give same result
    text3 = "Describe lung: Normal lung parenchyma."
    result_replace3 = text3.replace(p_text, "").strip()
    result_prefix3 = text3.removeprefix(p_text).strip()
    
    check(result_replace3 == result_prefix3,
          "Normal case: both methods give same result")
    
    print(f"\n  Verdict: text.replace() is indeed brittle. Should use removeprefix().")


# =====================================================================
# CONCERN 3: Hardcoded mask downsample grid (7×16×11)
# =====================================================================
def test_mask_grid_hardcoded():
    header("CONCERN 3: Hardcoded Mask Downsample Grid (7×16×11)")
    
    # medical_vlm.py line 91: f_d, f_h, f_w = 7, 16, 11
    # This must match ViT's token grid = (D/patch_D, H/patch_H, W/patch_W)
    
    # Current config: image_size=(112, 256, 352), patch_size=(16, 16, 32)
    image_size = (112, 256, 352)
    patch_size = (16, 16, 32)
    
    expected_grid = tuple(i // p for i, p in zip(image_size, patch_size))
    hardcoded_grid = (7, 16, 11)
    
    print(f"  image_size: {image_size}")
    print(f"  patch_size: {patch_size}")
    print(f"  Expected ViT grid: {expected_grid}")
    print(f"  Hardcoded grid: {hardcoded_grid}")
    
    match = expected_grid == hardcoded_grid
    check(match, f"Grid matches: {expected_grid} == {hardcoded_grid}")
    
    # Total tokens
    expected_tokens = expected_grid[0] * expected_grid[1] * expected_grid[2]
    hardcoded_tokens = hardcoded_grid[0] * hardcoded_grid[1] * hardcoded_grid[2]
    
    print(f"  Expected ViT tokens: {expected_tokens}")
    print(f"  Hardcoded grid tokens: {hardcoded_tokens}")
    check(expected_tokens == hardcoded_tokens, f"Token count matches: {expected_tokens}")
    
    # Check: no CLS token in this ViT
    # ViT.__init__ only creates cls_token if classification=True
    # MedicalVLM uses num_classes=0 → classification=False → no CLS
    print(f"\n  ViT uses classification=False, num_classes=0 → no CLS token")
    print(f"  seq_len = {expected_tokens} (pure patch tokens)")
    
    # Show what would happen if image/patch size changes
    print(f"\n  ⚠ If image/patch size changes:")
    alt_configs = [
        ((224, 224, 224), (16, 16, 16)),
        ((112, 256, 352), (16, 32, 32)),
        ((96, 192, 256), (16, 16, 32)),
    ]
    for img, patch in alt_configs:
        grid = tuple(i // p for i, p in zip(img, patch))
        tokens = grid[0] * grid[1] * grid[2]
        mismatch = grid != hardcoded_grid
        status = "MISMATCH ✗" if mismatch else "OK ✓"
        print(f"    {img} / {patch} → grid={grid}, tokens={tokens} [{status}]")
    
    check(True, "Grid correct for current config, but hardcoded — fragile if config changes", is_warn=True)
    
    # Read source to verify it's truly hardcoded (not derived from config)
    with open(os.path.join(project_dir, "medical_vlm.py"), 'r') as f:
        source = f.read()
    
    is_hardcoded = "f_d, f_h, f_w = 7, 16, 11" in source
    check(is_hardcoded, "Grid IS hardcoded as literal values (not computed from config)")


# =====================================================================
# CONCERN 4: Weighted loss — zero-weight samples dilute the mean
# =====================================================================
def test_weighted_loss_scaling():
    header("CONCERN 4: Weighted Loss — Zero Samples Dilute Mean")
    
    # Current code (medical_vlm.py lines 414-421):
    # sample_loss = sample_loss * flat_weights
    # lm_loss = sample_loss.mean()
    
    # With 12 organs per patient, batch_size=1:
    # flat_weights has 12 elements, some may be 0.0
    
    # Scenario: 8 organs with weight=1.0, 4 with weight=0.0
    sample_losses = torch.tensor([0.5] * 12)  # uniform loss per sample
    weights = torch.tensor([1.0]*8 + [0.0]*4)
    
    # Current method: mean over all 12
    weighted = sample_losses * weights
    current_loss = weighted.mean()
    
    # Alternative: mean over non-zero weights only
    nonzero_mask = weights > 0
    normalized_loss = weighted[nonzero_mask].mean() if nonzero_mask.any() else torch.tensor(0.0)
    
    # Another alternative: sum(weighted) / sum(weights)
    weighted_mean = weighted.sum() / (weights.sum() + 1e-8)
    
    print(f"  Sample losses: {sample_losses.tolist()}")
    print(f"  Weights: {weights.tolist()}")
    print(f"  Active samples: {int(nonzero_mask.sum())}/12")
    print(f"")
    print(f"  Current method (mean over all):     {current_loss.item():.4f}")
    print(f"  Alt 1 (mean over non-zero):         {normalized_loss.item():.4f}")
    print(f"  Alt 2 (sum/sum_weights):             {weighted_mean.item():.4f}")
    
    # The current method gives 0.3333 instead of 0.5
    # This means the effective learning rate is lower when more organs are dropped
    ratio = current_loss / normalized_loss
    print(f"\n  Current/Normalized ratio: {ratio.item():.4f}")
    print(f"  → Loss is scaled DOWN by {1-ratio.item():.0%} when 4/12 organs dropped")
    
    check(abs(current_loss.item() - normalized_loss.item()) > 0.01,
          f"CONFIRMS: zero-weight samples dilute loss ({current_loss.item():.4f} vs {normalized_loss.item():.4f})")
    
    # Impact analysis: how bad is this?
    print(f"\n  Impact analysis:")
    for n_dropped in range(0, 13):
        w = torch.tensor([1.0]*(12-n_dropped) + [0.0]*n_dropped)
        s = torch.tensor([0.5]*12)
        current = (s * w).mean().item()
        proper = (s * w).sum().item() / (w.sum().item() + 1e-8)
        scale = current / proper if proper > 0 else 0
        print(f"    {n_dropped}/12 dropped: loss={current:.3f} (should be {proper:.3f}, scale={scale:.1%})")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  4 HARDENING CONCERNS — VALIDATION")
    print("=" * 70)
    
    test_gpt2_eos_appending()
    test_eval_prompt_stripping()
    test_mask_grid_hardcoded()
    test_weighted_loss_scaling()
    
    print("\n" + "=" * 70)
    passes = sum(1 for r in results if r[0] == 'pass')
    warns = sum(1 for r in results if r[0] == 'warn')
    fails = sum(1 for r in results if r[0] == 'fail')
    
    print(f"  SUMMARY:")
    for status, msg in results:
        icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}[status]
        print(f"    {icon} {msg}")
    
    print(f"\n  {passes} passed, {warns} warnings, {fails} failed")
    print("=" * 70)

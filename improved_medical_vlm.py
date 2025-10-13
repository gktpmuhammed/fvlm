"""
Improved Medical VLM with LoRA for Vision Encoder + BioGPT Decoder
- LoRA adapters on last 4-6 layers of ViT
- BioGPT decoder (medical domain pretrained)
- Better training configuration
- NLP metrics for loss computation
"""

import torch
import torch.nn as nn
from transformers import BioGptConfig, BioGptForCausalLM, BioGptTokenizer
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from monai import transforms
import os
import math

# ==================== LoRA Implementation ====================
class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for efficient fine-tuning.
    Implements: h = W0*x + (B*A)*x where A and B are low-rank matrices
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices: A (in_features x rank), B (rank x out_features)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)

        # Initialize A with kaiming_uniform and B with zeros (as in original paper)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x, original_output):
        """
        x: input to the linear layer
        original_output: output from the frozen pretrained linear layer
        """
        # Compute LoRA contribution: x @ A @ B
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B
        return original_output + (lora_output * self.scaling)


class LoRALinear(nn.Module):
    """
    Wraps a frozen linear layer with LoRA adaptation
    """
    def __init__(self, linear_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.linear = linear_layer
        self.linear.weight.requires_grad = False  # Freeze original weights
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

    def forward(self, x):
        original_output = self.linear(x)
        return self.lora(x, original_output)


def apply_lora_to_attention(vit_model, start_layer=8, rank=8, alpha=16):
    """
    Apply LoRA to attention layers in ViT starting from start_layer
    The lavis ViT has 12 blocks, so start_layer=8 means last 4 layers
    """
    print(f"Applying LoRA to ViT attention layers from layer {start_layer} onwards...")

    num_layers_modified = 0

    # The lavis ViT has blocks attribute
    if not hasattr(vit_model, 'blocks'):
        print("Warning: ViT model doesn't have 'blocks' attribute")
        return vit_model

    blocks = vit_model.blocks
    total_blocks = len(blocks)
    print(f"Found {total_blocks} transformer blocks in ViT")

    for layer_idx in range(len(blocks)):
        if layer_idx < start_layer:
            continue

        block = blocks[layer_idx]

        # Find attention module (typically block.attn)
        if hasattr(block, 'attn'):
            attn = block.attn

            # Check for QKV projection (common in ViT implementations)
            if hasattr(attn, 'qkv') and isinstance(attn.qkv, nn.Linear):
                print(f"  Layer {layer_idx}: Adding LoRA to QKV projection")
                attn.qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha)
                num_layers_modified += 1
            else:
                # Separate Q, K, V projections
                modified_this_layer = False
                if hasattr(attn, 'q') and isinstance(attn.q, nn.Linear):
                    attn.q = LoRALinear(attn.q, rank=rank, alpha=alpha)
                    modified_this_layer = True
                if hasattr(attn, 'k') and isinstance(attn.k, nn.Linear):
                    attn.k = LoRALinear(attn.k, rank=rank, alpha=alpha)
                    modified_this_layer = True
                if hasattr(attn, 'v') and isinstance(attn.v, nn.Linear):
                    attn.v = LoRALinear(attn.v, rank=rank, alpha=alpha)
                    modified_this_layer = True

                if modified_this_layer:
                    print(f"  Layer {layer_idx}: Adding LoRA to separate Q/K/V projections")
                    num_layers_modified += 1

            # Also add LoRA to output projection if it exists
            if hasattr(attn, 'proj') and isinstance(attn.proj, nn.Linear):
                print(f"  Layer {layer_idx}: Adding LoRA to output projection")
                attn.proj = LoRALinear(attn.proj, rank=rank, alpha=alpha)

    print(f"Applied LoRA to {num_layers_modified} attention layers")
    return vit_model


# ==================== Model Architecture ====================
class ImprovedMedicalVLM(nn.Module):
    def __init__(
        self, 
        vision_encoder_path, 
        decoder_model_name="microsoft/biogpt",
        lora_rank=8,
        lora_alpha=16,
        vit_layers_to_adapt=4  # Adapt last N layers
    ):
        super().__init__()

        # 1. Load pretrained vision encoder with flexible checkpoint loading
        print("Loading pretrained vision encoder...")
        checkpoint = torch.load(vision_encoder_path, map_location='cpu')

        # Import the correct ViT from lavis
        from lavis.models.blip_models.vit import ViT

        # Use ONLY the parameters that the lavis ViT accepts
        self.vision_encoder = ViT(
            in_channels=1,
            img_size=(112, 256, 352),
            patch_size=(16, 16, 32),
            num_classes=0,  # No classification head
        )

        # Load weights from checkpoint - HANDLE MULTIPLE FORMATS
        print("Extracting vision encoder weights from checkpoint...")

        # Try different checkpoint formats
        vision_state = {}

        if 'state_dict' in checkpoint:
            # PyTorch Lightning format
            print("  Detected PyTorch Lightning checkpoint format (state_dict)")
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('visual_encoder.'):
                    new_key = k.replace('visual_encoder.', '')
                    vision_state[new_key] = v
        elif 'model' in checkpoint:
            # Model dict format
            print("  Detected model dict checkpoint format")
            for k, v in checkpoint['model'].items():
                if k.startswith('visual_encoder.'):
                    new_key = k.replace('visual_encoder.', '')
                    vision_state[new_key] = v
        else:
            # Assume checkpoint IS the state dict
            print("  Detected direct state dict format")
            for k, v in checkpoint.items():
                if k.startswith('visual_encoder.'):
                    new_key = k.replace('visual_encoder.', '')
                    vision_state[new_key] = v
                else:
                    # Maybe keys don't have prefix
                    vision_state[k] = v

        if len(vision_state) == 0:
            print("  WARNING: No vision encoder weights found in checkpoint!")
            print(f"  Checkpoint keys: {list(checkpoint.keys())[:5]}...")
        else:
            print(f"  Found {len(vision_state)} vision encoder parameters")

        missing_keys, unexpected_keys = self.vision_encoder.load_state_dict(vision_state, strict=False)
        print(f"Loaded vision encoder weights")
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")

        # Freeze all ViT parameters initially
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # Apply LoRA to last N layers
        # The lavis ViT typically has 12 blocks
        total_layers = 12
        start_layer = total_layers - vit_layers_to_adapt
        self.vision_encoder = apply_lora_to_attention(
            self.vision_encoder, 
            start_layer=start_layer,
            rank=lora_rank,
            alpha=lora_alpha
        )

        # 2. Load BioGPT decoder
        print("Loading BioGPT decoder...")
        self.tokenizer = BioGptTokenizer.from_pretrained(decoder_model_name)

        biogpt_config = BioGptConfig.from_pretrained(decoder_model_name)
        biogpt_config.is_decoder = True
        biogpt_config.add_cross_attention = True

        self.decoder = BioGptForCausalLM.from_pretrained(
            decoder_model_name,
            config=biogpt_config,
            ignore_mismatched_sizes=True
        )

        # Freeze BioGPT decoder
        for param in self.decoder.parameters():
            param.requires_grad = False

        # Unfreeze cross-attention layers in BioGPT
        # BioGPT has: self.decoder.biogpt.layers (list of decoder layers)
        if hasattr(self.decoder, 'biogpt') and hasattr(self.decoder.biogpt, 'layers'):
            print("Unfreezing cross-attention layers in BioGPT...")
            for layer in self.decoder.biogpt.layers:
                if hasattr(layer, 'encoder_attn'):  # Cross-attention layer
                    for param in layer.encoder_attn.parameters():
                        param.requires_grad = True
                    print(f"  Unfroze encoder_attn in layer")
        else:
            print("Warning: Could not find BioGPT layers for cross-attention unfreezing")

        # NEW: Resampler to reduce number of visual tokens
        self.visual_resampler = nn.AdaptiveAvgPool1d(512)

        # 3. Projection layer: 768 (ViT) -> 1024 (BioGPT hidden size)
        vit_dim = 768
        biogpt_dim = biogpt_config.hidden_size  # 1024 for BioGPT

        self.projection = nn.Sequential(
            nn.Linear(vit_dim, biogpt_dim * 2),
            nn.LayerNorm(biogpt_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(biogpt_dim * 2, biogpt_dim),
            nn.LayerNorm(biogpt_dim)
        )

        print("\nModel initialized:")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    def forward(self, images=None, input_ids=None, attention_mask=None, labels=None, prompt_embeds=None):
        # Handle generation case where we pass pre-computed prompt embeddings
        if prompt_embeds is not None:
            text_embeds = self.decoder.biogpt.embed_tokens(input_ids)
            inputs_embeds = torch.cat([prompt_embeds, text_embeds], dim=1)
            visual_attention_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=prompt_embeds.device)
            attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)
            
            # During generation, we don't have visual_labels
            if labels is not None:
                visual_labels = torch.full(prompt_embeds.shape[:2], -100, dtype=torch.long, device=prompt_embeds.device)
                labels = torch.cat([visual_labels, labels], dim=1)

            return self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )

        # Handle training/evaluation case
        visual_features = self.vision_encoder(images)
        if isinstance(visual_features, tuple):
            visual_features = visual_features[0]
        
        resampled_features = self.visual_resampler(visual_features.permute(0, 2, 1)).permute(0, 2, 1)
        projected_features = self.projection(resampled_features)

        text_embeds = self.decoder.biogpt.embed_tokens(input_ids)
        inputs_embeds = torch.cat([projected_features, text_embeds], dim=1)
        
        visual_attention_mask = torch.ones(projected_features.shape[:2], dtype=torch.long, device=images.device)
        combined_attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)

        if labels is not None:
            visual_labels = torch.full(projected_features.shape[:2], -100, dtype=torch.long, device=images.device)
            labels = torch.cat([visual_labels, labels], dim=1)

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=labels,
            return_dict=True
        )

        return outputs

    def generate(
        self, 
        images, 
        max_length=256, 
        num_beams=1,  # Greedy for now (beam search is complex to implement)
        temperature=1.0,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        early_stopping=True,
        log_attention=False
    ):
        """
        Generate text with visual prefix - Manual implementation
        BioGPT doesn't support inputs_embeds in .generate(), so we do it manually
        """
        self.eval()
        with torch.no_grad():
            batch_size = images.size(0)

            # 1. Encode and project visual features
            visual_features = self.vision_encoder(images)
            if isinstance(visual_features, tuple):
                visual_features = visual_features[0]

            resampled_features = self.visual_resampler(visual_features.permute(0, 2, 1)).permute(0, 2, 1)
            visual_embeds = self.projection(resampled_features)
            num_visual_tokens = visual_embeds.size(1)

            # 2. Start with BOS token
            generated_ids = torch.full(
                (batch_size, 1),
                self.tokenizer.bos_token_id,
                dtype=torch.long,
                device=images.device
            )

            # 3. Manual autoregressive generation
            for step in range(max_length):
                # Get text embeddings for current sequence
                text_embeds = self.decoder.biogpt.embed_tokens(generated_ids)

                # Concatenate visual prefix with text
                combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

                # Create attention mask
                seq_len = generated_ids.size(1)
                combined_attention_mask = torch.ones(
                    batch_size, num_visual_tokens + seq_len,
                    dtype=torch.long,
                    device=images.device
                )

                # Forward pass
                outputs = self.decoder(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    return_dict=True,
                    output_attentions=True
                )

                # Log attention weights if requested
                if log_attention:
                    # attentions is a tuple of (layer, batch, head, seq, seq)
                    last_layer_attention = outputs.attentions[-1] # (batch, head, seq, seq)
                    
                    # We want attention from the LAST token to all previous tokens
                    # The last token is the query
                    attention_from_last_token = last_layer_attention[:, :, -1, :] # (batch, head, seq)
                    
                    # Average across heads
                    attention_from_last_token = attention_from_last_token.mean(dim=1) # (batch, seq)
                    
                    # Attention to visual prefix (first N tokens)
                    attention_to_visual = attention_from_last_token[:, :num_visual_tokens].sum(dim=1) # (batch)

                    print(f"[Step {step}] Attention to visual prefix: {attention_to_visual.item():.4f}")

                # Get next token logits (from the last position)
                next_token_logits = outputs.logits[:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated_ids[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty

                # Get next token (greedy)
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_tokens], dim=1)

                # Check for EOS token (early stopping)
                if early_stopping and (next_tokens == self.tokenizer.eos_token_id).all():
                    break

            return generated_ids


# ==================== Dataset (unchanged) ====================
class MedicalReportDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image
        image_path = row['image_path']
        image = self.transform({'image': image_path})['image']

        # Combine findings and impression
        report_text = f"{row['findings']} {row['impression']}"

        # Tokenize
        encoding = self.tokenizer(
            report_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'images': image.squeeze(0),
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0)
        }
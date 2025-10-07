"""
Improved Medical VLM with LoRA for Vision Encoder
- LoRA adapters on last 4-6 layers of ViT
- Fixed BERT decoder (for now)
- Better training configuration
- NLP metrics for loss computation
- HANDLES MULTIPLE CHECKPOINT FORMATS
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertLMHeadModel, AutoTokenizer
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
        bert_model_name="/home/muhammedg/fvlm/BiomedVLP-CXR-BERT-specialized",
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

        # 2. Load BERT decoder (keeping frozen for now as requested)
        print("Loading BERT decoder...")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert_config = BertConfig.from_pretrained(bert_model_name)
        bert_config.is_decoder = True
        bert_config.add_cross_attention = True

        self.decoder = BertLMHeadModel.from_pretrained(
            bert_model_name,
            config=bert_config,
            ignore_mismatched_sizes=True
        )

        # Freeze BERT decoder
        for param in self.decoder.parameters():
            param.requires_grad = False

        # Unfreeze cross-attention layers
        for layer in self.decoder.bert.encoder.layer:
            if hasattr(layer, 'crossattention'):
                for param in layer.crossattention.parameters():
                    param.requires_grad = True

        # 3. Enhanced trainable projection layer
        self.projection = nn.Sequential(
            nn.Linear(768, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1536, 768),
            nn.LayerNorm(768)
        )

        print("Model initialized with:")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    def forward(self, images, input_ids, attention_mask, labels=None):
        # Encode images
        visual_features = self.vision_encoder(images)

        if isinstance(visual_features, tuple):
            visual_features = visual_features[0]
        encoder_hidden_states = self.projection(visual_features)

        # Create encoder attention mask
        batch_size = encoder_hidden_states.size(0)
        seq_len = encoder_hidden_states.size(1)
        encoder_attention_mask = torch.ones(
            batch_size, seq_len,
            dtype=torch.long,
            device=encoder_hidden_states.device
        )

        # Decode with cross-attention
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            return_dict=True
        )

        return outputs

    def generate(
        self, 
        images, 
        max_length=512, 
        num_beams=5,
        temperature=1.0,
        repetition_penalty=2.0,  # IMPORTANT: prevents repetition
        no_repeat_ngram_size=3,  # Prevents repeating 3-grams
        length_penalty=1.0,
        early_stopping=True
    ):
        """Generate text with anti-repetition mechanisms"""
        self.eval()
        with torch.no_grad():
            # Encode images
            visual_features = self.vision_encoder(images)
            if isinstance(visual_features, tuple):
                visual_features = visual_features[0]
            encoder_hidden_states = self.projection(visual_features)

            batch_size = encoder_hidden_states.size(0)
            seq_len = encoder_hidden_states.size(1)
            encoder_attention_mask = torch.ones(
                batch_size, seq_len,
                dtype=torch.long,
                device=encoder_hidden_states.device
            )

            # Start generation with [CLS] token
            input_ids = torch.full(
                (batch_size, 1),
                self.tokenizer.cls_token_id,
                dtype=torch.long,
                device=images.device
            )

            # Generate with improved decoding
            generated = self.decoder.generate(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                repetition_penalty=repetition_penalty,  # Key for preventing collapse
                no_repeat_ngram_size=no_repeat_ngram_size,  # Prevents n-gram repetition
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.sep_token_id,
                bos_token_id=self.tokenizer.cls_token_id,
            )

            return generated


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
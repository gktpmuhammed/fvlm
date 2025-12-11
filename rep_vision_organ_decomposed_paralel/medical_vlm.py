"""
Unified Medical Vision-Language Model
Feature: "One-Pass" ROI Pooling for Multi-Organ Generation
Optimization: Dynamic Batching (Filters empty organs to speed up training)
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    VisionEncoderDecoderModel, 
    VisionEncoderDecoderConfig,
    AutoTokenizer,
    AutoModelForCausalLM, 
    AutoConfig,
    ViTConfig,
    BartForCausalLM,
    BartConfig,
    BertConfig,
    BertModel
)
from transformers.modeling_outputs import BaseModelOutput

# Fix local import path for lavis
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class ROI_ViTWrapper(nn.Module):
    """
    Standard ViT -> ROI Pooling -> Decoder
    Extracts features for N organs from a single image pass.
    """
    def __init__(self, vit_model, config):
        super().__init__()
        self.vit = vit_model
        self.config = config
        self.main_input_name = "pixel_values"

    def forward(self, pixel_values, organ_masks=None, **kwargs):
        # 1. Run Vision Encoder ONCE
        # pixel_values: (Batch, 1, D, H, W)
        outputs = self.vit(pixel_values)
        if isinstance(outputs, tuple):
            image_feats = outputs[0] # (Batch, N_patches, Hidden)
        else:
            image_feats = outputs

        # 2. ROI Pooling (Extract Features for N organs simultaneously)
        if organ_masks is not None:
            # Handle dimensions: can be 5D (B,N,D,H,W) or 6D (B,N,C,D,H,W)
            if organ_masks.dim() == 6:
                # Shape: (B, N, 1, D, H, W)
                B, N_organs, C, D_m, H_m, W_m = organ_masks.shape
                # Reshape to (B * N, C, D, H, W)
                flat_masks = organ_masks.view(B * N_organs, C, D_m, H_m, W_m)
            else:
                # Shape: (B, N, D, H, W)
                B, N_organs, D_m, H_m, W_m = organ_masks.shape
                # Add channel dim: (B * N, 1, D, H, W)
                flat_masks = organ_masks.view(B * N_organs, 1, D_m, H_m, W_m)

            # Feature map dimensions (based on ViT patch size 16,16,32)
            # 112/16=7, 256/16=16, 352/32=11
            f_d, f_h, f_w = 7, 16, 11 
            
            # Interpolate to feature grid size (Area interpolation preserves small organs)
            masks_down = F.interpolate(flat_masks, size=(f_d, f_h, f_w), mode='area')
            
            # Flatten masks to match patch sequence: (B * N, 1232)
            masks_flat = masks_down.view(B, N_organs, -1)
            
            # Normalize masks (Weighted Average Pooling)
            mask_sums = masks_flat.sum(dim=2, keepdim=True) + 1e-6
            masks_norm = masks_flat / mask_sums
            
            # Feature Extraction (Batch Matrix Multiplication)
            # (B, N_organs, Patches) @ (B, Patches, Hidden) -> (B, N_organs, Hidden)
            organ_embeddings = torch.bmm(masks_norm, image_feats)
            
            # Flatten for Decoder ("Super Batch")
            # Output: (B * N_organs, 1, Hidden)
            final_embeddings = organ_embeddings.view(B * N_organs, 1, -1)
            
            return BaseModelOutput(last_hidden_state=final_embeddings)

        # Fallback if no masks provided
        return BaseModelOutput(last_hidden_state=image_feats)

class MedicalVLM(nn.Module):
    def __init__(
        self,
        vision_encoder_path,
        decoder_model_name="gpt2", 
        image_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        **kwargs
    ):
        super().__init__()
        
        # 1. SETUP ENCODER (3D ViT)
        hidden_size = 768 
        encoder_config = ViTConfig(
            hidden_size=hidden_size, num_hidden_layers=12, num_attention_heads=12, 
            intermediate_size=3072, image_size=image_size, patch_size=patch_size, num_channels=1
        )

        try:
            from lavis.models.blip_models.vit import ViT
            vision_encoder = ViT(in_channels=1, img_size=image_size, patch_size=patch_size, num_classes=0)
        except ImportError:
            raise ImportError("Could not find 'lavis.models.blip_models.vit'.")
        
        if os.path.exists(vision_encoder_path):
            checkpoint = torch.load(vision_encoder_path, map_location='cpu')
            vision_state = {}
            source = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            for k, v in source.items():
                if k.startswith('visual_encoder.'):
                    vision_state[k.replace('visual_encoder.', '')] = v
                elif not k.startswith('text_encoder') and not k.startswith('temp'):
                    vision_state[k] = v
            vision_encoder.load_state_dict(vision_state, strict=False)
            print("  ViT weights loaded successfully.")
        
        # Freeze ViT
        for param in vision_encoder.parameters():
            param.requires_grad = False

        # 2. WRAP ENCODER (ROI POOLING)
        wrapped_encoder = ROI_ViTWrapper(vision_encoder, encoder_config)

        # 3. SETUP DECODER
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        decoder_config = AutoConfig.from_pretrained(decoder_model_name)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True 
        decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name, config=decoder_config)

        # Surgical Freezing
        for param in decoder.parameters(): param.requires_grad = False
        keywords = ["crossattention", "ln_", "layer_norm", "lm_head", "output_projection"]
        for name, param in decoder.named_parameters():
            if any(k in name for k in keywords):
                param.requires_grad = True

        # 4. COMPILE
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.model = VisionEncoderDecoderModel(config=config)
        self.model.encoder = wrapped_encoder
        self.model.decoder = decoder
        
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.tokenizer.vocab_size

    def forward(self, pixel_values, organ_masks=None, labels=None, **kwargs):
        # 1. Encode (B -> B*N)
        # encoder_outputs.last_hidden_state shape: (B*N, 1, Hidden)
        encoder_outputs = self.model.encoder(pixel_values=pixel_values, organ_masks=organ_masks)
        
        # 2. Reshape Labels and Filter Empty Organs (OPTIMIZATION)
        flat_labels = None
        if labels is not None:
            B, N_organs, Seq_Len = labels.shape
            # Reshape to (B*N, Seq_Len)
            flat_labels = labels.view(B * N_organs, Seq_Len)
            
            # --- DYNAMIC BATCHING ---
            # Identify rows where at least one token is NOT -100
            valid_rows = (flat_labels != -100).any(dim=1)
            
            if valid_rows.any():
                # Filter Embeddings
                # Access embeddings: (B*N, 1, Hidden)
                original_embeds = encoder_outputs.last_hidden_state
                filtered_embeds = original_embeds[valid_rows]
                
                # Update the encoder output object
                encoder_outputs.last_hidden_state = filtered_embeds
                
                # Filter Labels
                flat_labels = flat_labels[valid_rows]
            else:
                # Edge case: All are -100 (unlikely but safe to handle)
                pass

        # 3. Decode (Runs only on valid organs)
        return self.model(
            encoder_outputs=encoder_outputs,
            labels=flat_labels, 
            return_dict=True, 
            **kwargs
        )

    def generate(self, pixel_values, organ_masks=None, **kwargs):
        encoder_outputs = self.model.encoder(pixel_values=pixel_values, organ_masks=organ_masks)
        return self.model.generate(encoder_outputs=encoder_outputs, **kwargs)

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
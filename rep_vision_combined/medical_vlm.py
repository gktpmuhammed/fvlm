"""
Unified Medical Vision-Language Model
Supports: GPT-2, BioGPT, and other Causal LM decoders.
Features:
- Custom 3D ViT Encoder
- Surgical Fine-Tuning (Cross-Attention + LayerNorms + Head)
- Auto-handling of weight tying issues
"""
import sys
import os
import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel, 
    VisionEncoderDecoderConfig,
    AutoTokenizer,
    AutoModelForCausalLM, 
    AutoConfig,
    ViTConfig
)
from transformers.modeling_outputs import BaseModelOutput

# ------------------------------------------------------------------
# FIX: Force local import of 'lavis'
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from lavis.models.blip_models.vit import ViT
except ImportError:
    raise ImportError("Could not find 'lavis.models.blip_models.vit'. Please check your directory structure.")

class ViTWrapper(nn.Module):
    """Wrapper to ensure ViT outputs match HuggingFace expectations"""
    def __init__(self, vit_model, config):
        super().__init__()
        self.vit = vit_model
        self.config = config
        # FIX: Required for HF generate() to know how to pass inputs
        self.main_input_name = "pixel_values"

    def forward(self, pixel_values, **kwargs):
        outputs = self.vit(pixel_values)
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = outputs
        return BaseModelOutput(last_hidden_state=last_hidden_state)

class MedicalVLM(nn.Module):
    def __init__(
        self,
        vision_encoder_path,
        decoder_model_name="microsoft/biogpt", 
        image_size=(112, 256, 352),
        patch_size=(16, 16, 32),
    ):
        super().__init__()
        print(f"Initializing Medical VLM with Decoder: {decoder_model_name}")

        # ------------------------------------------------------------------
        # 1. SETUP ENCODER (3D ViT)
        # ------------------------------------------------------------------
        encoder_config = ViTConfig(
            hidden_size=768, num_hidden_layers=12, num_attention_heads=12, 
            intermediate_size=3072, image_size=224, patch_size=16, num_channels=1
        )

        vision_encoder = ViT(in_channels=1, img_size=image_size, patch_size=patch_size, num_classes=0)
        
        # Load ViT weights
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
        else:
            print(f"WARNING: Vision encoder path {vision_encoder_path} not found. Using random init.")
        
        wrapped_encoder = ViTWrapper(vision_encoder, encoder_config)
        
        # Freeze Encoder
        for param in wrapped_encoder.parameters():
            param.requires_grad = False

        # ------------------------------------------------------------------
        # 2. SETUP DECODER (AutoModel handles GPT2 & BioGPT)
        # ------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        decoder_config = AutoConfig.from_pretrained(decoder_model_name)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True 

        decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name, config=decoder_config)

        # ------------------------------------------------------------------
        # 3. UNIFIED SURGICAL FREEZING
        # ------------------------------------------------------------------
        # First, freeze the entire decoder
        for param in decoder.parameters():
            param.requires_grad = False
            
        all_params = sum(p.numel() for p in decoder.parameters())
        
        # Keywords covering both GPT-2 and BioGPT parameter naming conventions
        # GPT2: crossattention, ln_
        # BioGPT: encoder_attn, layer_norm
        trainable_keywords = ["crossattention", "encoder_attn", "ln_", "layer_norm", "layernorm"]
        
        for name, param in decoder.named_parameters():
            if any(k in name for k in trainable_keywords):
                param.requires_grad = True

        # ------------------------------------------------------------------
        # 4. EXPLICITLY UNFREEZE LM HEAD
        # ------------------------------------------------------------------
        # GPT-2 ties input embeddings to output head. Freezing input embeds often
        # accidentally freezes the head. We must forcefully unfreeze the head 
        # to allow vocabulary adaptation.
        
        head_unfrozen = False
        
        # Try generic LM head names
        if hasattr(decoder, "lm_head"):
            for param in decoder.lm_head.parameters():
                param.requires_grad = True
            head_unfrozen = True
            print("  > Unfrozen 'lm_head'")
            
        # Try BioGPT specific head name
        if hasattr(decoder, "output_projection"):
            for param in decoder.output_projection.parameters():
                param.requires_grad = True
            head_unfrozen = True
            print("  > Unfrozen 'output_projection'")

        # Safety check: if we couldn't find the head attribute directly, look by name
        if not head_unfrozen:
            for name, param in decoder.named_parameters():
                if "lm_head" in name or "output_projection" in name:
                    param.requires_grad = True

        # Calculate final stats
        trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"  > Trainable Params: {trainable_params:,} / {all_params:,} ({(trainable_params/all_params)*100:.2f}%)")

        # ------------------------------------------------------------------
        # 5. COMPILE MODEL
        # ------------------------------------------------------------------
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.model = VisionEncoderDecoderModel(config=config)
        self.model.encoder = wrapped_encoder
        self.model.decoder = decoder
        
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.tokenizer.vocab_size

        # Handle dimension mismatch (e.g. ViT 768 -> BioGPT 1024)
        if encoder_config.hidden_size != decoder_config.hidden_size:
            if hasattr(self.model, 'enc_to_dec_proj'):
                 for param in self.model.enc_to_dec_proj.parameters():
                     param.requires_grad = True

    def forward(self, pixel_values, labels=None, **kwargs):
        return self.model(pixel_values=pixel_values, labels=labels, return_dict=True, **kwargs)

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
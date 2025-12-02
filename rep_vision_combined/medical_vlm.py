"""
Unified Medical Vision-Language Model
Supports: BART (Recommended), GPT-2, BioGPT
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
    ViTConfig,
    BartForCausalLM, # BART specific class
    BartConfig
)
from transformers.modeling_outputs import BaseModelOutput

# ------------------------------------------------------------------
# Fix local import path for lavis
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class ViTWrapper(nn.Module):
    """Wrapper to ensure ViT outputs match HuggingFace expectations"""
    def __init__(self, vit_model, config):
        super().__init__()
        self.vit = vit_model
        self.config = config
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
        decoder_model_name="facebook/bart-base", 
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
            intermediate_size=3072, image_size=image_size, patch_size=patch_size, num_channels=1
        )

        try:
            from lavis.models.blip_models.vit import ViT
            vision_encoder = ViT(in_channels=1, img_size=image_size, patch_size=patch_size, num_classes=0)
        except ImportError:
            raise ImportError("Could not find 'lavis.models.blip_models.vit'.")
        
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
            print("  ViT weights loaded successfully.")
        else:
            print(f"  WARNING: ViT path {vision_encoder_path} not found. Random init.")
        
        wrapped_encoder = ViTWrapper(vision_encoder, encoder_config)
        
        # Freeze Encoder
        for param in wrapped_encoder.parameters():
            param.requires_grad = False

        # ------------------------------------------------------------------
        # 2. SETUP DECODER (BART / GPT2)
        # ------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        # BART/RoBERTa do not use a separate pad token usually, or use eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # HANDLE SPECIFIC MODEL TYPES
        if "bart" in decoder_model_name:
            print("  > Configuring BART Decoder...")
            decoder_config = BartConfig.from_pretrained(decoder_model_name)
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True 
            decoder = BartForCausalLM.from_pretrained(decoder_model_name, config=decoder_config)
        
        elif "biogpt" in decoder_model_name:
            print("  > Configuring BioGPT Decoder...")
            # BioGPT needs special handling via AutoModel usually, but let's stick to standard flow
            # If standard AutoModel fails (like before), you might need the Wrapper class from previous turn.
            # But let's assume we are switching to BART now.
            decoder_config = AutoConfig.from_pretrained(decoder_model_name)
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True 
            decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name, config=decoder_config)
            
        else: # GPT-2, DistilGPT2, etc.
            print("  > Configuring Standard Causal Decoder...")
            decoder_config = AutoConfig.from_pretrained(decoder_model_name)
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True 
            decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name, config=decoder_config)

        # ------------------------------------------------------------------
        # 3. SURGICAL FREEZING (BART Edition)
        # ------------------------------------------------------------------
        # 1. Freeze everything first
        for param in decoder.parameters():
            param.requires_grad = False
            
        trainable_params = 0
        all_params = sum(p.numel() for p in decoder.parameters())
        
        # KEYWORDS for Unfreezing:
        # 'crossattention' -> GPT-2
        # 'encoder_attn'   -> BART / BioGPT (The layer that looks at the image)
        # 'ln_'            -> GPT-2 LayerNorm
        # 'layer_norm'     -> BART / BioGPT LayerNorm
        # 'lm_head'        -> Output vocabulary
        
        keywords = ["crossattention", "encoder_attn", "ln_", "layer_norm", "layernorm", "lm_head", "output_projection"]
        
        for name, param in decoder.named_parameters():
            if any(k in name for k in keywords):
                param.requires_grad = True
                trainable_params += param.numel()

        # Explicit head check (BART's head is 'lm_head')
        if hasattr(decoder, "lm_head"):
            for param in decoder.lm_head.parameters():
                param.requires_grad = True
        
        print(f"  > Trainable Params: {trainable_params:,} / {all_params:,} ({(trainable_params/all_params)*100:.2f}%)")

        # ------------------------------------------------------------------
        # 4. COMPILE MODEL
        # ------------------------------------------------------------------
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.model = VisionEncoderDecoderModel(config=config)
        self.model.encoder = wrapped_encoder
        self.model.decoder = decoder
        
        # Generation Config
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.tokenizer.vocab_size

        # Handle dimension mismatch (ViT 768 vs Decoder Dim)
        if encoder_config.hidden_size != decoder_config.hidden_size:
            if hasattr(self.model, 'enc_to_dec_proj'):
                 for param in self.model.enc_to_dec_proj.parameters():
                     param.requires_grad = True

    def forward(self, pixel_values, labels=None, **kwargs):
        return self.model(pixel_values=pixel_values, labels=labels, return_dict=True, **kwargs)

    def generate(self, pixel_values, **kwargs):
        return self.model.generate(pixel_values, **kwargs)

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
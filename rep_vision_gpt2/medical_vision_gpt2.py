"""
Medical Vision-Language Model using GPT2
Refined for Surgical Fine-Tuning (Cross-Attention Only)
"""

import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel, 
    VisionEncoderDecoderConfig,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    ViTConfig
)
from transformers.modeling_outputs import BaseModelOutput
import os

class ViTWrapper(nn.Module):
    """Wrapper to ensure ViT outputs match HuggingFace expectations"""
    def __init__(self, vit_model, config):
        super().__init__()
        self.vit = vit_model
        self.config = config

    def forward(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        # Forward pass through your pretrained ViT
        # We assume self.vit returns the sequence of hidden states (B, Seq_Len, Dim)
        outputs = self.vit(pixel_values)

        # Handle different output types from various ViT implementations
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        elif hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs

        # Return as BaseModelOutput for the VisionEncoderDecoder model to consume
        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=None,
            attentions=None
        )


class MedicalVisionGPT2(nn.Module):
    def __init__(
        self,
        vision_encoder_path,
        decoder_model_name="gpt2",
        image_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        # We remove specific freeze flags in favor of a smart init strategy
    ):
        super().__init__()

        print("="*80)
        print("Initializing Medical Vision-GPT2 (Surgical Fine-Tuning)")
        print("="*80)

        # ------------------------------------------------------------------
        # 1. SETUP ENCODER (Your Medical ViT)
        # ------------------------------------------------------------------
        encoder_config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224, 
            patch_size=16,
            num_channels=1,
        )

        print(f"Loading pretrained medical ViT from {vision_encoder_path}...")
        from lavis.models.blip_models.vit import ViT
        vision_encoder = ViT(
            in_channels=1,
            img_size=image_size,
            patch_size=patch_size,
            num_classes=0,
        )

        # Load weights
        checkpoint = torch.load(vision_encoder_path, map_location='cpu')
        vision_state = {}
        # Clean up state dict keys if necessary
        source_state = checkpoint['model'] if 'model' in checkpoint else checkpoint
        for k, v in source_state.items():
            if k.startswith('visual_encoder.'):
                vision_state[k.replace('visual_encoder.', '')] = v
            else:
                vision_state[k] = v
        
        vision_encoder.load_state_dict(vision_state, strict=False)
        wrapped_encoder = ViTWrapper(vision_encoder, encoder_config)

        # Freeze Encoder completely
        for param in wrapped_encoder.parameters():
            param.requires_grad = False

        # ------------------------------------------------------------------
        # 2. SETUP DECODER (GPT2)
        # ------------------------------------------------------------------
        print(f"Loading GPT2 decoder from {decoder_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Important: add_cross_attention=True creates NEW, RANDOM layers
        decoder_config = GPT2Config.from_pretrained(decoder_model_name)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True 

        decoder = GPT2LMHeadModel.from_pretrained(decoder_model_name, config=decoder_config)

        # ------------------------------------------------------------------
        # 3. SURGICAL FREEZING LOGIC
        # ------------------------------------------------------------------
        print("Applying surgical freezing to Decoder:")
        
        # Start by freezing EVERYTHING in decoder
        for param in decoder.parameters():
            param.requires_grad = False
            
        trainable_params = 0
        all_params = 0
        
        for name, param in decoder.named_parameters():
            all_params += param.numel()
            
            # UNFREEZE 1: Cross Attention (The bridge between Image and Text)
            # These are initialized randomly, so they MUST be trained.
            if "crossattention" in name:
                param.requires_grad = True
                trainable_params += param.numel()
                
            # UNFREEZE 2: Layer Norms (Crucial for stability in fine-tuning)
            elif "ln_" in name or "ln_1" in name or "ln_2" in name:
                param.requires_grad = True
                trainable_params += param.numel()
                
            # UNFREEZE 3: LM Head (To adapt to medical vocabulary)
            elif "lm_head" in name:
                param.requires_grad = True
                trainable_params += param.numel()
                
            # OPTIONAL: Unfreeze the very last transformer block entirely
            # allowing it to synthesize all features
            elif "h.11." in name: # Assuming gpt2 (12 layers, 0-11)
                param.requires_grad = True
                trainable_params += param.numel()

        print(f"  > Encoder: Frozen")
        print(f"  > Decoder Self-Attention: Frozen (preserving English)")
        print(f"  > Decoder Cross-Attention: Unfrozen (learning Image connection)")
        print(f"  > Trainable Parameters: {trainable_params:,} / {all_params:,} ({(trainable_params/all_params)*100:.2f}%)")

        # ------------------------------------------------------------------
        # 4. COMPILE MODEL
        # ------------------------------------------------------------------
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=encoder_config,
            decoder_config=decoder_config
        )

        self.model = VisionEncoderDecoderModel(config=config)
        self.model.encoder = wrapped_encoder
        self.model.decoder = decoder

        # Generation Config
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.tokenizer.vocab_size

        # If hidden sizes differ, a projection layer is created. It must be trainable.
        if encoder_config.hidden_size != decoder_config.hidden_size:
            if hasattr(self.model, 'enc_to_dec_proj'):
                 for param in self.model.enc_to_dec_proj.parameters():
                     param.requires_grad = True

    def forward(self, pixel_values, labels=None, **kwargs):
        return self.model(
            pixel_values=pixel_values,
            labels=labels,
            return_dict=True,
            **kwargs
        )

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
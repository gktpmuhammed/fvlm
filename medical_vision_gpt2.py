"""
Medical Vision-Language Model using GPT2 (WORKING!)
- Your pretrained medical ViT as encoder
- GPT2 as decoder (supports cross-attention)
- VisionEncoderDecoderModel framework
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
    """Wrapper for custom ViT to work with HuggingFace"""
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
        outputs = self.vit(pixel_values)

        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = outputs

        if return_dict:
            return BaseModelOutput(
                last_hidden_state=last_hidden_state,
                hidden_states=None,
                attentions=None
            )
        else:
            return (last_hidden_state,)


class MedicalVisionGPT2(nn.Module):
    """Medical VLM: Your ViT + GPT2"""

    def __init__(
        self,
        vision_encoder_path,
        decoder_model_name="gpt2",
        image_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        freeze_encoder=True,
        freeze_decoder_base=True,
    ):
        super().__init__()

        print("="*80)
        print("Initializing Medical Vision-GPT2 Model")
        print("="*80)

        # 1. Create encoder config
        encoder_config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            image_size=224,
            patch_size=16,
            num_channels=1,
        )

        # 2. Load YOUR pretrained medical ViT
        print(f"\nLoading pretrained medical ViT from {vision_encoder_path}...")
        from lavis.models.blip_models.vit import ViT

        vision_encoder = ViT(
            in_channels=1,
            img_size=image_size,
            patch_size=patch_size,
            num_classes=0,
        )

        # Load pretrained weights
        checkpoint = torch.load(vision_encoder_path, map_location='cpu')
        vision_state = {}

        if 'state_dict' in checkpoint:
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('visual_encoder.'):
                    vision_state[k.replace('visual_encoder.', '')] = v
        elif 'model' in checkpoint:
            for k, v in checkpoint['model'].items():
                if k.startswith('visual_encoder.'):
                    vision_state[k.replace('visual_encoder.', '')] = v
        else:
            for k, v in checkpoint.items():
                if k.startswith('visual_encoder.'):
                    vision_state[k.replace('visual_encoder.', '')] = v
                else:
                    vision_state[k] = v

        missing, unexpected = vision_encoder.load_state_dict(vision_state, strict=False)
        print(f"  Loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")

        wrapped_encoder = ViTWrapper(vision_encoder, encoder_config)

        if freeze_encoder:
            print("  Freezing encoder (preserving medical knowledge)")
            for param in wrapped_encoder.parameters():
                param.requires_grad = False

        # 3. Load GPT2 decoder with cross-attention support
        print(f"\nLoading GPT2 decoder from {decoder_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)

        # Add special tokens
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure GPT2 for cross-attention
        decoder_config = GPT2Config.from_pretrained(decoder_model_name)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True  # ← KEY: This works for GPT2!

        decoder = GPT2LMHeadModel.from_pretrained(
            decoder_model_name,
            config=decoder_config
        )

        # 4. Freeze decoder base if requested
        if freeze_decoder_base:
            print("  Freezing decoder base layers")
            for param in decoder.parameters():
                param.requires_grad = False

            # Unfreeze last 3 layers + cross-attention
            print("  Unfreezing last 3 layers + cross-attention...")
            num_layers = len(decoder.transformer.h)
            for layer_idx in range(num_layers - 3, num_layers):
                layer = decoder.transformer.h[layer_idx]
                for param in layer.parameters():
                    param.requires_grad = True
                print(f"    Unfroze layer {layer_idx}")

            # Unfreeze LM head
            for param in decoder.lm_head.parameters():
                param.requires_grad = True
            print("    Unfroze lm_head")

        # 5. Create VisionEncoderDecoderModel
        print("\nCreating VisionEncoderDecoderModel...")
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=encoder_config,
            decoder_config=decoder_config
        )

        self.model = VisionEncoderDecoderModel(config=config)

        # 6. Replace encoder and decoder
        print("  Replacing with custom encoder and GPT2 decoder...")
        self.model.encoder = wrapped_encoder
        self.model.decoder = decoder

        # 7. Configure generation
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.tokenizer.vocab_size

        # 8. Statistics
        print("\n" + "="*80)
        print("Model Statistics:")
        print("="*80)

        encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.model.decoder.parameters())

        trainable_encoder = sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)
        trainable_decoder = sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)
        trainable_total = trainable_encoder + trainable_decoder
        total_params = encoder_params + decoder_params

        print(f"  Encoder: {encoder_params:,} ({trainable_encoder:,} trainable)")
        print(f"  Decoder: {decoder_params:,} ({trainable_decoder:,} trainable)")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_total:,} ({100*trainable_total/total_params:.1f}%)")
        print("="*80)

    def forward(self, pixel_values, labels=None, **kwargs):
        return self.model(
            pixel_values=pixel_values,
            labels=labels,
            return_dict=True,
            **kwargs
        )

    def generate(self, pixel_values, max_length=256, num_beams=4, **kwargs):
        return self.model.generate(
            pixel_values=pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    @classmethod
    def from_pretrained(cls, model_path):
        loaded_model = VisionEncoderDecoderModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        instance = cls.__new__(cls)
        instance.model = loaded_model
        instance.tokenizer = tokenizer

        return instance

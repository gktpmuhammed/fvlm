"""
Medical VLM using VisionEncoderDecoderModel - Config Fix
"""

import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel, 
    VisionEncoderDecoderConfig,
    AutoTokenizer,
    BioGptForCausalLM,
    BioGptConfig,
    ViTConfig
)
from transformers.modeling_outputs import BaseModelOutput
import os


class ViTWrapper(nn.Module):
    """
    Wrapper around custom ViT to make it compatible with HuggingFace interface
    """
    def __init__(self, vit_model, config):
        super().__init__()
        self.vit = vit_model
        self.config = config  # Add config attribute

    def forward(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        """Forward pass compatible with HuggingFace interface"""
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


class MedicalVisionEncoderDecoder(nn.Module):
    """Medical VLM using HuggingFace's VisionEncoderDecoderModel"""

    def __init__(
        self,
        vision_encoder_path,
        decoder_model_name="microsoft/biogpt",
        image_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        freeze_encoder=True,
        freeze_decoder_base=True,
    ):
        super().__init__()

        print("="*80)
        print("Initializing Medical VisionEncoderDecoder")
        print("="*80)

        # 1. Create encoder config FIRST
        encoder_config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            image_size=224,
            patch_size=16,
            num_channels=1,
        )

        # 2. Load YOUR pretrained medical vision encoder
        print(f"\nLoading pretrained medical vision encoder from {vision_encoder_path}...")
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
        print(f"  Loaded vision encoder (missing: {len(missing)}, unexpected: {len(unexpected)})")

        # Wrap ViT with config
        wrapped_encoder = ViTWrapper(vision_encoder, encoder_config)

        # 3. Freeze encoder if requested
        if freeze_encoder:
            print("  Freezing vision encoder (keeping pretrained medical knowledge)")
            for param in wrapped_encoder.parameters():
                param.requires_grad = False

        # 4. Load BioGPT decoder
        print(f"\nLoading BioGPT decoder from {decoder_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)

        decoder_config = BioGptConfig.from_pretrained(decoder_model_name)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        decoder = BioGptForCausalLM.from_pretrained(
            decoder_model_name,
            config=decoder_config
        )

        # 5. Freeze decoder base layers if requested
        if freeze_decoder_base:
            print("  Freezing decoder base layers (keeping medical language knowledge)")
            for param in decoder.parameters():
                param.requires_grad = False

            # Unfreeze cross-attention and last few layers
            print("  Unfreezing cross-attention and last 3 decoder layers...")
            num_layers = len(decoder.biogpt.layers)
            for layer_idx in range(num_layers - 3, num_layers):
                layer = decoder.biogpt.layers[layer_idx]
                for param in layer.parameters():
                    param.requires_grad = True
                print(f"    Unfroze layer {layer_idx}")

            # Unfreeze embedding layer
            if hasattr(decoder.biogpt, 'embed_tokens'):
                for param in decoder.biogpt.embed_tokens.parameters():
                    param.requires_grad = True
                print("    Unfroze embed_tokens")

        # 6. Create VisionEncoderDecoderConfig
        print("\nCreating VisionEncoderDecoderModel...")
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=encoder_config,
            decoder_config=decoder_config
        )

        # Initialize VisionEncoderDecoderModel
        self.model = VisionEncoderDecoderModel(config=config)

        # 7. Replace encoder and decoder
        print("  Replacing encoder with pretrained medical ViT...")
        self.model.encoder = wrapped_encoder
        self.model.decoder = decoder

        # 8. Configure for generation
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.tokenizer.vocab_size
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.max_length = 512
        self.model.config.num_beams = 4

        # 9. Print parameter statistics
        print("\n" + "="*80)
        print("Model Statistics:")
        print("="*80)

        encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.model.decoder.parameters())
        total_params = encoder_params + decoder_params

        trainable_encoder = sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)
        trainable_decoder = sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)
        trainable_total = trainable_encoder + trainable_decoder

        print(f"  Encoder parameters: {encoder_params:,}")
        print(f"    Trainable: {trainable_encoder:,} ({100*trainable_encoder/encoder_params if encoder_params > 0 else 0:.1f}%)")
        print(f"  Decoder parameters: {decoder_params:,}")
        print(f"    Trainable: {trainable_decoder:,} ({100*trainable_decoder/decoder_params:.1f}%)")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Total trainable: {trainable_total:,} ({100*trainable_total/total_params:.1f}%)")
        print("="*80)

    def forward(self, pixel_values, labels=None, **kwargs):
        """Forward pass through VisionEncoderDecoderModel"""
        return self.model(
            pixel_values=pixel_values,
            labels=labels,
            return_dict=True,
            **kwargs
        )

    def generate(self, pixel_values, max_length=256, num_beams=4, **kwargs):
        """Generate text from images"""
        return self.model.generate(
            pixel_values=pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )

    def save_pretrained(self, output_dir):
        """Save model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    @classmethod
    def from_pretrained(cls, model_path, vision_encoder_path=None):
        """Load trained model"""
        loaded_model = VisionEncoderDecoderModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        instance = cls.__new__(cls)
        instance.model = loaded_model
        instance.tokenizer = tokenizer

        return instance

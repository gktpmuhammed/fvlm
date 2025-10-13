"""
Medical Vision-BART - PROPER FIX
Uses VisionEncoderDecoderModel.from_encoder_decoder_pretrained()
"""

import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    ViTConfig
)
from transformers.modeling_outputs import BaseModelOutput
import os


class ViTWrapper(nn.Module):
    """Wrapper for custom ViT"""
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


class MedicalVisionBART(nn.Module):
    """Medical VLM: ViT + BART (using proper HuggingFace method)"""

    def __init__(
        self,
        vision_encoder_path,
        decoder_model_name="facebook/bart-base",
        image_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        freeze_encoder=False,
    ):
        super().__init__()

        print("="*80)
        print("Initializing Medical Vision-BART (PROPER METHOD)")
        print("="*80)

        # 1. Load YOUR pretrained medical ViT
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

        # Wrap ViT
        encoder_config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
            patch_size=16,
            num_channels=1,
        )
        wrapped_encoder = ViTWrapper(vision_encoder, encoder_config)

        # 2. Create VisionEncoderDecoderModel using HuggingFace's method
        print(f"\nCreating VisionEncoderDecoderModel with {decoder_model_name}...")
        print("  Using from_encoder_decoder_pretrained() method...")

        # Use a dummy ViT encoder temporarily
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-224-in21k",  # Dummy, will be replaced
            decoder_model_name
        )

        # 3. Replace encoder with YOUR custom ViT
        print("  Replacing encoder with YOUR pretrained medical ViT...")
        self.model.encoder = wrapped_encoder

        # 4. Get tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)

        # 5. Configure generation
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # 6. Freeze/unfreeze
        if freeze_encoder:
            print("  Freezing encoder")
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        else:
            print("  Encoder UNFROZEN (allows vision-language alignment)")

        # Freeze decoder base, unfreeze last 3 layers
        print("  Freezing decoder base, unfreezing last 3 layers...")
        for param in self.model.decoder.parameters():
            param.requires_grad = False

        # Unfreeze last 3 decoder layers
        num_layers = len(self.model.decoder.model.decoder.layers)
        for layer_idx in range(num_layers - 3, num_layers):
            layer = self.model.decoder.model.decoder.layers[layer_idx]
            for param in layer.parameters():
                param.requires_grad = True
            print(f"    Unfroze decoder layer {layer_idx}")

        # Unfreeze LM head
        for param in self.model.decoder.lm_head.parameters():
            param.requires_grad = True
        print("    Unfroze lm_head")

        # 7. Statistics
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

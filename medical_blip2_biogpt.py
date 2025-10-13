#!/usr/bin/env python3

"""
Medical BLIP-2 Model with BioGPT - Using VisionEncoderDecoderModel Framework
- Your pretrained 3D medical ViT as encoder
- BioGPT as decoder (domain-specific biomedical GPT)
- Following the working GPT2 pattern
- FIXED: Generation method compatible with BioGPT
"""

import torch
import torch.nn as nn
from transformers import (
    BioGptForCausalLM,
    BioGptTokenizer,
    BioGptConfig,
    ViTConfig,
    BertConfig,
    BertModel,
)
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithCrossAttentions
import os


class ViTWrapper3D(nn.Module):
    """
    Wrapper for 3D medical ViT to work with HuggingFace
    Handles 3D volumes (D, H, W) properly
    """
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
        # Forward through 3D ViT
        outputs = self.vit(pixel_values)

        # Extract last hidden state
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


class QFormer(nn.Module):
    """
    Q-Former using BERT with cross-attention
    Compresses vision features into fixed number of query tokens
    """
    def __init__(self, hidden_size=768, num_query_tokens=32, num_hidden_layers=6):
        super().__init__()

        # BERT with cross-attention enabled
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            is_decoder=True,
            add_cross_attention=True,
        )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.num_query_tokens = num_query_tokens
        self.hidden_size = hidden_size

    def forward(self, query_embeds, encoder_hidden_states):
        """
        Args:
            query_embeds: [B, num_queries, hidden_size]
            encoder_hidden_states: [B, num_patches, hidden_size]
        """
        outputs = self.bert(
            inputs_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
        )
        return outputs.last_hidden_state


class MedicalBLIP2BioGPT(nn.Module):
    """
    Medical BLIP-2 with BioGPT
    Architecture: 3D ViT -> Q-Former -> BioGPT
    """

    def __init__(
        self,
        vision_encoder_path,
        decoder_model_name="microsoft/biogpt",
        image_size=(112, 256, 352),  # 3D: (D, H, W)
        patch_size=(16, 16, 32),      # 3D patches
        num_query_tokens=32,
        freeze_encoder=True,
        freeze_decoder_base=True,
        use_qformer=True,  # Option to use Q-Former or direct connection
    ):
        super().__init__()

        print("="*80)
        print("Initializing Medical BLIP-2 with BioGPT")
        print("="*80)

        self.use_qformer = use_qformer

        # 1. Create encoder config for 3D ViT
        print(f"\nConfiguring 3D ViT...")
        print(f"  Image size: {image_size}")
        print(f"  Patch size: {patch_size}")

        encoder_config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            image_size=224,  # Dummy for config (actual size handled by custom ViT)
            patch_size=16,   # Dummy for config
            num_channels=1,  # Medical images often single channel
        )

        # 2. Load YOUR pretrained 3D medical ViT
        print(f"\nLoading pretrained 3D medical ViT from {vision_encoder_path}...")

        try:
            # Try loading with LAVIS ViT (your working version)
            from lavis.models.blip_models.vit import ViT

            vision_encoder = ViT(
                in_channels=1,
                img_size=image_size,  # 3D dimensions
                patch_size=patch_size,  # 3D patches
                num_classes=0,
            )
            print("  Using LAVIS 3D ViT architecture")

        except ImportError:
            print("  LAVIS not found, using alternative ViT...")
            # Fallback: create a simple 3D ViT wrapper
            import timm
            vision_encoder = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=0,
                img_size=(112, 256),  # Will be overridden
                in_chans=112,  # Treat depth as channels for 2D ViT
            )

        # Load pretrained weights
        checkpoint = torch.load(vision_encoder_path, map_location='cpu')
        vision_state = {}

        if 'state_dict' in checkpoint:
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('visual_encoder.'):
                    vision_state[k.replace('visual_encoder.', '')] = v
                else:
                    vision_state[k] = v
        elif 'model' in checkpoint:
            for k, v in checkpoint['model'].items():
                if k.startswith('visual_encoder.'):
                    vision_state[k.replace('visual_encoder.', '')] = v
                else:
                    vision_state[k] = v
        else:
            for k, v in checkpoint.items():
                if k.startswith('visual_encoder.'):
                    vision_state[k.replace('visual_encoder.', '')] = v
                else:
                    vision_state[k] = v

        missing, unexpected = vision_encoder.load_state_dict(vision_state, strict=False)
        print(f"  Loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")

        # Get encoder hidden size
        if hasattr(vision_encoder, 'embed_dim'):
            encoder_hidden_size = vision_encoder.embed_dim
        elif hasattr(vision_encoder, 'num_features'):
            encoder_hidden_size = vision_encoder.num_features
        else:
            encoder_hidden_size = 768  # Default

        print(f"  Encoder hidden size: {encoder_hidden_size}")

        # Wrap the vision encoder
        wrapped_encoder = ViTWrapper3D(vision_encoder, encoder_config)

        if freeze_encoder:
            print("  Freezing encoder (preserving medical knowledge)")
            for param in wrapped_encoder.parameters():
                param.requires_grad = False

        self.encoder = wrapped_encoder

        # 3. Q-Former (optional compression layer)
        if use_qformer:
            print(f"\nInitializing Q-Former with {num_query_tokens} query tokens...")
            self.qformer = QFormer(
                hidden_size=encoder_hidden_size,
                num_query_tokens=num_query_tokens,
                num_hidden_layers=6,
            )

            # Learnable query tokens
            self.query_tokens = nn.Parameter(
                torch.zeros(1, num_query_tokens, encoder_hidden_size)
            )
            self.query_tokens.data.normal_(mean=0.0, std=0.02)

            qformer_output_size = encoder_hidden_size
        else:
            print("\nSkipping Q-Former (direct connection)")
            self.qformer = None
            self.query_tokens = None
            qformer_output_size = encoder_hidden_size

        # 4. Load BioGPT decoder
        print(f"\nLoading BioGPT decoder from {decoder_model_name}...")
        self.tokenizer = BioGptTokenizer.from_pretrained(decoder_model_name)

        # BioGPT uses eos_token as pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure BioGPT for cross-attention
        decoder_config = BioGptConfig.from_pretrained(decoder_model_name)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True  # Enable cross-attention

        decoder = BioGptForCausalLM.from_pretrained(
            decoder_model_name,
            config=decoder_config
        )

        print(f"  BioGPT loaded: {decoder_model_name}")
        print(f"  Vocab size: {len(self.tokenizer)}")
        print(f"  Hidden size: {decoder_config.hidden_size}")

        # 5. Projection layer (Q-Former/ViT output -> BioGPT input)
        self.projection = nn.Linear(qformer_output_size, decoder_config.hidden_size)
        print(f"\nProjection: {qformer_output_size} -> {decoder_config.hidden_size}")

        # 6. Freeze decoder base if requested
        if freeze_decoder_base:
            print("\nFreezing decoder base layers...")
            for param in decoder.parameters():
                param.requires_grad = False

            # Unfreeze last 3 layers + cross-attention
            print("  Unfreezing last 3 layers + cross-attention...")
            num_layers = len(decoder.biogpt.layers)
            for layer_idx in range(num_layers - 3, num_layers):
                layer = decoder.biogpt.layers[layer_idx]
                for param in layer.parameters():
                    param.requires_grad = True
                print(f"    Unfroze layer {layer_idx}")

            # Unfreeze LM head
            for param in decoder.output_projection.parameters():
                param.requires_grad = True
            print("    Unfroze output_projection (LM head)")

        self.decoder = decoder

        # 7. Statistics
        print("\n" + "="*80)
        print("Model Statistics:")
        print("="*80)

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        projection_params = sum(p.numel() for p in self.projection.parameters())

        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        projection_trainable = sum(p.numel() for p in self.projection.parameters() if p.requires_grad)

        if self.qformer:
            qformer_params = sum(p.numel() for p in self.qformer.parameters())
            qformer_trainable = sum(p.numel() for p in self.qformer.parameters() if p.requires_grad)
            print(f"  Vision Encoder: {encoder_params:,} ({encoder_trainable:,} trainable)")
            print(f"  Q-Former: {qformer_params:,} ({qformer_trainable:,} trainable)")
            print(f"  Projection: {projection_params:,} ({projection_trainable:,} trainable)")
            print(f"  BioGPT Decoder: {decoder_params:,} ({decoder_trainable:,} trainable)")

            total = encoder_params + qformer_params + projection_params + decoder_params
            trainable = encoder_trainable + qformer_trainable + projection_trainable + decoder_trainable
        else:
            print(f"  Vision Encoder: {encoder_params:,} ({encoder_trainable:,} trainable)")
            print(f"  Projection: {projection_params:,} ({projection_trainable:,} trainable)")
            print(f"  BioGPT Decoder: {decoder_params:,} ({decoder_trainable:,} trainable)")

            total = encoder_params + projection_params + decoder_params
            trainable = encoder_trainable + projection_trainable + decoder_trainable

        print(f"  ")
        print(f"  TOTAL: {total:,} ({trainable:,} trainable)")
        print(f"  Trainable: {trainable/total*100:.1f}%")
        print("="*80 + "\n")

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        batch_size = pixel_values.shape[0]

        # 1. Extract vision features
        encoder_outputs = self.encoder(pixel_values, return_dict=True)
        vision_features = encoder_outputs.last_hidden_state  # [B, num_patches, hidden_size]

        # 2. Q-Former compression (optional)
        if self.use_qformer:
            query_tokens = self.query_tokens.expand(batch_size, -1, -1)
            compressed_features = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=vision_features,
            )  # [B, num_queries, hidden_size]
        else:
            compressed_features = vision_features

        # 3. Project to decoder dimension
        projected_features = self.projection(compressed_features)  # [B, seq_len, decoder_hidden]

        # 4. Prepare decoder inputs
        if input_ids is not None:
            # Get text embeddings
            inputs_embeds = self.decoder.get_input_embeddings()(input_ids)

            # Concatenate vision and text
            inputs_embeds = torch.cat([projected_features, inputs_embeds], dim=1)

            # Adjust attention mask
            if attention_mask is not None:
                vision_attention = torch.ones(
                    batch_size, projected_features.shape[1],
                    dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([vision_attention, attention_mask], dim=1)

            # Adjust labels
            if labels is not None:
                vision_labels = torch.full(
                    (batch_size, projected_features.shape[1]),
                    fill_value=-100,
                    dtype=labels.dtype, device=labels.device
                )
                labels = torch.cat([vision_labels, labels], dim=1)
        else:
            inputs_embeds = projected_features

        # 5. Forward through BioGPT
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs

    def generate(self, pixel_values, max_length=256, num_beams=4, **kwargs):
        """
        Generate text from images

        FIXED: BioGPT doesn't support inputs_embeds in generate()
        Workaround: Use a custom generation loop with past_key_values
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device

        with torch.no_grad():
            # 1. Extract and project vision features
            encoder_outputs = self.encoder(pixel_values, return_dict=True)
            vision_features = encoder_outputs.last_hidden_state

            if self.use_qformer:
                query_tokens = self.query_tokens.expand(batch_size, -1, -1)
                compressed_features = self.qformer(
                    query_embeds=query_tokens,
                    encoder_hidden_states=vision_features,
                )
            else:
                compressed_features = vision_features

            vision_embeds = self.projection(compressed_features)  # [B, seq_len, hidden_size]

            # 2. Initialize with BOS token
            input_ids = torch.full(
                (batch_size, 1),
                self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id,
                dtype=torch.long,
                device=device
            )

            # 3. Get text embeddings and concatenate with vision
            text_embeds = self.decoder.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

            # 4. Create attention mask
            attention_mask = torch.ones(
                (batch_size, inputs_embeds.shape[1]),
                dtype=torch.long,
                device=device
            )

            # 5. Simple greedy/beam generation
            if num_beams == 1:
                # Greedy generation
                generated = self._greedy_generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    vision_seq_len=vision_embeds.shape[1],
                    **kwargs
                )
            else:
                # Beam search (simplified version)
                generated = self._beam_search_generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    vision_seq_len=vision_embeds.shape[1],
                    **kwargs
                )

            return generated

    def _greedy_generate(self, inputs_embeds, attention_mask, max_length, vision_seq_len, **kwargs):
        """Greedy decoding"""
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

        generated_ids = []
        past_key_values = None

        for step in range(max_length):
            if past_key_values is None:
                # First step: use full inputs_embeds
                outputs = self.decoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                # Subsequent steps: use last generated token
                last_token_id = generated_ids[-1]
                last_embeds = self.decoder.get_input_embeddings()(last_token_id.unsqueeze(1))

                outputs = self.decoder(
                    inputs_embeds=last_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            # Get next token
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            generated_ids.append(next_token_id)

            # Update attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), dtype=torch.long, device=device)
            ], dim=1)

            # Check for EOS
            if (next_token_id == self.tokenizer.eos_token_id).all():
                break

        # Stack and return (exclude vision tokens)
        generated_ids = torch.stack(generated_ids, dim=1)
        return generated_ids

    def _beam_search_generate(self, inputs_embeds, attention_mask, max_length, num_beams, vision_seq_len, **kwargs):
        """
        Simplified beam search
        For now, fall back to greedy if beam search is too complex
        """
        # TODO: Implement proper beam search
        # For now, use greedy as fallback
        return self._greedy_generate(inputs_embeds, attention_mask, max_length, vision_seq_len, **kwargs)

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # Save encoder
        torch.save(self.encoder.state_dict(), f"{output_dir}/encoder.pt")

        # Save Q-Former if used
        if self.use_qformer:
            torch.save(self.qformer.state_dict(), f"{output_dir}/qformer.pt")
            torch.save(self.query_tokens, f"{output_dir}/query_tokens.pt")

        # Save projection
        torch.save(self.projection.state_dict(), f"{output_dir}/projection.pt")

        # Save decoder and tokenizer
        self.decoder.save_pretrained(f"{output_dir}/decoder")
        self.tokenizer.save_pretrained(f"{output_dir}/tokenizer")

        # Save config
        config = {
            'use_qformer': self.use_qformer,
            'num_query_tokens': self.query_tokens.shape[1] if self.use_qformer else None,
        }
        torch.save(config, f"{output_dir}/config.pt")

        print(f"Model saved to {output_dir}")

    @classmethod
    def from_pretrained(cls, model_path, vision_encoder_path=None):
        """Load a pretrained model"""
        config = torch.load(f"{model_path}/config.pt")

        instance = cls(
            vision_encoder_path=vision_encoder_path or model_path,
            use_qformer=config['use_qformer'],
            num_query_tokens=config.get('num_query_tokens', 32),
        )

        # Load weights
        instance.encoder.load_state_dict(torch.load(f"{model_path}/encoder.pt"))
        instance.projection.load_state_dict(torch.load(f"{model_path}/projection.pt"))

        if instance.use_qformer:
            instance.qformer.load_state_dict(torch.load(f"{model_path}/qformer.pt"))
            instance.query_tokens = torch.load(f"{model_path}/query_tokens.pt")

        instance.decoder = BioGptForCausalLM.from_pretrained(f"{model_path}/decoder")
        instance.tokenizer = BioGptTokenizer.from_pretrained(f"{model_path}/tokenizer")

        return instance

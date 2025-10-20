#!/usr/bin/env python3
"""
Medical BLIP-2 - ENHANCED WITH ENCODER FINE-TUNING
Supports differential learning rates and partial encoder unfreezing
"""

import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertModel,
    AutoTokenizer,
    OPTForCausalLM,
)
import os

class MedicalBLIP2Official(nn.Module):
    """Medical BLIP-2 with encoder fine-tuning support"""

    def __init__(
        self,
        vision_encoder_path,
        opt_model="facebook/opt-2.7b",
        image_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        num_query_tokens=32,
        freeze_vision=True,
        freeze_opt=True,
        num_unfrozen_layers=0,  # NEW: Number of encoder layers to unfreeze
    ):
        super().__init__()

        print("="*80)
        print("Initializing Medical BLIP-2 (ENHANCED)")
        print("="*80)

        # 1. Vision encoder
        print(f"\nLoading Vision Encoder: {vision_encoder_path}")
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from lavis.models.blip_models.vit import ViT

            vision_encoder = ViT(
                in_channels=1,
                img_size=image_size,
                patch_size=patch_size,
                num_classes=0,
            )
            print("  Using LAVIS 3D ViT")
        except ImportError:
            import timm
            vision_encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)

        # Load checkpoint
        checkpoint = torch.load(vision_encoder_path, map_location='cpu', weights_only=False)
        vision_state = {}

        for source_dict in [checkpoint.get('state_dict', {}), checkpoint.get('model', {}), checkpoint]:
            for k, v in source_dict.items():
                key = k.replace('visual_encoder.', '')
                vision_state[key] = v
            if vision_state:
                break

        vision_encoder.load_state_dict(vision_state, strict=False)

        # ENHANCED: Selective unfreezing
        if freeze_vision:
            # Freeze all first
            for param in vision_encoder.parameters():
                param.requires_grad = False

            # Unfreeze last N layers if requested
            if num_unfrozen_layers > 0 and hasattr(vision_encoder, 'blocks'):
                total_layers = len(vision_encoder.blocks)
                start_layer = max(0, total_layers - num_unfrozen_layers)

                print(f"\nENCODER FINE-TUNING:")
                print(f"  Total layers: {total_layers}")
                print(f"  Unfrozen layers: {num_unfrozen_layers} (layers {start_layer}-{total_layers-1})")

                unfrozen_params = 0
                for i, block in enumerate(vision_encoder.blocks):
                    if i >= start_layer:
                        for param in block.parameters():
                            param.requires_grad = True
                            unfrozen_params += param.numel()

                print(f"  Unfrozen parameters: {unfrozen_params:,}")
            else:
                print("\nVision encoder: FULLY FROZEN")
        else:
            print("\nVision encoder: FULLY TRAINABLE")

        self.visual_encoder = vision_encoder
        self.num_unfrozen_layers = num_unfrozen_layers
        vision_width = getattr(vision_encoder, 'embed_dim', None) or getattr(vision_encoder, 'num_features', 768)

        # 2. Q-Former
        print(f"\nInitializing Q-Former (query tokens: {num_query_tokens})")
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.add_cross_attention = True
        encoder_config.is_decoder = True

        self.Qformer = BertModel(config=encoder_config, add_pooling_layer=False)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        # Only cross-attention trainable in Q-Former
        for name, param in self.Qformer.named_parameters():
            if "crossattention" in name or "cross_attention" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 3. OPT decoder
        print(f"\nLoading OPT: {opt_model}")
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model,
            torch_dtype=torch.float32,
        )

        if freeze_opt:
            for param in self.opt_model.parameters():
                param.requires_grad = False

        # Projection
        self.opt_proj = nn.Linear(encoder_config.hidden_size, self.opt_model.config.hidden_size)

        # Tokenizer
        self.opt_tokenizer.padding_side = "right"
        if self.opt_tokenizer.pad_token is None:
            self.opt_tokenizer.pad_token = self.opt_tokenizer.eos_token

        self.prompt = "A medical report describing this CT scan: "

        # Print statistics
        print("\n" + "="*80)
        print("PARAMETER STATISTICS:")
        print("="*80)

        vision_total = sum(p.numel() for p in self.visual_encoder.parameters())
        vision_train = sum(p.numel() for p in self.visual_encoder.parameters() if p.requires_grad)
        qformer_train = sum(p.numel() for p in self.Qformer.parameters() if p.requires_grad)
        proj_train = sum(p.numel() for p in self.opt_proj.parameters() if p.requires_grad)
        opt_train = sum(p.numel() for p in self.opt_model.parameters() if p.requires_grad)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Vision Encoder: {vision_train:,} / {vision_total:,} ({100*vision_train/vision_total:.1f}%)")
        print(f"Q-Former:       {qformer_train:,}")
        print(f"Projection:     {proj_train:,}")
        print(f"OPT:            {opt_train:,}")
        print(f"---")
        print(f"TOTAL:          {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        print("="*80 + "\n")

    def get_param_groups_with_lr(self, lr_vision=1e-5, lr_qformer=1e-4, lr_proj=1e-4, lr_opt=5e-5):
        """
        NEW: Get parameter groups with differential learning rates

        This allows fine-tuning encoder with much lower LR to prevent forgetting
        """
        param_groups = []

        # Vision encoder (if unfrozen)
        vision_params = [p for p in self.visual_encoder.parameters() if p.requires_grad]
        if vision_params:
            param_groups.append({
                'params': vision_params,
                'lr': lr_vision,
                'name': 'vision_encoder'
            })

        # Q-Former
        param_groups.append({
            'params': [p for p in self.Qformer.parameters() if p.requires_grad],
            'lr': lr_qformer,
            'name': 'qformer'
        })

        # Projection
        param_groups.append({
            'params': self.opt_proj.parameters(),
            'lr': lr_proj,
            'name': 'projection'
        })

        # OPT (if unfrozen)
        opt_params = [p for p in self.opt_model.parameters() if p.requires_grad]
        if opt_params:
            param_groups.append({
                'params': opt_params,
                'lr': lr_opt,
                'name': 'opt_decoder'
            })

        return param_groups

    def forward(self, image, text_input=None, text_output=None):
        """Forward pass"""
        batch_size = image.shape[0]
        device = image.device

        # 1. Vision features (with gradient if unfrozen)
        if self.num_unfrozen_layers > 0:
            # Allow gradients for unfrozen layers
            image_output = self.visual_encoder(image)
        else:
            # No gradients if fully frozen
            with torch.no_grad():
                image_output = self.visual_encoder(image)

        if isinstance(image_output, tuple):
            image_embeds = image_output[0]
        else:
            image_embeds = image_output
        image_embeds = image_embeds.float()

        # 2. Q-Former
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        query_output = self.Qformer(
            inputs_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # 3. Project
        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(device)

        # 4. Prompt
        if text_input is None:
            text_input = [self.prompt] * batch_size

        prompt_tokens = self.opt_tokenizer(
            text_input,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=32,
        ).to(device)

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_atts = prompt_tokens.attention_mask

        # 5. Concatenate
        inputs_embeds = torch.cat([inputs_opt, prompt_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, prompt_atts], dim=1)

        # 6. Training
        if text_output is not None:
            text_tokens = self.opt_tokenizer(
                text_output,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=256,
            ).to(device)

            targets_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
            targets_atts = text_tokens.attention_mask

            inputs_embeds = torch.cat([inputs_embeds, targets_embeds], dim=1)
            attention_mask = torch.cat([attention_mask, targets_atts], dim=1)

            empty_targets = (
                torch.ones([batch_size, inputs_opt.size(1) + prompt_embeds.size(1)],
                          dtype=torch.long).to(device).fill_(-100)
            )
            targets = torch.cat([empty_targets, text_tokens.input_ids], dim=1)

            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=targets,
                return_dict=True,
            )
            return outputs
        else:
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            return outputs

    @torch.no_grad()
    def generate(
        self,
        image,
        prompt=None,
        max_length=256,
        min_length=10,
        num_beams=1,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1.0,
    ):
        """Generation (same as before)"""
        batch_size = image.shape[0]
        device = image.device

        # Get vision features
        image_output = self.visual_encoder(image)
        if isinstance(image_output, tuple):
            image_embeds = image_output[0]
        else:
            image_embeds = image_output
        image_embeds = image_embeds.float()

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        # Q-Former
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_output = self.Qformer(
            inputs_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # Project
        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(device)

        # Prompt
        if prompt is None:
            prompt = [self.prompt] * batch_size
        elif isinstance(prompt, str):
            prompt = [prompt] * batch_size

        prompt_tokens = self.opt_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).to(device)

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_atts = prompt_tokens.attention_mask

        # Concatenate
        prefix_embeds = torch.cat([inputs_opt, prompt_embeds], dim=1)
        prefix_atts = torch.cat([atts_opt, prompt_atts], dim=1)

        # Generate
        generated_ids = self._simple_greedy_generate(
            prefix_embeds=prefix_embeds,
            prefix_atts=prefix_atts,
            max_length=max_length,
            min_length=min_length,
            eos_token_id=self.opt_tokenizer.eos_token_id,
            pad_token_id=self.opt_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
        )

        # Clip vocab
        vocab_size = self.opt_tokenizer.vocab_size
        generated_ids = torch.clamp(generated_ids, max=vocab_size - 1)

        # Decode
        generated_texts = self.opt_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts

    @torch.no_grad()
    def _simple_greedy_generate(
        self,
        prefix_embeds,
        prefix_atts,
        max_length,
        min_length,
        eos_token_id,
        pad_token_id,
        repetition_penalty=1.0,
    ):
        """Simple greedy generation"""
        batch_size = prefix_embeds.shape[0]
        device = prefix_embeds.device
        embedding_layer = self.opt_model.get_input_embeddings()

        generated = []
        unfinished = torch.ones(batch_size, dtype=torch.long, device=device)

        for step in range(max_length):
            if step == 0:
                cur_embeds = prefix_embeds
                cur_atts = prefix_atts
            else:
                gen_ids = torch.stack(generated, dim=1)
                gen_embeds = embedding_layer(gen_ids).detach()
                gen_atts = torch.ones(gen_ids.shape, dtype=torch.long, device=device)

                cur_embeds = torch.cat([prefix_embeds.detach(), gen_embeds], dim=1)
                cur_atts = torch.cat([prefix_atts, gen_atts], dim=1)

            with torch.no_grad():
                outputs = self.opt_model(
                    inputs_embeds=cur_embeds,
                    attention_mask=cur_atts,
                    return_dict=True,
                )

            next_logits = outputs.logits[:, -1, :].detach()

            # Repetition penalty
            if repetition_penalty != 1.0 and step > 0:
                for i in range(batch_size):
                    for token_id in set([t[i].item() for t in generated]):
                        next_logits[i, token_id] /= repetition_penalty

            # Prevent EOS before min_length
            if step < min_length:
                next_logits[:, eos_token_id] = -float("inf")

            # Get next token
            next_token = torch.argmax(next_logits, dim=-1)
            next_token = next_token * unfinished + pad_token_id * (1 - unfinished)
            generated.append(next_token.detach())

            # Update unfinished
            unfinished = (unfinished * (next_token != eos_token_id).long())
            if unfinished.max() == 0:
                break

        return torch.stack(generated, dim=1)

    def save_pretrained(self, output_dir):
        """Save model"""
        os.makedirs(output_dir, exist_ok=True)

        torch.save(self.visual_encoder.state_dict(), f"{output_dir}/vision_encoder.pth")
        self.Qformer.save_pretrained(f"{output_dir}/qformer")
        torch.save(self.query_tokens, f"{output_dir}/query_tokens.pth")
        torch.save(self.opt_proj.state_dict(), f"{output_dir}/opt_proj.pth")
        self.opt_tokenizer.save_pretrained(f"{output_dir}/tokenizer")

        config = {
            'num_query_tokens': self.query_tokens.shape[1],
            'opt_model_name': self.opt_model.config._name_or_path,
            'num_unfrozen_layers': self.num_unfrozen_layers,
        }
        torch.save(config, f"{output_dir}/config.pth")
        print(f"Model saved to {output_dir}")

    @classmethod
    def from_pretrained(cls, model_path, vision_encoder_path=None):
        """Load model"""
        config = torch.load(f"{model_path}/config.pth")

        instance = cls(
            vision_encoder_path=vision_encoder_path or model_path,
            opt_model=config['opt_model_name'],
            num_query_tokens=config['num_query_tokens'],
            num_unfrozen_layers=config.get('num_unfrozen_layers', 0),
        )

        instance.visual_encoder.load_state_dict(torch.load(f"{model_path}/vision_encoder.pth"))
        instance.Qformer = BertModel.from_pretrained(f"{model_path}/qformer")
        instance.query_tokens = torch.load(f"{model_path}/query_tokens.pth")
        instance.opt_proj.load_state_dict(torch.load(f"{model_path}/opt_proj.pth"))

        instance.opt_tokenizer = AutoTokenizer.from_pretrained(
            f"{model_path}/tokenizer", use_fast=False
        )
        if instance.opt_tokenizer.pad_token is None:
            instance.opt_tokenizer.pad_token = instance.opt_tokenizer.eos_token

        return instance

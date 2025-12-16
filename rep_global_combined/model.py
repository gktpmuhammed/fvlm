import sys
import os
import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel, VisionEncoderDecoderConfig,
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, ViTConfig, BartForCausalLM, BartConfig
)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from modules import GlobalAdapter, MaskedSingleAdapter, ROIPoolingAdapter, AttentionAdapter, AttentionQFormerAdapter, RobustQFormerAdapter

class MedicalVLM(nn.Module):
    def __init__(
        self,
        vision_encoder_path,
        decoder_model_name="gpt2", 
        strategy="global", 
        image_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        **kwargs
    ):
        super().__init__()
        self.strategy = strategy
        print(f"--- Initializing MedicalVLM (Strategy: {strategy}) ---")

        # 1. SETUP ENCODER (LAVIS ViT)
        enc_config = ViTConfig(
            hidden_size=768, num_hidden_layers=12, num_attention_heads=12, 
            intermediate_size=3072, image_size=image_size, patch_size=patch_size, num_channels=1
        )
        try:
            from lavis.models.blip_models.vit import ViT
            vision_encoder = ViT(in_channels=1, img_size=image_size, patch_size=patch_size, num_classes=0)
        except ImportError: raise ImportError("LAVIS not found.")

        if os.path.exists(vision_encoder_path):
            ckpt = torch.load(vision_encoder_path, map_location='cpu')
            state = {k.replace('visual_encoder.', ''): v for k, v in ckpt.get('model', ckpt).items() 
                     if 'visual_encoder.' in k or ('text_' not in k and 'temp' not in k)}
            vision_encoder.load_state_dict(state, strict=False)
            print("  ViT weights loaded.")
        
        for p in vision_encoder.parameters(): p.requires_grad = False

        # 2. SELECT ADAPTER
        if strategy == "global":
            self.adapter = GlobalAdapter(vision_encoder, enc_config, kwargs.get('use_qformer', False))
        elif strategy == "masked_single":
            self.adapter = MaskedSingleAdapter(vision_encoder, enc_config)
        elif strategy == "roi":
            self.adapter = ROIPoolingAdapter(vision_encoder, enc_config)
        elif strategy == "attention":
            # UPDATED: 12 Organs
            self.adapter = AttentionAdapter(vision_encoder, enc_config, num_organs=12)
        elif strategy == "attention_qformer":  # <--- NEW STRATEGY
            # Uses Q-Former + Masks + Queries
            self.adapter = RobustQFormerAdapter(vision_encoder, enc_config, num_organs=12)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 3. SETUP DECODER
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token

        if "bart" in decoder_model_name:
            dec_config = BartConfig.from_pretrained(decoder_model_name)
            dec_config.is_decoder = True; dec_config.add_cross_attention = True
            decoder = BartForCausalLM.from_pretrained(decoder_model_name, config=dec_config)
        else:
            dec_config = AutoConfig.from_pretrained(decoder_model_name)
            dec_config.is_decoder = True; dec_config.add_cross_attention = True
            decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name, config=dec_config)

        # 4. FREEZING
        for p in decoder.parameters(): p.requires_grad = False
        target_modules = ["crossattention", "ln_", "layer_norm", "lm_head", "output_projection"]
        for n, p in decoder.named_parameters():
            if any(t in n for t in target_modules): p.requires_grad = True

        # 5. COMPILE
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
        self.model = VisionEncoderDecoderModel(config=config)
        self.model.encoder = self.adapter
        self.model.decoder = decoder
        
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.tokenizer.vocab_size

        if enc_config.hidden_size != dec_config.hidden_size and hasattr(self.model, 'enc_to_dec_proj'):
             for p in self.model.enc_to_dec_proj.parameters(): p.requires_grad = True

    def forward(self, pixel_values, organ_masks=None, pixel_mask=None, labels=None, **kwargs):
        # Pass masks to encoder
        enc_args = {'pixel_mask': pixel_mask, 'organ_masks': organ_masks}
        encoder_outputs = self.model.encoder(pixel_values=pixel_values, **enc_args)

        # Reshape Labels for Parallel Strategies
        if self.strategy in ['roi', 'attention'] and labels is not None:
            B, N, Seq = labels.shape
            labels = labels.view(B * N, Seq)
            
            # Dynamic Batching: Filter empty targets (-100)
            valid_mask = (labels != -100).any(dim=1)
            if valid_mask.any():
                encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[valid_mask]
                labels = labels[valid_mask]
            else:
                return None 

        return self.model(encoder_outputs=encoder_outputs, labels=labels, return_dict=True, **kwargs)

    def generate(self, pixel_values, organ_masks=None, pixel_mask=None, decoder_input_ids=None, **kwargs):
        enc_args = {'pixel_mask': pixel_mask, 'organ_masks': organ_masks}
        encoder_outputs = self.model.encoder(pixel_values=pixel_values, **enc_args)
        
        return self.model.generate(
            encoder_outputs=encoder_outputs, 
            decoder_input_ids=decoder_input_ids, 
            **kwargs
        )

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
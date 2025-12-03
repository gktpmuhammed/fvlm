"""
Unified Medical Vision-Language Model
Supports: BART, GPT-2, BioGPT + Organ Masking
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

class QFormer(nn.Module):
    """Q-Former Module (BERT-based)."""
    def __init__(self, hidden_size, num_query_tokens=32, num_hidden_layers=2):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=12,
            intermediate_size=3072,
            is_decoder=True,
            add_cross_attention=True
        )
        self.bert = BertModel(config)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, hidden_size))
        self.query_tokens.data.normal_(mean=0.0, std=0.02)

    def forward(self, encoder_hidden_states):
        B = encoder_hidden_states.shape[0]
        query_embeds = self.query_tokens.expand(B, -1, -1)
        outputs = self.bert(
            inputs_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True
        )
        return outputs.last_hidden_state

class MaskedViTWrapper(nn.Module):
    """
    Combined Wrapper: ViT -> (Optional Masking) -> (Optional Q-Former) -> Decoder
    """
    def __init__(self, vit_model, config, use_qformer=False, qformer=None):
        super().__init__()
        self.vit = vit_model
        self.config = config
        self.use_qformer = use_qformer
        self.qformer = qformer
        self.main_input_name = "pixel_values"

    def forward(self, pixel_values, pixel_mask=None, **kwargs):
        # 1. Vision Encoder
        outputs = self.vit(pixel_values)
        if isinstance(outputs, tuple):
            image_feats = outputs[0]
        else:
            image_feats = outputs
        
        # 2. Apply Organ Masking (If mask provided)
        if pixel_mask is not None:
            # Mask shape: (B, 1, 112, 256, 352)
            # Feat shape: (B, N_patches, 768)
            # Patch grid: 112/16=7, 256/16=16, 352/32=11
            f_d, f_h, f_w = 7, 16, 11
            
            # Downsample mask to feature grid
            mask_down = F.interpolate(pixel_mask, size=(f_d, f_h, f_w), mode='area')
            # Flatten to (B, 1232, 1)
            mask_flat = mask_down.flatten(2).transpose(1, 2)
            
            # Apply Mask
            image_feats = image_feats * (mask_flat > 0.01).float()

        # 3. Q-Former
        if self.use_qformer and self.qformer is not None:
            image_feats = self.qformer(encoder_hidden_states=image_feats)
            
        return BaseModelOutput(last_hidden_state=image_feats)

class MedicalVLM(nn.Module):
    def __init__(
        self,
        vision_encoder_path,
        decoder_model_name="facebook/bart-base", 
        image_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        use_qformer=False,
        num_query_tokens=32
    ):
        super().__init__()
        print(f"Initializing Medical VLM (Decoder: {decoder_model_name}, Q-Former: {use_qformer})")

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
        
        for param in vision_encoder.parameters():
            param.requires_grad = False

        # 2. CONFIGURE ENCODER WRAPPER
        qformer = None
        if use_qformer:
            print(f"  > Initializing Q-Former with {num_query_tokens} tokens...")
            qformer = QFormer(hidden_size=hidden_size, num_query_tokens=num_query_tokens)
            for param in qformer.parameters():
                param.requires_grad = True 
        
        # Use the unified wrapper
        wrapped_encoder = MaskedViTWrapper(vision_encoder, encoder_config, use_qformer, qformer)

        # 3. SETUP DECODER
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if "bart" in decoder_model_name:
            decoder_config = BartConfig.from_pretrained(decoder_model_name)
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True 
            decoder = BartForCausalLM.from_pretrained(decoder_model_name, config=decoder_config)
        else: 
            decoder_config = AutoConfig.from_pretrained(decoder_model_name)
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True 
            decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name, config=decoder_config)

        # 4. STRICT SURGICAL FREEZING (Restored Logic)
        for param in decoder.parameters():
            param.requires_grad = False
            
        keywords = [
            "crossattention", "encoder_attn", # The Bridge
            "ln_", "layer_norm", "layernorm", # Stability
            "lm_head", "output_projection"    # Output Vocab
        ]
        
        for name, param in decoder.named_parameters():
            if any(k in name for k in keywords):
                param.requires_grad = True
        
        if hasattr(decoder, "lm_head"):
            for param in decoder.lm_head.parameters(): param.requires_grad = True
        if hasattr(decoder, "output_projection"):
            for param in decoder.output_projection.parameters(): param.requires_grad = True

        # 5. COMPILE MODEL
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.model = VisionEncoderDecoderModel(config=config)
        self.model.encoder = wrapped_encoder
        self.model.decoder = decoder
        
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.tokenizer.vocab_size

        # Restore Projection Unfreezing logic
        if encoder_config.hidden_size != decoder_config.hidden_size:
            if hasattr(self.model, 'enc_to_dec_proj'):
                 for param in self.model.enc_to_dec_proj.parameters():
                     param.requires_grad = True

    def forward(self, pixel_values, pixel_mask=None, labels=None, **kwargs):
        # We must manually pass pixel_mask to the encoder via kwargs if using standard call,
        # but VisionEncoderDecoderModel isn't built to pass kwargs to encoder easily unless we wrap it.
        # Explicit call:
        encoder_outputs = self.model.encoder(pixel_values=pixel_values, pixel_mask=pixel_mask)
        
        return self.model(
            encoder_outputs=encoder_outputs,
            labels=labels, 
            return_dict=True, 
            **kwargs
        )

    def generate(self, pixel_values, pixel_mask=None, **kwargs):
        encoder_outputs = self.model.encoder(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return self.model.generate(encoder_outputs=encoder_outputs, **kwargs)

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
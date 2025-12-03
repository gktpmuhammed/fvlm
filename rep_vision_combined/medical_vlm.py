"""
Unified Medical Vision-Language Model
Supports: BART, GPT-2, BioGPT
Features: 
- Optional Q-Former (BLIP-2 style)
- Custom 3D ViT Encoder
- STRICT Surgical Fine-Tuning (Cross-Attn + Norms + Head ONLY)
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

class ViTWrapper(nn.Module):
    """Standard Wrapper: ViT -> Decoder"""
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

class ViTWithQFormerWrapper(nn.Module):
    """Combined Wrapper: ViT -> Q-Former -> Decoder"""
    def __init__(self, vit_model, qformer, config):
        super().__init__()
        self.vit = vit_model
        self.qformer = qformer
        self.config = config
        self.main_input_name = "pixel_values"

    def forward(self, pixel_values, **kwargs):
        outputs = self.vit(pixel_values)
        if isinstance(outputs, tuple):
            image_feats = outputs[0]
        else:
            image_feats = outputs
        qformer_outputs = self.qformer(encoder_hidden_states=image_feats)
        return BaseModelOutput(last_hidden_state=qformer_outputs)

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

        # ------------------------------------------------------------------
        # 1. SETUP ENCODER (3D ViT)
        # ------------------------------------------------------------------
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
        
        # Freeze ViT
        for param in vision_encoder.parameters():
            param.requires_grad = False

        # ------------------------------------------------------------------
        # 2. CONFIGURE ENCODER WRAPPER
        # ------------------------------------------------------------------
        if use_qformer:
            print(f"  > Initializing Q-Former with {num_query_tokens} tokens...")
            qformer = QFormer(hidden_size=hidden_size, num_query_tokens=num_query_tokens)
            for param in qformer.parameters():
                param.requires_grad = True # Q-Former always trainable
            wrapped_encoder = ViTWithQFormerWrapper(vision_encoder, qformer, encoder_config)
        else:
            wrapped_encoder = ViTWrapper(vision_encoder, encoder_config)

        # ------------------------------------------------------------------
        # 3. SETUP DECODER
        # ------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if "bart" in decoder_model_name:
            print("  > Configuring BART Decoder...")
            decoder_config = BartConfig.from_pretrained(decoder_model_name)
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True 
            decoder = BartForCausalLM.from_pretrained(decoder_model_name, config=decoder_config)
        else: 
            print("  > Configuring Causal Decoder...")
            decoder_config = AutoConfig.from_pretrained(decoder_model_name)
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True 
            decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name, config=decoder_config)

        # ------------------------------------------------------------------
        # 4. STRICT SURGICAL FREEZING
        # ------------------------------------------------------------------
        # Step 1: Freeze everything in decoder
        for param in decoder.parameters():
            param.requires_grad = False
            
        all_decoder_params = sum(p.numel() for p in decoder.parameters())
        
        # Step 2: Unfreeze specific layers by NAME only
        # Includes: Cross Attention, Layer Norms, Output Heads
        keywords = [
            "crossattention", "encoder_attn", # The Bridge
            "ln_", "layer_norm", "layernorm", # Stability
            "lm_head", "output_projection"    # Output Vocab
        ]
        
        for name, param in decoder.named_parameters():
            if any(k in name for k in keywords):
                param.requires_grad = True
        
        # Step 3: Explicit Head Unfreeze (Double Check for weight tying)
        if hasattr(decoder, "lm_head"):
            for param in decoder.lm_head.parameters(): param.requires_grad = True
        if hasattr(decoder, "output_projection"):
            for param in decoder.output_projection.parameters(): param.requires_grad = True

        # Calculate Decoder Trainable Params
        trainable_decoder = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"  > Decoder Trainable Params: {trainable_decoder:,} / {all_decoder_params:,} ({(trainable_decoder/all_decoder_params)*100:.2f}%)")

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
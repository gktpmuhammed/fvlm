"""
Unified Medical Vision-Language Model
Feature: "One-Pass" ROI Pooling for Multi-Organ Generation
Optimization: Dynamic Batching (Filters empty organs to speed up training)
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

class Attentive_ROI_Wrapper(nn.Module):
    """
    ViT -> Masked Cross-Attention -> Decoder
    Replicates the fVLM approach: Learns a 'Query' for each organ that attends
    only to the relevant pixels of that organ.
    """
    def __init__(self, vit_model, config, num_organs=32):
        super().__init__()
        self.vit = vit_model
        self.config = config
        self.hidden_size = config.hidden_size
        
        # FIX: Required by Hugging Face VisionEncoderDecoderModel
        self.main_input_name = "pixel_values"
        
        # 1. Learnable Queries (The "Interviewer" for each organ)
        self.organ_queries = nn.Parameter(torch.randn(num_organs, self.hidden_size))
        
        # 2. Cross Attention Layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, 
            num_heads=8, 
            batch_first=True
        )
        
        # 3. Layer Norm for stability
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        # Initialize
        nn.init.normal_(self.organ_queries, std=0.02)

    def forward(self, pixel_values, organ_masks=None, **kwargs):
        # 1. Run Vision Encoder
        outputs = self.vit(pixel_values)
        
        # Handle specific LAVIS/BLIP output format
        if isinstance(outputs, tuple):
            image_feats = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            image_feats = outputs.last_hidden_state
        else:
            image_feats = outputs
            
        if organ_masks is not None:
            # FIX: Handle 6D input (Batch, N, C, D, H, W) -> Squeeze C
            if organ_masks.dim() == 6:
                organ_masks = organ_masks.squeeze(2)

            # organ_masks: (Batch, N_organs, D, H, W)
            B, N_organs, D_m, H_m, W_m = organ_masks.shape
            
            # --- A. Prepare the Mask for Attention ---
            f_d, f_h, f_w = 7, 16, 11 
            flat_masks = organ_masks.view(B * N_organs, 1, D_m, H_m, W_m)
            masks_down = F.interpolate(flat_masks, size=(f_d, f_h, f_w), mode='area')
            masks_flat = masks_down.view(B, N_organs, -1)
            
            # --- SAFEGUARD: Prevent NaNs from empty masks ---
            attn_bias = torch.zeros_like(masks_flat)
            # Set background to -inf
            attn_bias[masks_flat < 0.1] = float('-inf') 
            
            # If an organ is completely missing (all -inf), attend to everything (0.0)
            is_all_inf = (attn_bias == float('-inf')).all(dim=-1, keepdim=True)
            attn_bias = attn_bias.masked_fill(is_all_inf, 0.0)
            
            # Expand Mask for MultiHeadAttention
            num_heads = self.cross_attn.num_heads
            attn_bias = attn_bias.repeat_interleave(num_heads, dim=0)
            
            # --- B. Prepare Queries ---
            queries = self.organ_queries.unsqueeze(0).expand(B, -1, -1)
            
            # --- C. Cross Attention ---
            organ_embeddings, _ = self.cross_attn(
                query=queries,
                key=image_feats,
                value=image_feats,
                attn_mask=attn_bias
            )
            
            # --- D. Flatten for Decoder ---
            organ_embeddings = self.layer_norm(organ_embeddings)
            final_embeddings = organ_embeddings.view(B * N_organs, 1, -1)
            
            return BaseModelOutput(last_hidden_state=final_embeddings)

        # Fallback
        return BaseModelOutput(last_hidden_state=image_feats)

class MedicalVLM(nn.Module):
    def __init__(
        self,
        vision_encoder_path,
        decoder_model_name="gpt2", 
        image_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        **kwargs 
    ):
        super().__init__()
        
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
        
        # Freeze ViT
        for param in vision_encoder.parameters():
            param.requires_grad = False

        # 2. WRAP ENCODER (ROI Attention)
        # Ensure num_organs matches your dataset (31 or 32)
        wrapped_encoder = Attentive_ROI_Wrapper(vision_encoder, encoder_config, num_organs=32)

        # 3. SETUP DECODER
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        decoder_config = AutoConfig.from_pretrained(decoder_model_name)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True 
        decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name, config=decoder_config)

        # Surgical Freezing
        for param in decoder.parameters(): param.requires_grad = False
        keywords = ["crossattention", "ln_", "layer_norm", "lm_head", "output_projection"]
        for name, param in decoder.named_parameters():
            if any(k in name for k in keywords):
                param.requires_grad = True

        # 4. COMPILE
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.model = VisionEncoderDecoderModel(config=config)
        self.model.encoder = wrapped_encoder
        self.model.decoder = decoder
        
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.tokenizer.vocab_size

    def forward(self, pixel_values, organ_masks=None, labels=None, **kwargs):
        # 1. Encode (B -> B*N)
        encoder_outputs = self.model.encoder(pixel_values=pixel_values, organ_masks=organ_masks)
        
        # 2. Reshape Labels (B, N, Seq) -> (B*N, Seq)
        flat_labels = None
        if labels is not None:
            B, N_organs, Seq_Len = labels.shape
            flat_labels = labels.view(B * N_organs, Seq_Len)
            
        # 3. Decode
        return self.model(
            encoder_outputs=encoder_outputs,
            labels=flat_labels, 
            return_dict=True, 
            **kwargs
        )

    # FIX: Correctly mapping input_ids to decoder_input_ids for VisionEncoderDecoder
    def generate(self, pixel_values, organ_masks=None, input_ids=None, attention_mask=None, **kwargs):
        encoder_outputs = self.model.encoder(pixel_values=pixel_values, organ_masks=organ_masks)
        
        # We MUST pass the prompts as 'decoder_input_ids'
        # 'input_ids' argument in generate() is interpreted as Encoder inputs, which we don't want.
        return self.model.generate(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
            **kwargs
        )

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
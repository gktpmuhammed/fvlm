"""
Unified Medical Vision-Language Model
Feature: "One-Pass" ROI Pooling for Multi-Organ Generation
Architecture: 8 Visual Tokens per Organ (Matching MedGemma V3)
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
    AutoModel  
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
    Supports multiple queries per organ (e.g., 8 tokens/organ = 96 total).
    """
    def __init__(self, vit_model, config, num_organs=12):
        super().__init__()
        self.vit = vit_model
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_organs = num_organs
        
        # Required by Hugging Face VisionEncoderDecoderModel
        self.main_input_name = "pixel_values"
        
        # 1. Learnable Queries (Will be resized by MedicalVLM.__init__ if queries_per_organ > 1)
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
        
        if isinstance(outputs, tuple):
            image_feats = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            image_feats = outputs.last_hidden_state
        else:
            image_feats = outputs
            
        if organ_masks is not None:
            # Handle 6D input (Batch, N, C, D, H, W) -> Squeeze C
            if organ_masks.dim() == 6:
                organ_masks = organ_masks.squeeze(2)

            # organ_masks: (Batch, N_organs, D, H, W)
            B, N_organs, D_m, H_m, W_m = organ_masks.shape
            queries_per_organ = self.organ_queries.shape[0] // N_organs
            
            # --- A. Downsample masks FIRST (saves massive memory) ---
            f_d, f_h, f_w = 7, 16, 11 
            flat_masks = organ_masks.view(B * N_organs, 1, D_m, H_m, W_m)
            masks_down = F.adaptive_max_pool3d(flat_masks, output_size=(f_d, f_h, f_w))
            # (B, N_organs, f_d*f_h*f_w)
            masks_flat = masks_down.view(B, N_organs, -1)
            
            # --- EXPAND for multi-token AFTER downsampling (cheap on small tensor) ---
            masks_flat = masks_flat.repeat_interleave(queries_per_organ, dim=1)
            
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
            
            # --- D. Reshape for Decoder ---
            organ_embeddings = self.layer_norm(organ_embeddings)
            # (B, total_tokens, D) -> (B, 12, Q, D) -> (B*12, Q, D)
            final_embeddings = organ_embeddings.view(B, N_organs, queries_per_organ, -1)
            final_embeddings = final_embeddings.view(B * N_organs, queries_per_organ, -1)
            
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
        queries_per_organ=8,
        **kwargs 
    ):
        super().__init__()
        
        self.queries_per_organ = queries_per_organ
        self.num_organs = 12
        self.total_visual_tokens = self.num_organs * self.queries_per_organ

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
        
        # Unfreeze ViT Encoder
        for param in vision_encoder.parameters():
            param.requires_grad = True

        # 2. WRAP ENCODER (ROI Attention)
        wrapped_encoder = Attentive_ROI_Wrapper(vision_encoder, encoder_config, num_organs=12)
        
        # RESIZE QUERIES for multi-token: 12 -> 96
        wrapped_encoder.organ_queries = nn.Parameter(
            torch.randn(self.total_visual_tokens, encoder_config.hidden_size)
        )
        nn.init.normal_(wrapped_encoder.organ_queries, std=0.02)

        # 3. SETUP DECODER (Supports GPT-2, BART, and similar models)
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        decoder_config = AutoConfig.from_pretrained(decoder_model_name)
        decoder_config.is_decoder = True
        # Only set add_cross_attention if the model needs it (GPT-2 does, BART already has it)
        if not getattr(decoder_config, 'add_cross_attention', False):
            decoder_config.add_cross_attention = True 
        decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name, config=decoder_config)

        # Surgical Freezing: Freeze everything, then unfreeze cross-attention + norms + head
        for param in decoder.parameters(): param.requires_grad = False
        # Keywords cover both GPT-2 and BART layer naming conventions:
        #   GPT-2: crossattention, ln_1, ln_2, lm_head
        #   BART:  encoder_attn, layer_norm, final_layer_norm, lm_head, output_projection
        keywords = [
            "crossattention", "encoder_attn",   # Cross-attention layers
            "ln_", "layer_norm", "final_layer_norm",  # Normalization layers
            "lm_head", "output_projection"       # Output head
        ]
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

        # Self-Alignment Projection: ViT (768) -> Decoder Hidden Size
        # Auto-detect hidden size across architectures (GPT-2: n_embd, BART: d_model)
        self.llm_hidden_size = getattr(decoder_config, 'n_embd', None) or \
                               getattr(decoder_config, 'd_model', None) or \
                               getattr(decoder_config, 'hidden_size', 768)
        self.visual_projection = nn.Linear(hidden_size, self.llm_hidden_size)
        
        print(f"Model Summary:")
        print(f"  Vision Encoder: Trainable (ROI Masked, {self.queries_per_organ} tokens/organ)")
        print(f"  Decoder: {decoder_model_name} (Partially Frozen)")
    
    # InfoNCE Alignment Loss (Same as MedGemma V3)
    def compute_alignment_loss(self, visual_embeds, input_ids, labels=None, temperature=0.07):
        """
        Computes Image-Text Contrastive (ITC) Loss using GPT-2's own embeddings.
        visual_embeds: (B*12, Q, D) where Q = queries_per_organ
        """
        device = visual_embeds.device
        
        # 1. Get Visual Representation: Average over Q tokens -> (B*12, D)
        vis_rep = visual_embeds.mean(dim=1) 
        
        # 2. Get Text Representation: Avg Pool the text tokens -> (B*12, D)
        with torch.no_grad():
             text_embeds = self.model.decoder.get_input_embeddings()(input_ids)
        
        # Mask out padding/prompt for averaging
        if labels is not None:
             text_mask = (labels != -100).float().unsqueeze(-1)
        else:
             text_mask = (input_ids != self.tokenizer.pad_token_id).float().unsqueeze(-1)

        text_sum = (text_embeds * text_mask).sum(dim=1)
        text_count = text_mask.sum(dim=1).clamp(min=1e-8)
        text_rep = text_sum / text_count
        
        # 3. Normalize
        vis_rep = F.normalize(vis_rep, dim=-1)
        text_rep = F.normalize(text_rep, dim=-1)
        
        # 4. InfoNCE Loss (Per-patient contrastive over 12 organs)
        B_N, D = vis_rep.shape
        num_organs = self.num_organs
        B = B_N // num_organs
        
        vis_rep = vis_rep.view(B, num_organs, D)
        text_rep = text_rep.view(B, num_organs, D)
        
        total_loss = 0
        loss_fct = nn.CrossEntropyLoss()
        
        for i in range(B):
            v = vis_rep[i]
            t = text_rep[i]
            logits = torch.matmul(v, t.T) / temperature
            labels_idx = torch.arange(num_organs, device=logits.device)
            total_loss += loss_fct(logits, labels_idx)
            
        return total_loss / B

    # Forward with Sample Weights and InfoNCE Alignment Loss
    def forward(self, pixel_values, organ_masks=None, labels=None, sample_weights=None, **kwargs):
        # Strip kwargs that Trainer injects but VisionEncoderDecoderModel doesn't accept
        kwargs.pop('num_items_in_batch', None)
        
        # 1. Run Encoder
        encoder_outputs = self.model.encoder(pixel_values=pixel_values, organ_masks=organ_masks)
        
        if labels is not None:
            B, N_organs, Seq_Len = labels.shape
            flat_labels = labels.view(B * N_organs, Seq_Len)
            
            # Reconstruct input_ids for alignment loss
            flat_input_ids = flat_labels.clone()
            flat_input_ids[flat_input_ids == -100] = self.tokenizer.pad_token_id

            if sample_weights is not None:
                # --- Manual Loss Computation with Sample Weights ---
                flat_weights = sample_weights.view(B * N_organs)
                
                # Create decoder_input_ids (shift right)
                decoder_start_id = self.model.config.decoder_start_token_id
                decoder_input_ids = flat_labels.new_zeros(flat_labels.shape)
                decoder_input_ids[:, 1:] = flat_labels[:, :-1].clone()
                decoder_input_ids[:, 0] = decoder_start_id
                decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.tokenizer.pad_token_id)
                
                # Forward without internal loss (get logits only)
                outputs = self.model(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids,
                    labels=None,
                    return_dict=True,
                    **kwargs
                )
                
                logits = outputs.logits
                
                # Shift for loss computation
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = flat_labels[..., 1:].contiguous()
                
                # Per-token loss
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                token_losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                ).view(B * N_organs, -1)
                
                # Per-sample loss
                non_pad_mask = (shift_labels != -100).float()
                sample_loss = (token_losses * non_pad_mask).sum(dim=1) / (non_pad_mask.sum(dim=1) + 1e-8)
                
                # Apply weights
                flat_weights = flat_weights.to(sample_loss.device)
                sample_loss = sample_loss * flat_weights
                
                lm_loss = sample_loss.mean()
            else:
                # Standard loss (no weighting) - let HF compute it
                outputs = self.model(
                    encoder_outputs=encoder_outputs,
                    labels=flat_labels,
                    return_dict=True,
                    **kwargs
                )
                lm_loss = outputs.loss
            
            # 3. Add Alignment Loss during training
            if self.training:
                vis_feats = encoder_outputs.last_hidden_state  # (B*12, Q, 768)
                vis_embeds = self.visual_projection(vis_feats)
                align_loss = self.compute_alignment_loss(vis_embeds, flat_input_ids, labels=flat_labels)
                lm_loss = lm_loss + align_loss
            
            # Return a clean output with loss
            from transformers.modeling_outputs import Seq2SeqLMOutput
            return Seq2SeqLMOutput(
                loss=lm_loss,
                logits=outputs.logits,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            )
        else:
            # Inference (no labels)
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                return_dict=True,
                **kwargs
            )
            return outputs

    def generate(self, pixel_values, organ_masks=None, input_ids=None, attention_mask=None, **kwargs):
        encoder_outputs = self.model.encoder(pixel_values=pixel_values, organ_masks=organ_masks)
        
        return self.model.generate(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
            **kwargs
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the decoder."""
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
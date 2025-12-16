import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from transformers.modeling_outputs import BaseModelOutput

# --- BASE INTERFACE ---
class VisionAdapter(nn.Module):
    """Base interface for connecting ViT features to the Decoder."""
    def __init__(self, vit_model, config):
        super().__init__()
        self.vit = vit_model
        self.config = config
        # FIX: Required by Hugging Face generate()
        self.main_input_name = "pixel_values"

    def forward(self, pixel_values, **kwargs):
        raise NotImplementedError

# --- STRATEGY 1: GLOBAL (Whole Image -> Text) ---
class GlobalAdapter(VisionAdapter):
    def __init__(self, vit_model, config, use_qformer=False, num_query_tokens=32):
        super().__init__(vit_model, config)
        self.use_qformer = use_qformer
        if use_qformer:
            q_config = BertConfig(
                hidden_size=config.hidden_size,
                num_hidden_layers=2,
                num_attention_heads=12,
                intermediate_size=3072,
                is_decoder=True,
                add_cross_attention=True
            )
            self.qformer = BertModel(q_config)
            self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, config.hidden_size))
            nn.init.normal_(self.query_tokens, std=0.02)

    def forward(self, pixel_values, **kwargs):
        outputs = self.vit(pixel_values)
        image_feats = outputs[0] if isinstance(outputs, tuple) else outputs
        
        if hasattr(image_feats, "last_hidden_state"):
            image_feats = image_feats.last_hidden_state

        if self.use_qformer:
            B = image_feats.shape[0]
            query_embeds = self.query_tokens.expand(B, -1, -1)
            out = self.qformer(
                inputs_embeds=query_embeds,
                encoder_hidden_states=image_feats,
                return_dict=True
            )
            image_feats = out.last_hidden_state
            
        return BaseModelOutput(last_hidden_state=image_feats)

# --- STRATEGY 2: MASKED SINGLE (Flattened Dataset) ---
class MaskedSingleAdapter(VisionAdapter):
    def forward(self, pixel_values, pixel_mask=None, **kwargs):
        outputs = self.vit(pixel_values)
        image_feats = outputs[0] if isinstance(outputs, tuple) else outputs
        if hasattr(image_feats, "last_hidden_state"): image_feats = image_feats.last_hidden_state
        
        if pixel_mask is not None:
            # Downsample mask to feature grid (approx 7x16x11)
            f_d, f_h, f_w = 7, 16, 11
            mask_down = F.interpolate(pixel_mask, size=(f_d, f_h, f_w), mode='area')
            mask_flat = mask_down.flatten(2).transpose(1, 2) # (B, Seq, 1)
            
            # Soft Apply
            image_feats = image_feats * (mask_flat > 0.01).float()
            
        return BaseModelOutput(last_hidden_state=image_feats)

# --- STRATEGY 3: PARALLEL ROI POOLING (Hard Pooling) ---
class ROIPoolingAdapter(VisionAdapter):
    def forward(self, pixel_values, organ_masks=None, **kwargs):
        outputs = self.vit(pixel_values)
        image_feats = outputs[0] if isinstance(outputs, tuple) else outputs
        if hasattr(image_feats, "last_hidden_state"): image_feats = image_feats.last_hidden_state
        
        if organ_masks is None:
            return BaseModelOutput(last_hidden_state=image_feats)

        if organ_masks.dim() == 6: organ_masks = organ_masks.squeeze(2)
        B, N, D, H, W = organ_masks.shape
        
        # Interpolate
        flat_masks = organ_masks.view(B * N, 1, D, H, W)
        masks_down = F.interpolate(flat_masks, size=(7, 16, 11), mode='area')
        masks_flat = masks_down.view(B, N, -1)
        
        # Weighted Pooling
        masks_norm = masks_flat / (masks_flat.sum(dim=2, keepdim=True) + 1e-6)
        organ_embeds = torch.bmm(masks_norm, image_feats) # (B, N, Hidden)
        
        final = organ_embeds.view(B * N, 1, -1)
        return BaseModelOutput(last_hidden_state=final)

# --- STRATEGY 4: PARALLEL ATTENTION ---
class AttentionAdapter(VisionAdapter):
    def __init__(self, vit_model, config, num_organs=12):
        super().__init__(vit_model, config)
        self.hidden_size = config.hidden_size
        
        # Learnable Queries
        self.organ_queries = nn.Parameter(torch.randn(num_organs, self.hidden_size))
        
        self.cross_attn = nn.MultiheadAttention(self.hidden_size, 8, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        nn.init.normal_(self.organ_queries, std=0.02)

    def forward(self, pixel_values, organ_masks=None, **kwargs):
        outputs = self.vit(pixel_values)
        image_feats = outputs[0] if isinstance(outputs, tuple) else outputs
        if hasattr(image_feats, "last_hidden_state"): image_feats = image_feats.last_hidden_state
        
        if organ_masks is None: return BaseModelOutput(last_hidden_state=image_feats)
        
        if organ_masks.dim() == 6: organ_masks = organ_masks.squeeze(2)
        B, N, D, H, W = organ_masks.shape
        
        # A. Prepare Mask for Attention Bias
        f_d, f_h, f_w = 7, 16, 11
        flat_masks = organ_masks.view(B * N, 1, D, H, W)
        masks_down = F.interpolate(flat_masks, size=(f_d, f_h, f_w), mode='area')
        masks_flat = masks_down.view(B, N, -1)
        
        # B. Construct Bias
        attn_bias = torch.zeros_like(masks_flat)
        attn_bias[masks_flat < 0.1] = float('-inf')
        
        # C. Safeguard
        is_all_inf = (attn_bias == float('-inf')).all(dim=-1, keepdim=True)
        attn_bias = attn_bias.masked_fill(is_all_inf, 0.0)
        
        # Expand for heads
        attn_bias = attn_bias.repeat_interleave(8, dim=0) # 8 heads
        
        # D. Prepare Queries
        queries = self.organ_queries.unsqueeze(0).expand(B, -1, -1)
        
        # E. Cross Attention
        organ_embeds, _ = self.cross_attn(
            query=queries,
            key=image_feats,
            value=image_feats,
            attn_mask=attn_bias
        )
        
        organ_embeds = self.layer_norm(organ_embeds)
        final = organ_embeds.view(B * N, 1, -1)
        
        return BaseModelOutput(last_hidden_state=final)
    
# --- STRATEGY 5: ATTENTION + Q-FORMER ---
class AttentionQFormerAdapter(VisionAdapter):
    """
    Uses a BERT-based Q-Former to extract organ features.
    Corrected to handle CLS tokens and empty masks.
    """
    def __init__(self, vit_model, config, num_organs=12, num_query_tokens=1):
        super().__init__(vit_model, config)
        self.hidden_size = config.hidden_size
        
        # 1. Q-Former Configuration
        q_config = BertConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=2, 
            num_attention_heads=12,
            intermediate_size=3072,
            is_decoder=True,
            add_cross_attention=True
        )
        self.qformer = BertModel(q_config)
        
        # 2. Learnable Queries (1 vector per organ)
        self.organ_queries = nn.Parameter(torch.zeros(num_organs, num_query_tokens, config.hidden_size))
        nn.init.normal_(self.organ_queries, std=0.02)

    def forward(self, pixel_values, organ_masks=None, **kwargs):
        # 1. Vision Encoder
        outputs = self.vit(pixel_values)
        image_feats = outputs[0] if isinstance(outputs, tuple) else outputs
        if hasattr(image_feats, "last_hidden_state"): 
            image_feats = image_feats.last_hidden_state
        
        # image_feats shape: (B, Seq_Len, C) -> Usually (B, 1233, 768) including CLS
        B, Seq_Len_Img, C = image_feats.shape
        
        if organ_masks is None:
            return BaseModelOutput(last_hidden_state=image_feats)

        # 2. Process Masks
        if organ_masks.dim() == 6: organ_masks = organ_masks.squeeze(2)
        _, N_Organs, D, H, W = organ_masks.shape
        
        f_d, f_h, f_w = 7, 16, 11
        flat_masks = organ_masks.view(B * N_Organs, 1, D, H, W)
        masks_down = F.interpolate(flat_masks, size=(f_d, f_h, f_w), mode='area')
        
        # Flatten spatial masks
        masks_flat = masks_down.view(B * N_Organs, -1) # (B*N, 1232)
        
        # --- FIX 1: Handle CLS Token Alignment ---
        # If image features have 1 more token than the mask, it's the CLS token.
        # We must prepend a '1' to the mask so Q-Former always sees the CLS token.
        if Seq_Len_Img == masks_flat.shape[1] + 1:
            cls_mask = torch.ones((masks_flat.shape[0], 1), device=masks_flat.device)
            masks_flat = torch.cat([cls_mask, masks_flat], dim=1)
        
        # Create Binary Attention Mask
        attention_mask = (masks_flat > 0.1).long()
        
        # --- FIX 2: Handle Empty Masks (Blind Query Safeguard) ---
        # If an organ is missing, the mask (excluding CLS) is all zeros.
        # This causes NaNs. If empty, force attention to the whole image (Global Context).
        mask_sum = attention_mask.sum(dim=1)
        is_empty = (mask_sum <= 1) # <=1 means only CLS or nothing is visible
        
        if is_empty.any():
            attention_mask[is_empty] = 1 # Unmask everything for missing organs
            
        # 3. Prepare Inputs
        image_feats_expanded = image_feats.repeat_interleave(N_Organs, dim=0)
        queries_expanded = self.organ_queries.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N_Organs, 1, C)
        
        # 4. Q-Former Forward
        q_out = self.qformer(
            inputs_embeds=queries_expanded,
            encoder_hidden_states=image_feats_expanded,
            encoder_attention_mask=attention_mask,
            return_dict=True
        )
        
        return BaseModelOutput(last_hidden_state=q_out.last_hidden_state)
    
# --- STRATEGY 6: ROBUST Q-FORMER (With Mask Existence Gate) ---

class RobustQFormerAdapter(nn.Module):
    def __init__(self, vit_model, config, num_organs=12, num_query_tokens=1):
        super().__init__()
        # FIX 1: Store config so VisionEncoderDecoderModel can check hidden_size
        self.config = config 
        
        self.vit = vit_model
        self.num_organs = num_organs
        self.num_query_tokens = num_query_tokens
        self.main_input_name = "pixel_values"
        
        # Q-Former Setup
        q_config = BertConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=2, 
            num_attention_heads=12,
            intermediate_size=3072,
            is_decoder=True,
            add_cross_attention=True
        )
        self.qformer = BertModel(q_config)
        self.organ_queries = nn.Parameter(torch.zeros(num_organs, num_query_tokens, config.hidden_size))
        self.organ_queries.data.normal_(mean=0.0, std=0.02)

    def forward(self, pixel_values, organ_masks=None, **kwargs):
        # 1. Vision Encoder
        outputs = self.vit(pixel_values)
        image_feats = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        B, Seq_Len, C = image_feats.shape
        
        # Initialize Output Container: (Batch * Num_Organs, Tokens, Hidden)
        final_output_flat = torch.zeros(
            B * self.num_organs, self.num_query_tokens, C, 
            device=image_feats.device, dtype=image_feats.dtype
        )

        # 2. Logic Gate: Do we have masks?
        if organ_masks is not None:
            if organ_masks.dim() == 6: organ_masks = organ_masks.squeeze(2)
            _, N, D, H, W = organ_masks.shape
            
            # Interpolate & Flatten Masks
            f_d, f_h, f_w = 7, 16, 11 
            flat_masks = organ_masks.view(B * N, 1, D, H, W)
            masks_down = F.interpolate(flat_masks, size=(f_d, f_h, f_w), mode='area')
            masks_flat = masks_down.view(B * N, -1) 

            # CLS Token Alignment
            if Seq_Len == masks_flat.shape[1] + 1:
                cls_mask = torch.ones((masks_flat.shape[0], 1), device=masks_flat.device)
                masks_flat = torch.cat([cls_mask, masks_flat], dim=1)
            
            attention_mask = (masks_flat > 0.1).long()

            # Filter: Which organs actually exist?
            valid_organs_mask = masks_flat[:, 1:].sum(dim=1) > 0.1
            
            if valid_organs_mask.any():
                # Extract valid inputs
                valid_attn_mask = attention_mask[valid_organs_mask]
                
                # Expand & Filter Image Feats
                image_feats_expanded = image_feats.repeat_interleave(N, dim=0)
                valid_image_feats = image_feats_expanded[valid_organs_mask]
                
                # Expand & Filter Queries
                queries_expanded = self.organ_queries.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N, -1, C)
                valid_queries = queries_expanded[valid_organs_mask]

                # Run Q-Former
                q_out = self.qformer(
                    inputs_embeds=valid_queries,
                    encoder_hidden_states=valid_image_feats,
                    encoder_attention_mask=valid_attn_mask,
                    return_dict=True
                )
                
                # Scatter results back
                final_output_flat[valid_organs_mask] = q_out.last_hidden_state
        
        return BaseModelOutput(last_hidden_state=final_output_flat)
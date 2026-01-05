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
    AutoModel  # ### NEW: Required for loading BERT
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
    """
    def __init__(self, vit_model, config, num_organs=12):
        super().__init__()
        self.vit = vit_model
        self.config = config
        self.hidden_size = config.hidden_size
        
        # FIX: Required by Hugging Face VisionEncoderDecoderModel
        self.main_input_name = "pixel_values"
        
        # 1. Learnable Queries (The "Interviewer" for each organ)
        # Updated to 12 based on the refined list
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
        bert_model_path="/home/muhammedg/fvlm/BiomedVLP-CXR-BERT-specialized", # ### NEW: Argument for BERT path
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
        
        # Unfreeze ViT Encoder
        for param in vision_encoder.parameters():
            param.requires_grad = True

        # 2. WRAP ENCODER (ROI Attention)
        wrapped_encoder = Attentive_ROI_Wrapper(vision_encoder, encoder_config, num_organs=12)

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

        # ### NEW: Load CXR-BERT Teacher (If path provided)
        if bert_model_path and os.path.exists(bert_model_path):
            print(f"  Loading CXR-BERT Teacher from: {bert_model_path}")
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path, trust_remote_code=True)
            self.bert_model = AutoModel.from_pretrained(bert_model_path, trust_remote_code=True)
            
            # Freeze BERT (Teacher only)
            for param in self.bert_model.parameters():
                param.requires_grad = False
            
            # Projection Layer: ViT (768) -> BERT Hidden Size (usually 128 or 768)
            # We assume CXR-BERT-specialized uses 128 dim projection in original paper, 
            # but the raw HuggingFace model usually outputs 768. 
            # We project to match the BERT hidden size.
            bert_dim = self.bert_model.config.hidden_size
            self.visual_projection = nn.Linear(768, bert_dim)
        else:
            print("  No BERT path provided or path invalid. Semantic Loss disabled.")
            self.bert_model = None

    # ### NEW: Semantic Loss Calculation Helper
    def get_semantic_loss(self, visual_features, label_ids):
        """
        visual_features: (Batch * N_Organs, Hidden_Size)
        label_ids: (Batch * N_Organs, Seq_Len)
        """
        device = visual_features.device
        
        # 1. Decode Labels to Text (Batch Decode)
        # We replace -100 with pad_token_id to decode correctly
        clean_labels = label_ids.clone()
        clean_labels[clean_labels == -100] = self.tokenizer.pad_token_id
        
        decoded_texts = self.tokenizer.batch_decode(clean_labels, skip_special_tokens=True)
        
        # Filter out empty strings (missing organs) to avoid BERT errors
        # Create a mask for valid texts
        valid_indices = [i for i, t in enumerate(decoded_texts) if len(t.strip()) > 3]
        if not valid_indices:
            return torch.tensor(0.0, device=device)
            
        valid_texts = [decoded_texts[i] for i in valid_indices]
        valid_visuals = visual_features[valid_indices]
        
        # 2. Get Text Embeddings from BERT
        with torch.no_grad():
            inputs = self.bert_tokenizer(
                valid_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(device)
            bert_outputs = self.bert_model(**inputs)
            # Use [CLS] token (index 0)
            text_embeds = bert_outputs.last_hidden_state[:, 0, :]
            
        # 3. Project Visual Embeddings
        vis_proj = self.visual_projection(valid_visuals)
        
        # 4. Normalize
        vis_norm = F.normalize(vis_proj, dim=-1)
        text_norm = F.normalize(text_embeds, dim=-1)
        
        # 5. Cosine Similarity Loss (Maximize similarity -> Minimize 1 - Sim)
        similarity = (vis_norm * text_norm).sum(dim=-1)
        loss = 1.0 - similarity.mean()
        
        return loss

    # ### MODIFIED: Forward to include Semantic Loss
    def forward(self, pixel_values, organ_masks=None, labels=None, **kwargs):
        # 1. Run Encoder explicitly to get embeddings
        encoder_outputs = self.model.encoder(pixel_values=pixel_values, organ_masks=organ_masks)
        
        flat_labels = None
        if labels is not None:
            B, N_organs, Seq_Len = labels.shape
            flat_labels = labels.view(B * N_organs, Seq_Len)
            
        # 2. Run Decoder (Standard generation loss)
        outputs = self.model(
            encoder_outputs=encoder_outputs,
            labels=flat_labels, 
            return_dict=True, 
            **kwargs
        )
        
        # 3. Add Semantic Loss if BERT is loaded and we are training
        if labels is not None and self.bert_model is not None and self.training:
            # encoder_outputs.last_hidden_state is (Batch * N, 1, 768)
            # Squeeze to (Batch * N, 768)
            vis_feats = encoder_outputs.last_hidden_state.squeeze(1)
            
            sem_loss = self.get_semantic_loss(vis_feats, flat_labels)
            
            # Combine Losses (Weight semantic loss)
            # 0.5 is a conservative weight; can be increased to 1.0 if hallucinations persist
            outputs.loss += (0.5 * sem_loss)
            
        return outputs

    def generate(self, pixel_values, organ_masks=None, input_ids=None, attention_mask=None, **kwargs):
        encoder_outputs = self.model.encoder(pixel_values=pixel_values, organ_masks=organ_masks)
        
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
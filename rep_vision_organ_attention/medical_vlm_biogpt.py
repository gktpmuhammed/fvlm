"""
Unified Medical Vision-Language Model
Feature: "One-Pass" ROI Pooling for Multi-Organ Generation
Optimization: Dynamic Batching (Filters empty organs to speed up training)
"""
import sys
import os
import copy
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
# Attempt to import BioGpt classes; handle errors if not present
try:
    from transformers import BioGptForCausalLM, BioGptModel
    HAS_BIOGPT = True
except ImportError:
    HAS_BIOGPT = False

from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithCrossAttentions

# Fix local import path for lavis
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ### FIX: BioGPT Wrapper for Visual Prefix Injection ###
if HAS_BIOGPT:
    class BioGptLM_Fixed(BioGptForCausalLM):
        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None, # This contains our Visual Features!
            encoder_attention_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
        ):
            # 1. Get Text Embeddings
            if inputs_embeds is None:
                inputs_embeds = self.biogpt.embed_tokens(input_ids)

            # 2. Inject Visual Prefix (Only if we have visual feats and no history yet)
            visual_len = 0
            if encoder_hidden_states is not None:
                if past_key_values is None:
                    # Capture length to trim later
                    visual_len = encoder_hidden_states.shape[1]
                    
                    # Concatenate [Visual_Feats, Text_Feats]
                    inputs_embeds = torch.cat([encoder_hidden_states, inputs_embeds], dim=1)
                    
                    # Extend Attention Mask
                    if attention_mask is not None:
                        B, N_Vis = encoder_hidden_states.shape[:2]
                        vis_mask = torch.ones((B, N_Vis), device=attention_mask.device, dtype=attention_mask.dtype)
                        attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
            
            # 3. Call Internal BioGPT Model
            outputs = self.biogpt(
                input_ids=None, 
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

            hidden_states = outputs[0]
            lm_logits = self.output_projection(hidden_states)

            # ### CRITICAL FIX: Trim Visual Prefix from Logits ###
            # VisionEncoderDecoderModel calculates loss using 'labels' which correspond ONLY to text.
            # We must remove the visual logits so shapes match.
            if visual_len > 0 and past_key_values is None:
                lm_logits = lm_logits[:, visual_len:, :]

            loss = None
            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                cross_attentions=outputs.cross_attentions,
            )

class Attentive_ROI_Wrapper(nn.Module):
    """
    ViT -> Masked Cross-Attention -> Decoder
    """
    def __init__(self, vit_model, config, num_organs=12, decoder_hidden_size=768):
        super().__init__()
        self.vit = vit_model
        
        # Update internal config to match Decoder Size
        self.config = copy.deepcopy(config)
        self.hidden_size = config.hidden_size 
        self.config.hidden_size = decoder_hidden_size 
        
        self.main_input_name = "pixel_values"
        
        # Project Vision Features
        self.decoder_hidden_size = decoder_hidden_size
        if self.hidden_size != self.decoder_hidden_size:
            self.vis_project = nn.Linear(self.hidden_size, self.decoder_hidden_size)
        else:
            self.vis_project = nn.Identity()

        self.organ_queries = nn.Parameter(torch.randn(num_organs, self.decoder_hidden_size))
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.decoder_hidden_size, 
            num_heads=8, 
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(self.decoder_hidden_size)
        nn.init.normal_(self.organ_queries, std=0.02)

    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, new_embeddings):
        pass

    def forward(self, pixel_values, organ_masks=None, **kwargs):
        outputs = self.vit(pixel_values)
        
        if isinstance(outputs, tuple):
            image_feats = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            image_feats = outputs.last_hidden_state
        else:
            image_feats = outputs
            
        image_feats = self.vis_project(image_feats)
            
        if organ_masks is not None:
            if organ_masks.dim() == 6:
                organ_masks = organ_masks.squeeze(2)

            B, N_organs, D_m, H_m, W_m = organ_masks.shape
            
            f_d, f_h, f_w = 7, 16, 11 
            flat_masks = organ_masks.view(B * N_organs, 1, D_m, H_m, W_m)
            masks_down = F.interpolate(flat_masks, size=(f_d, f_h, f_w), mode='area')
            masks_flat = masks_down.view(B, N_organs, -1)
            
            attn_bias = torch.zeros_like(masks_flat)
            attn_bias[masks_flat < 0.1] = float('-inf') 
            is_all_inf = (attn_bias == float('-inf')).all(dim=-1, keepdim=True)
            attn_bias = attn_bias.masked_fill(is_all_inf, 0.0)
            
            num_heads = self.cross_attn.num_heads
            attn_bias = attn_bias.repeat_interleave(num_heads, dim=0)
            
            queries = self.organ_queries.unsqueeze(0).expand(B, -1, -1)
            
            organ_embeddings, _ = self.cross_attn(
                query=queries,
                key=image_feats,
                value=image_feats,
                attn_mask=attn_bias
            )
            
            organ_embeddings = self.layer_norm(organ_embeddings)
            final_embeddings = organ_embeddings.view(B * N_organs, 1, -1)
            
            return BaseModelOutput(last_hidden_state=final_embeddings)

        return BaseModelOutput(last_hidden_state=image_feats)

class MedicalVLM(nn.Module):
    def __init__(
        self,
        vision_encoder_path,
        decoder_model_name="gpt2", 
        image_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        bert_model_path="/home/muhammedg/fvlm/BiomedVLP-CXR-BERT-specialized", 
        **kwargs 
    ):
        super().__init__()
        
        # 1. SETUP ENCODER
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
            param.requires_grad = True

        # 3. SETUP DECODER
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        decoder_config = AutoConfig.from_pretrained(decoder_model_name)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True 
        
        if HAS_BIOGPT and ("biogpt" in decoder_model_name.lower()):
            print("  Using Fixed BioGPT Decoder class.")
            decoder = BioGptLM_Fixed.from_pretrained(decoder_model_name, config=decoder_config)
        else:
            decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name, config=decoder_config)
        
        dec_hidden_size = decoder_config.hidden_size

        # 2. WRAP ENCODER
        wrapped_encoder = Attentive_ROI_Wrapper(
            vision_encoder, 
            encoder_config, 
            num_organs=12, 
            decoder_hidden_size=dec_hidden_size
        )

        for param in decoder.parameters(): param.requires_grad = False
        
        keywords = ["crossattention", "encoder_attn", "ln_", "layer_norm", "lm_head", "output_projection"]
        print("  Unfreezing Decoder Layers:")
        for name, param in decoder.named_parameters():
            if any(k in name for k in keywords):
                param.requires_grad = True

        # 4. COMPILE
        ved_enc_config = copy.deepcopy(encoder_config)
        ved_enc_config.hidden_size = dec_hidden_size 
        
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(ved_enc_config, decoder_config)
        
        self.model = VisionEncoderDecoderModel(config=config, encoder=wrapped_encoder, decoder=decoder)
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.tokenizer.vocab_size

        if bert_model_path and os.path.exists(bert_model_path):
            print(f"  Loading CXR-BERT Teacher from: {bert_model_path}")
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path, trust_remote_code=True)
            self.bert_model = AutoModel.from_pretrained(bert_model_path, trust_remote_code=True)
            
            for param in self.bert_model.parameters():
                param.requires_grad = False
            
            bert_dim = self.bert_model.config.hidden_size
            self.visual_projection = nn.Linear(dec_hidden_size, bert_dim)
        else:
            print("  No BERT path provided or path invalid. Semantic Loss disabled.")
            self.bert_model = None

    def get_semantic_loss(self, visual_features, label_ids):
        device = visual_features.device
        clean_labels = label_ids.clone()
        clean_labels[clean_labels == -100] = self.tokenizer.pad_token_id
        decoded_texts = self.tokenizer.batch_decode(clean_labels, skip_special_tokens=True)
        valid_indices = [i for i, t in enumerate(decoded_texts) if len(t.strip()) > 3]
        if not valid_indices: return torch.tensor(0.0, device=device)
        valid_texts = [decoded_texts[i] for i in valid_indices]
        valid_visuals = visual_features[valid_indices]
        with torch.no_grad():
            inputs = self.bert_tokenizer(valid_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            bert_outputs = self.bert_model(**inputs)
            text_embeds = bert_outputs.last_hidden_state[:, 0, :]
        vis_proj = self.visual_projection(valid_visuals)
        vis_norm = F.normalize(vis_proj, dim=-1)
        text_norm = F.normalize(text_embeds, dim=-1)
        similarity = (vis_norm * text_norm).sum(dim=-1)
        loss = 1.0 - similarity.mean()
        return loss

    def forward(self, pixel_values, organ_masks=None, labels=None, **kwargs):
        encoder_outputs = self.model.encoder(pixel_values=pixel_values, organ_masks=organ_masks)
        flat_labels = None
        if labels is not None:
            B, N_organs, Seq_Len = labels.shape
            flat_labels = labels.view(B * N_organs, Seq_Len)
        outputs = self.model(encoder_outputs=encoder_outputs, labels=flat_labels, return_dict=True, **kwargs)
        if labels is not None and self.bert_model is not None and self.training:
            vis_feats = encoder_outputs.last_hidden_state.squeeze(1)
            sem_loss = self.get_semantic_loss(vis_feats, flat_labels)
            outputs.loss += (0.5 * sem_loss)
        return outputs

    def generate(self, pixel_values, organ_masks=None, input_ids=None, attention_mask=None, **kwargs):
        encoder_outputs = self.model.encoder(pixel_values=pixel_values, organ_masks=organ_masks)
        return self.model.generate(encoder_outputs=encoder_outputs, decoder_input_ids=input_ids, decoder_attention_mask=attention_mask, **kwargs)

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
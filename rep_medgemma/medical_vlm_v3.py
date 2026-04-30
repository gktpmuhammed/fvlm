"""
Medical VLM — CNN Stem + ViT architecture (V3)
Copied and patched to return attention weights from cross-attention.
Only the CNNStem + Attentive_ROI_Wrapper + MedicalVLM shell are kept;
LLM is not loaded when used for attention visualization.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    ViTConfig,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType

# Fix local import path for lavis
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)


class CNNStem(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Downsample 2x
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True),
            # Layer 2: Downsample 2x (Total 4x)
            nn.Conv3d(32, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class Attentive_ROI_Wrapper(nn.Module):
    """
    ViT -> Masked Cross-Attention -> Organ Specific Embeddings
    Uses adaptive_max_pool3d for mask downsampling (preserves organ presence).
    """
    def __init__(self, vit_model, config, num_organs=12):
        super().__init__()
        self.vit = vit_model
        self.hidden_size = config.hidden_size
        
        # Learnable Queries
        self.organ_queries = nn.Parameter(torch.randn(num_organs, self.hidden_size))
        
        # Cross Attention Layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, 
            num_heads=8, 
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        nn.init.normal_(self.organ_queries, std=0.02)

    def forward(self, pixel_values, organ_masks=None):
        pixel_values = pixel_values.to(dtype=torch.float32)
        with torch.cuda.amp.autocast(enabled=False):
            outputs = self.vit(pixel_values)
        
        if isinstance(outputs, tuple):
            image_feats = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            image_feats = outputs.last_hidden_state
        else:
            image_feats = outputs
            
        if organ_masks is not None:
            if organ_masks.dim() == 6:
                organ_masks = organ_masks.squeeze(2)

            B, N_organs, D_m, H_m, W_m = organ_masks.shape
            
            queries_per_organ = self.organ_queries.shape[0] // N_organs
            organ_masks = organ_masks.repeat_interleave(queries_per_organ, dim=1)
            
            f_d, f_h, f_w = 7, 16, 11 
            
            flat_masks = organ_masks.view(B * organ_masks.shape[1], 1, D_m, H_m, W_m)
            # Use MaxPool to preserve ANY organ presence in the patch
            masks_down = F.adaptive_max_pool3d(flat_masks, output_size=(f_d, f_h, f_w))
            masks_flat = masks_down.view(B, organ_masks.shape[1], -1)
            
            attn_bias = torch.zeros_like(masks_flat)
            attn_bias[masks_flat < 0.1] = float('-inf') 
            
            is_all_inf = (attn_bias == float('-inf')).all(dim=-1, keepdim=True)
            attn_bias = attn_bias.masked_fill(is_all_inf, 0.0)
            
            num_heads = self.cross_attn.num_heads
            attn_bias = attn_bias.repeat_interleave(num_heads, dim=0)
            
            queries = self.organ_queries.unsqueeze(0).expand(B, -1, -1)
            
            organ_embeddings, attn_weights = self.cross_attn(
                query=queries,
                key=image_feats,
                value=image_feats,
                attn_mask=attn_bias
            )
            
            return self.layer_norm(organ_embeddings), attn_weights

        return self.layer_norm(image_feats), None


class MedicalVLM(nn.Module):
    def __init__(
        self,
        vision_encoder_path,
        decoder_model_name="google/medgemma-4b-it",
        image_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        queries_per_organ=8,
        **kwargs 
    ):
        super().__init__()
        
        self.queries_per_organ = queries_per_organ
        self.num_organs = 12
        self.total_visual_tokens = self.num_organs * self.queries_per_organ

        # --- CNN Stem ---
        self.stem = CNNStem(out_channels=64)
        
        # --- ViT (takes stem output) ---
        vit_img_size = (28, 64, 88)
        vit_patch_size = (4, 4, 8)
        vit_in_chans = 64
        vit_hidden_size = 768
        
        encoder_config = ViTConfig(
            hidden_size=vit_hidden_size, image_size=vit_img_size, patch_size=vit_patch_size
        )

        try:
            from lavis.models.blip_models.vit import ViT
            vision_encoder = ViT(
                in_channels=vit_in_chans, 
                img_size=vit_img_size, 
                patch_size=vit_patch_size, 
                num_classes=0
            )
        except ImportError:
            raise ImportError("Could not find 'lavis.models.blip_models.vit'. Please check python path.")
        
        self.vision_encoder = Attentive_ROI_Wrapper(vision_encoder, encoder_config, num_organs=12)

        # RESIZE QUERIES for multi-token
        self.vision_encoder.organ_queries = nn.Parameter(
            torch.randn(self.total_visual_tokens, encoder_config.hidden_size)
        )
        nn.init.normal_(self.vision_encoder.organ_queries, std=0.02)

        for param in self.vision_encoder.parameters():
            param.requires_grad = True
        for param in self.stem.parameters():
            param.requires_grad = True

        # --- LLM (same as base8t for weight loading compat) ---
        print(f"Loading Frozen Decoder: {decoder_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading Decoder in BF16 (No Quantization)...")
        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        )
        
        special_tokens_dict = {'additional_special_tokens': ['<vis>', '<end_vis>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.vis_token_id = self.tokenizer.convert_tokens_to_ids('<vis>')
        self.end_vis_token_id = self.tokenizer.convert_tokens_to_ids('<end_vis>')
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

        print("Applying LoRA to Decoder...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.decoder = get_peft_model(self.decoder, peft_config)
        
        with torch.no_grad():
            proxy_ids = self.tokenizer(["image", "end"], add_special_tokens=False).input_ids
            img_id = proxy_ids[0][0] if proxy_ids and len(proxy_ids[0]) > 0 else 2
            end_id = proxy_ids[1][0] if len(proxy_ids) > 1 and len(proxy_ids[1]) > 0 else 2
            input_embeddings = self.decoder.get_input_embeddings().weight
            input_embeddings[self.vis_token_id] = input_embeddings[img_id]
            input_embeddings[self.end_vis_token_id] = input_embeddings[end_id]
            output_embeddings = self.decoder.get_output_embeddings().weight
            output_embeddings[self.vis_token_id] = output_embeddings[img_id]
            output_embeddings[self.end_vis_token_id] = output_embeddings[end_id]
            
        self.decoder.print_trainable_parameters()
            
        # --- Projector ---
        if hasattr(self.decoder.config, "hidden_size"):
            self.llm_hidden_size = self.decoder.config.hidden_size
        elif hasattr(self.decoder.config, "text_config") and hasattr(self.decoder.config.text_config, "hidden_size"):
            self.llm_hidden_size = self.decoder.config.text_config.hidden_size
        else:
            self.llm_hidden_size = getattr(self.decoder.config, "d_model", 768)
        self.visual_projection = nn.Linear(vit_hidden_size, self.llm_hidden_size)
        self.projector_layernorm = nn.LayerNorm(self.llm_hidden_size)
        nn.init.normal_(self.visual_projection.weight, std=0.01)
        nn.init.zeros_(self.visual_projection.bias)
        
        self.visual_pos_embed = nn.Parameter(torch.randn(1, self.total_visual_tokens, self.llm_hidden_size) * 0.02)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_encoder.to(device)
        self.stem.to(device)
        self.visual_projection.to(device)
        self.projector_layernorm.to(device)
        
        for param in self.visual_projection.parameters():
            param.requires_grad = True
        for param in self.projector_layernorm.parameters():
            param.requires_grad = True
        self.visual_pos_embed.requires_grad = True

        self.is_parallelizable = True
        self.model_parallel = True

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.stem.state_dict(), os.path.join(output_dir, "stem.bin"))
        torch.save(self.vision_encoder.state_dict(), os.path.join(output_dir, "vision_encoder.bin"))
        torch.save(self.visual_projection.state_dict(), os.path.join(output_dir, "projector.bin"))
        torch.save(self.projector_layernorm.state_dict(), os.path.join(output_dir, "projector_layernorm.bin"))
        torch.save(self.visual_pos_embed, os.path.join(output_dir, "visual_pos_embed.bin"))
        self.tokenizer.save_pretrained(output_dir)
        self.decoder.save_pretrained(output_dir)

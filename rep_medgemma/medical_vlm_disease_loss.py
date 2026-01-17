"""
Medical VLM: Frozen LLM + Trainable Vision Encoder/Projector
Architecture: [Visual_Token] + [Instruction_Prompt] -> [Report_Generation]
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

# Fix local import path for lavis
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class Attentive_ROI_Wrapper(nn.Module):
    """
    ViT -> Masked Cross-Attention -> Organ Specific Embeddings
    """
    def __init__(self, vit_model, config, num_organs=12):
        super().__init__()
        self.vit = vit_model
        self.hidden_size = config.hidden_size
        
        # Learnable Queries (The "Interviewer" for each organ)
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
        # 1. Run Vision Encoder
        pixel_values = pixel_values.to(dtype=torch.float32) # Force FP32 for ViT
        with torch.cuda.amp.autocast(enabled=False):
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

            B, N_organs, D_m, H_m, W_m = organ_masks.shape
            
            # --- Downsample Mask for Attention ---
            # ViT reduces 112x256x352 -> approx 7x16x22 depending on patch size
            # Ensure these dimensions match your specific ViT output
            f_d, f_h, f_w = 7, 16, 11 
            
            flat_masks = organ_masks.view(B * N_organs, 1, D_m, H_m, W_m)
            masks_down = F.interpolate(flat_masks, size=(f_d, f_h, f_w), mode='area')
            masks_flat = masks_down.view(B, N_organs, -1)
            
            # --- Create Attention Bias ---
            attn_bias = torch.zeros_like(masks_flat)
            # -inf blocks attention to background
            attn_bias[masks_flat < 0.1] = float('-inf') 
            
            # Safeguard: If mask is empty, attend to everything (0.0)
            is_all_inf = (attn_bias == float('-inf')).all(dim=-1, keepdim=True)
            attn_bias = attn_bias.masked_fill(is_all_inf, 0.0)
            
            # Expand for MultiHead
            num_heads = self.cross_attn.num_heads
            attn_bias = attn_bias.repeat_interleave(num_heads, dim=0)
            
            # --- Cross Attention ---
            queries = self.organ_queries.unsqueeze(0).expand(B, -1, -1)
            
            organ_embeddings, _ = self.cross_attn(
                query=queries,
                key=image_feats,
                value=image_feats,
                attn_mask=attn_bias
            )
            
            return self.layer_norm(organ_embeddings)

        return self.layer_norm(image_feats)

class MedicalVLM(nn.Module):
    def __init__(
        self,
        vision_encoder_path,
        decoder_model_name="google/medgemma-4b-it", # Note: Check HF for exact ID
        image_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        **kwargs 
    ):
        super().__init__()
        
        # --- 1. VISION ENCODER (Trainable) ---
        print("Initializing Trainable Vision Encoder...")
        vit_hidden_size = 768 
        encoder_config = ViTConfig(
            hidden_size=vit_hidden_size, image_size=image_size, patch_size=patch_size
        )

        # Import ViT from LAVIS
        try:
            from lavis.models.blip_models.vit import ViT
            vision_encoder = ViT(in_channels=1, img_size=image_size, patch_size=patch_size, num_classes=0)
        except ImportError:
            raise ImportError("Could not find 'lavis.models.blip_models.vit'. Please check python path.")
        
        if os.path.exists(vision_encoder_path):
            print(f"  Loading ViT weights from {vision_encoder_path}")
            checkpoint = torch.load(vision_encoder_path, map_location='cpu')
            vision_state = {}
            source = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            for k, v in source.items():
                if k.startswith('visual_encoder.'):
                    vision_state[k.replace('visual_encoder.', '')] = v
                elif not k.startswith('text_encoder') and not k.startswith('temp'):
                    vision_state[k] = v
            vision_encoder.load_state_dict(vision_state, strict=False)

        self.vision_encoder = Attentive_ROI_Wrapper(vision_encoder, encoder_config, num_organs=12)

        # Ensure ViT is Trainable
        for param in self.vision_encoder.parameters():
            param.requires_grad = True

        # --- 2. DECODER / LLM (Frozen) ---
        print(f"Loading Frozen Decoder: {decoder_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        self.tokenizer.padding_side = 'right'
        # Gemma requires setting pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load in 4-bit (Quantization)
        print("Loading Decoder in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="eager" # SDPA sometimes issues with 4bit
        )

        # Freeze LLM
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
            
        # --- 3. PROJECTOR (Trainable) ---
        # Projects ViT (768) -> Gemma Hidden Size (e.g., 2048, 3072, etc)
        if hasattr(self.decoder.config, "hidden_size"):
            self.llm_hidden_size = self.decoder.config.hidden_size
        elif hasattr(self.decoder.config, "text_config") and hasattr(self.decoder.config.text_config, "hidden_size"):
            self.llm_hidden_size = self.decoder.config.text_config.hidden_size
        else:
            print(f"WARNING: Could not determine hidden_size from config. Using default 768.")
            self.llm_hidden_size = getattr(self.decoder.config, "d_model", 768)
        self.visual_projection = nn.Linear(vit_hidden_size, self.llm_hidden_size)
        
        # --- 4. Disease Classifier Head (Auxiliary Loss) ---
        from disease_classifier import DiseaseAuxiliaryLoss
        self.disease_loss_fn = DiseaseAuxiliaryLoss(embedding_dim=vit_hidden_size)
        self.disease_loss_fn.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Move Trainable Components to GPU (Required since is_model_parallel=True prevents Trainer from doing it)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_encoder.to(device)
        self.visual_projection.to(device)
        self.disease_loss_fn.to(device)
        
        # Ensure Projector is Trainable
        for param in self.visual_projection.parameters():
            param.requires_grad = True

        print(f"Model Summary:")
        print(f"  Vision Encoder: Trainable (ROI Masked)")
        print(f"  Projector:      Trainable (768 -> {self.llm_hidden_size})")
        print(f"  MedGemma:       FROZEN (4-bit)")
        print(f"  Disease Heads:  Trainable (Auxiliary Loss)")
        
        # Helper to tell Trainer NOT to wrap in DataParallel
        self.is_parallelizable = True
        self.model_parallel = True

    def forward(self, pixel_values, organ_masks=None, input_ids=None, attention_mask=None, labels=None, disease_labels=None, **kwargs):
        """
        Forward pass for Training.
        Data flow: Image -> ViT -> Projector -> [Visual_Embed] + [Text_Embed] -> Decoder -> Loss
        """
        B_batch = pixel_values.shape[0]

        # 1. Get Visual Features (Trainable)
        # Output: (Batch, N_Organs, ViT_Dim)
        visual_feats = self.vision_encoder(pixel_values, organ_masks) 
        
        # --- Auxiliary Disease Loss ---
        aux_loss = 0.0
        if disease_labels is not None:
             aux_loss, loss_details = self.disease_loss_fn(visual_feats, disease_labels)
        
        # 2. Project to LLM Space (Trainable)
        # Output: (Batch, N_Organs, LLM_Dim)
        visual_embeds = self.visual_projection(visual_feats)
        visual_embeds = visual_embeds.to(self.decoder.dtype)
        
        # 3. Reshape for Processing
        # We process (Batch * N_Organs) as independent sequences
        if labels is not None:
            B, N, S = input_ids.shape
            
            # Flatten: (B*N, 1, D)
            visual_embeds = visual_embeds.view(B * N, 1, -1)
            
            # Flatten Text: (B*N, S)
            input_ids = input_ids.view(B * N, S)
            labels = labels.view(B * N, S)
            attention_mask = attention_mask.view(B * N, S)

            # 4. Get Text Embeddings (Frozen LLM)
            # Use model's embedding layer
            # Ensure inputs are on the same device as the embedding weights
            embed_device = self.decoder.get_input_embeddings().weight.device
            input_ids = input_ids.to(embed_device)
            
            with torch.no_grad():
                text_embeds = self.decoder.get_input_embeddings()(input_ids)
            
            # 5. Concatenate: [Visual, Text]
            # Move visual embeds to the same device as text embeds
            visual_embeds = visual_embeds.to(text_embeds.device)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            
            # 6. Adjust Labels and Mask
            # Prepend -100 to labels (ignore loss on visual token)
            ignore_pad = torch.full((B * N, 1), -100, dtype=labels.dtype, device=labels.device)
            concat_labels = torch.cat([ignore_pad, labels], dim=1)
            
            # Prepend 1 to attention mask (attend to visual token)
            att_pad = torch.ones((B * N, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            concat_mask = torch.cat([att_pad, attention_mask], dim=1)

            # 7. Calculate Loss
            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=concat_mask,
                labels=concat_labels,
                return_dict=True,
                use_cache=False # Critical for memory saving during training
            )
            
            # Ensure loss is on the Input Device (Fix for Trainer multi-gpu check)
            # Trainer expects loss to be on the same device as the input batch
            if outputs.loss is not None:
                outputs.loss = outputs.loss.to(pixel_values.device)
                
                # Add Auxiliary Loss
                outputs.loss += aux_loss
            
            return outputs

    def generate(self, pixel_values, organ_masks=None, input_ids=None, attention_mask=None, **kwargs):
        """
        Forward pass for Inference.
        """
        # 1. Visual Path
        visual_feats = self.vision_encoder(pixel_values, organ_masks)
        visual_embeds = self.visual_projection(visual_feats)
        
        B, N, _ = visual_embeds.shape
        visual_embeds = visual_embeds.view(B * N, 1, -1)
        
        # 2. Prepare Prompts
        if input_ids is not None:
            input_ids = input_ids.view(B * N, -1)
            attention_mask = attention_mask.view(B * N, -1)
            
            with torch.no_grad():
                text_embeds = self.decoder.get_input_embeddings()(input_ids)
            
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            
            att_pad = torch.ones((B * N, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            concat_mask = torch.cat([att_pad, attention_mask], dim=1)
        else:
            inputs_embeds = visual_embeds
            concat_mask = torch.ones((B * N, 1), device=visual_embeds.device)

        # 3. Generate
        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=concat_mask,
            **kwargs
        )
        
        return outputs

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # We only save the trainable parts to save space
        torch.save(self.vision_encoder.state_dict(), os.path.join(output_dir, "vision_encoder.bin"))
        torch.save(self.visual_projection.state_dict(), os.path.join(output_dir, "projector.bin"))
        self.tokenizer.save_pretrained(output_dir)
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
            # Project to ViT dimensions is implicit if ViT in_channels = out_channels
        )
    def forward(self, x):
        return self.net(x)

class Attentive_ROI_Wrapper(nn.Module):
    """
    CNN Stem -> ViT -> Masked Cross-Attention -> Organ Specific Embeddings
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
        # 1. Run Vision Encoder (Stem + ViT)
        pixel_values = pixel_values.to(dtype=torch.float32) # Force FP32
        
        # NOTE: pixel_values here should already be processed by Stem if Stem is outside?
        # Or we assume vit_model includes the stem?
        # Current design: MedicalVLM calls vision_encoder(pixel_values).
        # So Attentive_ROI_Wrapper should handle everything.
        # But we want to inject Stem.
        # Let's assume 'vit_model' passed here is just the ViT.
        # We need to add 'self.stem' here or pass 'stem_output' to forward.
        # See MedicalVLM below for integration.
        
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
            
            # --- EXPAND MASKS FOR MULTI-TOKEN (N=8) ---
            # We have N_organs masks, but N_organs * Q queries.
            # We repeat each mask Q times.
            # organ_masks: (B, 12, ...) -> (B, 96, ...)
            
            queries_per_organ = self.organ_queries.shape[0] // N_organs
            organ_masks = organ_masks.repeat_interleave(queries_per_organ, dim=1)
            
            # --- Downsample Mask for Attention ---
            # ViT Grid is now: (112//4)//4, (256//4)//4, (352//4)//8
            # Stem: 4x downsample.
            # ViT Patch: (4,4,8).
            # Total Downsample: 16x, 16x, 32x.
            # Grid: 7, 16, 11 (Matches Original 1232 tokens).
            # Ensure these dimensions match your specific ViT output
            f_d, f_h, f_w = 7, 16, 11 
            
            # Now flatten with the expanded count
            flat_masks = organ_masks.view(B * organ_masks.shape[1], 1, D_m, H_m, W_m)
            
            # Use MaxPool to preserve ANY organ presence in the patch
            # F.interpolate(mode='area') averages, which dilutes small organs.
            masks_down = F.adaptive_max_pool3d(flat_masks, output_size=(f_d, f_h, f_w))
            
            masks_flat = masks_down.view(B, organ_masks.shape[1], -1)
            
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
        decoder_model_name="google/medgemma-4b-it", # Note: Check HF for exact ID
        image_size=(112, 256, 352),
        patch_size=(16, 16, 32), # DEPRECATED: Overridden by logic below
        queries_per_organ=8, # NEW: Default to 8 tokens per organ
        **kwargs 
    ):
        super().__init__()
        
        self.queries_per_organ = queries_per_organ
        self.num_organs = 12
        self.total_visual_tokens = self.num_organs * self.queries_per_organ

        # --- 1. VISION ENCODER (CNN Stem + ViT) ---
        print("Initializing Logic: Hybrid CNN Stem + ViT")
        
        # STEM Configuration
        # Downsample 4x (Stride 2 * 2)
        # Input: (112, 256, 352) -> (28, 64, 88)
        self.stem = CNNStem(out_channels=64)
        
        # ViT Configuration
        # Input to ViT is output of Stem: (28, 64, 88), 64 channels
        vit_img_size = (28, 64, 88)
        vit_patch_size = (4, 4, 8) # Reducing to (4,4,8) to match original token count (1232)
        vit_in_chans = 64
        vit_hidden_size = 768
        
        print(f"  Stem Output / ViT Input: {vit_img_size}, Channels: {vit_in_chans}")
        print(f"  ViT Patch Size: {vit_patch_size}")
        
        encoder_config = ViTConfig(
            hidden_size=vit_hidden_size, image_size=vit_img_size, patch_size=vit_patch_size
        )

        # Import ViT from LAVIS
        try:
            from lavis.models.blip_models.vit import ViT
            # Note: We use the modified ViT class that accepts stride if needed.
            # Currently using patch_size as stride (default).
            vision_encoder = ViT(
                in_channels=vit_in_chans, 
                img_size=vit_img_size, 
                patch_size=vit_patch_size, 
                num_classes=0
            )
        except ImportError:
            raise ImportError("Could not find 'lavis.models.blip_models.vit'. Please check python path.")
        
        # Note: We are creating a NEW architecture, so we cannot load 'vision_encoder_path' directly
        # onto this new structure. We will start the ViT from scratch (or partial load if possible).
        # Given the drastic change (Stem + New Patch Size), it's better to initialize ViT from scratch
        # or transfer weights carefully if they were pretrained on standard ImageNet (but 3D is different).
        # We will assume training from scratch or loading what we can.
        
        print("  WARNING: Architecture changed. Initializing Vision Encoder from scratch.")
        # If you really want to load weights, implement partial loading here.
        
        self.vision_encoder = Attentive_ROI_Wrapper(vision_encoder, encoder_config, num_organs=12)
        
        # RESIZE QUERIES
        # Override the default 12 queries with 12 * Q queries
        self.vision_encoder.organ_queries = nn.Parameter(
            torch.randn(self.total_visual_tokens, encoder_config.hidden_size)
        )
        nn.init.normal_(self.vision_encoder.organ_queries, std=0.02)

        # Ensure ViT is Trainable
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
        for param in self.stem.parameters():
            param.requires_grad = True

        # --- 2. DECODER / LLM (Frozen) ---
        print(f"Loading Frozen Decoder: {decoder_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        self.tokenizer.padding_side = 'right'
        # Gemma requires setting pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load in 4-bit (Quantization)
        # print("Loading Decoder in 4-bit...")
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )
        
        print("Loading Decoder in BF16 (No Quantization)...")

        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_model_name,
            # quantization_config=bnb_config,
            torch_dtype=torch.bfloat16, # Use BF16 directly
            # device_map="auto", # REMOVED: Conflicts with Trainer DataParallel
            attn_implementation="eager" # SDPA sometimes issues with 4bit
        )
        
        # --- 2b. SENTINEL TOKENS ---
        # Add <vis> and <end_vis> to tokenizer
        special_tokens_dict = {'additional_special_tokens': ['<vis>', '<end_vis>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.vis_token_id = self.tokenizer.convert_tokens_to_ids('<vis>')
        self.end_vis_token_id = self.tokenizer.convert_tokens_to_ids('<end_vis>')
        
        # Resize LLM embeddings to accommodate new tokens
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        # Freeze LLM Base
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

        # --- APPLY LORA (Trainable Adapters) ---
        print("Applying LoRA to Decoder...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.05,
            # Target ALL linear layers for maximum plasticity (helps break the "normalcy bias")
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.decoder = get_peft_model(self.decoder, peft_config)
        
        # --- SMART INITIALIZATION OF SENTINEL TOKENS ---
        # Since we are NOT training valid embeddings, we must initialize them to something meaningful
        # instead of random noise.
        with torch.no_grad():
            # Get IDs for "image" and "end" to use as proxies
            proxy_ids = self.tokenizer(["image", "end"], add_special_tokens=False).input_ids
            # Ensure we have valid IDs
            img_id = proxy_ids[0][0] if proxy_ids and len(proxy_ids[0]) > 0 else 2
            end_id = proxy_ids[1][0] if len(proxy_ids) > 1 and len(proxy_ids[1]) > 0 else 2
            
            # Copy weights: <vis> <= "image", <end_vis> <= "end"
            input_embeddings = self.decoder.get_input_embeddings().weight
            input_embeddings[self.vis_token_id] = input_embeddings[img_id]
            input_embeddings[self.end_vis_token_id] = input_embeddings[end_id]
            
            # Also sync output embeddings (lm_head) if separate
            output_embeddings = self.decoder.get_output_embeddings().weight
            output_embeddings[self.vis_token_id] = output_embeddings[img_id]
            output_embeddings[self.end_vis_token_id] = output_embeddings[end_id]
            
        self.decoder.print_trainable_parameters()
            
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
        
        # --- NEW: Added LayerNorm for stability ---
        self.projector_layernorm = nn.LayerNorm(self.llm_hidden_size)
        
        # --- NEW: Better Weight Initialization ---
        # Initialize projector to small values so it starts as a "neutral" input
        nn.init.normal_(self.visual_projection.weight, std=0.01)
        nn.init.zeros_(self.visual_projection.bias)
        
        # --- NEW: Learned Visual Position Embeddings ---
        # (1, 96, D) to allow broadcasting across batch
        self.visual_pos_embed = nn.Parameter(torch.randn(1, self.total_visual_tokens, self.llm_hidden_size) * 0.02)
        
        # Move Trainable Components to GPU (Required since is_model_parallel=True prevents Trainer from doing it)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.vision_encoder.to(device)
        # self.visual_projection.to(device)
        # self.projector_layernorm.to(device)
        
        # Ensure Projector is Trainable
        for param in self.visual_projection.parameters():
            param.requires_grad = True
        for param in self.projector_layernorm.parameters():
            param.requires_grad = True
        self.visual_pos_embed.requires_grad = True

        print(f"Model Summary:")
        print(f"  Vision Encoder: Trainable (ROI Masked, {self.queries_per_organ} tokens/organ)")
        print(f"  Projector:      Trainable (768 -> {self.llm_hidden_size})")
        print(f"  MedGemma:       FROZEN (4-bit)")
        
        # Helper to tell Trainer NOT to wrap in DataParallel
        # self.is_parallelizable = True
        # self.model_parallel = True

    def forward(self, pixel_values, organ_masks=None, input_ids=None, attention_mask=None, labels=None, sample_weights=None, **kwargs):
        """
        Forward pass for Training.
        Data flow: Image -> ViT -> Projector -> [Visual_Embed] + [Text_Embed] -> Decoder -> Loss
        """
        B_batch = pixel_values.shape[0]

        # 1. Get Visual Features (Trainable)
        pixel_values = pixel_values.to(dtype=torch.float32)
        
        # Run CNN Stem
        stem_feats = self.stem(pixel_values)
        
        # Output: (Batch, N_Organs * Q, ViT_Dim)
        # We pass stem_feats to Attentive_ROI_Wrapper
        visual_feats, attn_weights = self.vision_encoder(stem_feats, organ_masks) 
        
        # 2. Project to LLM Space (Trainable)
        # Output: (Batch, N_Organs * Q, LLM_Dim)
        visual_embeds = self.visual_projection(visual_feats)
        
        # Add Learned Position Embeddings (Broadcasts to Batch)
        # visual_embeds: (B, 96, D)
        visual_embeds = visual_embeds + self.visual_pos_embed.to(visual_embeds.device)
        
        # 2. STABILIZATION: Apply LayerNorm and Scale Matching
        # This keeps visual embeddings in the same numerical range as MedGemma's vocabulary
        visual_embeds = self.projector_layernorm(visual_embeds)
        visual_embeds = visual_embeds.to(self.decoder.dtype)
        
        # Match variance to text embeddings to prevent "Embedding Shock"
        with torch.no_grad():
            text_std = self.decoder.get_input_embeddings().weight.std()
        visual_embeds = visual_embeds * (text_std / (visual_embeds.std() + 1e-6))
        
        # 3. Reshape for Processing
        # We process (Batch * N_Organs * Q) as independent sequences
        # BUT wait, the dataset repeats text 12 times. We have 96 visual tokens.
        # Logic: 
        #   Case A: We process 12 samples per image. Each sample has 8 independent visual tokens? NO.
        #   Case B: We process 96 samples per image? NO.
        
        #   CORRECT LOGIC:
        #   The dataset treats each ORGAN as a sample -> returns 12 items.
        #   For 'LUNG', we want to give it 8 visual tokens.
        #   So visual_embeds for LUNG should be (1, 8, D).
        #   Currently visual_embeds is (B, 96, D).
        #   We need to reshape visual_embeds to (B, 12, 8, D).
        #   Then for each of the 12 organ samples, we pass (1, 8, D) as the visual context.
        
        # Reshape: (B, 12, 8, D)
        visual_embeds = visual_embeds.view(B_batch, self.num_organs, self.queries_per_organ, -1)

        if labels is not None:
            B, N, S = input_ids.shape # N=12
            
            # Flatten Inputs: (B*12, S)
            input_ids = input_ids.view(B * N, S)
            labels = labels.view(B * N, S)
            attention_mask = attention_mask.view(B * N, S)

            # Flatten Visuals: (B*12, 8, D)
            # This aligns the 8 tokens with the correct organ text
            visual_embeds = visual_embeds.view(B * N, self.queries_per_organ, -1)
            
            # Flatten Weights: (B*N)
            if sample_weights is not None:
                sample_weights = sample_weights.view(B * N)

            # 4. Get Text Embeddings (Frozen LLM)
            embed_device = self.decoder.get_input_embeddings().weight.device
            input_ids = input_ids.to(embed_device)
            
            with torch.no_grad():
                text_embeds = self.decoder.get_input_embeddings()(input_ids)
                
            # 5. VISUAL SENTINEL TOKENS
            # We insert <vis> (id) before and <end_vis> (id) after the visual token
            # Sequence: [BOS] + <vis> + [Visual_Token] + ... + [Visual_Token] + <end_vis> + [Instruction...]
            
            # Create embedding for <vis> and <end_vis>
            # Shape: (1, 1, D) -> (B*N, 1, D)
            embed_device = self.decoder.get_input_embeddings().weight.device
            
            vis_embed = self.decoder.get_input_embeddings()(
                torch.tensor([[self.vis_token_id]], device=embed_device)
            ).expand(B * N, -1, -1)
            
            end_vis_embed = self.decoder.get_input_embeddings()(
                torch.tensor([[self.end_vis_token_id]], device=embed_device)
            ).expand(B * N, -1, -1)
            
            inputs_embeds = torch.cat([
                text_embeds[:, :1, :],   # [BOS]
                vis_embed,               # <vis>
                visual_embeds,           # [Visual] (8 tokens)
                end_vis_embed,           # <end_vis>
                text_embeds[:, 1:, :]    # [Instruction...]
            ], dim=1)
            
            # Adjust Labels: ignore loss on <vis>, <end_vis>, and Visual Tokens (-100)
            # We added 2 + 8 = 10 tokens usually? 
            # Wait, previously we added 1 visual + 2 sentinels = 3.
            # Now we add 8 visual + 2 sentinels = 10.
            
            num_new_tokens = self.queries_per_organ + 2
            ignore_vis_block = torch.full((B * N, num_new_tokens), -100, dtype=labels.dtype, device=labels.device)
            concat_labels = torch.cat([labels[:, :1], ignore_vis_block, labels[:, 1:]], dim=1)
            
            # Adjust Mask
            att_vis_block = torch.ones((B * N, num_new_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
            concat_mask = torch.cat([attention_mask[:, :1], att_vis_block, attention_mask[:, 1:]], dim=1)
            
            # 7. Calculate LM Loss
            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=concat_mask,
                labels=None, # Don't calculating loss inside model
                return_dict=True,
                use_cache=False 
            )

            # Logits: (B*N, SeqLen, Vocab)
            logits = outputs.logits
            
            # Shift Logits and Labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = concat_labels[..., 1:].contiguous()
            
            # Flatten to (Batch * Seq, Vocab)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            
            # Calculate Cross Entropy (Reduction=None to keep per-token loss)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            token_losses = loss_fct(flat_logits, flat_labels)
            
            # Reshape back to (B*N, SeqLen-1)
            token_losses = token_losses.view(B * N, -1)
            
            # Mean loss per sample (ignoring masked tokens)
            non_pad_mask = (shift_labels != -100).float()
            sample_loss_sum = (token_losses * non_pad_mask).sum(dim=1)
            sample_tokens = non_pad_mask.sum(dim=1)
            
            # Avoid division by zero
            sample_loss = sample_loss_sum / (sample_tokens + 1e-8)
            
            # Apply Organ Weights
            if sample_weights is not None:
                sample_weights = sample_weights.to(sample_loss.device)
                sample_loss = sample_loss * sample_weights
                
            # Final Mean Loss
            lm_loss = sample_loss.mean()
            
            # --- 8. ALIGNMENT LOSS (ITC) ---
            # We enforce that the Visual Representation of an organ (Avg of 8 tokens)
            # aligns with the Text Representation of that organ (Avg of text tokens).
            # This is done via InfoNCE loss over the 12 organs within the patient.
            
            # visual_embeds is currently (B*12, 8, D)
            # input_ids is (B*12, S)
            
            alignment_loss = self.compute_alignment_loss(
                visual_embeds, # (B*N, 8, D)
                input_ids,     # (B*N, S)
                attention_mask, # (B*N, S)
                labels=labels   # Pass labels for correct masking
            )
            
            # Combine Losses
            # We weight alignment loss higher (5.0) to force the model to respect the visual signal
            total_loss = lm_loss + 5.0 * alignment_loss
            
            # Use dict return for Trainer compatibility
            return {
                "loss": total_loss,
                "logits": logits,
                "lm_loss": lm_loss,
                "alignment_loss": alignment_loss,
                "attention_weights": attn_weights
            }

    def compute_alignment_loss(self, visual_embeds, input_ids, attention_mask, labels=None, temperature=0.07):
        """
        Computes Image-Text Contrastive (ITC) Loss.
        Goal: Match the visual representation of an organ to its text description.
        If labels is provided, we align ONLY with the 'Response' tokens (where labels != -100).
        """
        # 1. Get Visual Representation: Avg Pool the 8 tokens -> (B*12, D)
        vis_rep = visual_embeds.mean(dim=1) 
        
        # 2. Get Text Representation: Avg Pool the text tokens -> (B*12, D)
        with torch.no_grad():
             text_embeds = self.decoder.get_input_embeddings()(input_ids) # (B*12, S, D)
        
        # Mask out padding/prompt for averaging
        if labels is not None:
             # Use labels to identify the "Response" part (labels != -100)
             # This aligns the visual embedding specifically with the FINDINGS, ignoring the User Prompt.
             text_mask = (labels != -100).float().unsqueeze(-1)
        else:
             # Fallback to attention mask (includes User Prompt)
             text_mask = attention_mask.unsqueeze(-1).float()

        text_sum = (text_embeds * text_mask).sum(dim=1)
        text_count = text_mask.sum(dim=1).clamp(min=1e-8)
        text_rep = text_sum / text_count # (B*12, D)
        
        # 3. Project to Shared Space (Optional? No, we assume same space)
        # Normalize
        vis_rep = F.normalize(vis_rep, dim=-1)
        text_rep = F.normalize(text_rep, dim=-1)
        
        # 4. InfoNCE Loss
        # We have B*12 samples. 
        # Actually, because we reshaped inputs to (B*12), we lost the Batch dimension structure.
        # We need to reshape back to (B, 12, D) to do contrastive within each patient.
        
        B_N, D = vis_rep.shape
        num_organs = self.num_organs # 12
        B = B_N // num_organs
        
        vis_rep = vis_rep.view(B, num_organs, D)
        text_rep = text_rep.view(B, num_organs, D)
        
        total_loss = 0
        loss_fct = nn.CrossEntropyLoss()
        
        for i in range(B):
            # For patient i
            v = vis_rep[i] # (12, D)
            t = text_rep[i] # (12, D)
            
            # Similarity Matrix: (12, 12)
            logits = torch.matmul(v, t.T) / temperature
            
            # Targets: 0..11
            labels = torch.arange(num_organs, device=logits.device)
            
            # Loss (Symmetric? Usually text->image and image->text)
            # Let's do Visual->Text (standard)
            loss_i = loss_fct(logits, labels)
            total_loss += loss_i
            
        return total_loss / B

    def generate(self, pixel_values, organ_masks=None, input_ids=None, attention_mask=None, **kwargs):
        """
        Forward pass for Inference.
        """
        # 1. Visual Path
        pixel_values = pixel_values.to(dtype=torch.float32)
        stem_feats = self.stem(pixel_values)
        visual_feats, attn_weights = self.vision_encoder(stem_feats, organ_masks)
        visual_embeds = self.visual_projection(visual_feats)
        
        # Add Learned Position Embeddings
        visual_embeds = visual_embeds + self.visual_pos_embed.to(visual_embeds.device)
        
        # STABILIZATION
        visual_embeds = self.projector_layernorm(visual_embeds)
        visual_embeds = visual_embeds.to(self.decoder.dtype)
        with torch.no_grad():
            text_std = self.decoder.get_input_embeddings().weight.std()
        visual_embeds = visual_embeds * (text_std / (visual_embeds.std() + 1e-6))
        
        B, N, _ = visual_embeds.shape
        visual_embeds = visual_embeds.view(B * N, 1, -1)
        
        # 2. Prepare Prompts
        if input_ids is not None:
            B, N, S = input_ids.shape
            
            # Reshape Visuals (B, 12, 8, D) -> (B*12, 8, D)
            visual_embeds = visual_embeds.view(B * N, self.queries_per_organ, -1)
            
            input_ids = input_ids.view(B * N, -1)
            attention_mask = attention_mask.view(B * N, -1)
            
            with torch.no_grad():
                text_embeds = self.decoder.get_input_embeddings()(input_ids)
                text_embeds = text_embeds.to(self.decoder.dtype)
            
            # BOS-Aware Insertion with Sentinels: [BOS] + <vis> + [Visual_Tokens] + <end_vis> + [Rest]
            
            embed_device = self.decoder.get_input_embeddings().weight.device
            
            vis_embed = self.decoder.get_input_embeddings()(
                torch.tensor([[self.vis_token_id]], device=embed_device)
            ).expand(B * N, -1, -1)
            
            end_vis_embed = self.decoder.get_input_embeddings()(
                torch.tensor([[self.end_vis_token_id]], device=embed_device)
            ).expand(B * N, -1, -1)
            
            inputs_embeds = torch.cat([
                text_embeds[:, :1, :],
                vis_embed,
                visual_embeds, # 8 tokens
                end_vis_embed,
                text_embeds[:, 1:, :]
            ], dim=1)
            
            # Mask: + (queries_per_organ + 2) tokens
            num_new = self.queries_per_organ + 2
            att_vis_block = torch.ones((B * N, num_new), dtype=attention_mask.dtype, device=attention_mask.device)
            concat_mask = torch.cat([attention_mask[:, :1], att_vis_block, attention_mask[:, 1:]], dim=1)
        else:
            inputs_embeds = visual_embeds
            concat_mask = torch.ones((visual_embeds.shape[0], visual_embeds.shape[1]), device=visual_embeds.device)

        # 3. Generate
        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=concat_mask,
            **kwargs
        )
        
        return outputs
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            """
            Activates gradient checkpointing for the current model.
            Wrapper to enable it on the decoder (LLM) which is the heaviest part.
            """
            self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # We only save the trainable parts to save space
        torch.save(self.stem.state_dict(), os.path.join(output_dir, "stem.bin")) # NEW
        torch.save(self.vision_encoder.state_dict(), os.path.join(output_dir, "vision_encoder.bin"))
        torch.save(self.visual_projection.state_dict(), os.path.join(output_dir, "projector.bin"))
        torch.save(self.projector_layernorm.state_dict(), os.path.join(output_dir, "projector_layernorm.bin"))
        torch.save(self.visual_pos_embed, os.path.join(output_dir, "visual_pos_embed.bin"))
        self.tokenizer.save_pretrained(output_dir)
        # Save LoRA Adapters
        self.decoder.save_pretrained(output_dir)
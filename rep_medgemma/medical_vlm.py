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
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # Key Mapping for Contrastive Encoder (which has 'visual_encoder.' prefix)
            # We need to map `visual_encoder.blocks...` -> `blocks...`
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("visual_encoder."):
                    # Strip prefix
                    new_k = k.replace("visual_encoder.", "")
                    new_state_dict[new_k] = v
                elif k.startswith("module."): # Common DDP prefix
                    new_k = k.replace("module.", "")
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            
            # Load weights (strict=False because the checkpoint might contain extra heads like text_encoder, queues etc)
            msg = vision_encoder.load_state_dict(new_state_dict, strict=False)
            print(f"Vision Encoder Loaded with msg: {msg}")
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
        # (1, 12, D) to allow broadcasting across batch
        self.visual_pos_embed = nn.Parameter(torch.randn(1, 12, self.llm_hidden_size) * 0.02)
        
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
        print(f"  Vision Encoder: Trainable (ROI Masked)")
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
        # Output: (Batch, N_Organs, ViT_Dim)
        visual_feats = self.vision_encoder(pixel_values, organ_masks) 
        
        # 2. Project to LLM Space (Trainable)
        # Output: (Batch, N_Organs, LLM_Dim)
        visual_embeds = self.visual_projection(visual_feats)
        
        # Add Learned Position Embeddings (Broadcasts to Batch)
        # visual_embeds: (B, 12, D)
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
        # We process (Batch * N_Organs) as independent sequences
        if labels is not None:
            B, N, S = input_ids.shape
            
            # Flatten: (B*N, 1, D)
            visual_embeds = visual_embeds.view(B * N, 1, -1)
            
            # Flatten Text: (B*N, S)
            input_ids = input_ids.view(B * N, S)
            labels = labels.view(B * N, S)
            attention_mask = attention_mask.view(B * N, S)

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
            # Sequence: [BOS] + <vis> + [Visual_Token] + <end_vis> + [Instruction...]
            
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
                visual_embeds,           # [Visual]
                end_vis_embed,           # <end_vis>
                text_embeds[:, 1:, :]    # [Instruction...]
            ], dim=1)
            
            # Adjust Labels: ignore loss on <vis>, <end_vis>, and Visual Token (-100)
            # We added 2 new tokens (<vis>, <end_vis>), so we need 2 more -100s
            ignore_vis_block = torch.full((B * N, 3), -100, dtype=labels.dtype, device=labels.device)
            concat_labels = torch.cat([labels[:, :1], ignore_vis_block, labels[:, 1:]], dim=1)
            
            # Adjust Mask: add '1' for the visual block (<vis>, Visual, <end_vis>)
            att_vis_block = torch.ones((B * N, 3), dtype=attention_mask.dtype, device=attention_mask.device)
            concat_mask = torch.cat([attention_mask[:, :1], att_vis_block, attention_mask[:, 1:]], dim=1)
            
            # 7. Calculate Loss Manually for Weighting
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
            # labels are already shifted inside CausalLM value usually, BUT since we passed Labels=None, 
            # we must do shift manually.
            # Shift: logits[..., :-1, :], labels[..., 1:]
            
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
            # Find non-mask tokens count
            # Use shift_labels != -100
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
            loss = sample_loss.mean()
            
            # Use dict return for Trainer compatibility
            return {
                "loss": loss,
                "logits": logits
            }

    def generate(self, pixel_values, organ_masks=None, input_ids=None, attention_mask=None, **kwargs):
        """
        Forward pass for Inference.
        """
        # 1. Visual Path
        visual_feats = self.vision_encoder(pixel_values, organ_masks)
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
            input_ids = input_ids.view(B * N, -1)
            attention_mask = attention_mask.view(B * N, -1)
            
            with torch.no_grad():
                text_embeds = self.decoder.get_input_embeddings()(input_ids)
                text_embeds = text_embeds.to(self.decoder.dtype)
            
            # BOS-Aware Insertion with Sentinels: [BOS] + <vis> + [Visual] + <end_vis> + [Rest]
            
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
                visual_embeds,
                end_vis_embed,
                text_embeds[:, 1:, :]
            ], dim=1)
            
            # Mask: +3 tokens (<vis>, Vis, <end_vis>)
            att_vis_block = torch.ones((B * N, 3), dtype=attention_mask.dtype, device=attention_mask.device)
            concat_mask = torch.cat([attention_mask[:, :1], att_vis_block, attention_mask[:, 1:]], dim=1)
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
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            """
            Activates gradient checkpointing for the current model.
            Wrapper to enable it on the decoder (LLM) which is the heaviest part.
            """
            self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # We only save the trainable parts to save space
        torch.save(self.vision_encoder.state_dict(), os.path.join(output_dir, "vision_encoder.bin"))
        torch.save(self.visual_projection.state_dict(), os.path.join(output_dir, "projector.bin"))
        torch.save(self.projector_layernorm.state_dict(), os.path.join(output_dir, "projector_layernorm.bin"))
        torch.save(self.visual_pos_embed, os.path.join(output_dir, "visual_pos_embed.bin"))
        self.tokenizer.save_pretrained(output_dir)
        # Save LoRA Adapters
        self.decoder.save_pretrained(output_dir)
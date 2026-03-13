"""
Medical VLM using MedGemma full multimodal stack (vision encoder + text decoder).

The existing dataset/preprocessing pipeline is preserved:
- input CT volumes stay 3D in the dataset
- per-organ binary masks stay the same
- report/chat tokenization stays in train.py

This wrapper converts each organ mask into one organ-focused 2D slice and feeds it to
MedGemma's native vision encoder through Gemma3ForConditionalGeneration.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType


class MedicalVLM(nn.Module):
    def __init__(
        self,
        vision_encoder_path=None,
        decoder_model_name="google/medgemma-4b-it",
        num_organs=12,
        organ_chunk_size=1,
        apply_lora=True,
        use_4bit=False,
        local_files_only=True,
        device_map=None,
        **kwargs,
    ):
        super().__init__()
        del vision_encoder_path  # Kept for backward CLI compatibility.

        self.num_organs = num_organs
        self.organ_chunk_size = max(1, int(organ_chunk_size))
        self.local_files_only = local_files_only

        self.tokenizer = AutoTokenizer.from_pretrained(
            decoder_model_name,
            local_files_only=local_files_only,
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "dtype": torch.bfloat16,
            "attn_implementation": "eager",
            "local_files_only": local_files_only,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.decoder = Gemma3ForConditionalGeneration.from_pretrained(
            decoder_model_name,
            **model_kwargs,
        )

        # Freeze the base model, then train LoRA adapters only.
        for param in self.decoder.parameters():
            param.requires_grad = False

        if apply_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            self.decoder = get_peft_model(self.decoder, peft_config)
            # Required when using gradient checkpointing with a mostly frozen model.
            # Without this, checkpointed blocks may get inputs with requires_grad=False,
            # resulting in a detached loss ("does not require grad").
            if hasattr(self.decoder, "enable_input_require_grads"):
                self.decoder.enable_input_require_grads()
            self.decoder.print_trainable_parameters()

        cfg = self.decoder.config
        self.image_token_id = cfg.image_token_id
        self.boi_token_id = cfg.boi_token_index
        self.eoi_token_id = cfg.eoi_token_index
        self.mm_tokens_per_image = cfg.mm_tokens_per_image
        self.image_size = cfg.vision_config.image_size

    def _decoder_device(self):
        return next(self.decoder.parameters()).device

    def _build_organ_images(self, pixel_values, organ_masks):
        """
        Convert 3D CT + per-organ masks into MedGemma 2D RGB image inputs.
        Strategy: pick the depth slice with maximum organ area for each organ.
        """
        if organ_masks is None:
            raise ValueError("organ_masks is required for organ-specific prompts.")

        # pixel_values: (B, 1, D, H, W)
        if pixel_values.dim() != 5:
            raise ValueError(f"Expected pixel_values with 5 dims (B,1,D,H,W), got {tuple(pixel_values.shape)}")

        # organ_masks: (B, N, D, H, W) or (B, N, 1, D, H, W)
        if organ_masks.dim() == 6:
            organ_masks = organ_masks.squeeze(2)
        if organ_masks.dim() != 5:
            raise ValueError(f"Expected organ_masks with 5 dims (B,N,D,H,W), got {tuple(organ_masks.shape)}")

        bsz, n_organs, depth, height, width = organ_masks.shape
        volume = pixel_values[:, 0]  # (B, D, H, W)

        organ_masks = organ_masks.float()
        area_per_slice = organ_masks.sum(dim=(-1, -2))  # (B, N, D)
        best_slice_idx = area_per_slice.argmax(dim=-1)  # (B, N)
        has_pixels = area_per_slice.max(dim=-1).values > 0
        center_idx = torch.full_like(best_slice_idx, depth // 2)
        best_slice_idx = torch.where(has_pixels, best_slice_idx, center_idx)

        volume_expanded = volume.unsqueeze(1).expand(-1, n_organs, -1, -1, -1)  # (B, N, D, H, W)
        gather_idx = best_slice_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, height, width)
        organ_slices = torch.gather(volume_expanded, dim=2, index=gather_idx).squeeze(2)  # (B, N, H, W)

        organ_slices = organ_slices.reshape(bsz * n_organs, 1, height, width)
        organ_images = F.interpolate(
            organ_slices,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        organ_images = organ_images.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        # Match Gemma3 image processor normalization: mean=0.5, std=0.5
        organ_images = (organ_images - 0.5) / 0.5
        return organ_images

    def _flatten_organs(self, input_ids, attention_mask, labels, batch_size, num_organs):
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask are required.")

        if input_ids.dim() == 3:
            b, n, s = input_ids.shape
            if b != batch_size:
                raise ValueError(f"Batch mismatch: input_ids B={b}, pixel_values B={batch_size}")
            flat_input_ids = input_ids.reshape(b * n, s)
            flat_attention_mask = attention_mask.reshape(b * n, s)
            flat_labels = labels.reshape(b * n, s) if labels is not None else None
            return flat_input_ids, flat_attention_mask, flat_labels

        if input_ids.dim() == 2:
            if input_ids.shape[0] != batch_size * num_organs:
                raise ValueError(
                    f"Expected input_ids[0]={batch_size * num_organs} for flattened organ prompts, "
                    f"got {input_ids.shape[0]}"
                )
            return input_ids, attention_mask, labels

        raise ValueError(f"Unsupported input_ids shape: {tuple(input_ids.shape)}")

    def _prepend_image_tokens(self, input_ids, attention_mask, labels=None):
        bsz = input_ids.shape[0]
        device = input_ids.device
        dtype = input_ids.dtype

        boi = torch.full((bsz, 1), self.boi_token_id, dtype=dtype, device=device)
        image_soft_tokens = torch.full(
            (bsz, self.mm_tokens_per_image),
            self.image_token_id,
            dtype=dtype,
            device=device,
        )
        eoi = torch.full((bsz, 1), self.eoi_token_id, dtype=dtype, device=device)

        # Keep BOS at index 0 and insert image block after BOS.
        input_ids = torch.cat(
            [input_ids[:, :1], boi, image_soft_tokens, eoi, input_ids[:, 1:]],
            dim=1,
        )

        image_block_len = self.mm_tokens_per_image + 2
        attn_block = torch.ones((bsz, image_block_len), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat(
            [attention_mask[:, :1], attn_block, attention_mask[:, 1:]],
            dim=1,
        )

        if labels is not None:
            ignore_block = torch.full((bsz, image_block_len), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([labels[:, :1], ignore_block, labels[:, 1:]], dim=1)

        return input_ids, attention_mask, labels

    def forward(
        self,
        pixel_values,
        organ_masks=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
        sample_weights=None,
        **kwargs,
    ):
        del sample_weights, kwargs  # Not used in this full MedGemma pipeline.

        batch_size = pixel_values.shape[0]
        if organ_masks is None:
            raise ValueError("organ_masks is required.")
        if organ_masks.dim() == 6:
            num_organs = organ_masks.shape[1]
        else:
            num_organs = organ_masks.shape[1]

        organ_images = self._build_organ_images(pixel_values, organ_masks)
        flat_input_ids, flat_attention_mask, flat_labels = self._flatten_organs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            batch_size=batch_size,
            num_organs=num_organs,
        )
        flat_input_ids, flat_attention_mask, flat_labels = self._prepend_image_tokens(
            flat_input_ids,
            flat_attention_mask,
            flat_labels,
        )

        model_device = self._decoder_device()
        flat_input_ids = flat_input_ids.to(model_device)
        flat_attention_mask = flat_attention_mask.to(model_device)
        if flat_labels is not None:
            flat_labels = flat_labels.to(model_device)
        organ_images = organ_images.to(model_device, dtype=self.decoder.dtype)

        total = flat_input_ids.shape[0]
        chunk = self.organ_chunk_size
        if chunk >= total:
            return self.decoder(
                input_ids=flat_input_ids,
                pixel_values=organ_images,
                attention_mask=flat_attention_mask,
                labels=flat_labels,
                return_dict=True,
                use_cache=False,
            )

        weighted_loss = None
        total_tokens = None
        last_outputs = None
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            c_input_ids = flat_input_ids[start:end]
            c_attention_mask = flat_attention_mask[start:end]
            c_labels = flat_labels[start:end] if flat_labels is not None else None
            c_images = organ_images[start:end]

            c_outputs = self.decoder(
                input_ids=c_input_ids,
                pixel_values=c_images,
                attention_mask=c_attention_mask,
                labels=c_labels,
                return_dict=True,
                use_cache=False,
            )
            last_outputs = c_outputs

            if c_labels is not None:
                # Model loss is mean CE over valid shifted labels, so re-weight by valid token count.
                valid_tokens = (c_labels[:, 1:] != -100).sum().to(c_outputs.loss.device, dtype=c_outputs.loss.dtype)
                if weighted_loss is None:
                    weighted_loss = c_outputs.loss * valid_tokens
                    total_tokens = valid_tokens
                else:
                    weighted_loss = weighted_loss + (c_outputs.loss * valid_tokens)
                    total_tokens = total_tokens + valid_tokens

        if flat_labels is not None:
            loss = weighted_loss / total_tokens.clamp(min=1.0)
            return {"loss": loss}
        return last_outputs

    def generate(self, pixel_values, organ_masks=None, input_ids=None, attention_mask=None, **kwargs):
        if organ_masks is None:
            raise ValueError("organ_masks is required.")
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask are required for generation.")

        batch_size = pixel_values.shape[0]
        num_organs = organ_masks.shape[1] if organ_masks.dim() >= 5 else self.num_organs
        organ_images = self._build_organ_images(pixel_values, organ_masks)
        flat_input_ids, flat_attention_mask, _ = self._flatten_organs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            batch_size=batch_size,
            num_organs=num_organs,
        )
        flat_input_ids, flat_attention_mask, _ = self._prepend_image_tokens(
            flat_input_ids,
            flat_attention_mask,
            labels=None,
        )

        model_device = self._decoder_device()
        flat_input_ids = flat_input_ids.to(model_device)
        flat_attention_mask = flat_attention_mask.to(model_device)
        organ_images = organ_images.to(model_device, dtype=self.decoder.dtype)

        total = flat_input_ids.shape[0]
        chunk = self.organ_chunk_size
        if chunk >= total:
            return self.decoder.generate(
                input_ids=flat_input_ids,
                pixel_values=organ_images,
                attention_mask=flat_attention_mask,
                **kwargs,
            )

        generated = []
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            out = self.decoder.generate(
                input_ids=flat_input_ids[start:end],
                pixel_values=organ_images[start:end],
                attention_mask=flat_attention_mask[start:end],
                **kwargs,
            )
            generated.append(out)
        return torch.cat(generated, dim=0)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)
        self.decoder.save_pretrained(output_dir)

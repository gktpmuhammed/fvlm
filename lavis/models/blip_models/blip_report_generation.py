"""
Simple BLIP model for report generation: Vision Encoder + Text Decoder only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from lavis.models.med import XBertLMHeadDecoder
from transformers import BertTokenizer


@registry.register_model("blip_report_generation")
class BlipReportGeneration(BaseModel):
    """
    Simplified BLIP model for medical report generation.
    Only uses Vision Encoder + Text Decoder (no text encoder for contrastive learning).
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_report_generation.yaml",
    }

    def __init__(
        self,
        image_encoder,
        text_decoder,
        max_txt_len=256,
        use_autoregressive_training=False,
    ):
        super().__init__()

        self.visual_encoder = image_encoder
        self.text_decoder = text_decoder
        self.max_txt_len = max_txt_len
        self.use_autoregressive_training = use_autoregressive_training
        
        # Initialize tokenizer
        self.tokenizer = self.init_tokenizer()
        
        # Resize token embeddings to match tokenizer
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

    def init_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained("BiomedVLP-CXR-BERT-specialized")
        tokenizer.add_special_tokens({"bos_token": "[CLS]"})
        return tokenizer

    def forward(self, samples):
        """
        Forward pass for training: Image → Vision Encoder → Text Decoder
        Can use either teacher forcing or autoregressive training.
        """
        # Add training mode parameter (can be set via config)
        use_autoregressive_training = getattr(self, 'use_autoregressive_training', False)
        
        if use_autoregressive_training:
            return self._forward_autoregressive(samples)
        else:
            return self._forward_teacher_forcing(samples)
    
    def _forward_teacher_forcing(self, samples):
        """Original teacher forcing training"""
        image = samples["image"]
        text_input = samples["text_input"]  # Target text (findings + impressions)
        
        # Encode image
        image_embeds, _ = self.visual_encoder(image)
        
        # Tokenize target text
        text = self.tokenizer(
            text_input,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image_embeds.device)
        
        # Create attention mask for image embeddings
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )
        
        batch_size = text.input_ids.size(0)
        
        # Create decoder input: [CLS] + target_text[:-1]
        decoder_input_ids = torch.full(
            (batch_size, 1), 
            self.tokenizer.cls_token_id, 
            dtype=torch.long, 
            device=image_embeds.device
        )
        decoder_input_ids = torch.cat([decoder_input_ids, text.input_ids[:, :-1]], dim=1)
        
        # Create attention mask for decoder input
        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        decoder_attention_mask[decoder_input_ids == self.tokenizer.pad_token_id] = 0
        
        # Prepare labels: target[1:] (what we want to predict)
        decoder_targets = text.input_ids.clone()
        decoder_targets = decoder_targets.masked_fill(
            decoder_targets == self.tokenizer.pad_token_id, -100
        )
        
        # Forward through decoder with cross-attention to image
        decoder_output = self.text_decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True,
        )
        
        return {"loss": decoder_output.loss}
    
    def _forward_autoregressive(self, samples):
        """Autoregressive training - matches inference behavior"""
        image = samples["image"]
        text_input = samples["text_input"]  # Target text (findings + impressions)
        
        # Encode image
        image_embeds, _ = self.visual_encoder(image)
        
        # Tokenize target text
        text = self.tokenizer(
            text_input,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image_embeds.device)
        
        batch_size = image.size(0)
        device = image.device
        
        # Start with CLS token for each sample
        generated_ids = torch.full(
            (batch_size, 1), 
            self.tokenizer.cls_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        # Create attention mask for image embeddings
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        
        total_loss = 0.0
        num_tokens = 0
        
        # Generate tokens one by one (like in inference)
        max_length = min(self.max_txt_len, text.input_ids.size(1))
        
        for step in range(max_length - 1):  # -1 because we start with CLS
            # Create attention mask for current sequence
            attention_mask = torch.ones_like(generated_ids)
            
            # Forward pass to get next token logits
            decoder_output = self.text_decoder(
                generated_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            # Get logits for next token prediction
            next_token_logits = decoder_output.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Get target tokens for this step
            if step + 1 < text.input_ids.size(1):
                target_tokens = text.input_ids[:, step + 1]  # Next token in ground truth
                
                # Compute loss for this step
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
                step_loss = loss_fn(next_token_logits, target_tokens)
                
                # Only add to loss if not padding
                valid_mask = (target_tokens != self.tokenizer.pad_token_id)
                if valid_mask.any():
                    total_loss += step_loss * valid_mask.float().mean()
                    num_tokens += 1
                
                # Use ground truth for next input (can be changed to use predictions)
                next_tokens = target_tokens.unsqueeze(1)
            else:
                # Beyond ground truth length, stop
                break
            
            # Append next token to sequence
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
            
            # Stop if all sequences hit SEP token
            if (next_tokens.squeeze() == self.tokenizer.sep_token_id).all():
                break
        
        # Average loss over all tokens
        if num_tokens > 0:
            avg_loss = total_loss / num_tokens
        else:
            avg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return {"loss": avg_loss}

    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=100,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Generate medical reports from CT images
        """
        image = samples["image"]
        
        # Encode image
        image_embeds, _ = self.visual_encoder(image)
        
        # Create prompt (start with CLS token)
        batch_size = image.size(0)
        prompt_ids = torch.full((batch_size, 1), self.tokenizer.cls_token_id, dtype=torch.long, device=image.device)
        
        # Create tokenizer-like object for compatibility
        class PromptTokens:
            def __init__(self, input_ids):
                self.input_ids = input_ids
                
        prompt = PromptTokens(prompt_ids)
        
        # Generate text
        outputs = self.text_decoder.generate_from_encoder(
            tokenized_prompt=prompt,
            visual_embeds=image_embeds,
            sep_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_nucleus_sampling=use_nucleus_sampling,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        
        # Decode to text
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text

    @classmethod
    def from_config(cls, cfg):
        # Load vision encoder (ViT)
        import torch
        from lavis.models.blip_models.vit import ViT

        model = ViT(
            in_channels=1,
            img_size=(112, 256, 352),
            patch_size=(16, 16, 32),
            num_classes=0,
            dropout_rate=0.1,
            qkv_bias=True
        )
        
        # Load pretrained ViT weights
        ckpt = torch.load(
            'mae_pretrain_vit_base.pth',
            map_location='cpu'
        )

        from collections import OrderedDict
        new_ckpt = OrderedDict()
        for key, value in ckpt['model'].items():
            if key.startswith("decoder") or key == 'mask_token' or key == "cls_token" or key.startswith("patch_embed"):
                continue
            
            if key.startswith("pos_embed"):
                value = value[0, 1:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
                value = F.interpolate(value, size=(16, 11), mode='bilinear', align_corners=False)
                value = value.unsqueeze(2).repeat(1, 1, 7, 1, 1).flatten(2).permute(0, 2, 1)
                new_ckpt['patch_embedding.position_embeddings'] = value
                continue

            new_ckpt[key.replace('fc', 'linear').replace('proj', 'out_proj')] = value
        model.load_state_dict(new_ckpt, strict=False)

        image_encoder = model

        # Load text decoder only (no text encoder)
        text_decoder = XBertLMHeadDecoder.from_config(cfg, from_pretrained=True)

        max_txt_len = cfg.get("max_txt_len", 256)
        use_autoregressive_training = cfg.get("use_autoregressive_training", False)

        model = cls(
            image_encoder=image_encoder,
            text_decoder=text_decoder,
            max_txt_len=max_txt_len,
            use_autoregressive_training=use_autoregressive_training,
        )

        return model

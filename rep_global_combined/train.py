import sys
import os
import argparse
import logging
import wandb
import torch # Needed for AdamW
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from data import UnifiedMedicalDataset, ModularCollator, build_transforms
from model import MedicalVLM

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logging.basicConfig(level=logging.INFO)
    os.environ["WANDB_PROJECT"] = "thesis"
    
    # Map CLI strategy to Dataset Mode
    data_mode = 'global'
    if args.strategy == 'masked_single': data_mode = 'masked_single'
    if args.strategy in ['roi', 'attention', 'attention_qformer']: data_mode = 'parallel'

    print(f"Training Strategy: {args.strategy} | Data Mode: {data_mode}")

    model = MedicalVLM(
        vision_encoder_path=args.vision_encoder_path,
        decoder_model_name=args.decoder_model,
        strategy=args.strategy,
        use_qformer=args.use_qformer
    )

    transform = build_transforms()
    
    # Initialize Datasets
    train_dataset = UnifiedMedicalDataset(
        args.csv_file, args.json_file, model.tokenizer, transform, 
        mode=data_mode, split='training', max_length=args.max_length, subset_size=args.subset_size
    )
    val_dataset = UnifiedMedicalDataset(
        args.csv_file, args.json_file, model.tokenizer, transform, 
        mode=data_mode, split='validation', max_length=args.max_length, subset_size=args.subset_size
    )

    run_name = f"{args.decoder_model.split('/')[-1]}_{args.strategy}_12organs_diffLR"
    
    # --- 1. CUSTOM OPTIMIZER LOGIC ---
    # Goal: High LR for Q-Former (New), Low LR for Decoder (Pre-trained)
    
    # Hyperparameters
    lr_qformer = 1e-4   # Fast learning for new weights
    lr_decoder = 5e-5   # Slow fine-tuning for GPT-2
    weight_decay = 0.05

    # Separate parameters by name
    qformer_params = []
    decoder_params = []
    
    # We iterate all named parameters to sort them
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue # Skip frozen weights (like ViT)
            
        # "encoder" in the HF VisionEncoderDecoderModel refers to our Adapter/Q-Former
        # "organ_queries", "qformer", "output_proj" are all inside the encoder
        if "encoder" in name or "qformer" in name or "organ_queries" in name:
            qformer_params.append(param)
        else:
            # "decoder", "lm_head" etc. go here
            decoder_params.append(param)

    print(f"\n--- Optimizer Groups ---")
    print(f"Q-Former/Adapter Params (LR={lr_qformer}): {len(qformer_params)} tensors")
    print(f"GPT-2/Decoder Params    (LR={lr_decoder}): {len(decoder_params)} tensors")

    # Create Grouped List
    optimizer_grouped_parameters = [
        {
            "params": qformer_params, 
            "lr": lr_qformer, 
            "weight_decay": weight_decay
        },
        {
            "params": decoder_params, 
            "lr": lr_decoder, 
            "weight_decay": weight_decay
        }
    ]

    # Initialize AdamW manually
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    # ---------------------------------

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,
        report_to="wandb",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8, 
        # learning_rate=2e-4, <-- REMOVED (Overridden by optimizer)
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=ModularCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        # PASS THE CUSTOM OPTIMIZER HERE
        # We pass (optimizer, None) so Trainer creates the standard Scheduler for us automatically
        optimizers=(optimizer, None) 
    )

    # Sanity Check print
    print("\n--- Trainable Parameters Check ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "crossattention" in name or "qformer" in name or "ln_2" in name:
                print(f"Training: {name}")
                break 

    trainer.train()
    model.save_pretrained(f"{args.output_dir}/final_model")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, required=True, choices=['global', 'masked_single', 'roi', 'attention', 'attention_qformer'])
    parser.add_argument('--decoder_model', type=str, default='gpt2')
    parser.add_argument('--vision_encoder_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth')
    parser.add_argument('--csv_file', type=str, default='/home/muhammedg/fvlm/data/image_first_dataset.csv')
    parser.add_argument('--json_file', type=str, default='/home/muhammedg/fvlm/data/combined_desc_conc.json')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--subset_size', type=int, default=None)
    parser.add_argument('--use_qformer', action='store_true')
    
    args = parser.parse_args()
    # Clean up any potential conflicts with CLI args regarding LR
    # (Though optimizer override usually takes precedence)
    main(args)
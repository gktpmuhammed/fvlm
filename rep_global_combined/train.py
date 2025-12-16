import sys
import os
import argparse
import logging
import wandb
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from data import UnifiedMedicalDataset, ModularCollator, build_transforms
from model import MedicalVLM

def main(args):
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

    run_name = f"{args.decoder_model.split('/')[-1]}_{args.strategy}_12organs"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,
        report_to="wandb",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8, 
        learning_rate=2e-4,
        weight_decay=0.01,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print("\n--- Trainable Parameters ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            # We want to see 'qformer' and 'crossattention' here
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
    main(args)
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
from transformers import AutoTokenizer
from torchmetrics.text.rouge import ROUGEScore
import logging

# Local import from the same directory
from simple_medical_vlm import SimpleMedicalVLM
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    SpatialPadd,
    CenterSpatialCropd,
    Transposed,
)

# Suppress Hugging Face tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- DATASET DEFINITION ---
# This class is now synced with the training script's version
class ImageFirstDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform, max_length=512, subset_size=None, split='validation'):
        df = pd.read_csv(csv_file)
        
        # Filter by split
        df = df[df['split'] == split].copy()
        
        if subset_size:
            if subset_size > len(df):
                subset_size = len(df)
            self.samples = df.sample(n=subset_size, random_state=42).reset_index(drop=True)
        else:
            self.samples = df.reset_index(drop=True)
            
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        image_path = sample['image_path']
        
        # Apply the MONAI transforms to load and process the image
        if os.path.exists(image_path):
            try:
                data = self.transform({"image": image_path})
                pixel_values = data["image"]
            except Exception as e:
                logging.warning(f"MONAI transform failed for {image_path}: {e}. Using zero tensor.")
                pixel_values = torch.zeros((1, 112, 256, 352))
        else:
            logging.warning(f"Image not found at {image_path}. Using zero tensor.")
            pixel_values = torch.zeros((1, 112, 256, 352)) # Return a zero tensor of the correct size
        
        findings = str(sample.get("findings", "")).strip() if pd.notna(sample.get("findings", "")) else ""
        impressions = str(sample.get("impressions", "")).strip() if pd.notna(sample.get("impressions", "")) else ""
        
        if findings and impressions:
            text = f"[FINDINGS] {findings} [IMPRESSION] {impressions}"
        elif impressions:
            text = f"[IMPRESSION] {impressions}"
        elif findings:
            text = f"[FINDINGS] {findings}"
        else:
            text = "[NORMAL]"

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'pixel_values': pixel_values,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze().clone(),
            'text': text,
            'patient_id': sample['patient_id']
        }


# --- CUSTOM COLLATE FUNCTION ---
# This function correctly batches tensors while keeping text fields as lists
def custom_collate_fn(batch):
    # Separate non-tensor data
    texts = [item.pop('text') for item in batch]
    patient_ids = [item.pop('patient_id') for item in batch]

    # Batch the remaining tensor items using the default logic
    # This works because all remaining items in the list of dicts are tensors.
    # PyTorch's default collate can handle this.
    collated_batch = torch.utils.data.dataloader.default_collate(batch)
    
    # Add the non-tensor data back
    collated_batch['text'] = texts
    collated_batch['patient_id'] = patient_ids
    
    return collated_batch


# --- EVALUATION FUNCTION ---
def evaluate_model(model, eval_dataset, device, batch_size=4):
    model.eval()
    model.to(device)
    tokenizer = model.tokenizer
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=custom_collate_fn  # Use our custom collate function
    )
    
    results = []
    rouge = ROUGEScore()

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            
            # Generate report from image
            generated_ids = model.generate(
                pixel_values,
                max_length=256,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                repetition_penalty=2.0
            )

            # Get ground truth and patient IDs directly from the batch
            ground_truth_texts = batch['text']
            patient_ids = batch['patient_id']
            
            # Decode generated text
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Store results for this batch
            for i in range(len(generated_texts)):
                results.append({
                    "patient_id": patient_ids[i],
                    "generated_report": generated_texts[i],
                    "ground_truth": ground_truth_texts[i]
                })

    # Calculate ROUGE scores on the full set of results
    generated_reports = [r['generated_report'] for r in results]
    ground_truth_reports = [r['ground_truth'] for r in results]
    
    # Handle the case where reports might be empty
    if not generated_reports or not ground_truth_reports:
        logging.warning("No reports were generated or no ground truth available. Skipping ROUGE calculation.")
        return {}, pd.DataFrame(results)
        
    rouge_scores = rouge(generated_reports, ground_truth_reports)
    
    return rouge_scores, pd.DataFrame(results)


if __name__ == "__main__":
    # The path to the directory where your trained model is saved
    final_model_path = "/home/muhammedg/fvlm/outputs/ImageFirst_VLM_Final_Subset_Test"
    
    # The path to your dataset CSV file
    dataset_csv_path = "/home/muhammedg/fvlm/image_first_dataset.csv"
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    # Note: The model was saved as a state_dict, so we need to initialize the model
    # structure first and then load the weights.
    model = SimpleMedicalVLM(
        vision_encoder_path="/home/muhammedg/fvlm/checkpoints/model.pth"
    )
    
    # The Trainer saves the model inside a 'pytorch_model.bin' file
    model_checkpoint_path = os.path.join(final_model_path, 'pytorch_model.bin')
    if not os.path.exists(model_checkpoint_path):
        print(f"Error: Model checkpoint not found at {model_checkpoint_path}")
        exit()
        
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    tokenizer = model.tokenizer
    print("✅ Model loaded successfully.")

    # --- Main execution ---
    # Define the MONAI transform pipeline
    transform = Compose([
        LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
        Transposed(keys=["image"], indices=(0, 3, 2, 1)),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1150, a_max=350,
            b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(
            keys=["image"], spatial_size=(112, 256, 352),
            mode="constant", constant_values=0
        ),
        CenterSpatialCropd(
            keys=["image"], roi_size=(112, 256, 352)
        ),
    ])

    print(f"\n📊 Loading validation dataset with MONAI transforms...")
    eval_dataset = ImageFirstDataset(
        csv_file=dataset_csv_path,
        tokenizer=tokenizer,
        transform=transform,
        max_length=512,
        subset_size=100,
        split='validation'
    )
    
    # Run the evaluation on the validation subset
    rouge_scores, results_df = evaluate_model(
        model=model,
        eval_dataset=eval_dataset,
        device=device,
        batch_size=4
    )

    print("\n✅ Evaluation Complete!")
    print("\n📊 ROUGE Scores:")
    for key, value in rouge_scores.items():
        print(f"  - {key}: {value.item():.4f}")
        
    # Save results to CSV
    save_path = os.path.join(os.path.dirname(final_model_path), "evaluation_results.csv")
    results_df.to_csv(save_path, index=False)
    print(f"\n💾 Results saved to: {save_path}")

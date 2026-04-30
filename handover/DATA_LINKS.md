# Data Links and Metadata

This project expects symlinked CT volumes and local metadata files.

## Keep in handover bundle
- `data_sym/image_first_dataset.csv`
- `data_sym/combined_desc_conc_v2.json`
- `data_sym/organ_loss_weights.json` (if used)
- `data_sym/organ_sampling_probs.json` (if used)
- `decomposed_data/train_findings.json`
- `decomposed_data/train_impressions.json`
- `decomposed_data/val_findings.json`
- `decomposed_data/val_impressions.json`
- `decomposed_data/combined_train_val.json`
- `decomposed_data/combine_data.py`

## Recreate symlinks for data_sym
Recommended (single command):
```bash
bash scripts/setup/recreate_data_symlinks.sh
```
Default dataset root in script: `/mnt/nas/Data_WholeBody/CT-Rate/dataset`

Optional override:
```bash
bash scripts/setup/recreate_data_symlinks.sh /path/to/CT_RATE/dataset
```
This one command recreates all expected links:
- `data_sym/train/images` + `data_sym/train/masks`
- `data_sym/valid/images` + `data_sym/valid/masks`
- `data_sym/metadata`
- `data_sym/radiology_text_reports`
- `data_sym/multi_abnormality_labels`

The script auto-detects mask directories using either:
- `train_TS` + `valid_TS` (your current CT-Rate layout), or
- `train_mask_process` + `valid_mask_process`

Manual example:
```bash
python data/create_symlinks.py \
  --source_images /path/to/CT_RATE/train_process \
  --source_masks /path/to/CT_RATE/train_TS \
  --dest_root data_sym/train

python data/create_symlinks.py \
  --source_images /path/to/CT_RATE/valid_process \
  --source_masks /path/to/CT_RATE/valid_TS \
  --dest_root data_sym/valid
```

## Update metadata/report symlinks
Set these to your dataset root:
- `data_sym/metadata`
- `data_sym/radiology_text_reports`
- `data_sym/multi_abnormality_labels`

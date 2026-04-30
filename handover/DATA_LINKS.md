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
bash scripts/setup/recreate_data_symlinks.sh /path/to/CT_RATE/dataset
```

Manual example:
```bash
python data/create_symlinks.py \
  --source_images /path/to/CT_RATE/train_process \
  --source_masks /path/to/CT_RATE/train_mask_process \
  --dest_root data_sym/train

python data/create_symlinks.py \
  --source_images /path/to/CT_RATE/valid_process \
  --source_masks /path/to/CT_RATE/valid_mask_process \
  --dest_root data_sym/valid
```

## Update metadata/report symlinks
Set these to your dataset root:
- `data_sym/metadata`
- `data_sym/radiology_text_reports`
- `data_sym/multi_abnormality_labels`

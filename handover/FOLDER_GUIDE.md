# FVLM Folder Guide

This guide explains what the main folders in this local `fvlm` workspace are for and how to run the important model variants. It is meant as a practical handover note for people continuing the experiments on this machine.

## Top-level folders

| Path | Purpose |
| --- | --- |
| `data/` | Original FVLM/CT-RATE preprocessing scripts and, in some setups, local processed metadata. Prefer `data_sym/` for current training inputs. |
| `data_sym/` | Symlink-based CT-RATE data layout used by the current training/evaluation scripts. It should contain `train/`, `valid/`, `metadata/`, reports, masks, and generated files such as `image_first_dataset.csv`. |
| `decomposed_data/` | Anatomy-wise report decomposition JSONs and helper scripts. These files are used to build `data_sym/combined_desc_conc_v2.json`. |
| `BiomedVLP-CXR-BERT-specialized/` | Local Hugging Face snapshot of the CXR-BERT encoder used by the original FVLM/CT-CLIP setup. This is a large local asset and should not be committed. |
| `checkpoints/` | Local model checkpoints. `checkpoints/model.pth` is the released FVLM/CT-RATE vision checkpoint used as the vision encoder seed. Training runs also write here. |
| `results/` | Generated reports and metric outputs from evaluation runs. |
| `envs/` | Environment lock/requirement files used by setup scripts. |
| `scripts/setup/` | Bootstrap and setup helpers for Conda environments, data symlinks, and lockfile regeneration. |
| `lavis/` | Local LAVIS/FVLM model code used by the original training stack and the custom ViT implementation. |
| `rep_medgemma/` | MedGemma-based report generation experiments. Several variants live here; choose the one that matches the experiment you want to reproduce. |
| `rep_vision_organ_attention/` | Earlier organ-attention VLM experiments using non-MedGemma decoders such as GPT-2/BioBART-style models. |
| `rep_vision_bert/` | BERT-decoder/clinical-BERT experiments. Useful as a lighter comparison branch, but not the main MedGemma path. |
| `handover/` | Local documentation for the next person using this workspace. |

## Smoke-tested MedGemma path: `rep_medgemma/perceiver_resampler`

This is the MedGemma variant that was smoke-tested in this workspace. That does not mean it is the best or final variant; it is simply known to run end-to-end here with the commands below. It contains:

- `train.py`: training entrypoint.
- `evaluate.py`: generation/evaluation entrypoint that writes `generated_reports_gemma.csv`.
- `medical_vlm.py`: model definition with trainable ViT, organ masks, Perceiver resampler, visual projection, frozen MedGemma, and LoRA adapters.

Minimal MedGemma smoke training command:

```bash
cd /home/muhammedg/fvlm

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=disabled \
python rep_medgemma/perceiver_resampler/train.py \
  --decoder_model google/medgemma-4b-it \
  --vision_encoder_path checkpoints/model.pth \
  --csv_file data_sym/image_first_dataset.csv \
  --json_file data_sym/combined_desc_conc_v2.json \
  --output_dir checkpoints/smoke_medgemma_1epoch \
  --batch_size 1 \
  --num_epochs 1 \
  --subset_size 2 \
  --eval_steps 1 \
  --logging_steps 1 \
  --queries_per_organ 1
```

Evaluation for that smoke checkpoint:

```bash
cd /home/muhammedg/fvlm

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=disabled \
python rep_medgemma/perceiver_resampler/evaluate.py \
  --checkpoint_dir checkpoints/smoke_medgemma_1epoch/final \
  --vision_encoder_path checkpoints/model.pth \
  --decoder_model google/medgemma-4b-it \
  --csv_file data_sym/image_first_dataset.csv \
  --output_dir results/smoke_medgemma_eval \
  --subset_size 2 \
  --batch_size 1 \
  --queries_per_organ 1
```

Run lightweight metrics:

```bash
cd /home/muhammedg/fvlm

conda run -n radevalmetrics python rep_medgemma/radeval_metrics.py \
  --input_csv results/smoke_medgemma_eval/generated_reports_gemma.csv \
  --ground_truth_json data_sym/combined_desc_conc_v2.json \
  --output_dir results/smoke_medgemma_metrics \
  --metrics bleu rouge meteor \
  --subset 2 \
  --device cpu
```

Notes:

- Keep `CUDA_VISIBLE_DEVICES=0` or another single GPU for small smoke runs. If multiple GPUs are visible, Hugging Face Trainer may use `DataParallel`, which can duplicate the model and cause out-of-memory errors.
- `google/medgemma-4b-it` is gated. Log in with `huggingface-cli login` and accept the model terms before first use.
- If using a shared account, run `huggingface-cli logout` after downloading/running. The model cache may remain, but the account token should be removed.
- If you train with `--queries_per_organ 1`, evaluate with `--queries_per_organ 1`. The checkpoint shape depends on this value.

## Other `rep_medgemma` experiment folders

Most subfolders under `rep_medgemma/` are experiment snapshots. They follow the same rough pattern: `medical_vlm.py`, `train.py`, and `evaluate.py`, with architectural changes between folders.

| Folder | What it represents |
| --- | --- |
| `perceiver_resampler/` | Smoke-tested variant in this workspace. Uses a Perceiver-style resampler over organ queries and visual features. |
| `medgemma_architecture_v3/` | Earlier MedGemma architecture baseline with organ visual tokens. |
| `medical_vlm_8_tokens_full/` | Variant using 8 visual tokens per organ. |
| `medical_vlm_8_tokens_full_maxpool/` | 8-token variant with max-pooling mask handling. |
| `medgemma_lora_vis_token_pos_embed/` | LoRA plus visual token positional embedding experiment. |
| `lora_with_vis_tokens_pos_embed_undersampling/` | LoRA/visual-token variant with undersampling logic. |
| `lora_with_vis_tokens_pos_embed_weight_loss/` | LoRA/visual-token variant with weighted loss. |
| `medgemma_alignment_v1/` | Alignment-loss experiment for stronger image/text grounding. |
| `curriculum_learning/` | Curriculum-learning experiment branch. |
| `hard_example_mining/` | Hard-example-mining experiment branch. |
| `multiscale_vit_fpn/` | Multi-scale ViT/FPN experiment branch. |

There are also shared helpers:

- `rep_medgemma/radeval_metrics.py`: metrics runner used after generated reports exist.
- `rep_medgemma/run_experiment.sh`: older end-to-end script for train, eval, and metrics. Check paths before using it.
- `rep_medgemma/visualize_attention.py`: attention visualization helper.

## Organ-attention branch: `rep_vision_organ_attention`

This branch predates the MedGemma setup. It uses organ masks to extract organ-aware visual features and trains a smaller decoder model.

Files:

- `train.py`: trains the organ-attention VLM.
- `evaluate.py`: generates reports from a saved `final_model`.
- `medical_vlm.py`: model definition for organ attention and decoder integration.
- `radeval_metrics.py`: metrics wrapper for this branch.
- `run_training_parallel.sh`: older multi-run helper.

Small smoke training:

```bash
cd /home/muhammedg/fvlm

CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=disabled \
python rep_vision_organ_attention/train.py \
  --decoder_model gpt2 \
  --vision_encoder_path checkpoints/model.pth \
  --csv_file data_sym/image_first_dataset.csv \
  --json_file data_sym/combined_desc_conc_v2.json \
  --output_dir checkpoints/smoke_organ_attention \
  --batch_size 1 \
  --num_epochs 1 \
  --subset_size 2 \
  --eval_steps 1 \
  --logging_steps 1 \
  --queries_per_organ 1
```

Evaluate that checkpoint:

```bash
cd /home/muhammedg/fvlm

CUDA_VISIBLE_DEVICES=0 \
python rep_vision_organ_attention/evaluate.py \
  --model_path checkpoints/smoke_organ_attention/final_model \
  --decoder_model gpt2 \
  --vision_encoder_path checkpoints/model.pth \
  --csv_file data_sym/image_first_dataset.csv \
  --json_file data_sym/combined_desc_conc_v2.json \
  --output_dir results/smoke_organ_attention_eval \
  --subset_size 2 \
  --queries_per_organ 1
```

Use this branch when you want a lighter, non-MedGemma comparison or want to debug organ-mask behavior without loading a 4B decoder.

## BERT branch: `rep_vision_bert`

This branch uses a clinical/BioBERT-style decoder path. It is useful for comparison experiments with smaller text models, but its scripts are less current than `rep_medgemma/perceiver_resampler`.

Files:

- `train.py`: training entrypoint. Default decoder is `emilyalsentzer/Bio_ClinicalBERT`.
- `evaluate.py`: evaluates a saved model path.
- `medical_vlm.py`: BERT-branch model definition.
- `radeval_metrics.py`: branch-local metrics wrapper.
- `run_training_e2e_bert.sh`: older end-to-end helper. Parts of training/eval are currently commented, so inspect before using.

Small smoke training:

```bash
cd /home/muhammedg/fvlm

CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=disabled \
python rep_vision_bert/train.py \
  --decoder_model emilyalsentzer/Bio_ClinicalBERT \
  --vision_encoder_path checkpoints/model.pth \
  --csv_file data_sym/image_first_dataset.csv \
  --json_file data_sym/combined_desc_conc_v2.json \
  --output_dir checkpoints/smoke_vision_bert \
  --batch_size 1 \
  --num_epochs 1 \
  --subset_size 2 \
  --eval_steps 1 \
  --logging_steps 1 \
  --queries_per_organ 1
```

Evaluate that checkpoint:

```bash
cd /home/muhammedg/fvlm

CUDA_VISIBLE_DEVICES=0 \
python rep_vision_bert/evaluate.py \
  --model_path checkpoints/smoke_vision_bert/final_model \
  --decoder_model emilyalsentzer/Bio_ClinicalBERT \
  --vision_encoder_path checkpoints/model.pth \
  --csv_file data_sym/image_first_dataset.csv \
  --json_file data_sym/combined_desc_conc_v2.json \
  --output_dir results/smoke_vision_bert_eval \
  --subset_size 2 \
  --queries_per_organ 1
```

## Environment usage

Use these Conda environments:

- `fvlm_training_clean`: training and evaluation for FVLM, MedGemma, organ-attention, and BERT branches.
- `radevalmetrics`: RadEval/NLP metric calculation.
- `decompose`: report decomposition with vLLM in the separate `report_decomposition` repo.

Examples:

```bash
conda activate fvlm_training_clean
conda activate radevalmetrics
conda activate decompose
```

or from scripts:

```bash
conda run -n fvlm_training_clean python ...
conda run -n radevalmetrics python ...
```

## Expected local assets

Before running training/evaluation, check these exist:

```text
/home/muhammedg/fvlm/checkpoints/model.pth
/home/muhammedg/fvlm/data_sym/image_first_dataset.csv
/home/muhammedg/fvlm/data_sym/combined_desc_conc_v2.json
/home/muhammedg/fvlm/data_sym/train/images
/home/muhammedg/fvlm/data_sym/valid/images
/home/muhammedg/fvlm/data_sym/train/masks
/home/muhammedg/fvlm/data_sym/valid/masks
```

Quick checks:

```bash
cd /home/muhammedg/fvlm

conda run -n fvlm_training_clean python -c "import torch, transformers; print('training env OK')"
conda run -n fvlm_training_clean python -c "import pandas as pd; df=pd.read_csv('data_sym/image_first_dataset.csv'); print(df.shape); print(df['split'].value_counts())"
```

## Git and storage notes

- Do not commit large local artifacts: model snapshots, CT volumes, checkpoints, caches, or generated results.
- Keep permanent documentation in `handover/` or `README.md`.
- Use `git status --short` before handing off changes.

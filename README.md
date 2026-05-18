# Fine-grained Vision-language Pre-training for Enhanced CT Image Understanding
[Paper](https://openreview.net/pdf?id=nYpPAT4L3D) (ICLR 2025 Spotlight)

## Fork Note: Medical VLM Thesis Experiments

This fork contains my thesis-related experiments on 3D CT report generation and medical vision-language modeling. The original repository implements Fine-grained Vision-language Pre-training for CT understanding. My additions build on that codebase to explore MedGemma-based report generation, organ-aware visual tokens, LoRA variants, training strategies, evaluation scripts, and attention visualization.

The main experimental code is under:

```text
rep_medgemma/
rep_vision_organ_attention/
rep_vision_bert/
```

Key additions include:

- MedGemma-based 3D CT report generation experiments.
- Custom `medical_vlm.py` model variants for visual-token and organ-aware conditioning.
- LoRA-based fine-tuning setups.
- Positional embedding and visual-token ablation experiments.
- Curriculum learning and hard-example mining variants.
- Organ-level attention visualization scripts and generated examples.
- Report-generation evaluation utilities, including language metrics and model comparison tables.

Useful entry points:

| Path | Purpose |
| --- | --- |
| [`handover/clean-migration`](https://github.com/gktpmuhammed/fvlm/tree/handover/clean-migration) | Clean migration branch with setup notes, environment files, smoke tests, and code-only handover structure |
| `rep_medgemma/perceiver_resampler/` | Smoke-tested MedGemma path in the clean migration branch |
| `rep_medgemma/medgemma_lora_vis_token_pos_embed/` | LoRA + visual-token + positional embedding experiments |
| `rep_medgemma/medical_vlm_8_tokens_full/` | 8 visual tokens per organ variant |
| `rep_medgemma/multiscale_vit_fpn/` | Multi-scale ViT/FPN experiment branch |
| `rep_medgemma/visualize_attention.py` | Attention visualization helper |
| `rep_vision_organ_attention/` | Organ-aware vision-language experiments |
| `rep_vision_bert/` | BERT-decoder comparison branch |

Thesis-final result tables and figures are summarized in the project page:

- [medical-vlm-radiology-report-generation](https://github.com/gktpmuhammed/medical-vlm-radiology-report-generation)

High-level thesis findings:

- Within MedGemma ablations, Base-8T improves over Base-1T on GREEN, RadGraph, and most lexical-semantic metrics.
- In cross-decoder comparison, the fVLM-aligned BERT-base baseline is strongest on GREEN, RadGraph, ROUGE-L, and BERTScore.
- MedGemma shows substantially lower template reuse and higher output diversity than the other decoder families.

This fork remains the implementation workspace; the separate project page is the concise portfolio view.


## Data processing

- Download the [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) dataset into the data folder.

- Download ImageNet pre-trained ViT weights from [link](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth), and BiomedVLP-CXR-BERT-specialized text encoder from [link](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized), as used by CT-CLIP.

- Download the decomposed anatomy-wise descriptions from our provided supplementary materials [link](https://drive.google.com/drive/folders/10bz2UFxqxDPzl2P9NohESSNyBuld_Iek?usp=drive_link), and process the CT volume with the following commands.

  ```bash
  cd data
  python fix_data.py --split [train/valid]
  python generate_mask.py --split [train/valid]
  python resize.py --split [train/valid]
  python preprocess.py --split [train/valid]
  ```

  The processed results.

  ```bash
  |-- BiomedVLP-CXR-BERT
  |-- data
  |   |-- train
  |   |-- valid
  |   |-- train_fixed
  |   |-- valid_fixed
  |   |-- train_mask
  |   |-- valid_mask
  |   |-- resized_train_images
  |   |-- resized_train_masks
  |   |-- resized_valid_images
  |   |-- resized_valid_masks
  |   |-- processed_train_images
  |   |-- processed_train_masks
  |   |-- processed_valid_images
  |   |-- processed_valid_masks
  |   |-- multi_abnormality_labels
  |   |-- desc_info.json
  |   |-- conc_info.json
  |-- mae_pretrain_vit_base.pth
  ```



## Training

```shell
torchrun --nproc_per_node=4 train.py
```

[Pre-trained weights](https://drive.google.com/drive/folders/15BnMo1lIAlOH_8KLdB2NugiHnmj9AWSD?usp=drive_link) of CT-RATE are released. 


## Evaluation

```bash
torchrun --nproc_per_node=4 eval.py
```

Then, you can calculate the metrics using the generated CSV file.

```bash
python calc_metrics.py --csv_file res/xxx.csv
```

## Citation
If you find this repository useful, please cite:
```
@inproceedings{fvlm_iclr25,
  title={Large-scale and fine-grained vision-language pre-training for enhanced CT image understanding},
  author={Zhongyi Shui, Jianpeng Zhang, Weiwei Cao, Sinuo Wang, Ruizhe Guo, Le Lu, Lin Yang, Xianghua Ye, Tingbo Liang, Qi Zhang, Ling Zhang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  pages={},
  year={2025}
}
```

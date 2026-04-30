# Smoke Tests

## Env imports
```bash
conda activate fvlm_training_clean
python -c "import torch, transformers"

conda activate radevalmetrics
python -c "from RadEval import RadEval"

conda activate ct-rate
python -c "from vllm import LLM"
```

## Data link checks
```bash
readlink -f data_sym/train/images/train/train_1/train_1_a/train_1_a_1.nii.gz
readlink -f data_sym/valid/images/valid/valid_1/valid_1_a/valid_1_a_1.nii.gz
python - <<'PY'
import json, pandas as pd
pd.read_csv('data_sym/image_first_dataset.csv').head(1)
json.load(open('data_sym/combined_desc_conc_v2.json'))
print('metadata OK')
PY
```

## Functional smoke
```bash
python ../src/ct_rate/report_decomposition_vllm.py --train_csv /path/to/train_reports.csv --val_csv /path/to/validation_reports.csv --output_dir decomposed_data --sample 2
python decomposed_data/combine_data.py
```

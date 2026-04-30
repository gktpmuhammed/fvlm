#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${1:-}"

if [[ -z "$DATASET_ROOT" ]]; then
  echo "Usage: $0 <CT_RATE_DATASET_ROOT>"
  echo "Expected subfolders: train_process, valid_process, train_mask_process, valid_mask_process, metadata, radiology_text_reports, multi_abnormality_labels"
  exit 1
fi

DATA_SYM_ROOT="${DATA_SYM_ROOT:-$PROJECT_ROOT/data_sym}"
mkdir -p "$DATA_SYM_ROOT/train" "$DATA_SYM_ROOT/valid"

python "$PROJECT_ROOT/data/create_symlinks.py" \
  --source_images "$DATASET_ROOT/train_process" \
  --source_masks "$DATASET_ROOT/train_mask_process" \
  --dest_root "$DATA_SYM_ROOT/train"

python "$PROJECT_ROOT/data/create_symlinks.py" \
  --source_images "$DATASET_ROOT/valid_process" \
  --source_masks "$DATASET_ROOT/valid_mask_process" \
  --dest_root "$DATA_SYM_ROOT/valid"

ln -sfn "$DATASET_ROOT/metadata" "$DATA_SYM_ROOT/metadata"
ln -sfn "$DATASET_ROOT/radiology_text_reports" "$DATA_SYM_ROOT/radiology_text_reports"
ln -sfn "$DATASET_ROOT/multi_abnormality_labels" "$DATA_SYM_ROOT/multi_abnormality_labels"

echo "Symlink recreation complete under $DATA_SYM_ROOT"

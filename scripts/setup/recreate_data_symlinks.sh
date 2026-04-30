#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${1:-}"

if [[ -z "$DATASET_ROOT" ]]; then
  echo "Usage: $0 <CT_RATE_DATASET_ROOT>"
  echo "Expected subfolders:"
  echo "  train_process, valid_process,"
  echo "  either train_TS+valid_TS OR train_mask_process+valid_mask_process,"
  echo "  metadata, radiology_text_reports, multi_abnormality_labels"
  exit 1
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "ERROR: neither 'python' nor 'python3' is available on PATH." >&2
  exit 1
fi

required_dirs=(
  "train_process"
  "valid_process"
  "metadata"
  "radiology_text_reports"
  "multi_abnormality_labels"
)
for d in "${required_dirs[@]}"; do
  if [[ ! -d "$DATASET_ROOT/$d" ]]; then
    echo "ERROR: missing required directory: $DATASET_ROOT/$d" >&2
    exit 1
  fi
done

if [[ -d "$DATASET_ROOT/train_TS" && -d "$DATASET_ROOT/valid_TS" ]]; then
  TRAIN_MASK_SUBDIR="train_TS"
  VALID_MASK_SUBDIR="valid_TS"
elif [[ -d "$DATASET_ROOT/train_mask_process" && -d "$DATASET_ROOT/valid_mask_process" ]]; then
  TRAIN_MASK_SUBDIR="train_mask_process"
  VALID_MASK_SUBDIR="valid_mask_process"
else
  echo "ERROR: could not find mask directories under $DATASET_ROOT" >&2
  echo "Expected either:" >&2
  echo "  - train_TS and valid_TS" >&2
  echo "  - train_mask_process and valid_mask_process" >&2
  exit 1
fi

DATA_SYM_ROOT="${DATA_SYM_ROOT:-$PROJECT_ROOT/data_sym}"
mkdir -p "$DATA_SYM_ROOT/train" "$DATA_SYM_ROOT/valid"

"$PYTHON_BIN" "$PROJECT_ROOT/data/create_symlinks.py" \
  --source_images "$DATASET_ROOT/train_process" \
  --source_masks "$DATASET_ROOT/$TRAIN_MASK_SUBDIR" \
  --dest_root "$DATA_SYM_ROOT/train"

"$PYTHON_BIN" "$PROJECT_ROOT/data/create_symlinks.py" \
  --source_images "$DATASET_ROOT/valid_process" \
  --source_masks "$DATASET_ROOT/$VALID_MASK_SUBDIR" \
  --dest_root "$DATA_SYM_ROOT/valid"

ln -sfn "$DATASET_ROOT/metadata" "$DATA_SYM_ROOT/metadata"
ln -sfn "$DATASET_ROOT/radiology_text_reports" "$DATA_SYM_ROOT/radiology_text_reports"
ln -sfn "$DATASET_ROOT/multi_abnormality_labels" "$DATA_SYM_ROOT/multi_abnormality_labels"

echo "Symlink recreation complete under $DATA_SYM_ROOT"
echo "Mask folders used: $TRAIN_MASK_SUBDIR and $VALID_MASK_SUBDIR"

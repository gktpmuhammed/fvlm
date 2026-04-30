#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/handover}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$OUT_DIR/fvlm_handover_${STAMP}.tar.gz"

mkdir -p "$OUT_DIR"

# Build a clean code + metadata bundle (no heavy results/checkpoints)
tar -czf "$ARCHIVE" \
  --exclude-vcs \
  --exclude='checkpoints*' \
  --exclude='**/results*' \
  --exclude='**/wandb' \
  --exclude='**/*.pth' \
  --exclude='**/*.log' \
  --exclude='data_sym/train/images' \
  --exclude='data_sym/train/masks' \
  --exclude='data_sym/valid/images' \
  --exclude='data_sym/valid/masks' \
  -C "$(dirname "$PROJECT_ROOT")" "$(basename "$PROJECT_ROOT")"

echo "Created: $ARCHIVE"

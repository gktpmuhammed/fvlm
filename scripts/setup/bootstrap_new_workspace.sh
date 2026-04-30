#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATASET_ROOT=""
CONDA_BASE_ARG=""
REPORT_REPO_ROOT=""
RUN_SMOKE=1

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup/bootstrap_new_workspace.sh [options]

Options:
  --dataset-root <path>       CT-RATE dataset root for symlink recreation
  --conda-base <path>         Conda base path (example: /path/to/miniconda3)
  --report-repo-root <path>   Optional repo root containing src/ct_rate/report_decomposition_vllm.py
  --skip-smoke                Skip smoke tests
  -h, --help                  Show this help

Examples:
  bash scripts/setup/bootstrap_new_workspace.sh \
    --conda-base /mnt/nas/Users/Students_Homes/<user>/miniconda3 \
    --dataset-root /mnt/nas/datasets/CT_RATE

  bash scripts/setup/bootstrap_new_workspace.sh --skip-smoke
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="${2:-}"
      shift 2
      ;;
    --conda-base)
      CONDA_BASE_ARG="${2:-}"
      shift 2
      ;;
    --report-repo-root)
      REPORT_REPO_ROOT="${2:-}"
      shift 2
      ;;
    --skip-smoke)
      RUN_SMOKE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "$CONDA_BASE_ARG" ]]; then
  export CONDA_BASE="$CONDA_BASE_ARG"
fi

CONDA_CMD=()

setup_conda() {
  if command -v conda >/dev/null 2>&1 && conda --version >/dev/null 2>&1; then
    CONDA_CMD=(conda)
    return 0
  fi

  local parent_root
  parent_root="$(cd "$PROJECT_ROOT/.." && pwd)"
  local bases=(
    "${CONDA_BASE:-}"
    "$parent_root/miniconda3"
    "$HOME/miniconda3"
    "$HOME/anaconda3"
    "$HOME/miniforge3"
    "$HOME/mambaforge"
  )

  local base
  for base in "${bases[@]}"; do
    [[ -z "$base" ]] && continue
    if [[ -x "$base/bin/python" ]] && "$base/bin/python" -m conda --version >/dev/null 2>&1; then
      CONDA_CMD=("$base/bin/python" -m conda)
      return 0
    fi
  done

  echo "ERROR: Could not find a working conda installation." >&2
  echo "Pass --conda-base /path/to/miniconda3" >&2
  exit 1
}

conda_cmd() {
  "${CONDA_CMD[@]}" "$@"
}

echo "[1/4] Creating environments from lock files..."
bash scripts/setup/create_env_fvlm_training_clean.sh
bash scripts/setup/create_env_radevalmetrics.sh
bash scripts/setup/create_env_ct_rate.sh

echo "[2/4] Recreating dataset symlinks (if dataset root provided)..."
if [[ -n "$DATASET_ROOT" ]]; then
  bash scripts/setup/recreate_data_symlinks.sh "$DATASET_ROOT"
else
  echo "Skipping data symlink recreation (no --dataset-root provided)."
fi

echo "[3/4] Running environment import checks..."
setup_conda
conda_cmd run -n fvlm_training_clean python -c "import torch, transformers; print('fvlm_training_clean OK')"
conda_cmd run -n radevalmetrics python -c "from RadEval import RadEval; print('radevalmetrics OK')"
conda_cmd run -n ct-rate python -c "from vllm import LLM; print('ct-rate OK')"

if [[ "$RUN_SMOKE" -eq 1 ]]; then
  echo "[4/4] Running lightweight code smoke checks..."
  conda_cmd run -n fvlm_training_clean python "$PROJECT_ROOT/decomposed_data/combine_data.py" --help >/dev/null
  if [[ -n "$REPORT_REPO_ROOT" ]] && [[ -f "$REPORT_REPO_ROOT/src/ct_rate/report_decomposition_vllm.py" ]]; then
    conda_cmd run -n ct-rate python "$REPORT_REPO_ROOT/src/ct_rate/report_decomposition_vllm.py" --help >/dev/null
  fi
  echo "Smoke checks completed."
else
  echo "Skipping smoke tests (--skip-smoke)."
fi

echo "Setup completed successfully."

#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_CMD=()
CONDA_BASE_PATH=""

setup_conda() {
  if command -v conda >/dev/null 2>&1 && conda --version >/dev/null 2>&1; then
    CONDA_CMD=(conda)
    CONDA_BASE_PATH="$(conda info --base 2>/dev/null || true)"
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
      CONDA_BASE_PATH="$base"
      return 0
    fi
  done

  echo "ERROR: Could not find a working conda." >&2
  exit 1
}

conda_cmd() {
  "${CONDA_CMD[@]}" "$@"
}

setup_conda
if [[ -z "$CONDA_BASE_PATH" ]]; then
  CONDA_BASE_PATH="$(conda_cmd info --base | tail -n1)"
fi
conda_cmd create -n fvlm_training_clean -y python=3.10
conda_cmd install -n fvlm_training_clean -y --file envs/fvlm_training_clean.explicit.txt || true
ENV_PYTHON="$CONDA_BASE_PATH/envs/fvlm_training_clean/bin/python"
if [[ -x "$ENV_PYTHON" ]]; then
  "$ENV_PYTHON" -m pip install -r envs/fvlm_training_clean.pip.txt || true
else
  conda_cmd run -n fvlm_training_clean python -m pip install -r envs/fvlm_training_clean.pip.txt || true
fi

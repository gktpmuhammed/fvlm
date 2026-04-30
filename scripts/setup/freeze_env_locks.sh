#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_CMD=()
CONDA_BASE_PATH=""

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
      CONDA_BASE_PATH="$base"
      return 0
    fi

    if [[ -f "$base/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "$base/etc/profile.d/conda.sh" || true
      if command -v conda >/dev/null 2>&1 && conda --version >/dev/null 2>&1; then
        CONDA_CMD=(conda)
        CONDA_BASE_PATH="$(conda info --base 2>/dev/null || true)"
        return 0
      fi
    fi
  done

  echo "ERROR: Could not find a working conda after workspace relocation." >&2
  echo "Try: CONDA_BASE=/mnt/nas/Users/Students_Homes/muhammedg/miniconda3 bash scripts/setup/freeze_env_locks.sh" >&2
  exit 1
}

conda_cmd() {
  "${CONDA_CMD[@]}" "$@"
}

run_freeze() {
  local env_name="$1"
  local prefix="$2"
  conda_cmd list -n "$env_name" --explicit > "envs/${prefix}.explicit.txt"
  local env_python="$CONDA_BASE_PATH/envs/$env_name/bin/python"
  if [[ -x "$env_python" ]]; then
    "$env_python" -m pip freeze > "envs/${prefix}.pip.txt"
  else
    # Fallback for unusual installations
    conda_cmd run -n "$env_name" python -m pip freeze > "envs/${prefix}.pip.txt"
  fi
}

setup_conda
if [[ -z "$CONDA_BASE_PATH" ]]; then
  CONDA_BASE_PATH="$(conda_cmd info --base | tail -n1)"
fi
run_freeze "fvlm_training_clean" "fvlm_training_clean"
run_freeze "radevalmetrics" "radevalmetrics"
run_freeze "ct-rate" "ct-rate"

echo "Wrote lock files under envs/."

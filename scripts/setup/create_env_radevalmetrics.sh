#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

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

  echo "ERROR: Could not find a working conda." >&2
  exit 1
}

conda_cmd() {
  "${CONDA_CMD[@]}" "$@"
}

setup_conda
conda_cmd create -n radevalmetrics -y python=3.10
conda_cmd install -n radevalmetrics -y --file envs/radevalmetrics.explicit.txt || true
conda_cmd run -n radevalmetrics pip install -r envs/radevalmetrics.pip.txt || true

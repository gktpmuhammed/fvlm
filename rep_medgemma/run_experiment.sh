#!/usr/bin/env bash
set -e
set -o pipefail

#############################
# CONFIG
#############################

EXP_NAME="medgemma_architecture_v3_resized_synonyms_new_dataset"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BASE_DIR="${BASE_DIR:-$SCRIPT_DIR}"
LOG_ROOT="$BASE_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$LOG_ROOT/${EXP_NAME}_${TIMESTAMP}"

CHECKPOINT_DIR="$BASE_DIR/checkpoints/$EXP_NAME"
RESULT_DIR="$BASE_DIR/results/$EXP_NAME"

#############################
# SETUP
#############################

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

mkdir -p "$RUN_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$RESULT_DIR"

LOGFILE="$RUN_DIR/run.log"

echo "==========================================="
echo "Experiment: $EXP_NAME"
echo "Run dir   : $RUN_DIR"
echo "Started at: $(date)"
echo "==========================================="

#############################
# TRAIN
#############################

echo "[1/3] TRAINING..." | tee -a "$LOGFILE"

conda activate fvlm_training_clean

python train.py \
  --output_dir "$CHECKPOINT_DIR" --num_epochs 1 \
  2>&1 | tee -a "$LOGFILE"

#############################
# EVALUATE
#############################

echo "[2/3] EVALUATION..." | tee -a "$LOGFILE"

python evaluate.py \
  --checkpoint_dir "$CHECKPOINT_DIR/final" \
  --output_dir "$RESULT_DIR" \
  2>&1 | tee -a "$LOGFILE"

#############################
# METRICS
#############################

echo "[3/3] METRICS..." | tee -a "$LOGFILE"

conda activate radevalmetrics

python radeval_metrics.py \
  --input_csv "$RESULT_DIR/generated_reports_gemma.csv" \
  --output_dir "$RESULT_DIR" \
  --metrics all \
  2>&1 | tee -a "$LOGFILE"

#############################
# DONE
#############################

echo "==========================================="
echo "FINISHED SUCCESSFULLY"
echo "Ended at: $(date)"
echo "Logs: $LOGFILE"
echo "==========================================="

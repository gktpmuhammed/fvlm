#!/usr/bin/env bash
set -euo pipefail

#############################
# CONFIG (override via env)
#############################

BASE_DIR="/home/muhammedg/fvlm/rep_medgemma"
MODEL_DIR="$BASE_DIR/medgemma_standalone"

EXP_NAME="${EXP_NAME:-medgemma_standalone_full}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ORGAN_CHUNK_SIZE="${ORGAN_CHUNK_SIZE:-1}"
EVAL_STEPS="${EVAL_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SUBSET_SIZE="${SUBSET_SIZE:-}"  # leave empty for full dataset

TRAIN_ENV="${TRAIN_ENV:-fvlm_training_clean}"
METRICS_ENV="${METRICS_ENV:-radevalmetrics}"
WANDB_MODE="${WANDB_MODE:-offline}"
TRAIN_DISABLE_EVAL="${TRAIN_DISABLE_EVAL:-1}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"

MEDGEMMA_MODEL="${MEDGEMMA_MODEL:-/home/muhammedg/.cache/huggingface/hub/models--google--medgemma-4b-it/snapshots/290cda5eeccbee130f987c4ad74a59ae6f196408}"
GROUND_TRUTH_JSON="${GROUND_TRUTH_JSON:-/home/muhammedg/fvlm/data_sym/combined_desc_conc_v2.json}"

LOG_ROOT="$BASE_DIR/logs"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="$LOG_ROOT/${EXP_NAME}_${TIMESTAMP}"
CHECKPOINT_DIR="$BASE_DIR/checkpoints/$EXP_NAME"
RESULT_DIR="$BASE_DIR/results/$EXP_NAME"

#############################
# SETUP
#############################

source "/home/muhammedg/miniconda3/etc/profile.d/conda.sh"

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$RESULT_DIR"
LOGFILE="$RUN_DIR/run.log"

# GPU auto-detection:
# - If NUM_GPUS is not provided, default to 2 when at least 2 GPUs exist.
# - If GPU_IDS is not provided, choose "0,1" for multi-GPU and "0" for single-GPU.
if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
else
  DETECTED_GPUS="1"
fi

if [[ -z "${NUM_GPUS:-}" ]]; then
  if (( DETECTED_GPUS >= 2 )); then
    NUM_GPUS=2
  else
    NUM_GPUS=1
  fi
fi

if [[ -z "${GPU_IDS:-}" ]]; then
  if (( NUM_GPUS > 1 )); then
    GPU_IDS="0,1"
  else
    GPU_IDS="0"
  fi
fi

if [[ -n "$SUBSET_SIZE" ]]; then
  SUBSET_ARGS=(--subset_size "$SUBSET_SIZE")
else
  SUBSET_ARGS=()
fi

if [[ "$TRAIN_DISABLE_EVAL" == "1" ]]; then
  TRAIN_EVAL_ARGS=(--disable_eval)
else
  TRAIN_EVAL_ARGS=(--eval_steps "$EVAL_STEPS")
fi

echo "===========================================" | tee -a "$LOGFILE"
echo "Experiment : $EXP_NAME" | tee -a "$LOGFILE"
echo "Run dir    : $RUN_DIR" | tee -a "$LOGFILE"
echo "Checkpoint : $CHECKPOINT_DIR" | tee -a "$LOGFILE"
echo "Results    : $RESULT_DIR" | tee -a "$LOGFILE"
echo "Model path : $MEDGEMMA_MODEL" | tee -a "$LOGFILE"
echo "GPUs       : NUM_GPUS=$NUM_GPUS, GPU_IDS=$GPU_IDS (detected=$DETECTED_GPUS)" | tee -a "$LOGFILE"
echo "Started at : $(date)" | tee -a "$LOGFILE"
echo "===========================================" | tee -a "$LOGFILE"

#############################
# TRAIN
#############################

echo "[1/3] TRAINING..." | tee -a "$LOGFILE"
conda activate "$TRAIN_ENV"

cd "$MODEL_DIR"
if (( NUM_GPUS > 1 )); then
  CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun --standalone --nproc_per_node="$NUM_GPUS" train.py \
    --decoder_model "$MEDGEMMA_MODEL" \
    --output_dir "$CHECKPOINT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --organ_chunk_size "$ORGAN_CHUNK_SIZE" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --dataloader_num_workers "$DATALOADER_WORKERS" \
    --use_4bit \
    --local_files_only \
    "${TRAIN_EVAL_ARGS[@]}" \
    "${SUBSET_ARGS[@]}" \
    2>&1 | tee -a "$LOGFILE"
else
  WANDB_MODE="$WANDB_MODE" python train.py \
    --decoder_model "$MEDGEMMA_MODEL" \
    --output_dir "$CHECKPOINT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --organ_chunk_size "$ORGAN_CHUNK_SIZE" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --dataloader_num_workers "$DATALOADER_WORKERS" \
    --use_4bit \
    --local_files_only \
    "${TRAIN_EVAL_ARGS[@]}" \
    "${SUBSET_ARGS[@]}" \
    2>&1 | tee -a "$LOGFILE"
fi

#############################
# EVALUATE
#############################

echo "[2/3] EVALUATION..." | tee -a "$LOGFILE"
CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" WANDB_MODE="$WANDB_MODE" python evaluate.py \
  --decoder_model "$MEDGEMMA_MODEL" \
  --checkpoint_dir "$CHECKPOINT_DIR/final" \
  --output_dir "$RESULT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --organ_chunk_size "$ORGAN_CHUNK_SIZE" \
  --use_4bit \
  --local_files_only \
  "${SUBSET_ARGS[@]}" \
  2>&1 | tee -a "$LOGFILE"

conda deactivate

#############################
# METRICS
#############################

echo "[3/3] METRICS..." | tee -a "$LOGFILE"
PRED_CSV="$RESULT_DIR/generated_reports_gemma.csv"

if [[ ! -f "$PRED_CSV" ]]; then
  echo "ERROR: Prediction CSV not found at $PRED_CSV" | tee -a "$LOGFILE"
  exit 1
fi

conda activate "$METRICS_ENV"
python "$BASE_DIR/radeval_metrics.py" \
  --input_csv "$PRED_CSV" \
  --ground_truth_json "$GROUND_TRUTH_JSON" \
  --output_dir "$RESULT_DIR" \
  --metrics all \
  2>&1 | tee -a "$LOGFILE"
conda deactivate

#############################
# DONE
#############################

echo "===========================================" | tee -a "$LOGFILE"
echo "FINISHED SUCCESSFULLY" | tee -a "$LOGFILE"
echo "Ended at  : $(date)" | tee -a "$LOGFILE"
echo "Log file  : $LOGFILE" | tee -a "$LOGFILE"
echo "===========================================" | tee -a "$LOGFILE"

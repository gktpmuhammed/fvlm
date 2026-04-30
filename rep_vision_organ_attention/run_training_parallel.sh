#!/bin/bash
# run_training_parallel.sh
# End-to-end pipeline: Train -> Evaluate -> RadEval Metrics
# Runs GPT-2 on GPU 0 and BioBART on GPU 1 in parallel

# Initialize Conda
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Global Settings
export WANDB_PROJECT="thesis_retrain_v3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_SYM_ROOT="${DATA_SYM_ROOT:-$PROJECT_ROOT/data_sym}"
VISION_ENCODER_PATH="${VISION_ENCODER_PATH:-$PROJECT_ROOT/checkpoints/model.pth}"
DATA_CSV="$DATA_SYM_ROOT/image_first_dataset.csv"
DATA_JSON="$DATA_SYM_ROOT/combined_desc_conc_v2.json"
CHECKPOINTS_ROOT="$SCRIPT_DIR/checkpoints"
RESULTS_ROOT="$SCRIPT_DIR/results"
METRICS_SCRIPT="${METRICS_SCRIPT:-$PROJECT_ROOT/rep_medgemma/radeval_metrics.py}"

# Function to run a full pipeline for a decoder model
run_pipeline() {
    DECODER_MODEL=$1
    GPU_ID=$2
    MODEL_NAME=$3   # Short name for directories
    
    CKPT_DIR="$CHECKPOINTS_ROOT/$MODEL_NAME"
    RESULT_DIR="$RESULTS_ROOT/$MODEL_NAME"
    
    echo "=================================================="
    echo "Starting pipeline for $MODEL_NAME ($DECODER_MODEL) on GPU $GPU_ID"
    echo "=================================================="
    
    cd "$SCRIPT_DIR" || exit

    # ---------------------------------------------------------
    # 1. Train (Env: fvlm_training_clean)
    # ---------------------------------------------------------
    conda activate fvlm_training_clean
    echo "[$(date '+%H:%M:%S')] [Env: fvlm_training_clean] Training $MODEL_NAME..."
    
    export WANDB_NAME="$MODEL_NAME"
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        --vision_encoder_path "$VISION_ENCODER_PATH" \
        --decoder_model "$DECODER_MODEL" \
        --num_epochs 1 \
        --batch_size 1 \
        --queries_per_organ 8 \
        --output_dir "$CKPT_DIR" \
        --csv_file "$DATA_CSV" \
        --json_file "$DATA_JSON" \
        > "$SCRIPT_DIR/${MODEL_NAME}_train.log" 2>&1
    
    TRAIN_EXIT=$?
    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "ERROR: Training failed for $MODEL_NAME (exit code: $TRAIN_EXIT)"
        echo "Check log: $SCRIPT_DIR/${MODEL_NAME}_train.log"
        conda deactivate
        return 1
    fi
    echo "[$(date '+%H:%M:%S')] Training complete for $MODEL_NAME."

    # ---------------------------------------------------------
    # 2. Evaluate (Env: fvlm_training_clean)
    # ---------------------------------------------------------
    echo "[$(date '+%H:%M:%S')] [Env: fvlm_training_clean] Evaluating $MODEL_NAME..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py \
        --vision_encoder_path "$VISION_ENCODER_PATH" \
        --decoder_model "$DECODER_MODEL" \
        --queries_per_organ 8 \
        --model_path "$CKPT_DIR/final_model" \
        --output_dir "$RESULT_DIR" \
        --csv_file "$DATA_CSV" \
        --json_file "$DATA_JSON" \
        > "$SCRIPT_DIR/${MODEL_NAME}_eval.log" 2>&1
    
    EVAL_EXIT=$?
    if [ $EVAL_EXIT -ne 0 ]; then
        echo "ERROR: Evaluation failed for $MODEL_NAME (exit code: $EVAL_EXIT)"
        echo "Check log: $SCRIPT_DIR/${MODEL_NAME}_eval.log"
        conda deactivate
        return 1
    fi
    echo "[$(date '+%H:%M:%S')] Evaluation complete for $MODEL_NAME."
    conda deactivate

    # ---------------------------------------------------------
    # 3. Metrics (Env: radevalmetrics)
    # ---------------------------------------------------------
    conda activate radevalmetrics
    echo "[$(date '+%H:%M:%S')] [Env: radevalmetrics] Calculating metrics for $MODEL_NAME..."
    
    PRED_FILE="$RESULT_DIR/generated_reports.csv"
    
    if [ -f "$PRED_FILE" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python "$METRICS_SCRIPT" \
            --input_csv "$PRED_FILE" \
            --ground_truth_json "$DATA_JSON" \
            --output_dir "$RESULT_DIR" \
            > "$SCRIPT_DIR/${MODEL_NAME}_metrics.log" 2>&1
        echo "[$(date '+%H:%M:%S')] Metrics complete for $MODEL_NAME."
    else
        echo "WARNING: Prediction file not found: $PRED_FILE"
    fi
    
    conda deactivate
    echo "Finished $MODEL_NAME"
}

# ===== GPU ASSIGNMENTS =====

run_gpu0() {
    run_pipeline "gpt2" 0 "gpt2_v3"
}

run_gpu1() {
    run_pipeline "GanjinZero/biobart-v2-base" 1 "biobart_v3"
}

# ===== START PARALLEL EXECUTION =====
echo "Starting Parallel Training: GPT-2 (GPU 0) + BioBART (GPU 1)"
echo "Logs: $SCRIPT_DIR/*_train.log, *_eval.log, *_metrics.log"
echo ""

run_gpu0 &
PID_0=$!

run_gpu1 &
PID_1=$!

wait $PID_0
STATUS_0=$?

wait $PID_1
STATUS_1=$?

echo ""
echo "=================================================="
echo "All jobs completed."
echo "  GPT-2:    $([ $STATUS_0 -eq 0 ] && echo '✅ SUCCESS' || echo '❌ FAILED')"
echo "  BioBART:  $([ $STATUS_1 -eq 0 ] && echo '✅ SUCCESS' || echo '❌ FAILED')"
echo "=================================================="

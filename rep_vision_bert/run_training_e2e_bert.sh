#!/bin/bash
# run_training_e2e_bert.sh
# End-to-end pipeline: Train -> Evaluate -> RadEval Metrics
# Runs Bio_ClinicalBERT

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Global Settings
export WANDB_PROJECT="thesis_retrain_v3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_CSV="/home/muhammedg/fvlm/data_sym/image_first_dataset.csv"
DATA_JSON="/home/muhammedg/fvlm/data_sym/combined_desc_conc_v2.json"
CHECKPOINTS_ROOT="$SCRIPT_DIR/checkpoints"
RESULTS_ROOT="$SCRIPT_DIR/results"

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
    # echo "[$(date '+%H:%M:%S')] [Env: fvlm_training_clean] Training $MODEL_NAME..."
    
    # export WANDB_NAME="$MODEL_NAME"
    # CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    #     --vision_encoder_path "/home/muhammedg/fvlm/checkpoints/model.pth" \
    #     --decoder_model "$DECODER_MODEL" \
    #     --num_epochs 1 \
    #     --batch_size 1 \
    #     --max_length 150 \
    #     --queries_per_organ 8 \
    #     --align_loss_weight 1.0 \
    #     --output_dir "$CKPT_DIR" \
    #     --csv_file "$DATA_CSV" \
    #     --json_file "$DATA_JSON" \
    #     > "$SCRIPT_DIR/${MODEL_NAME}_train.log" 2>&1
    
    # TRAIN_EXIT=$?
    # if [ $TRAIN_EXIT -ne 0 ]; then
    #     echo "ERROR: Training failed for $MODEL_NAME (exit code: $TRAIN_EXIT)"
    #     echo "Check log: $SCRIPT_DIR/${MODEL_NAME}_train.log"
    #     conda deactivate
    #     return 1
    # fi
    # echo "[$(date '+%H:%M:%S')] Training complete for $MODEL_NAME."

    # ---------------------------------------------------------
    # 2. Evaluate (Env: fvlm_training_clean)
    # ---------------------------------------------------------
    echo "[$(date '+%H:%M:%S')] [Env: fvlm_training_clean] Evaluating $MODEL_NAME..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py \
        --vision_encoder_path "/home/muhammedg/fvlm/checkpoints/model.pth" \
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
        CUDA_VISIBLE_DEVICES=$GPU_ID python "$SCRIPT_DIR/radeval_metrics.py" \
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

# ===== START EXECUTION =====
echo "Starting End-To-End Training: Bio_ClinicalBERT"
echo "Logs: $SCRIPT_DIR/bio_clinicalbert_v3_train.log, _eval.log, _metrics.log"
echo ""

# We'll run it on GPU 0. (Change to 1, 2, or 3 if you specifically want to use another GPU)
run_pipeline "emilyalsentzer/Bio_ClinicalBERT" 0 "bio_clinicalbert_v3"

echo ""
echo "=================================================="
echo "Job completed."
echo "=================================================="

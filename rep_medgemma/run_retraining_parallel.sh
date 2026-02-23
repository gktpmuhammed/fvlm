#!/bin/bash
# run_retraining_parallel.sh

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Global WandB Settings
export WANDB_PROJECT="thesis_retrain_v3"


# Function to run a full pipeline for a model
run_pipeline() {
    MODEL_DIR=$1
    GPU_ID=$2
    QUERIES=$3
    
    echo "=================================================="
    echo "Starting pipeline for $MODEL_DIR on GPU $GPU_ID"
    echo "=================================================="
    
    cd $MODEL_DIR || exit
    
    # ---------------------------------------------------------
    # 1. Train & Evaluate (Env: fvlm_training_clean)
    # ---------------------------------------------------------
    conda activate fvlm_training_clean
    echo "[Env: fvlm_training_clean] Training $MODEL_DIR..."
    
    export WANDB_NAME="$MODEL_DIR"
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        --num_epochs 1 \
        --batch_size 1 \
        --output_dir "../checkpoints_retrain/$MODEL_DIR" \
        > train.log 2>&1
    
    echo "[Env: fvlm_training_clean] Evaluating $MODEL_DIR..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py \
        --queries_per_organ $QUERIES \
        --batch_size 1 \
        --checkpoint_dir "../checkpoints_retrain/$MODEL_DIR/final" \
        --output_dir "../results_retrain/$MODEL_DIR" \
        > eval.log 2>&1
        
    conda deactivate

    # ---------------------------------------------------------
    # 3. Metrics (Env: radevalmetrics)
    # ---------------------------------------------------------
    conda activate radevalmetrics
    echo "[Env: radevalmetrics] Calculating metrics for $MODEL_DIR..."
    
    PRED_FILE="../results_retrain/$MODEL_DIR/generated_reports_gemma.csv"
    OUTPUT_DIR="../results_retrain/$MODEL_DIR"
    GT_FILE="../data_sym/combined_desc_conc_v2.json"
    
    if [ -f "$PRED_FILE" ]; then
        cd .. # Go back to root to find radeval_metrics.py
        CUDA_VISIBLE_DEVICES=$GPU_ID python radeval_metrics.py \
            --input_csv "$MODEL_DIR/$PRED_FILE" \
            --output_dir "$MODEL_DIR/$OUTPUT_DIR" \
            --ground_truth_json "$GT_FILE" \
            > "$MODEL_DIR/$OUTPUT_DIR/metrics.log" 2>&1
        cd $MODEL_DIR # Return to model dir
    else
        echo "WARNING: Prediction file not found: $PRED_FILE"
    fi
    
    conda deactivate
    
    cd .. # Return to root for next loop
    echo "Finished $MODEL_DIR"
}

# GPU 0 List (1 Query Models + others)
run_gpu0() {
    # # 1 Query Models
    # run_pipeline "medgemma_lora_vis_token_pos_embed" 0 1
    # run_pipeline "lora_with_vis_tokens_pos_embed_weight_loss" 0 1
    # run_pipeline "lora_with_vis_tokens_pos_embed_undersampling" 0 1
    # run_pipeline "curriculum_learning" 0 8
    run_pipeline "hard_example_mining" 0 8
}

# GPU 1 List (8 Query Models)
run_gpu1() {
    # run_pipeline "medical_vlm_8_tokens_full" 1 8
    # run_pipeline "medgemma_alignment_v1" 1 8
    # run_pipeline "medgemma_architecture_v3" 1 8
    # run_pipeline "medical_vlm_8_tokens_full_maxpool" 1 8
    # run_pipeline "multiscale_vit_fpn" 1 8
    run_pipeline "perceiver_resampler" 1 8
}

# Start Parallel Execution
echo "Starting Parallel Retraining..."

run_gpu0 &
PID_0=$!

run_gpu1 &
PID_1=$!

wait $PID_0
wait $PID_1

echo "All jobs completed."

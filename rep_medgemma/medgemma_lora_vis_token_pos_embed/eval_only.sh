#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fvlm_training_clean

cd /home/muhammedg/fvlm/rep_medgemma/medgemma_lora_vis_token_pos_embed

# # 1-token eval on GPU 0
# echo "Evaluating 1-token model on GPU 0..."
# CUDA_VISIBLE_DEVICES=0 python evaluate.py \
#     --queries_per_organ 1 \
#     --batch_size 1 \
#     --checkpoint_dir "../checkpoints_retrain/medgemma_lora_vis_token_pos_embed_1tokens/final" \
#     --output_dir "../results_retrain/medgemma_lora_vis_token_pos_embed_1tokens" \
#     > eval_1tokens_rerun.log 2>&1 &
# PID1=$!

# # 8-token eval on GPU 1
# echo "Evaluating 8-token model on GPU 1..."
# CUDA_VISIBLE_DEVICES=1 python evaluate.py \
#     --queries_per_organ 8 \
#     --batch_size 1 \
#     --checkpoint_dir "../checkpoints_retrain/medgemma_lora_vis_token_pos_embed_8tokens/final" \
#     --output_dir "../results_retrain/medgemma_lora_vis_token_pos_embed_8tokens" \
#     > eval_8tokens_rerun.log 2>&1 &
# PID2=$!

# echo "Waiting for both evaluations to finish..."
# wait $PID1
# echo "1-token evaluation done (exit code: $?)"
# wait $PID2
# echo "8-token evaluation done (exit code: $?)"

# Metrics
conda activate radevalmetrics
cd /home/muhammedg/fvlm/rep_medgemma

echo "Calculating metrics for 1-token..."
CUDA_VISIBLE_DEVICES=0 python radeval_metrics.py \
    --input_csv "results_retrain/medgemma_lora_vis_token_pos_embed_1tokens/generated_reports_gemma.csv" \
    --output_dir "results_retrain/medgemma_lora_vis_token_pos_embed_1tokens" \
    --ground_truth_json "/home/muhammedg/fvlm/data_sym/combined_desc_conc_v2.json" \
    > "results_retrain/medgemma_lora_vis_token_pos_embed_1tokens/metrics.log" 2>&1 &
PID3=$!

echo "Calculating metrics for 8-token..."
CUDA_VISIBLE_DEVICES=1 python radeval_metrics.py \
    --input_csv "results_retrain/medgemma_lora_vis_token_pos_embed_8tokens/generated_reports_gemma.csv" \
    --output_dir "results_retrain/medgemma_lora_vis_token_pos_embed_8tokens" \
    --ground_truth_json "/home/muhammedg/fvlm/data_sym/combined_desc_conc_v2.json" \
    > "results_retrain/medgemma_lora_vis_token_pos_embed_8tokens/metrics.log" 2>&1 &
PID4=$!

wait $PID3
wait $PID4
echo "All done!"

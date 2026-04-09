#!/bin/bash

# ==============================================================================
# Experiment 1: Main Performance Table (End-to-End Speedup)
# Goal: Demonstrate that Fed-SD provides the highest wall-clock speedup.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Default Hyperparameters (Aligned with OSD ICML 2024 defaults)
K=5
I=8
LR="1e-5"
TARGET_MODEL="vicuna-7b"
DRAFT_MODEL="llama-160m"
NUM_SAMPLES=50
EXP_NAME="Exp1_MainTable"
GPUS="0"

# Parse command-line arguments for easy extensibility
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --k) K="$2"; shift ;;
        --i) I="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --target_model) TARGET_MODEL="$2"; shift ;;
        --draft_model) DRAFT_MODEL="$2"; shift ;;
        --num_samples) NUM_SAMPLES="$2"; shift ;;
        --exp_name) EXP_NAME="$2"; shift ;;
        --gpus) GPUS="$2"; shift ;;
        --use_wandb) USE_WANDB="--use_wandb" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "============================================================"
echo "Starting Experiment 1: Main Performance Table"
echo "============================================================"
echo "Target Model : $TARGET_MODEL"
echo "Draft Model  : $DRAFT_MODEL"
echo "Speculation K: $K"
echo "Update Int I : $I"
echo "Learning Rate: $LR"
echo "GPUs         : $GPUS"
echo "W&B Enabled  : ${USE_WANDB:-"No"}"
echo "============================================================"

# Ensure the logs directory exists
mkdir -p logs

# Run the python evaluation pipeline
CUDA_VISIBLE_DEVICES=$GPUS python -m run_experiments \
    --exp_name "${EXP_NAME}_K${K}_I${I}" \
    --target_model "$TARGET_MODEL" \
    --draft_model "$DRAFT_MODEL" \
    --datasets spider gsm8k code_search_python alpaca_finance \
    --methods target_only draft_only vanilla_sd osd ours_fed_sd \
    --k "$K" \
    --update_interval "$I" \
    --lr "$LR" \
    --num_samples "$NUM_SAMPLES" \
    $USE_WANDB 2>&1 | tee "logs/${EXP_NAME}_K${K}_I${I}_$(date +%Y%m%d_%H%M%S).log"

echo "Experiment 1 Completed Successfully!"
#!/bin/bash

echo "Launching Parallel Ablation Studies across 6 GPUs..."
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/private/wsx/federated:$PYTHONPATH
export PYTHONPATH=/private/wsx/federated/eva/OSD:$PYTHONPATH
export PYTHONPATH=/private/wsx/federated/eva:$PYTHONPATH
export WANDB_API_KEY=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

# Launch K=4 on GPUs 0,1
nohup ./run_exp1_main_table.sh --k 4 --exp_name "Ablation_K4" --gpus 0,1 --use_wandb > logs/k4_out.log 2>&1 &

# Launch K=5 on GPUs 2,3
nohup ./run_exp1_main_table.sh --k 5 --exp_name "Ablation_K5" --gpus 2,3 --use_wandb > logs/k5_out.log 2>&1 &

# Launch K=8 on GPUs 4,5
nohup ./run_exp1_main_table.sh --k 8 --exp_name "Ablation_K8" --gpus 4,5 --use_wandb > logs/k8_out.log 2>&1 &

echo "All 3 experiments dispatched! Use 'tail -f logs/k5_out.log' to monitor."
wait
echo "All parallel runs finished."
#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=nlprx-lab
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH -J gpqa_qwen3_4b
#SBATCH -o logs/xreasoning_qwen3_4b_instruct_gpqa_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

# Evaluate GPQA for Qwen3-4B-Instruct (Non-ST)
python run_xreasoning_evaluation.py \
    --dataset "gpqa" \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1

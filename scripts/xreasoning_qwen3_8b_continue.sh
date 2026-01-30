#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=nlprx-lab
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH -J xr_qwen3_8b_cont
#SBATCH -o logs/xreasoning_qwen3_8b_continue_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

# Complete AIME 2024 runs 3-5 for Qwen3-8B
python run_xreasoning_evaluation.py \
    --dataset "aime2024" \
    --model_name "Qwen/Qwen3-8B" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1

# Evaluate AIME 2025 (5 runs)
python run_xreasoning_evaluation.py \
    --dataset "aime2025" \
    --model_name "Qwen/Qwen3-8B" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1

# Evaluate GPQA (1 run)
python run_xreasoning_evaluation.py \
    --dataset "gpqa" \
    --model_name "Qwen/Qwen3-8B" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1

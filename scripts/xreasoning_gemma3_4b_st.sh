#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -J xr_g3_4b_st
#SBATCH -o logs/xreasoning_gemma3_4b_st_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

export HF_TOKEN="${HF_TOKEN}"

# Evaluate AIME 2024 with soft thinking
python run_xreasoning_evaluation.py \
    --dataset "aime2024" \
    --model_name "google/gemma-3-4b-it" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1 \
    --enable_soft_thinking

# Evaluate AIME 2025 with soft thinking
python run_xreasoning_evaluation.py \
    --dataset "aime2025" \
    --model_name "google/gemma-3-4b-it" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1 \
    --enable_soft_thinking

# Evaluate GPQA with soft thinking
python run_xreasoning_evaluation.py \
    --dataset "gpqa" \
    --model_name "google/gemma-3-4b-it" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1 \
    --enable_soft_thinking

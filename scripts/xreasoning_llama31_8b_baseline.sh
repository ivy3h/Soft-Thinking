#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -J xr_l31_8b_base
#SBATCH -o logs/xreasoning_llama31_8b_baseline_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

export HF_TOKEN="${HF_TOKEN}"

# Evaluate AIME 2024
python run_xreasoning_evaluation.py \
    --dataset "aime2024" \
    --model_name "meta-llama/Llama-3.1-8B" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1

# Evaluate AIME 2025
python run_xreasoning_evaluation.py \
    --dataset "aime2025" \
    --model_name "meta-llama/Llama-3.1-8B" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1

# Evaluate GPQA
python run_xreasoning_evaluation.py \
    --dataset "gpqa" \
    --model_name "meta-llama/Llama-3.1-8B" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1
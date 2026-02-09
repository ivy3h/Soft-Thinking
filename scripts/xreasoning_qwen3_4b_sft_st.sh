#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -J xr_q3_4b_sft_st
#SBATCH -o logs/xreasoning_qwen3_4b_sft_st_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

# Evaluate AIME 2024
python run_xreasoning_evaluation.py \
    --dataset "aime2024" \
    --model_name "/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-4b/full/sft_gsm8k/checkpoint-2299" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1 \
    --enable_soft_thinking

# Evaluate AIME 2025
python run_xreasoning_evaluation.py \
    --dataset "aime2025" \
    --model_name "/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-4b/full/sft_gsm8k/checkpoint-2299" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1 \
    --enable_soft_thinking

# Evaluate GPQA
python run_xreasoning_evaluation.py \
    --dataset "gpqa" \
    --model_name "/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-4b/full/sft_gsm8k/checkpoint-2299" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1 \
    --enable_soft_thinking

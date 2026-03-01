#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -x spd-13,samantha
#SBATCH -J xr_q3_8b_sft_a25st
#SBATCH -o logs/xreasoning_qwen3_8b_sft_aime25_st_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

# Evaluate AIME 2025 - Soft-Thinking only
python run_xreasoning_evaluation.py \
    --dataset "aime2025" \
    --model_name "/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-8b/full/sft_gsm8k" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 2 \
    --num_samples 1 \
    --enable_soft_thinking

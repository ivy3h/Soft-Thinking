#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J xr_aime2024_qwen3_4b_instruct_2507_st
#SBATCH -o /nethome/jhe478/flash/Soft-Thinking/logs/xr_aime2024_qwen3_4b_instruct_2507_st_%j.log

# Evaluate Qwen3-4B-Instruct-2507 on XReasoning-AIME2024 (Soft-Thinking)

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

export TRANSFORMERS_CACHE=/coc/pskynet6/jhe478/huggingface/transformers
export HF_HOME=/coc/pskynet6/jhe478/huggingface
export HF_DATASETS_CACHE=/coc/pskynet6/jhe478/huggingface/datasets

python run_xreasoning_evaluation.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --dataset aime2024 \
    --num_runs 5 \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.6 \
    --num_gpus 1 \
    --num_samples 1 \
    --enable_soft_thinking \

echo "Evaluation completed!"

#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH -J mgsm_g3_27b_st
#SBATCH -o logs/mgsm_gemma3_27b_st_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

# Use local HF cache
export HF_HOME=/coc/pskynet6/jhe478/huggingface
# export HF_TOKEN="your_token_here"

# Very long watchdog timeout for FlashInfer JIT compilation
python run_mgsm_evaluation.py \
    --model_name "google/gemma-3-27b-it" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --start_idx 0 \
    --end_idx 250 \
    --num_gpus 4 \
    --num_samples 1 \
    --single_engine \
    --watchdog_timeout 3600 \
    --enable_soft_thinking \
    --resume

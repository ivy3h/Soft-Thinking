#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -J mgsm_g3_12b_base
#SBATCH -o logs/mgsm_gemma3_12b_baseline_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

# HF_TOKEN should be set in environment or ~/.huggingface/token
export HF_TOKEN="${HF_TOKEN}"

python run_mgsm_evaluation.py \
    --model_name "google/gemma-3-12b-it" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.6 \
    --start_idx 0 \
    --end_idx 250 \
    --num_gpus 2 \
    --num_samples 1 \
    --single_engine \
    --resume

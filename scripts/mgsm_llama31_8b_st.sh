#!/bin/bash
#SBATCH -p nlprx-lab
#SBATCH --account=nlprx-lab
#SBATCH -t 04:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -J mgsm_l31_8b_st
#SBATCH -o logs/mgsm_llama31_8b_st_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

export HF_TOKEN="${HF_TOKEN}"

python run_mgsm_evaluation.py \
    --model_name "meta-llama/Llama-3.1-8B" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.6 \
    --start_idx 0 \
    --end_idx 250 \
    --num_gpus 1 \
    --num_samples 1 \
    --single_engine \
    --enable_soft_thinking \
    --resume
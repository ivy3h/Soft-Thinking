#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=nlprx-lab
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH -J qwen3_8b_te
#SBATCH -o logs/mgsm_qwen3_8b_missing_te_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

# Complete missing 'te' language for Qwen3-8B Non-ST
python run_mgsm_evaluation.py \
    --model_name "Qwen/Qwen3-8B" \
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
    --resume

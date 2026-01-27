#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=nlprx-lab
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH -J st_qwen3_8b_base_new
#SBATCH -o logs/mgsm_qwen3_8b_base_st_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

# Start Qwen3-8B-Base MGSM with Soft-Thinking (all 11 languages)
python run_mgsm_evaluation.py \
    --model_name "Qwen/Qwen3-8B-Base" \
    --max_generated_tokens 32768 \
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

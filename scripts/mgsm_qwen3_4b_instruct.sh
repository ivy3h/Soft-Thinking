#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=nlprx-lab
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH -J mgsm_qwen3_4b_instruct
#SBATCH -o mgsm_qwen3_4b_instruct_%j.log

source ~/.bashrc
conda activate tinker

python run_mgsm_evaluation.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --start_idx 0 \
    --end_idx 250 \
    --num_gpus 1 \
    --num_samples 1

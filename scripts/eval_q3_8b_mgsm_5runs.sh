#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -x starrysky,heistotron,deebot,nestor,cheetah,chitti,tachikoma,optimistprime,uniblab,puma,perseverance,clippy,xaea-12,megazord,trublu,baymax
#SBATCH -J mgsm_8b_5r
#SBATCH -o logs/eval_q3_8b_mgsm_5runs_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

MODEL="Qwen/Qwen3-8B"

python run_mgsm_evaluation.py \
    --model_name "$MODEL" \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1 \
    --num_runs 5 \
    --resume

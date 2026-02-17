#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -x starrysky,heistotron,deebot,nestor,cheetah,chitti,tachikoma,optimistprime,uniblab,puma,perseverance,clippy,xaea-12,megazord,trublu,ig-88,brainiac,randotron,consu,chappie,cyborg,spot
#SBATCH -J xr_ms1k_a25
#SBATCH -o logs/eval_q3_4b_ms1k_aime25_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

python run_xreasoning_evaluation.py \
    --dataset "aime2025" \
    --model_name "/coc/pskynet6/jhe478/Soft-Thinking/merged_models/Qwen3-4B-SFT-MS1K" \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.9 \
    --num_gpus 1 \
    --num_samples 1 \
    --single_engine

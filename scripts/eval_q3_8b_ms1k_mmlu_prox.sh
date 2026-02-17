#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -x starrysky,heistotron,deebot,nestor,cheetah,chitti,tachikoma,optimistprime,uniblab,puma,perseverance,clippy,xaea-12,megazord,trublu,baymax,spd-13
#SBATCH -J mmlu_8bms1k
#SBATCH -o logs/eval_q3_8b_ms1k_mmlu_prox_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

MODEL="/coc/pskynet6/jhe478/LlamaFactory/saves/symlinks/Qwen3-8B-SFT-MS1K"

python run_xreasoning_evaluation.py \
    --dataset "mmlu_prox" \
    --model_name "$MODEL" \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.85 \
    --num_gpus 2 \
    --num_samples 1 \
    --single_engine

#!/bin/bash
#SBATCH -p nlprx-lab
#SBATCH --account=nlprx-lab
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -x ig-88,brainiac,randotron,consu,chappie,cyborg,spot,sonny,major,gundam,omgwth,protocol,conroy,baymax,spd-13,puma,samantha,shakey,crushinator
#SBATCH -J inst_a24_nl
#SBATCH -o /coc/pskynet6/jhe478/Soft-Thinking/logs/eval_q3_4b_instruct_aime24_nlprx_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

python run_xreasoning_evaluation.py \
    --dataset "aime2024" \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.85 \
    --num_gpus 4 \
    --num_samples 1 \
    --single_engine

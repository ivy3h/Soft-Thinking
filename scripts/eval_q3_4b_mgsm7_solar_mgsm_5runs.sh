#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -x starrysky,heistotron,deebot,nestor,cheetah,chitti,tachikoma,optimistprime,uniblab,puma,perseverance,clippy,xaea-12,megazord,trublu,ig-88,brainiac,randotron,consu,chappie,cyborg,spot,sonny,major,gundam,omgwth,protocol,conroy,baymax,spd-13,samantha,robby,shakey,crushinator
#SBATCH -J mgsm_4bm7s_5r
#SBATCH -o logs/eval_q3_4b_mgsm7_solar_mgsm_5runs_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

MODEL="/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-4b/full/sft_ms1k_mgsm7_solar_val"

python run_mgsm_evaluation.py \
    --model_name "$MODEL" \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.85 \
    --num_gpus 2 \
    --num_samples 1 \
    --num_runs 5 \
    --single_engine \
    --resume

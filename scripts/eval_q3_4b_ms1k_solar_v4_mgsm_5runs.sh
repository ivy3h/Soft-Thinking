#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -x ig-88,brainiac,randotron,consu,chappie,cyborg,spot,sonny,major,gundam,omgwth,protocol,robby,spd-13,samantha,deebot,shakey,tachikoma,nestor,crushinator,trublu,chitti,hk47,megabot,perseverance,megazord,johnny5,dave,kitt
#SBATCH -J mgsm_v4
#SBATCH -o logs/eval_q3_4b_ms1k_solar_v4_mgsm_5runs_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

MODEL="/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-4b/full/sft_ms1k_solar_val_v4"

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
    --num_runs 10 \
    --languages en zh fr ja sw te th \
    --single_engine \
    --resume

#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -x ig-88,brainiac,randotron,consu,chappie,cyborg,spot,sonny,major,gundam,omgwth,spd-13,tachikoma,robby,protocol,crushinator
#SBATCH -J aime24_v3
#SBATCH -o logs/eval_solar_v3_aime2024_5runs_%j.log

source ~/.bashrc
conda activate st
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOB_ID

cd /coc/pskynet6/jhe478/Soft-Thinking

MODEL="/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-4b/full/sft_ms1k_solar_val_v3"

python run_xreasoning_evaluation.py \
    --dataset "aime2024" \
    --model_name "$MODEL" \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.85 \
    --num_gpus 1 \
    --num_samples 1 \
    --num_runs 10 \
    --languages en zh fr ja sw te th \
    --single_engine

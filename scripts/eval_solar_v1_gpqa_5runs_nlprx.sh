#!/bin/bash
#SBATCH -p nlprx-lab
#SBATCH --account=nlprx-lab
#SBATCH --qos long
#SBATCH -t 7-00:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -x bishop,perseverance,megazord,johnny5,crushinator,omgwth,robby,protocol,tachikoma
#SBATCH -J gpqa_v1_nl
#SBATCH -o logs/eval_solar_v1_gpqa_5runs_nlprx_%j.log

source ~/.bashrc
conda activate st
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOB_ID

cd /coc/pskynet6/jhe478/Soft-Thinking

MODEL="/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-4b/full/sft_ms1k_solar_val"

python run_xreasoning_evaluation.py \
    --dataset "gpqa" \
    --model_name "$MODEL" \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.85 \
    --num_gpus 1 \
    --num_samples 1 \
    --num_runs 5 \
    --languages en zh fr ja sw te th \
    --single_engine

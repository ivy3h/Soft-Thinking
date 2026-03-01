#!/bin/bash
#SBATCH -p nlprx-lab
#SBATCH --account=nlprx-lab
#SBATCH --qos short
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -x bishop,baymax,samantha,deebot,nestor,chitti,crushinator,trublu,johnny5,perseverance,megazord,hk47,megabot,dave
#SBATCH -J mgsm_v3_nl
#SBATCH -o logs/eval_solar_v3_mgsm_5runs_nlprx_%j.log

source ~/.bashrc
conda activate st
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOB_ID

cd /coc/pskynet6/jhe478/Soft-Thinking

MODEL="/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-4b/full/sft_ms1k_solar_val_v3"

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
    --languages en zh fr ja sw te th \
    --single_engine \
    --resume

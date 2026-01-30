#!/bin/bash
#SBATCH -p nlprx-lab
#SBATCH --account=nlprx-lab
#SBATCH -t 04:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J eval_sft_cot
#SBATCH -o /nethome/jhe478/flash/Soft-Thinking/logs/eval_sft_mgsm_cot_%j.log

# Evaluate STA-SFT fine-tuned model on MGSM (CoT baseline, no soft-thinking)
# Base: Qwen3-4B-Instruct-2507, SFT checkpoint-934

source ~/.bashrc

cd /coc/pskynet6/jhe478/Soft-Thinking

export TRANSFORMERS_CACHE=/coc/pskynet6/jhe478/huggingface/transformers
export HF_HOME=/coc/pskynet6/jhe478/huggingface
export HF_DATASETS_CACHE=/coc/pskynet6/jhe478/huggingface/datasets

ADAPTER_PATH="/coc/pskynet6/jhe478/soft-token-alignment/outputs/sft_qwen3_4b/checkpoint-934"
MERGED_DIR="/coc/pskynet6/jhe478/Soft-Thinking/merged_models/sft_qwen3_4b_ckpt934"

echo "=========================================="
echo "Step 1: Merge LoRA adapter"
echo "=========================================="

mkdir -p "$(dirname "$MERGED_DIR")"

# Use tinker env for merge (has peft)
conda activate tinker
python merge_lora_and_eval.py \
    --adapter_path "$ADAPTER_PATH" \
    --output_dir "$MERGED_DIR" \
    --cache_dir /coc/pskynet6/jhe478/huggingface

echo "=========================================="
echo "Step 2: MGSM CoT Baseline Evaluation"
echo "=========================================="

# Switch to st env for evaluation
conda activate st

python run_mgsm_evaluation.py \
    --model_name "$MERGED_DIR" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.6 \
    --start_idx 0 \
    --end_idx 250 \
    --num_gpus 1 \
    --num_samples 1 \
    --single_engine \
    --resume

echo "Evaluation completed!"

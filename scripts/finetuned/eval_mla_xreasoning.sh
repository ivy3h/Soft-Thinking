#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J eval_mla_xr
#SBATCH -o logs/eval_mla_xreasoning_%j.log

# Evaluate MLA fine-tuned model on XReasoning (AIME2024 + AIME2025 + GPQA)
# Model: MLA on Qwen3-4B-Instruct-2507 (checkpoint-12000 from latest resume)

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

export TRANSFORMERS_CACHE=/coc/pskynet6/jhe478/huggingface/transformers
export HF_HOME=/coc/pskynet6/jhe478/huggingface
export HF_DATASETS_CACHE=/coc/pskynet6/jhe478/huggingface/datasets

ADAPTER_PATH="/coc/pskynet6/jhe478/tinker-cookbook/outputs/mla_qwen3_4b_resume_20260128_221632/checkpoint-12000"
MERGED_DIR="/coc/pskynet6/jhe478/Soft-Thinking/merged_models/mla_qwen3_4b_ckpt12000"

echo "=========================================="
echo "Step 1: Merge LoRA adapter (if needed)"
echo "=========================================="

python merge_lora_and_eval.py \
    --adapter_path "$ADAPTER_PATH" \
    --output_dir "$MERGED_DIR" \
    --cache_dir /coc/pskynet6/jhe478/huggingface

echo "=========================================="
echo "Step 2: XReasoning Evaluation - AIME2024"
echo "=========================================="

python run_xreasoning_evaluation.py \
    --dataset aime2024 \
    --model_name "$MERGED_DIR" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1

echo "=========================================="
echo "Step 3: XReasoning Evaluation - AIME2025"
echo "=========================================="

python run_xreasoning_evaluation.py \
    --dataset aime2025 \
    --model_name "$MERGED_DIR" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1

echo "=========================================="
echo "Step 4: XReasoning Evaluation - GPQA"
echo "=========================================="

python run_xreasoning_evaluation.py \
    --dataset gpqa \
    --model_name "$MERGED_DIR" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1

echo "All XReasoning evaluations completed!"

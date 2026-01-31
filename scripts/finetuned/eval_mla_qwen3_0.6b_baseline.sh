#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J eval_mla_0.6b_baseline
#SBATCH -o logs/eval_mla_qwen3_0.6b_baseline_%j.log

# Evaluate MLA fine-tuned Qwen3-0.6B model on MGSM (Baseline - No Soft-Thinking)
# Model: MLA on Qwen/Qwen3-0.6B (Job 2403716 - completed)
# Training output: /coc/pskynet6/jhe478/tinker-cookbook/outputs/mla_qwen3_0.6b_20260130_125635/final/

echo "=========================================="
echo "MLA Qwen3-0.6B Baseline Evaluation"
echo "Model trained with MLA (Job 2403716)"
echo "Evaluation: MGSM without soft-thinking"
echo "=========================================="

source ~/.bashrc

cd /coc/pskynet6/jhe478/Soft-Thinking

export TRANSFORMERS_CACHE=/coc/pskynet6/jhe478/huggingface/transformers
export HF_HOME=/coc/pskynet6/jhe478/huggingface
export HF_DATASETS_CACHE=/coc/pskynet6/jhe478/huggingface/datasets

ADAPTER_PATH="/coc/pskynet6/jhe478/tinker-cookbook/outputs/mla_qwen3_0.6b_20260130_125635/final"
MERGED_DIR="/coc/pskynet6/jhe478/Soft-Thinking/merged_models/mla_qwen3_0.6b_final"

echo "=========================================="
echo "Step 1: Merge LoRA adapter (using tinker env with peft)"
echo "Adapter: $ADAPTER_PATH"
echo "Output: $MERGED_DIR"
echo "=========================================="

conda activate tinker
python merge_lora_and_eval.py \
    --adapter_path "$ADAPTER_PATH" \
    --output_dir "$MERGED_DIR" \
    --cache_dir /coc/pskynet6/jhe478/huggingface

echo "=========================================="
echo "Step 2: MGSM Baseline Evaluation (using st env)"
echo "=========================================="

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

echo "Baseline evaluation completed!"

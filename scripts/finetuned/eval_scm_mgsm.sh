#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J eval_scm_mgsm
#SBATCH -o logs/eval_scm_mgsm_%j.log

# Evaluate SCM fine-tuned model on MGSM
# Model: SCM-GRPO on Qwen3-4B-Instruct-2507 (checkpoint-50 from resume)

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

export TRANSFORMERS_CACHE=/coc/pskynet6/jhe478/huggingface/transformers
export HF_HOME=/coc/pskynet6/jhe478/huggingface
export HF_DATASETS_CACHE=/coc/pskynet6/jhe478/huggingface/datasets

ADAPTER_PATH="/coc/pskynet6/jhe478/tinker-cookbook/outputs/scm_qwen3_4b_resume_20260128_190147/checkpoint-50"
MERGED_DIR="/coc/pskynet6/jhe478/Soft-Thinking/merged_models/scm_qwen3_4b_ckpt50"

echo "=========================================="
echo "Step 1: Merge LoRA adapter"
echo "Adapter: $ADAPTER_PATH"
echo "Output: $MERGED_DIR"
echo "=========================================="

python merge_lora_and_eval.py \
    --adapter_path "$ADAPTER_PATH" \
    --output_dir "$MERGED_DIR" \
    --cache_dir /coc/pskynet6/jhe478/huggingface

echo "=========================================="
echo "Step 2: MGSM Evaluation"
echo "=========================================="

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

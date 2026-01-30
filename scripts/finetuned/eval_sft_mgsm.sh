#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J eval_sft_mgsm
#SBATCH -o logs/eval_sft_mgsm_%j.log

# Evaluate STA-SFT fine-tuned model on MGSM
# Model: SFT (Soft Token Alignment) on Qwen3-4B-Instruct-2507
# NOTE: Update ADAPTER_PATH once SFT training completes and produces checkpoints

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

export TRANSFORMERS_CACHE=/coc/pskynet6/jhe478/huggingface/transformers
export HF_HOME=/coc/pskynet6/jhe478/huggingface
export HF_DATASETS_CACHE=/coc/pskynet6/jhe478/huggingface/datasets

# TODO: Update this path once SFT training produces a checkpoint
ADAPTER_PATH="/coc/pskynet6/jhe478/soft-token-alignment/outputs/sft_qwen3_4b/checkpoint-XXXX"
MERGED_DIR="/coc/pskynet6/jhe478/Soft-Thinking/merged_models/sft_qwen3_4b"

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: Adapter path does not exist: $ADAPTER_PATH"
    echo "SFT training may not have completed yet. Update ADAPTER_PATH and resubmit."
    exit 1
fi

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

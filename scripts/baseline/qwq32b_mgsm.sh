#!/bin/bash
# MGSM (Multilingual Grade School Math) Baseline Evaluation Script
# Evaluates on all 11 languages without soft thinking

python ./models/download.py --model_name "Qwen/QwQ-32B"

export OPENAI_API_KEY=""

# Run MGSM evaluation on all languages without soft thinking
python run_mgsm_evaluation.py \
    --model_name "./models/Qwen/QwQ-32B" \
    --max_topk 10 \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --start_idx 0 \
    --end_idx 250 \
    --num_gpus 8 \
    --num_samples 1 \
    --use_llm_judge \
    --judge_model_name "gpt-4.1-2025-04-14"

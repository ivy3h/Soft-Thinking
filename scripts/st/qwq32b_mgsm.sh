#!/bin/bash
# MGSM (Multilingual Grade School Math) Evaluation Script
# Evaluates on all 11 languages: en, es, fr, de, ru, zh, ja, th, sw, bn, te

python ./models/download.py --model_name "Qwen/QwQ-32B"

export OPENAI_API_KEY=""

# Run MGSM evaluation on all languages with soft thinking enabled
python run_mgsm_evaluation.py \
    --model_name "./models/Qwen/QwQ-32B" \
    --max_topk 10 \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --after_thinking_temperature 0.6 \
    --after_thinking_top_p 0.95 \
    --after_thinking_top_k 30 \
    --after_thinking_min_p 0.0 \
    --early_stopping_entropy_threshold 0.01 \
    --early_stopping_length_threshold 256 \
    --mem_fraction_static 0.8 \
    --start_idx 0 \
    --end_idx 250 \
    --num_gpus 8 \
    --num_samples 1 \
    --enable_soft_thinking \
    --use_llm_judge \
    --judge_model_name "gpt-4.1-2025-04-14"

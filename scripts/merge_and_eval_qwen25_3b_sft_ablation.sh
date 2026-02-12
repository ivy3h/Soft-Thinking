#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -J eval_q25_sft_abl
#SBATCH -o logs/eval_qwen25_3b_sft_ablation_%j.log

source ~/.bashrc

cd /coc/pskynet6/jhe478/Soft-Thinking

MERGED_DIR="/coc/pskynet6/jhe478/Soft-Thinking/merged_models/Qwen2.5-3B-SFT-ABLATION"

# Step 1: Merge LoRA using llama-factory env (has peft)
if [ ! -f "$MERGED_DIR/config.json" ]; then
    echo "Merging LoRA weights..."
    conda activate /coc/pskynet6/jhe478/envs/llama-factory-new
    python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, '/nethome/jhe478/flash/LlamaFactory/saves/qwen25-3b/lora/sft_gsm8k_solar/checkpoint-575')
merged = model.merge_and_unload()
merged.save_pretrained('$MERGED_DIR')
tokenizer = AutoTokenizer.from_pretrained('/nethome/jhe478/flash/LlamaFactory/saves/qwen25-3b/lora/sft_gsm8k_solar/checkpoint-575')
tokenizer.save_pretrained('$MERGED_DIR')
print('Merge complete!')
"
else
    echo "Merged model already exists, skipping merge."
fi

# Step 2: Run MGSM evaluation using st env
conda activate st
python run_mgsm_evaluation.py \
    --model_name "$MERGED_DIR" \
    --max_generated_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1

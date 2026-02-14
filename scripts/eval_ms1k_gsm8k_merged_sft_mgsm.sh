#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -x starrysky,heistotron,deebot,nestor,cheetah,chitti,tachikoma,optimistprime,uniblab,puma,perseverance,clippy,xaea-12,megazord,trublu,ig-88
#SBATCH -J eval_merged_mgsm
#SBATCH -o logs/eval_ms1k_gsm8k_merged_sft_mgsm_%j.log

source ~/.bashrc
cd /coc/pskynet6/jhe478/Soft-Thinking

MERGED_DIR="/coc/pskynet6/jhe478/Soft-Thinking/merged_models/Qwen3-4B-MS1K-GSM8K-Merged-SFT"

if [ ! -f "$MERGED_DIR/config.json" ]; then
    echo "Merging LoRA weights..."
    conda activate /coc/pskynet6/jhe478/envs/llama-factory-new
    python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B-Instruct-2507', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, '/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-4b/lora/sft_ms1k_gsm8k_merged')
merged = model.merge_and_unload()
merged.save_pretrained('$MERGED_DIR')
tokenizer = AutoTokenizer.from_pretrained('/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-4b/lora/sft_ms1k_gsm8k_merged')
tokenizer.save_pretrained('$MERGED_DIR')
print('Merge complete!')
"
else
    echo "Merged model already exists, skipping merge."
fi

conda activate st
python run_mgsm_evaluation.py \
    --model_name "$MERGED_DIR" \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --mem_fraction_static 0.8 \
    --num_gpus 1 \
    --num_samples 1

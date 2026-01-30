#!/bin/bash
#SBATCH -p nlprx-lab
#SBATCH --account=nlprx-lab
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J remerge_test
#SBATCH -o /nethome/jhe478/flash/Soft-Thinking/logs/remerge_test_%j.log

# Re-merge MLA model with fp16 on GPU and test

source ~/.bashrc
conda activate tinker

cd /coc/pskynet6/jhe478/Soft-Thinking

export TRANSFORMERS_CACHE=/coc/pskynet6/jhe478/huggingface/transformers
export HF_HOME=/coc/pskynet6/jhe478/huggingface

echo "=========================================="
echo "Re-merging MLA model with fp16 on GPU"
echo "=========================================="

ADAPTER_PATH="/coc/pskynet6/jhe478/tinker-cookbook/outputs/mla_qwen3_4b_resume_20260129_022440/checkpoint-epoch-1"
OUTPUT_DIR="/coc/pskynet6/jhe478/Soft-Thinking/merged_models/mla_qwen3_4b_epoch1_fp16"

python remerge_lora_fp16.py \
    --adapter_path "$ADAPTER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir /coc/pskynet6/jhe478/huggingface

echo ""
echo "=========================================="
echo "Testing merged model"
echo "=========================================="

# Test the merged model
cd /coc/pskynet6/jhe478/soft-token-alignment

python3 << 'EOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/coc/pskynet6/jhe478/Soft-Thinking/merged_models/mla_qwen3_4b_epoch1_fp16"
print(f"\nTesting: {model_path}\n")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model (fp16, cuda:0)...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    trust_remote_code=True,
)
model.eval()

text = "Hello, world!"
encoded = tokenizer(text, return_tensors="pt").to("cuda:0")
input_ids = encoded["input_ids"]

print("Running forward pass with output_hidden_states=True...")
with torch.no_grad():
    outputs = model(input_ids=input_ids, output_hidden_states=True)

print(f"Number of layers: {len(outputs.hidden_states)}")

has_nan = False
for i, hs in enumerate(outputs.hidden_states):
    if torch.isnan(hs).any():
        print(f"  Layer {i}: NaN [FAIL]")
        has_nan = True
        if i > 5:
            print("  ... (stopping check)")
            break
    elif i == 0 or i == len(outputs.hidden_states) - 1:
        print(f"  Layer {i}: {hs.shape} min={hs.min():.3f} max={hs.max():.3f} [OK]")

if not has_nan:
    print("  ... (all intermediate layers OK)")

logits_has_nan = torch.isnan(outputs.logits).any()
print(f"\nLogits: {'NaN [FAIL]' if logits_has_nan else 'OK'}")

print("\n" + "="*50)
if has_nan or logits_has_nan:
    print("Result: FAIL - NaN detected")
else:
    print("Result: PASS - No NaN, model is valid!")
print("="*50)
EOF

echo ""
echo "Test completed!"

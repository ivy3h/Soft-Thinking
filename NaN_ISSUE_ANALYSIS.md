# NaN Issue in Fine-tuned Model Pilot Experiments

## Problem Summary

The pilot experiments for fine-tuned models (SFT and MLA) were producing NaN values for **all samples** (not just samples 46-49) across all languages, resulting in zero metrics and failed visualizations.

## Root Cause

The pilot experiment script was attempting to load **LoRA checkpoint directories** as full models:
- SFT checkpoint: `/coc/pskynet6/jhe478/soft-token-alignment/outputs/sft_qwen3_4b/checkpoint-934`
- MLA checkpoint: `/coc/pskynet6/jhe478/tinker-cookbook/outputs/mla_qwen3_4b_resume_20260129_022440/checkpoint-epoch-1`

The script uses:
```python
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,  # This was a LoRA checkpoint path
    cache_dir=args.cache_dir,
    torch_dtype=torch.float16 if args.fp16 else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)
```

**Issue**: LoRA checkpoints only contain adapter weights, not full model weights. Loading them directly with `AutoModelForCausalLM.from_pretrained()` results in an incomplete model that produces NaN outputs.

## Evidence

From the logs (`eval_mla_pilot_2392498.log`, `eval_sft_pilot_2392499.log`):
```
2026-01-29 11:26:38,330 - __main__ - WARNING - NaN/Inf detected in discrete hidden states for English, sample 0, layer -1
2026-01-29 11:26:38,585 - __main__ - WARNING - NaN/Inf detected in soft hidden states for English, sample 0, layer -1
[... continues for ALL samples and ALL languages ...]
```

This pattern shows that:
1. NaN appears starting from sample 0 (not just 46-49)
2. Affects both discrete and soft hidden states
3. Occurs across all 10 languages
4. Results in 500/500 zero vectors after NaN replacement

## Solution

The MGSM CoT evaluation scripts already had the correct approach:
1. **Merge LoRA adapter with base model** using `merge_lora_and_eval.py`
2. **Save merged model** to `/coc/pskynet6/jhe478/Soft-Thinking/merged_models/`
3. **Use merged model** for evaluation

### Changes Made

Updated pilot experiment scripts to use merged models:

**eval_sft_pilot.sh**:
```bash
MERGED_DIR="/coc/pskynet6/jhe478/Soft-Thinking/merged_models/sft_qwen3_4b_ckpt934"
# Changed from checkpoint path to merged model path
python pilot_experiment/run_hidden_states_experiment.py \
    --model_name "$MERGED_DIR" \  # Was: "$ADAPTER_PATH"
    ...
```

**eval_mla_pilot.sh**:
```bash
MERGED_DIR="/coc/pskynet6/jhe478/Soft-Thinking/merged_models/mla_qwen3_4b_epoch1"
# Changed from checkpoint path to merged model path
python pilot_experiment/run_hidden_states_experiment.py \
    --model_name "$MERGED_DIR" \  # Was: "$ADAPTER_PATH"
    ...
```

### Resubmitted Jobs

- Job 2392626: SFT pilot experiment with merged model
- Job 2392627: MLA pilot experiment with merged model

## Previous Attempts

The following defensive measures were added but couldn't fix the underlying issue:
1. NaN detection and replacement with zeros in hidden states (lines 302-312, 330-340)
2. NaN checks in silhouette score computation (lines 374-407)
3. Data validation before t-SNE visualization (lines 442-464)
4. Data validation before connected plot visualization (lines 486-507)

These measures prevented crashes but resulted in meaningless zero metrics because the root cause (incomplete model) wasn't addressed.

## Alternative Solution (Not Implemented)

The pilot experiment script could be modified to support LoRA loading:
```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    cache_dir=args.cache_dir,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()  # Merge adapter into base model
```

This approach wasn't used because:
1. The merge step is already implemented in the MGSM evaluation workflow
2. Merged models are reusable across different evaluation scripts
3. Merging once is more efficient than merging for each experiment

## Expected Results

With the merged models, the pilot experiments should now:
1. Produce valid (non-NaN) hidden states for all samples
2. Generate meaningful similarity and clustering metrics
3. Create informative t-SNE and connected plot visualizations
4. Allow comparison of soft vs discrete token representations for fine-tuned models

## Date
2026-01-29

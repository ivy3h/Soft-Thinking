# Fine-tuned Models Evaluation Status

**Date**: 2026-01-29
**Status**: MGSM CoT evaluations running successfully, Pilot experiments encountering NaN issues

## Summary

We have two fine-tuned models ready for evaluation:
1. **SFT Model**: Standard supervised fine-tuning on checkpoint-934
2. **MLA Model**: Middle-layer alignment fine-tuning on checkpoint-epoch-1

Both models have been merged with their base model (Qwen/Qwen3-4B-Instruct-2507) and the merged models are stored at:
- `/coc/pskynet6/jhe478/Soft-Thinking/merged_models/sft_qwen3_4b_ckpt934/`
- `/coc/pskynet6/jhe478/Soft-Thinking/merged_models/mla_qwen3_4b_epoch1/`

## Current Job Status

### ✅ MGSM CoT Baseline Evaluations (Working)
- **SFT**: Job 2392422 (running ~1h, nlprx-lab/major)
- **MLA**: Job 2392425 (running ~1h, nlprx-lab/ig-88)

Both jobs are successfully generating text and evaluating on MGSM datasets. The models are functional for text generation tasks.

### ❌ Pilot Experiments (NaN Issues)
- **SFT**: Job 2392670 (running, encountering NaN)
- **MLA**: Job 2392671 (running, encountering NaN)

Both pilot experiments produce NaN values in hidden states extraction for **ALL samples** across all languages.

## The NaN Problem

### What We Know

1. **Scope**: NaN appears in BOTH discrete and soft hidden states for every sample (0-49) in all 10 languages
2. **Pattern**: `NaN/Inf detected in discrete hidden states for [language], sample [N], layer -1`
3. **Impact**: Results in 500/500 zero vectors after NaN replacement, leading to meaningless metrics (all 0.0)
4. **Contrast**: MGSM text generation works fine, but hidden states extraction (`output_hidden_states=True`) produces NaN

### What We've Tried

1. ✅ **Fixed model loading**: Changed from LoRA checkpoint paths to merged model paths
2. ✅ **Added NaN handling**: Detect and replace NaN with zeros to prevent crashes
3. ✅ **Tried fp16**: Added `--fp16` flag (currently testing in jobs 2392670/2392671)
4. ❌ **Result**: NaN persists across all attempts

### Root Cause Hypothesis

The fine-tuned models produce valid text generation (logits) but invalid hidden states. Possible causes:

1. **LoRA merge issue**: The merge process may not correctly merge hidden state computations
2. **Training artifact**: Middle-layer alignment or SFT training may have introduced numerical instability in intermediate layers
3. **Model compatibility**: The merged models may require special handling for `output_hidden_states=True`
4. **Precision issue**: Despite trying fp16, there may be mixed precision problems in hidden state extraction

### Evidence

**MGSM Evaluation Log** (works):
```
[2026-01-29 11:44:19] Decode batch. #running-req: 14, #token: 126440, token usage: 0.94, gen throughput (token/s): 272.37
```

**Pilot Experiment Log** (fails):
```
2026-01-29 11:57:39,390 - __main__ - WARNING - NaN/Inf detected in soft hidden states for Spanish, sample 20, layer -1
2026-01-29 11:57:39,434 - __main__ - WARNING - NaN/Inf detected in discrete hidden states for French, sample 20, layer -1
[... continues for ALL samples ...]
```

## Comparison with Base Models

Base model pilot experiments (Jan 28) all succeeded:
- `pilot_results/` (Qwen3-4B-Instruct-2507)
- `pilot_results_qwen3_4b/` (Qwen3-4B-Instruct-2507)
- `pilot_results_qwen3_8b/` (Qwen3-8B)
- `pilot_results_qwen3_8b_base/` (Qwen3-8B-Base)

All produced valid metrics, t-SNE visualizations, and connected plots.

## Files and Logs

### Scripts
- `/nethome/jhe478/flash/Soft-Thinking/scripts/finetuned/eval_sft_mgsm_cot.sh`
- `/nethome/jhe478/flash/Soft-Thinking/scripts/finetuned/eval_mla_mgsm_cot.sh`
- `/nethome/jhe478/flash/soft-token-alignment/scripts/eval_sft_pilot.sh`
- `/nethome/jhe478/flash/soft-token-alignment/scripts/eval_mla_pilot.sh`

### Logs
- SFT MGSM: `/nethome/jhe478/flash/Soft-Thinking/logs/eval_sft_mgsm_cot_2392422.log`
- MLA MGSM: `/nethome/jhe478/flash/Soft-Thinking/logs/eval_mla_mgsm_cot_2392425.log`
- SFT Pilot: `/nethome/jhe478/flash/soft-token-alignment/logs/eval_sft_pilot_2392670.log`
- MLA Pilot: `/nethome/jhe478/flash/soft-token-alignment/logs/eval_mla_pilot_2392671.log`

### Training
- SFT: `/nethome/jhe478/flash/soft-token-alignment/logs/sft/sta_sft_qwen3_4b_2386837.log`
- MLA: `/nethome/jhe478/flash/tinker-cookbook/logs/mla/mla_resume_qwen3_4b_2390432.log`

## Next Steps

### Option 1: Wait for Current Jobs
Let jobs 2392670 and 2392671 complete to see if `--fp16` flag resolves the issue (unlikely based on current logs).

### Option 2: Alternative Approaches
1. **Try without hidden states**: Modify pilot experiment to use logits directly instead of hidden states
2. **Use base model layers**: Load base model and manually apply LoRA adapters instead of using merged model
3. **Skip pilot experiments**: Focus on MGSM CoT evaluation results only
4. **Debug merge process**: Re-merge models with different settings (different dtype, device_map, etc.)

### Option 3: Investigate LoRA Merge
The `merge_lora_and_eval.py` script loads models on CPU with bf16:
```python
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    cache_dir=cache_dir,
)
```

Then merges with:
```python
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()
```

This could be causing issues with hidden state extraction. We could try:
- Merging on GPU instead of CPU
- Using fp16 instead of bf16
- Verifying merged weights are valid

## Working Evaluations

While we troubleshoot the pilot experiments, the MGSM CoT evaluations are proceeding:

| Model | Job | Status | Queue | Expected Time |
|---|---|---|---|---|
| SFT | 2392422 | Running | nlprx-lab | 2-3 hours |
| MLA | 2392425 | Running | nlprx-lab | 2-3 hours |

These will provide the main evaluation metrics for the fine-tuned models on MGSM (11 languages, 250 samples each).

##Additional Context

All 16 missing base model evaluations are also running on overcap queue (jobs 2392479-2392494), covering:
- MGSM Soft-Thinking for Qwen3-8B-Base
- XReasoning (AIME2024, AIME2025, GPQA) for all three base models with CoT and Soft-Thinking methods

---

**Note**: The NaN issue is perplexing because the same merged models work perfectly for text generation but fail for hidden states extraction. This suggests the issue is specific to how PyTorch computes and returns intermediate layer outputs when `output_hidden_states=True` is set.

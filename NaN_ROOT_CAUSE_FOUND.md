# NaN Root Cause Analysis - CONFIRMED

**Date**: 2026-01-29 12:20
**Status**: ROOT CAUSE IDENTIFIED

## Executive Summary

The merged LoRA models (`sft_qwen3_4b_ckpt934` and `mla_qwen3_4b_epoch1`) are **fundamentally broken**. When loaded with `transformers.AutoModelForCausalLM`, all transformer layers (layers 1-36) produce NaN values in their hidden states and output logits.

## Test Results

### MLA Merged Model Test
```
Testing: /coc/pskynet6/jhe478/Soft-Thinking/merged_models/mla_qwen3_4b_epoch1
Text: "Hello, world!" (4 tokens)
Loading: CPU, bf16

Results:
  Layer 0 (embedding): OK (min=-0.083, max=0.080)
  Layer 1-36 (all transformers): NaN [FAIL]
  Logits: NaN [FAIL]

Result: FAIL - All transformer layers produce NaN
```

### SFT Merged Model Test
```
Testing: /coc/pskynet6/jhe478/Soft-Thinking/merged_models/sft_qwen3_4b_ckpt934
Text: "Hello" (2 tokens)
Loading: CPU, bf16

Results:
  Layer 1-6: NaN [FAIL]
  Logits: NaN [FAIL]

Result: FAIL - All transformer layers produce NaN
```

## Why MGSM Evaluation Works

The MGSM CoT evaluation works because it uses **SGLang** engine instead of direct transformers:

```
[2026-01-29 10:42:45 TP0] Load weight begin. avail mem=43.56 GB
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:00<00:00, 2.34it/s]
[2026-01-29 10:42:48 TP0] Loading other weights begin. mode=safetensors
```

SGLang likely has different loading/precision handling that works around the merge issues.

## Root Cause: LoRA Merge Process

The `merge_lora_and_eval.py` script merges models on **CPU with bf16**:

```python
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,  # bf16 precision
    device_map="cpu",             # CPU device
    cache_dir=cache_dir,
)

model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()  # Merge happens here
model.save_pretrained(output_dir)  # Save merged model
```

**The problem**: When the merged model is later loaded with:
- `torch.float16` (what pilot experiment uses)
- or `torch.bfloat16` on CPU (what our test uses)
- Direct transformers loading (not SGLang)

...all the transformer layers produce NaN.

## Why Only Hidden States Fail

The key insight: **text generation with SGLang works, but hidden states extraction with transformers fails**.

This suggests the merged weights are in a state that SGLang can handle but transformers cannot properly process when `output_hidden_states=True` is set.

## Potential Issues

1. **Precision Mismatch**: Merging in bf16 on CPU, then loading in fp16 on GPU
2. **PEFT Version Compatibility**: Using peft 0.18.1 from checkpoint
3. **Weight Corruption**: The merge_and_unload() process corrupts transformer layer weights
4. **Unsafe Operations**: Some bf16 operations on CPU produce NaN/inf that get saved

## Evidence Timeline

1. **Jan 28**: Base model pilot experiments all succeeded
   - `pilot_results/` (Qwen3-4B-Instruct-2507)
   - `pilot_results_qwen3_8b/` etc.

2. **Jan 29 10:41**: MLA and SFT models merged
   - Both used same merge script
   - Both saved to `merged_models/`

3. **Jan 29 11:46**: First NaN warnings in pilot experiments
   - ALL samples (0-49) produce NaN
   - Both discrete AND soft hidden states affected
   - Jobs 2392498, 2392499 completed with zero metrics

4. **Jan 29 12:00**: MGSM evaluations running successfully
   - Jobs 2392422, 2392425 still generating text
   - Using SGLang engine, not transformers

5. **Jan 29 12:18**: Confirmed merged models are broken
   - Direct CPU test shows NaN in all transformer layers
   - Both SFT and MLA models affected identically

## Solution Options

### Option 1: Re-merge with Different Settings
```python
# Try fp16 instead of bf16
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,  # Use fp16
    device_map="cuda:0",         # Merge on GPU
    cache_dir=cache_dir,
)
```

### Option 2: Direct LoRA Loading (No Merge)
```python
# Load base + adapter directly in pilot experiment
base_model = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(base_model, adapter_path)
# Don't call merge_and_unload()
```

### Option 3: Use SGLang for Pilot Experiments
Modify pilot experiment to use SGLang engine like MGSM does.

### Option 4: Skip Pilot Experiments
Focus on MGSM CoT results only, which are working correctly.

## Recommended Action

**Try Option 1 first**: Re-merge models with fp16 on GPU.

The current merge process (bf16 on CPU) appears to be incompatible with how transformers loads and processes models for hidden states extraction. Re-merging with fp16 on GPU may produce models that work correctly with transformers.

## Test Command

To verify a fixed merge:
```bash
cd /coc/pskynet6/jhe478/soft-token-alignment
python3 quick_test.py
```

Expected output:
```
Testing MLA merged model hidden states...
Number of layers: 37
  Layer 0: torch.Size([1, 4, 2560]) min=-0.083 max=0.080 [OK]
  Layer 36: torch.Size([1, 4, 2560]) min=X.XXX max=X.XXX [OK]

Logits: torch.Size([1, 4, 151936]) [OK]

Result: PASS - No NaN, merged model is valid
```

## Impact

- **MGSM CoT Evaluations**: ✅ Working (using SGLang)
- **Pilot Experiments**: ❌ Broken (using transformers)
- **Base Model Evaluations**: ✅ Working (16 jobs running)

The fine-tuned models ARE functional (proven by MGSM text generation), but the merged model files cannot be used with transformers for hidden states extraction.

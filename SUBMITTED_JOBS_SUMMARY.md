# Evaluation Jobs Submission Summary

**Date**: 2026-01-29
**Queue**: overcap (12 hours time limit)

## ✅ Successfully Submitted Jobs

### Total: 20 evaluation jobs

#### Fine-tuned Model Evaluations (4 jobs - nlprx-lab queue)

| Job ID | Model | Dataset | Method | Status | Node |
|--------|-------|---------|--------|--------|------|
| 2392422 | SFT fine-tuned | MGSM | CoT | Running (~44 min) | major |
| 2392425 | MLA fine-tuned | MGSM | CoT | Running (~40 min) | ig-88 |
| 2392456 | MLA fine-tuned | Pilot | Soft vs Discrete | Running | major |
| 2392457 | SFT fine-tuned | Pilot | Soft vs Discrete | Running | spot |

#### Missing Base Model Evaluations (16 jobs - overcap queue)

##### Qwen3-4B-Instruct-2507 (5 jobs)

| Job ID | Dataset | Method | Status |
|--------|---------|--------|--------|
| 2392480 | XReasoning-AIME2024 | Soft-Thinking | Running |
| 2392484 | XReasoning-AIME2025 | CoT | Running |
| 2392485 | XReasoning-AIME2025 | Soft-Thinking | Running |
| 2392489 | XReasoning-GPQA | CoT | Running |
| 2392490 | XReasoning-GPQA | Soft-Thinking | Running |

##### Qwen3-8B-Base (6 jobs)

| Job ID | Dataset | Method | Status |
|--------|---------|--------|--------|
| 2392479 | MGSM | Soft-Thinking | Running |
| 2392481 | XReasoning-AIME2024 | Soft-Thinking | Running |
| 2392486 | XReasoning-AIME2025 | CoT | Running |
| 2392487 | XReasoning-AIME2025 | Soft-Thinking | Running |
| 2392491 | XReasoning-GPQA | CoT | Running |
| 2392492 | XReasoning-GPQA | Soft-Thinking | Running |

##### Qwen3-8B (5 jobs)

| Job ID | Dataset | Method | Status |
|--------|---------|--------|--------|
| 2392482 | XReasoning-AIME2024 | CoT | Running |
| 2392483 | XReasoning-AIME2024 | Soft-Thinking | Running |
| 2392488 | XReasoning-AIME2025 | Soft-Thinking | Running |
| 2392493 | XReasoning-GPQA | CoT | Running |
| 2392494 | XReasoning-GPQA | Soft-Thinking | Running |

## 📊 Expected Outcomes

After all jobs complete, we will have:

### MGSM Dataset (11 languages)
- ✅ Qwen3-4B-Instruct-2507: CoT + Soft-Thinking (Complete)
- 🔄 Qwen3-8B-Base: CoT (Complete) + **Soft-Thinking (Running)**
- ✅ Qwen3-8B: CoT + Soft-Thinking (Complete)

### XReasoning Datasets

#### AIME2024 (5 runs each)
- 🔄 Qwen3-4B-Instruct-2507: CoT (Complete) + **Soft-Thinking (Running)**
- 🔄 Qwen3-8B-Base: CoT (Complete) + **Soft-Thinking (Running)**
- 🔄 Qwen3-8B: **CoT (Running)** + **Soft-Thinking (Running)**

#### AIME2025 (5 runs each)
- 🔄 Qwen3-4B-Instruct-2507: **CoT (Running)** + **Soft-Thinking (Running)**
- 🔄 Qwen3-8B-Base: **CoT (Running)** + **Soft-Thinking (Running)**
- 🔄 Qwen3-8B: CoT (Complete) + **Soft-Thinking (Running)**

#### GPQA (1 run each)
- 🔄 Qwen3-4B-Instruct-2507: **CoT (Running)** + **Soft-Thinking (Running)**
- 🔄 Qwen3-8B-Base: **CoT (Running)** + **Soft-Thinking (Running)**
- 🔄 Qwen3-8B: **CoT (Running)** + **Soft-Thinking (Running)**

## 📁 Output Locations

### Logs
- Fine-tuned models: `/nethome/jhe478/flash/Soft-Thinking/logs/eval_*_*.log`
- Base models: `/nethome/jhe478/flash/Soft-Thinking/logs/mgsm_*.log`, `xr_*.log`

### Results
- MGSM: `/coc/pskynet6/jhe478/Soft-Thinking/results/results/mgsm/`
- XReasoning: `/coc/pskynet6/jhe478/Soft-Thinking/results/results/xreasoning_*/`
- Pilot experiments: `/coc/pskynet6/jhe478/soft-token-alignment/pilot_results_*/`

## 🔧 Scripts Generated

All scripts are in: `/nethome/jhe478/flash/Soft-Thinking/scripts/missing_evals/`

To resubmit if needed:
```bash
bash /nethome/jhe478/flash/Soft-Thinking/scripts/missing_evals/submit_all.sh
```

## ⏱️ Estimated Completion Time

- **MGSM evaluations**: 2-3 hours (250 samples × 11 languages)
- **XReasoning AIME**: 1-2 hours per run (5 runs = 5-10 hours)
- **XReasoning GPQA**: 2-3 hours (1 run, 50 samples)
- **Pilot experiments**: 5-10 minutes (50 samples)

All jobs have a 12-hour time limit and include resume functionality where supported.

## 📝 Next Steps

1. Monitor job progress: `squeue -u jhe478`
2. Check logs for errors: `tail -f /nethome/jhe478/flash/Soft-Thinking/logs/<job_name>_<jobid>.log`
3. After completion, run analysis: `python /nethome/jhe478/flash/Soft-Thinking/analyze_results_v2.py`
4. Generate updated summary report with complete results

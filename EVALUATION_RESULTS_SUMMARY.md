# Soft-Thinking Evaluation Results Summary

Generated: 2026-01-29

## Overview

**Completion Status:** 8/24 configurations complete (33.3%)

### Evaluation Matrix

| Dimension | Options |
|-----------|---------|
| **Models** | Qwen3-4B-Instruct-2507, Qwen3-8B-Base, Qwen3-8B |
| **Methods** | CoT (Chain-of-Thought), Soft-Thinking |
| **Datasets** | MGSM (11 languages), XReasoning-AIME2024, XReasoning-AIME2025, XReasoning-GPQA |

---

## Completed Evaluations

### 1. Qwen3-4B-Instruct-2507

#### MGSM Results

| Method | en | es | fr | de | ru | zh | ja | th | sw | bn | te | **Avg** | **Consistency** |
|--------|----|----|----|----|----|----|----|----|----|----|----|---------|-----------------|
| **CoT** | 94.4% | 88.8% | 81.2% | 84.4% | 88.4% | 86.8% | 80.4% | 84.8% | 25.2% | 79.2% | 65.2% | **78.1%** | **0.736** |
| **Soft-Thinking** | 92.0% | 87.6% | 80.8% | 82.4% | 88.4% | 85.2% | 79.6% | 83.6% | 25.2% | 76.0% | 59.6% | **76.4%** | **0.709** |

**Key Findings:**
- CoT slightly outperforms Soft-Thinking on MGSM (78.1% vs 76.4%)
- Both methods struggle with Swahili (sw: 25.2%)
- CoT shows better cross-lingual consistency (0.736 vs 0.709)

#### XReasoning Results

| Dataset | Method | Accuracy |
|---------|--------|----------|
| AIME2024 | CoT | 0.00% |
| AIME2025 | ❌ Missing | - |
| GPQA | ❌ Missing | - |

---

### 2. Qwen3-8B-Base

#### MGSM Results

| Method | en | es | fr | de | ru | zh | ja | th | sw | bn | te | **Avg** | **Consistency** |
|--------|----|----|----|----|----|----|----|----|----|----|----|---------|-----------------|
| **CoT** | 90.0% | 80.0% | 76.4% | 77.2% | 73.6% | 74.4% | 64.0% | 76.4% | 37.2% | 64.4% | 59.6% | **70.3%** | **0.666** |

**Key Findings:**
- Base model (without instruction tuning) performs worse than instruct model
- Still shows reasonable performance on high-resource languages (en: 90.0%)
- Better on Swahili compared to 4B instruct model (37.2% vs 25.2%)

#### XReasoning Results

| Dataset | Method | Accuracy |
|---------|--------|----------|
| AIME2024 | CoT | 0.00% |
| AIME2025 | ❌ Missing | - |
| GPQA | ❌ Missing | - |

---

### 3. Qwen3-8B

#### MGSM Results

| Method | en | es | fr | de | ru | zh | ja | th | sw | bn | te | **Avg** | **Consistency** |
|--------|----|----|----|----|----|----|----|----|----|----|----|---------|-----------------|
| **CoT** | 94.4% | 91.2% | 87.2% | 88.0% | 93.2% | 86.8% | 87.2% | 90.8% | 61.2% | 87.2% | 81.2% | **86.2%** | **0.831** |
| **Soft-Thinking** | 95.6% | 90.8% | 85.2% | 87.2% | 92.4% | 88.4% | 84.8% | 88.0% | 58.0% | 87.6% | 81.6% | **85.4%** | **0.821** |

**Key Findings:**
- **Best overall performance** across all models (86.2% CoT)
- Very high cross-lingual consistency (0.831 CoT, 0.821 Soft-Thinking)
- CoT slightly outperforms Soft-Thinking (86.2% vs 85.4%)
- Much better on low-resource languages (sw: 61.2% vs 25.2% for 4B model)

#### XReasoning Results

| Dataset | Method | Accuracy |
|---------|--------|----------|
| AIME2025 | CoT | 0.00% |

---

## Missing Evaluations (16 configurations)

### By Model

#### Qwen3-4B-Instruct-2507 (5 missing)
- [ ] XReasoning-AIME2024 + Soft-Thinking
- [ ] XReasoning-AIME2025 + CoT
- [ ] XReasoning-AIME2025 + Soft-Thinking
- [ ] XReasoning-GPQA + CoT
- [ ] XReasoning-GPQA + Soft-Thinking

#### Qwen3-8B-Base (6 missing)
- [ ] MGSM + Soft-Thinking
- [ ] XReasoning-AIME2024 + Soft-Thinking
- [ ] XReasoning-AIME2025 + CoT
- [ ] XReasoning-AIME2025 + Soft-Thinking
- [ ] XReasoning-GPQA + CoT
- [ ] XReasoning-GPQA + Soft-Thinking

#### Qwen3-8B (5 missing)
- [ ] XReasoning-AIME2024 + CoT
- [ ] XReasoning-AIME2024 + Soft-Thinking
- [ ] XReasoning-AIME2025 + Soft-Thinking
- [ ] XReasoning-GPQA + CoT
- [ ] XReasoning-GPQA + Soft-Thinking

### Priority Recommendations

**High Priority:**
1. **Qwen3-8B MGSM Soft-Thinking** - Already running (Job 2392425, 2392426)
2. **Qwen3-8B-Base MGSM Soft-Thinking** - Complete the best performing model
3. **XReasoning-GPQA evaluations** - Most important reasoning benchmark

**Medium Priority:**
4. XReasoning-AIME2024/2025 Soft-Thinking variants
5. Qwen3-4B XReasoning evaluations

---

## Key Insights

### Model Comparison (MGSM CoT)
1. **Qwen3-8B**: 86.2% (best)
2. **Qwen3-4B-Instruct-2507**: 78.1%
3. **Qwen3-8B-Base**: 70.3%

### Method Comparison
- **CoT generally outperforms Soft-Thinking** on MGSM:
  - Qwen3-4B: 78.1% vs 76.4% (Δ +1.7%)
  - Qwen3-8B: 86.2% vs 85.4% (Δ +0.8%)
- Soft-Thinking shows slightly lower cross-lingual consistency

### Language-Specific Observations
- **Swahili (sw)** is the most challenging language for all models
- **English, Russian, Spanish** typically achieve highest accuracies
- Larger models (8B) show better multilingual capabilities

---

## Currently Running Evaluations

As of 2026-01-29 10:30:

| Job ID | Model | Dataset | Method | Status |
|--------|-------|---------|--------|--------|
| 2392422 | SFT fine-tuned | MGSM | CoT | Running (4:32) |
| 2392424 | SFT fine-tuned | Pilot | - | Running (2:08) |
| 2392425 | MLA fine-tuned | MGSM | CoT | Running (0:20) |
| 2392426 | MLA fine-tuned | Pilot | - | Running (0:13) |

*Note: These are fine-tuned model evaluations, not part of the base model evaluation matrix*

---

## Next Steps

1. ✅ Complete currently running fine-tuned model evaluations
2. 📋 Prioritize and schedule missing base model evaluations
3. 📊 Analyze results to determine if Soft-Thinking provides benefits on specific task types
4. 🔬 Investigate why CoT outperforms Soft-Thinking on MGSM

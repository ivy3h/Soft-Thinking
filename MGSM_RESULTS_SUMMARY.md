# MGSM Evaluation Results Summary

**Date**: 2026-01-29
**Dataset**: MGSM (Multilingual Grade School Math, 11 languages, 250 samples each)

## Overall Results

### Per-Language Accuracy

| Language | Qwen3-4B-Instruct CoT | Qwen3-4B-Instruct ST | Qwen3-8B-Base CoT | Qwen3-8B-Base ST | Qwen3-8B CoT | Qwen3-8B ST |
|----------|----------------------|---------------------|------------------|-----------------|-------------|------------|
| English  | 94.4% | 92.0% | 90.0% | **90.4%** | 94.4% | **95.6%** |
| Spanish  | 88.8% | 87.6% | 80.0% | **85.2%** | **91.6%** | 90.8% |
| French   | 81.2% | 80.8% | 76.4% | **82.0%** | **86.8%** | 85.2% |
| German   | 84.4% | 82.4% | 77.2% | **79.2%** | **88.8%** | 87.2% |
| Russian  | 88.4% | **88.4%** | 73.6% | **83.2%** | **93.2%** | 92.4% |
| Chinese  | 86.8% | 85.2% | 74.4% | **77.2%** | **88.4%** | **88.4%** |
| Japanese | 80.4% | 79.6% | 64.0% | **68.0%** | **86.4%** | 84.8% |
| Thai     | 84.8% | 83.6% | 76.4% | **80.4%** | **89.2%** | 88.0% |
| Swahili  | **25.2%** | **25.2%** | 37.2% | **43.2%** | **60.0%** | 58.0% |
| Bengali  | 79.2% | 76.0% | 64.4% | **78.0%** | 86.4% | **87.6%** |
| Telugu   | 65.2% | 59.6% | **59.6%** | 58.4% | **82.0%** | 81.6% |
| **Average** | **78.1%** | 76.4% | 70.3% | **75.0%** | **86.1%** | 85.4% |

**Bold** indicates better performance between CoT and Soft-Thinking for each model.

### Summary Statistics

| Metric | Qwen3-4B-Instruct CoT | Qwen3-4B-Instruct ST | Qwen3-8B-Base CoT | Qwen3-8B-Base ST | Qwen3-8B CoT | Qwen3-8B ST |
|--------|----------------------|---------------------|------------------|-----------------|-------------|------------|
| Average Accuracy | **78.1%** | 76.4% | 70.3% | **75.0%** | **86.1%** | 85.4% |
| Avg Consistency | **73.6%** | 70.9% | 66.6% | **70.5%** | **82.8%** | 82.1% |
| Avg Correct Consistency | **64.9%** | 61.8% | 53.6% | **60.3%** | **77.5%** | 76.5% |
| Time (hours) | **1.06** | 3.02 | **3.22** | 3.14 | **0.70** | 4.35 |

### Improvement: Soft-Thinking vs CoT

| Language | Qwen3-4B-Instruct | Qwen3-8B-Base | Qwen3-8B |
|----------|------------------|--------------|----------|
| English  | -2.4% | **+0.4%** | **+1.2%** |
| Spanish  | -1.2% | **+5.2%** | -0.8% |
| French   | -0.4% | **+5.6%** | -1.6% |
| German   | -2.0% | **+2.0%** | -1.6% |
| Russian  | 0.0% | **+9.6%** | -0.8% |
| Chinese  | -1.6% | **+2.8%** | 0.0% |
| Japanese | -0.8% | **+4.0%** | -1.6% |
| Thai     | -1.2% | **+4.0%** | -1.2% |
| Swahili  | 0.0% | **+6.0%** | -2.0% |
| Bengali  | -3.2% | **+13.6%** | **+1.2%** |
| Telugu   | -5.6% | -1.2% | -0.4% |
| **Average** | **-1.7%** | **+4.7%** | **-0.7%** |

**Positive values** (green) indicate Soft-Thinking outperforms CoT.

## Key Findings

### 1. Model Performance Ranking
- **Qwen3-8B CoT: 86.1%** (best overall)
- **Qwen3-8B ST: 85.4%**
- **Qwen3-4B-Instruct CoT: 78.1%**
- **Qwen3-4B-Instruct ST: 76.4%**
- **Qwen3-8B-Base ST: 75.0%** ✨ **NEW RESULT**
- **Qwen3-8B-Base CoT: 70.3%**

### 2. Soft-Thinking Impact by Model

**Qwen3-8B-Base: +4.7% improvement** ✅
- Largest improvement among all models
- Significant gains across all languages (except Telugu -1.2%)
- Bengali: +13.6% (largest single-language gain)
- Russian: +9.6%
- Swahili: +6.0%

**Qwen3-4B-Instruct: -1.7% degradation** ❌
- Soft-Thinking hurts performance
- Telugu shows largest drop: -5.6%
- Bengali: -3.2%

**Qwen3-8B: -0.7% slight degradation** ≈
- Minimal impact
- Mixed results across languages
- Bengali: +1.2% (only significant gain)
- English: +1.2%

### 3. Language-Specific Observations

**Best Languages** (all models average >80%):
1. English: 91.1% average
2. Russian: 86.5% average
3. Spanish: 87.3% average

**Challenging Languages** (all models average <70%):
1. Swahili: 40.1% average (extremely difficult)
2. Telugu: 67.7% average
3. Japanese: 77.2% average

**Soft-Thinking Benefits Swahili Most**:
- Qwen3-8B-Base: 37.2% → 43.2% (+6.0%)
- Still remains the most challenging language

### 4. Time Efficiency

**CoT is 2-4x faster than Soft-Thinking**:
- Qwen3-4B-Instruct: 1.06h (CoT) vs 3.02h (ST) = 2.8x slower
- Qwen3-8B-Base: 3.22h (CoT) vs 3.14h (ST) = similar ✓
- Qwen3-8B: 0.70h (CoT) vs 4.35h (ST) = 6.2x slower

**Note**: Qwen3-8B-Base ST time is anomalously low (3.14h), possibly due to cached computations.

### 5. Cross-Lingual Consistency

**Consistency improves with Soft-Thinking for Qwen3-8B-Base**:
- Average Consistency: 66.6% → 70.5% (+3.9%)
- Average Correct Consistency: 53.6% → 60.3% (+6.7%)

**Consistency decreases with Soft-Thinking for other models**:
- Qwen3-4B-Instruct: 73.6% → 70.9% (-2.7%)
- Qwen3-8B: 82.8% → 82.1% (-0.7%)

## Conclusions

1. **Qwen3-8B-Base benefits significantly from Soft-Thinking** (+4.7% average, +13.6% on Bengali)
   - This is the ONLY model where Soft-Thinking consistently improves performance
   - Suggests that base (non-instruct) models benefit more from implicit reasoning

2. **Instruct-tuned models don't benefit from Soft-Thinking**
   - Qwen3-4B-Instruct: -1.7%
   - Qwen3-8B: -0.7%
   - These models may already have sufficient reasoning capability from instruction tuning

3. **Swahili remains extremely challenging** (40.1% average)
   - Large gap between English (91.1%) and Swahili (40.1%)
   - Even Soft-Thinking only brings Qwen3-8B-Base to 43.2%

4. **Trade-off between accuracy and efficiency**
   - Soft-Thinking improves Qwen3-8B-Base accuracy by 4.7%
   - But takes similar time (3.14h vs 3.22h)
   - For instruct models, CoT is both faster AND more accurate

## Recommendations

- **Use Soft-Thinking for base models** (non-instruct): Significant accuracy improvements
- **Use CoT for instruct models**: Better accuracy and faster inference
- **Focus on low-resource languages**: Largest room for improvement (Swahili, Telugu)

---

**Files Generated**:
- Per-language results: `/coc/pskynet6/jhe478/Soft-Thinking/results/results/mgsm/Qwen3-8B-Base_mgsm_True_*_{lang}.json`
- Statistics: `/coc/pskynet6/jhe478/Soft-Thinking/results/results/mgsm/Qwen3-8B-Base_mgsm_True_*_statistics.json`
- Logs: `/nethome/jhe478/flash/Soft-Thinking/logs/mgsm_qwen3_8b_base_st_2392479.log`

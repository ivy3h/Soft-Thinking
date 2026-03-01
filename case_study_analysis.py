#!/usr/bin/env python3
"""
Case study analysis for SOLAR paper.
Finds:
  Case 1: SOLAR correct but Baseline AND SFT both wrong
  Case 2: Language switch (baseline reasons in English, SFT/SOLAR in target language)
"""

import json
import os
import re
import sys
import io
from collections import defaultdict

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RESULTS_DIR = "/coc/pskynet6/jhe478/Soft-Thinking/results/results/mgsm"

# File name templates
BASE_TEMPLATE = "Qwen3-4B-Instruct-2507_mgsm_False_1_0.6_0.95_30_0.001_1.0_1.0_15_32768_0.0_256_run{run}_{lang}.json"
SFT_TEMPLATE = "sft_ms1k_val_mgsm_False_1_0.6_0.95_30_0.001_1.0_1.0_15_32768_0.0_256_run{run}_{lang}.json"
SOLAR_TEMPLATE = "sft_ms1k_solar_val_mgsm_False_1_0.6_0.95_30_0.001_1.0_1.0_15_32768_0.0_256_run{run}_{lang}.json"

LANGUAGES = ["zh", "ja", "th", "te"]
LANG_NAMES = {"zh": "Chinese", "ja": "Japanese", "th": "Thai", "te": "Telugu"}

def load_json(filepath):
    """Load a JSON file, return None if not found."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def get_correct(entry):
    """Extract correctness from judge_info."""
    ji = entry["judge_info"]
    if isinstance(ji, list):
        return ji[0].get("finally_judge_result", False)
    elif isinstance(ji, dict):
        return ji.get("finally_judge_result", ji.get("correct", False))
    return False

def extract_think_block(text):
    """Extract content inside <think>...</think> tags."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1)
    # If no closing tag, return everything after <think>
    match = re.search(r"<think>(.*)", text, re.DOTALL)
    if match:
        return match.group(1)
    return text

def count_cjk(text):
    """Count CJK characters (Chinese/Japanese/Korean)."""
    return sum(1 for c in text if '\u4e00' <= c <= '\u9fff')

def count_latin(text):
    """Count Latin alphabetic characters."""
    return sum(1 for c in text if c.isascii() and c.isalpha())

def count_thai(text):
    """Count Thai characters."""
    return sum(1 for c in text if '\u0e00' <= c <= '\u0e7f')

def count_telugu(text):
    """Count Telugu characters."""
    return sum(1 for c in text if '\u0c00' <= c <= '\u0c7f')

def count_japanese_kana(text):
    """Count Hiragana + Katakana characters."""
    return sum(1 for c in text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')

def target_lang_ratio(text, lang):
    """Compute ratio of target language characters to total meaningful characters."""
    latin = count_latin(text)
    if lang == "zh":
        target = count_cjk(text)
    elif lang == "ja":
        target = count_cjk(text) + count_japanese_kana(text)
    elif lang == "th":
        target = count_thai(text)
    elif lang == "te":
        target = count_telugu(text)
    else:
        target = 0
    total = target + latin
    if total == 0:
        return 0.0
    return target / total

def get_reasoning_text(entry, has_think_tags):
    """Get the reasoning text from a completion entry."""
    text = entry["completion"][0]
    if has_think_tags:
        return extract_think_block(text)
    return text

def find_common_runs(lang):
    """Find run numbers that exist for all three models for a given language."""
    runs = []
    for r in range(1, 6):
        base_f = os.path.join(RESULTS_DIR, BASE_TEMPLATE.format(run=r, lang=lang))
        sft_f = os.path.join(RESULTS_DIR, SFT_TEMPLATE.format(run=r, lang=lang))
        solar_f = os.path.join(RESULTS_DIR, SOLAR_TEMPLATE.format(run=r, lang=lang))
        if all(os.path.exists(f) for f in [base_f, sft_f, solar_f]):
            runs.append(r)
    return runs


# =============================================================================
# CASE 1: SOLAR correct but Baseline AND SFT both wrong
# =============================================================================
print("=" * 100)
print("CASE 1: SOLAR correct but Baseline AND SFT both wrong")
print("=" * 100)

for lang in LANGUAGES:
    print(f"\n{'='*80}")
    print(f"Language: {LANG_NAMES[lang]} ({lang})")
    print(f"{'='*80}")

    common_runs = find_common_runs(lang)
    print(f"Common runs available: {common_runs}")

    if not common_runs:
        print("  No common runs found, skipping.")
        continue

    # For each run, find cases where SOLAR is correct but both baseline and SFT are wrong
    # Then aggregate across runs to find consistent patterns
    all_solar_only_correct = defaultdict(list)  # idx -> list of run numbers where condition holds

    for run in common_runs:
        base_data = load_json(os.path.join(RESULTS_DIR, BASE_TEMPLATE.format(run=run, lang=lang)))
        sft_data = load_json(os.path.join(RESULTS_DIR, SFT_TEMPLATE.format(run=run, lang=lang)))
        solar_data = load_json(os.path.join(RESULTS_DIR, SOLAR_TEMPLATE.format(run=run, lang=lang)))

        if not all([base_data, sft_data, solar_data]):
            continue

        # Index by idx
        base_by_idx = {e["idx"]: e for e in base_data}
        sft_by_idx = {e["idx"]: e for e in sft_data}
        solar_by_idx = {e["idx"]: e for e in solar_data}

        for idx in sorted(base_by_idx.keys()):
            if idx not in sft_by_idx or idx not in solar_by_idx:
                continue
            base_correct = get_correct(base_by_idx[idx])
            sft_correct = get_correct(sft_by_idx[idx])
            solar_correct = get_correct(solar_by_idx[idx])

            if not base_correct and not sft_correct and solar_correct:
                all_solar_only_correct[idx].append(run)

    # Report: idx values where SOLAR is uniquely correct in at least 1 run
    print(f"\n  Total unique idx values where SOLAR correct & both others wrong (any run): {len(all_solar_only_correct)}")

    # Sort by how many runs the pattern holds
    sorted_cases = sorted(all_solar_only_correct.items(), key=lambda x: -len(x[1]))

    print(f"  idx values (sorted by consistency across runs):")
    for idx, runs_list in sorted_cases:
        print(f"    idx={idx}: in {len(runs_list)}/{len(common_runs)} runs ({runs_list})")

    # Show detailed examples for the first 3 most consistent cases
    print(f"\n  --- Detailed examples (top 3 most consistent) ---")
    shown = 0
    for idx, runs_list in sorted_cases[:3]:
        # Use the first run where the pattern holds
        run = runs_list[0]
        base_data = load_json(os.path.join(RESULTS_DIR, BASE_TEMPLATE.format(run=run, lang=lang)))
        sft_data = load_json(os.path.join(RESULTS_DIR, SFT_TEMPLATE.format(run=run, lang=lang)))
        solar_data = load_json(os.path.join(RESULTS_DIR, SOLAR_TEMPLATE.format(run=run, lang=lang)))

        base_entry = {e["idx"]: e for e in base_data}[idx]
        sft_entry = {e["idx"]: e for e in sft_data}[idx]
        solar_entry = {e["idx"]: e for e in solar_data}[idx]

        print(f"\n  Example {shown+1}: idx={idx}, consistent in {len(runs_list)}/{len(common_runs)} runs")
        print(f"  Ground truth: {base_entry['ground_truth']}")
        print(f"  Question (from prompt): {base_entry['prompt'][:300]}...")

        print(f"\n  Baseline (WRONG, run {run}):")
        base_reasoning = get_reasoning_text(base_entry, False)
        print(f"    {base_reasoning[:200]}...")

        print(f"\n  SFT (WRONG, run {run}):")
        sft_reasoning = get_reasoning_text(sft_entry, True)
        print(f"    {sft_reasoning[:200]}...")

        print(f"\n  SOLAR (CORRECT, run {run}):")
        solar_reasoning = get_reasoning_text(solar_entry, True)
        print(f"    {solar_reasoning[:200]}...")

        # Also show extracted answers
        print(f"\n  Extracted answers:")
        print(f"    Baseline: {base_entry.get('extracted_answer', 'N/A')}")
        print(f"    SFT:      {sft_entry.get('extracted_answer', 'N/A')}")
        print(f"    SOLAR:    {solar_entry.get('extracted_answer', 'N/A')}")
        shown += 1


# =============================================================================
# Also report overall stats: how many times each model got it right across all runs
# =============================================================================
print("\n\n" + "=" * 100)
print("OVERALL CORRECTNESS COMPARISON (per run, per language)")
print("=" * 100)

for lang in LANGUAGES:
    common_runs = find_common_runs(lang)
    if not common_runs:
        continue
    print(f"\n  {LANG_NAMES[lang]} ({lang}), {len(common_runs)} common runs:")

    for run in common_runs:
        base_data = load_json(os.path.join(RESULTS_DIR, BASE_TEMPLATE.format(run=run, lang=lang)))
        sft_data = load_json(os.path.join(RESULTS_DIR, SFT_TEMPLATE.format(run=run, lang=lang)))
        solar_data = load_json(os.path.join(RESULTS_DIR, SOLAR_TEMPLATE.format(run=run, lang=lang)))

        base_acc = sum(1 for e in base_data if get_correct(e)) / len(base_data) * 100
        sft_acc = sum(1 for e in sft_data if get_correct(e)) / len(sft_data) * 100
        solar_acc = sum(1 for e in solar_data if get_correct(e)) / len(solar_data) * 100

        print(f"    Run {run}: Baseline={base_acc:.1f}%, SFT={sft_acc:.1f}%, SOLAR={solar_acc:.1f}%")


# =============================================================================
# CASE 2: Language switch analysis (focus on Chinese zh)
# =============================================================================
print("\n\n" + "=" * 100)
print("CASE 2: Language switch analysis")
print("Baseline reasons in English/Latin, but SFT/SOLAR reason in target language")
print("=" * 100)

for lang in LANGUAGES:
    print(f"\n{'='*80}")
    print(f"Language: {LANG_NAMES[lang]} ({lang})")
    print(f"{'='*80}")

    common_runs = find_common_runs(lang)
    if not common_runs:
        print("  No common runs, skipping.")
        continue

    # Use run 1 for analysis
    run = common_runs[0]
    base_data = load_json(os.path.join(RESULTS_DIR, BASE_TEMPLATE.format(run=run, lang=lang)))
    sft_data = load_json(os.path.join(RESULTS_DIR, SFT_TEMPLATE.format(run=run, lang=lang)))
    solar_data = load_json(os.path.join(RESULTS_DIR, SOLAR_TEMPLATE.format(run=run, lang=lang)))

    base_by_idx = {e["idx"]: e for e in base_data}
    sft_by_idx = {e["idx"]: e for e in sft_data}
    solar_by_idx = {e["idx"]: e for e in solar_data}

    # Compute language ratios for all entries
    lang_switch_cases = []
    all_base_ratios = []
    all_sft_ratios = []
    all_solar_ratios = []

    for idx in sorted(base_by_idx.keys()):
        if idx not in sft_by_idx or idx not in solar_by_idx:
            continue

        base_text = get_reasoning_text(base_by_idx[idx], False)
        sft_text = get_reasoning_text(sft_by_idx[idx], True)
        solar_text = get_reasoning_text(solar_by_idx[idx], True)

        base_ratio = target_lang_ratio(base_text, lang)
        sft_ratio = target_lang_ratio(sft_text, lang)
        solar_ratio = target_lang_ratio(solar_text, lang)

        all_base_ratios.append(base_ratio)
        all_sft_ratios.append(sft_ratio)
        all_solar_ratios.append(solar_ratio)

        # Language switch: baseline low target-lang ratio, SFT/SOLAR high
        if base_ratio < 0.5 and sft_ratio > 0.7 and solar_ratio > 0.7:
            lang_switch_cases.append({
                "idx": idx,
                "base_ratio": base_ratio,
                "sft_ratio": sft_ratio,
                "solar_ratio": solar_ratio,
                "base_correct": get_correct(base_by_idx[idx]),
                "sft_correct": get_correct(sft_by_idx[idx]),
                "solar_correct": get_correct(solar_by_idx[idx]),
            })

    # Summary stats
    avg_base = sum(all_base_ratios) / len(all_base_ratios) if all_base_ratios else 0
    avg_sft = sum(all_sft_ratios) / len(all_sft_ratios) if all_sft_ratios else 0
    avg_solar = sum(all_solar_ratios) / len(all_solar_ratios) if all_solar_ratios else 0

    print(f"\n  Average target-language ratio in reasoning (run {run}):")
    print(f"    Baseline: {avg_base:.3f}")
    print(f"    SFT:      {avg_sft:.3f}")
    print(f"    SOLAR:    {avg_solar:.3f}")
    print(f"\n  Language switch cases (base<0.5, SFT>0.7, SOLAR>0.7): {len(lang_switch_cases)}/{len(all_base_ratios)}")

    # Distribution of base ratios
    low_base = sum(1 for r in all_base_ratios if r < 0.3)
    mid_base = sum(1 for r in all_base_ratios if 0.3 <= r < 0.7)
    high_base = sum(1 for r in all_base_ratios if r >= 0.7)
    print(f"  Baseline target-lang ratio distribution: <0.3: {low_base}, 0.3-0.7: {mid_base}, >=0.7: {high_base}")

    low_sft = sum(1 for r in all_sft_ratios if r < 0.3)
    mid_sft = sum(1 for r in all_sft_ratios if 0.3 <= r < 0.7)
    high_sft = sum(1 for r in all_sft_ratios if r >= 0.7)
    print(f"  SFT target-lang ratio distribution:      <0.3: {low_sft}, 0.3-0.7: {mid_sft}, >=0.7: {high_sft}")

    low_solar = sum(1 for r in all_solar_ratios if r < 0.3)
    mid_solar = sum(1 for r in all_solar_ratios if 0.3 <= r < 0.7)
    high_solar = sum(1 for r in all_solar_ratios if r >= 0.7)
    print(f"  SOLAR target-lang ratio distribution:    <0.3: {low_solar}, 0.3-0.7: {mid_solar}, >=0.7: {high_solar}")

    if not lang_switch_cases:
        # Try relaxed threshold
        relaxed = [c for c in range(len(all_base_ratios))
                   if all_base_ratios[c] < all_sft_ratios[c] - 0.2
                   and all_base_ratios[c] < all_solar_ratios[c] - 0.2]
        print(f"  (No strong switch cases. Relaxed: {len(relaxed)} cases where SFT/SOLAR ratio > base+0.2)")

    # Sort by largest difference (biggest language switch)
    lang_switch_cases.sort(key=lambda x: x["sft_ratio"] + x["solar_ratio"] - 2 * x["base_ratio"], reverse=True)

    # Show top 3 examples
    print(f"\n  --- Top 3 language switch examples ---")
    for i, case in enumerate(lang_switch_cases[:3]):
        idx = case["idx"]
        print(f"\n  Example {i+1}: idx={idx}")
        print(f"    Target-lang ratio: Baseline={case['base_ratio']:.3f}, SFT={case['sft_ratio']:.3f}, SOLAR={case['solar_ratio']:.3f}")
        print(f"    Correct:           Baseline={case['base_correct']}, SFT={case['sft_correct']}, SOLAR={case['solar_correct']}")
        print(f"    Ground truth: {base_by_idx[idx]['ground_truth']}")

        print(f"\n    Baseline completion (first 400 chars):")
        base_comp = base_by_idx[idx]["completion"][0]
        print(f"      {base_comp[:400]}")

        print(f"\n    SFT completion (first 400 chars):")
        sft_comp = sft_by_idx[idx]["completion"][0]
        print(f"      {sft_comp[:400]}")

        print(f"\n    SOLAR completion (first 400 chars):")
        solar_comp = solar_by_idx[idx]["completion"][0]
        print(f"      {solar_comp[:400]}")


# =============================================================================
# BONUS: For Chinese specifically, also look at broader language distribution
# =============================================================================
print("\n\n" + "=" * 100)
print("BONUS: Detailed Chinese (zh) language analysis across ALL runs")
print("=" * 100)

lang = "zh"
common_runs = find_common_runs(lang)

# Aggregate ratios across all runs
from collections import Counter

base_all = []
sft_all = []
solar_all = []

for run in common_runs:
    base_data = load_json(os.path.join(RESULTS_DIR, BASE_TEMPLATE.format(run=run, lang=lang)))
    sft_data = load_json(os.path.join(RESULTS_DIR, SFT_TEMPLATE.format(run=run, lang=lang)))
    solar_data = load_json(os.path.join(RESULTS_DIR, SOLAR_TEMPLATE.format(run=run, lang=lang)))

    for e in base_data:
        base_all.append(target_lang_ratio(get_reasoning_text(e, False), lang))
    for e in sft_data:
        sft_all.append(target_lang_ratio(get_reasoning_text(e, True), lang))
    for e in solar_data:
        solar_all.append(target_lang_ratio(get_reasoning_text(e, True), lang))

print(f"\n  Across {len(common_runs)} runs, {len(base_all)} total entries per model:")
print(f"  Baseline avg CJK ratio: {sum(base_all)/len(base_all):.3f}")
print(f"  SFT avg CJK ratio:      {sum(sft_all)/len(sft_all):.3f}")
print(f"  SOLAR avg CJK ratio:    {sum(solar_all)/len(solar_all):.3f}")

# Histogram-style breakdown
bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]

print(f"\n  CJK ratio distribution:")
print(f"  {'Bin':>12s}  {'Baseline':>10s}  {'SFT':>10s}  {'SOLAR':>10s}")
for lo, hi in bins:
    b_cnt = sum(1 for r in base_all if lo <= r < hi)
    s_cnt = sum(1 for r in sft_all if lo <= r < hi)
    sol_cnt = sum(1 for r in solar_all if lo <= r < hi)
    print(f"  [{lo:.1f}, {hi:.1f})  {b_cnt:>10d}  {s_cnt:>10d}  {sol_cnt:>10d}")

print("\n\nDone!")

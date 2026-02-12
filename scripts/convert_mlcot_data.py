"""
Convert Multilingual-Long-CoT benchmarks (MT-MATH-500, MMLU-ProX-Lite)
to Soft-Thinking JSON format for evaluation.

Usage:
    python scripts/convert_mlcot_data.py
"""

import json
import os
from datasets import load_from_disk

MLCOT_DIR = "/coc/pskynet6/jhe478/Multilingual-Long-CoT/eval/benchmarks"
OUTPUT_DIR = "/coc/pskynet6/jhe478/Soft-Thinking/datasets"

MATH500_LANGUAGES = ["en", "zh", "fr", "ja", "lv", "sw", "te", "th", "af", "mr"]
MMLU_LANGUAGES = ["en", "zh", "fr", "ja", "sw", "te", "th", "af", "mr"]  # no Latvian

LANGUAGE_NAMES = {
    "en": "English", "zh": "Chinese", "fr": "French", "ja": "Japanese",
    "lv": "Latvian", "sw": "Swahili", "te": "Telugu", "th": "Thai",
    "af": "Afrikaans", "mr": "Marathi",
}


def convert_math500():
    """Convert MT-MATH-500 to Soft-Thinking format."""
    print("Loading MT-MATH-500...")
    ds = load_from_disk(os.path.join(MLCOT_DIR, "mt-math-500"))
    print(f"  Loaded {len(ds)} rows, columns: {ds.column_names}")

    all_samples = []
    for lang in MATH500_LANGUAGES:
        lang_samples = []
        for idx, row in enumerate(ds):
            sample = {
                "prompt": [{"from": "user", "value": row[lang]}],
                "final_answer": str(row["answer"]),
                "question_id": idx,
                "language": lang,
                "language_name": LANGUAGE_NAMES[lang],
            }
            lang_samples.append(sample)
            all_samples.append(sample)

        # Write per-language file
        outpath = os.path.join(OUTPUT_DIR, f"mt_math500_{lang}.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(lang_samples, f, ensure_ascii=False, indent=2)
        print(f"  Wrote {len(lang_samples)} samples to {outpath}")

    # Write combined file
    outpath = os.path.join(OUTPUT_DIR, "mt_math500_all.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {len(all_samples)} total samples to {outpath}")


def convert_mmlu():
    """Convert MMLU-ProX-Lite to Soft-Thinking format."""
    print("\nLoading MMLU-ProX-Lite...")
    ds = load_from_disk(os.path.join(MLCOT_DIR, "mmlu-prox-lite"))
    print(f"  Loaded {len(ds)} rows, columns: {ds.column_names}")

    all_samples = []
    for lang in MMLU_LANGUAGES:
        lang_samples = []
        for idx, row in enumerate(ds):
            question_text = row[lang]
            # Add boxed answer instruction (matching GPQA convention)
            question_text += "\n\nPlease write your final answer as \\boxed{LETTER}."
            sample = {
                "prompt": [{"from": "user", "value": question_text}],
                "final_answer": f"\\boxed{{{row['answer']}}}",
                "choices": {},
                "question_id": idx,
                "language": lang,
                "language_name": LANGUAGE_NAMES[lang],
            }
            lang_samples.append(sample)
            all_samples.append(sample)

        outpath = os.path.join(OUTPUT_DIR, f"mmlu_prox_{lang}.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(lang_samples, f, ensure_ascii=False, indent=2)
        print(f"  Wrote {len(lang_samples)} samples to {outpath}")

    outpath = os.path.join(OUTPUT_DIR, "mmlu_prox_all.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {len(all_samples)} total samples to {outpath}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    convert_math500()
    convert_mmlu()
    print("\nDone!")

"""
Download MGSM dataset to local JSON files.

Run this script on a machine with internet access before running evaluation
on compute nodes without internet.

Usage:
    python datasets/download_mgsm.py
"""

from datasets import load_dataset
import json
import os

# All MGSM languages
MGSM_LANGUAGES = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"]

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "th": "Thai",
    "sw": "Swahili",
    "bn": "Bengali",
    "te": "Telugu"
}


def download_mgsm(output_dir=None):
    """Download MGSM dataset for all languages and save to JSON files."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    all_data = {}

    for lang in MGSM_LANGUAGES:
        print(f"Downloading MGSM {lang} ({LANGUAGE_NAMES[lang]})...")
        try:
            ds = load_dataset("juletxara/mgsm", lang)

            data = []
            for idx, example in enumerate(ds["test"]):
                data.append({
                    "prompt": [{"from": "user", "value": example["question"]}],
                    "final_answer": str(example["answer_number"]),
                    "question_id": idx,
                    "language": lang,
                    "language_name": LANGUAGE_NAMES[lang],
                    "full_answer": example["answer"],
                    "equation_solution": example["equation_solution"]
                })

            all_data[lang] = data

            # Save individual language file
            lang_file = os.path.join(output_dir, f"mgsm_{lang}.json")
            with open(lang_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(data)} samples to {lang_file}")

        except Exception as e:
            print(f"  Error downloading {lang}: {e}")
            all_data[lang] = []

    # Save combined file
    combined_file = os.path.join(output_dir, "mgsm_all.json")
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved combined data to {combined_file}")

    # Print summary
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    for lang, data in all_data.items():
        print(f"  {lang} ({LANGUAGE_NAMES[lang]:10s}): {len(data)} samples")
    print(f"\nTotal: {sum(len(d) for d in all_data.values())} samples")


if __name__ == "__main__":
    download_mgsm()

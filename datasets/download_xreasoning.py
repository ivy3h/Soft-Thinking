"""
Download XReasoning datasets to local JSON files.

XReasoning datasets contain all languages in a single split with columns:
id, question, answer

The language is typically encoded in the 'id' field (e.g., 'en_0', 'zh_1').

Usage:
    python datasets/download_xreasoning.py
    python datasets/download_xreasoning.py --dataset aime2024
"""

from datasets import load_dataset
import json
import os
import argparse
import re

# All XReasoning languages (same as MGSM)
XREASONING_LANGUAGES = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"]

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

# Dataset HuggingFace paths
XREASONING_DATASETS = {
    "aime2024": "shanchen/aime_2024_multilingual",
    "aime2025": "shanchen/aime_2025_multilingual",
    "gpqa": "shanchen/gpqa_diamond_mc_multilingual"
}


def extract_language_from_id(sample_id):
    """Extract language code from sample ID.

    Expected formats:
    - 'en_0', 'zh_1', etc.
    - 'en-0', 'zh-1', etc.
    - 'english_0', 'chinese_0', etc.
    """
    sample_id = str(sample_id).lower()

    # Direct language code match at start
    for lang in XREASONING_LANGUAGES:
        if sample_id.startswith(lang + "_") or sample_id.startswith(lang + "-"):
            return lang

    # Full language name match
    lang_name_map = {
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "russian": "ru",
        "chinese": "zh",
        "japanese": "ja",
        "thai": "th",
        "swahili": "sw",
        "bengali": "bn",
        "telugu": "te"
    }

    for name, code in lang_name_map.items():
        if sample_id.startswith(name):
            return code

    return None


def download_xreasoning(dataset_name, output_dir=None):
    """Download XReasoning dataset and organize by language."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    hf_dataset = XREASONING_DATASETS[dataset_name]
    all_data = {lang: [] for lang in XREASONING_LANGUAGES}

    print(f"\nDownloading {dataset_name} from {hf_dataset}")
    print("=" * 50)

    try:
        # Load dataset
        ds = load_dataset(hf_dataset, trust_remote_code=True)

        # Get available splits
        print(f"Available splits: {list(ds.keys())}")

        # Use test split if available, otherwise first split
        split_name = "test" if "test" in ds else list(ds.keys())[0]
        print(f"Using split: {split_name}")
        print(f"Total samples: {len(ds[split_name])}")

        # Print first sample to understand structure
        if len(ds[split_name]) > 0:
            first_sample = ds[split_name][0]
            print(f"Sample columns: {list(first_sample.keys())}")
            print(f"First sample: {first_sample}")

        # Process each sample
        lang_question_ids = {lang: 0 for lang in XREASONING_LANGUAGES}

        for example in ds[split_name]:
            sample_id = example.get("id", "")
            question = example.get("question", "")
            answer = example.get("answer", "")

            # Extract language from ID
            lang = extract_language_from_id(sample_id)

            if lang is None:
                # Try to detect language from other fields or default to en
                print(f"  Warning: Could not determine language for id={sample_id}")
                continue

            if lang not in XREASONING_LANGUAGES:
                continue

            sample = {
                "prompt": [{"from": "user", "value": str(question)}],
                "final_answer": str(answer),
                "question_id": lang_question_ids[lang],
                "original_id": sample_id,
                "language": lang,
                "language_name": LANGUAGE_NAMES.get(lang, lang)
            }

            # For GPQA, add choices if available
            if dataset_name == "gpqa":
                choices = {}
                for choice_key in ["A", "B", "C", "D"]:
                    if choice_key in example:
                        choices[choice_key] = str(example[choice_key])
                if choices:
                    sample["choices"] = choices

            all_data[lang].append(sample)
            lang_question_ids[lang] += 1

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

    # Save individual language files
    for lang, data in all_data.items():
        if data:
            lang_file = os.path.join(output_dir, f"xreasoning_{dataset_name}_{lang}.json")
            with open(lang_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(data)} samples to {lang_file}")

    # Save combined file
    combined_file = os.path.join(output_dir, f"xreasoning_{dataset_name}_all.json")
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved combined data to {combined_file}")

    # Print summary
    print("\n" + "=" * 50)
    print(f"{dataset_name} Download Summary")
    print("=" * 50)
    for lang, data in all_data.items():
        print(f"  {lang} ({LANGUAGE_NAMES[lang]:10s}): {len(data)} samples")
    print(f"\nTotal: {sum(len(d) for d in all_data.values())} samples")

    # Warn if missing languages
    missing = [lang for lang, data in all_data.items() if not data]
    if missing:
        print(f"\nWARNING: Missing languages: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(description='Download XReasoning datasets')
    parser.add_argument('--dataset', type=str, choices=['aime2024', 'aime2025', 'gpqa', 'all'],
                        default='all', help='Dataset to download (default: all)')
    args = parser.parse_args()

    if args.dataset == 'all':
        for ds_name in ['aime2024', 'aime2025', 'gpqa']:
            download_xreasoning(ds_name)
    else:
        download_xreasoning(args.dataset)


if __name__ == "__main__":
    main()

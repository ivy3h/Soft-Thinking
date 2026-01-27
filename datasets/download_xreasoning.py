"""
Download XReasoning datasets to local JSON files.

Run this script on a machine with internet access before running evaluation
on compute nodes without internet.

Usage:
    python datasets/download_xreasoning.py
    python datasets/download_xreasoning.py --dataset aime2024
"""

from datasets import load_dataset
import json
import os
import argparse

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


def download_xreasoning(dataset_name, output_dir=None):
    """Download XReasoning dataset for all languages and save to JSON files."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    hf_dataset = XREASONING_DATASETS[dataset_name]
    all_data = {lang: [] for lang in XREASONING_LANGUAGES}

    print(f"\nDownloading {dataset_name} from {hf_dataset}")
    print("=" * 50)

    # First try loading with language as config
    try:
        print("Trying to load with language configs...")
        for lang in XREASONING_LANGUAGES:
            print(f"  Trying {lang}...")
            ds = load_dataset(hf_dataset, lang)
            split_name = "test" if "test" in ds else list(ds.keys())[0]

            data = []
            for idx, example in enumerate(ds[split_name]):
                question = (example.get("question") or example.get("problem") or
                           example.get("Question") or example.get("Problem"))
                answer = (example.get("answer") or example.get("Answer") or
                         example.get("final_answer") or example.get("correct_answer"))

                sample = {
                    "prompt": [{"from": "user", "value": question}],
                    "final_answer": str(answer),
                    "question_id": idx,
                    "language": lang,
                    "language_name": LANGUAGE_NAMES[lang]
                }

                if dataset_name == "gpqa":
                    choices = {}
                    for choice_key in ["A", "B", "C", "D"]:
                        if choice_key in example:
                            choices[choice_key] = example[choice_key]
                    if choices:
                        sample["choices"] = choices

                data.append(sample)

            all_data[lang] = data
            print(f"    Loaded {len(data)} samples")

    except Exception as e:
        print(f"  Language configs not available: {e}")
        print("  Trying 'default' config and filtering by language field...")

        # Load default config and filter by language
        try:
            ds = load_dataset(hf_dataset, "default")
            split_name = "test" if "test" in ds else list(ds.keys())[0]

            print(f"  Found {len(ds[split_name])} total samples")
            print(f"  Sample keys: {list(ds[split_name][0].keys()) if len(ds[split_name]) > 0 else 'N/A'}")

            # Group by language
            for example in ds[split_name]:
                # Try to find language field
                lang = (example.get("language") or example.get("lang") or
                       example.get("Language") or "en")

                if lang not in XREASONING_LANGUAGES:
                    continue

                question = (example.get("question") or example.get("problem") or
                           example.get("Question") or example.get("Problem"))
                answer = (example.get("answer") or example.get("Answer") or
                         example.get("final_answer") or example.get("correct_answer"))

                idx = len(all_data[lang])
                sample = {
                    "prompt": [{"from": "user", "value": question}],
                    "final_answer": str(answer),
                    "question_id": idx,
                    "language": lang,
                    "language_name": LANGUAGE_NAMES.get(lang, lang)
                }

                if dataset_name == "gpqa":
                    choices = {}
                    for choice_key in ["A", "B", "C", "D"]:
                        if choice_key in example:
                            choices[choice_key] = example[choice_key]
                    if choices:
                        sample["choices"] = choices

                all_data[lang].append(sample)

        except Exception as e2:
            print(f"  Error loading default config: {e2}")

            # Try loading without any config
            try:
                print("  Trying to load without config...")
                ds = load_dataset(hf_dataset)
                split_name = "test" if "test" in ds else list(ds.keys())[0]

                print(f"  Found {len(ds[split_name])} total samples")
                if len(ds[split_name]) > 0:
                    print(f"  Sample keys: {list(ds[split_name][0].keys())}")
                    print(f"  First sample: {ds[split_name][0]}")

                for example in ds[split_name]:
                    lang = (example.get("language") or example.get("lang") or
                           example.get("Language") or "en")

                    if lang not in XREASONING_LANGUAGES:
                        continue

                    question = (example.get("question") or example.get("problem") or
                               example.get("Question") or example.get("Problem"))
                    answer = (example.get("answer") or example.get("Answer") or
                             example.get("final_answer") or example.get("correct_answer"))

                    idx = len(all_data[lang])
                    sample = {
                        "prompt": [{"from": "user", "value": question}],
                        "final_answer": str(answer),
                        "question_id": idx,
                        "language": lang,
                        "language_name": LANGUAGE_NAMES.get(lang, lang)
                    }

                    if dataset_name == "gpqa":
                        choices = {}
                        for choice_key in ["A", "B", "C", "D"]:
                            if choice_key in example:
                                choices[choice_key] = example[choice_key]
                        if choices:
                            sample["choices"] = choices

                    all_data[lang].append(sample)

            except Exception as e3:
                print(f"  Error: {e3}")

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

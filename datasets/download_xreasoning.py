"""
Download XReasoning datasets to local JSON files.

Run this script on a machine with internet access before running evaluation
on compute nodes without internet.

Usage:
    python datasets/download_xreasoning.py
    python datasets/download_xreasoning.py --dataset aime2024
    python datasets/download_xreasoning.py --dataset aime2025
    python datasets/download_xreasoning.py --dataset gpqa
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
    all_data = {}

    print(f"\nDownloading {dataset_name} from {hf_dataset}")
    print("=" * 50)

    for lang in XREASONING_LANGUAGES:
        print(f"Downloading {dataset_name} {lang} ({LANGUAGE_NAMES[lang]})...")
        try:
            ds = load_dataset(hf_dataset, lang)
            split_name = "test" if "test" in ds else list(ds.keys())[0]

            data = []
            for idx, example in enumerate(ds[split_name]):
                # Handle different column naming conventions
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

                # For GPQA, add choices if available
                if dataset_name == "gpqa":
                    choices = {}
                    for choice_key in ["A", "B", "C", "D"]:
                        if choice_key in example:
                            choices[choice_key] = example[choice_key]
                    if choices:
                        sample["choices"] = choices

                data.append(sample)

            all_data[lang] = data

            # Save individual language file
            lang_file = os.path.join(output_dir, f"xreasoning_{dataset_name}_{lang}.json")
            with open(lang_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(data)} samples to {lang_file}")

        except Exception as e:
            print(f"  Error downloading {lang}: {e}")
            all_data[lang] = []

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

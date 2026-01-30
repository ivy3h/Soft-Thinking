"""
Download XReasoning datasets to local JSON files.

XReasoning datasets have splits named by language codes (en, zh, ja, etc.)
Each split contains: id, question, answer

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

        # Get available splits (these are the language codes)
        available_splits = list(ds.keys())
        print(f"Available splits (languages): {available_splits}")

        # Process each language split
        for lang in XREASONING_LANGUAGES:
            if lang not in ds:
                print(f"  {lang}: not available in dataset")
                continue

            print(f"Processing {lang} ({LANGUAGE_NAMES[lang]})...")

            data = []
            for idx, example in enumerate(ds[lang]):
                # HF datasets use different field names: "problem" for AIME, "question" for GPQA
                question = (example.get("problem") or example.get("question") or
                           example.get("Question") or example.get("Problem") or "")
                answer = (example.get("answer") or example.get("Answer") or
                         example.get("final_answer") or example.get("correct_answer") or "")
                sample = {
                    "prompt": [{"from": "user", "value": str(question)}],
                    "final_answer": str(answer),
                    "question_id": idx,
                    "original_id": str(example.get("id", idx)),
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

                data.append(sample)

            all_data[lang] = data
            print(f"  Loaded {len(data)} samples")

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

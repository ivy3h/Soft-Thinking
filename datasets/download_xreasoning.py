"""
Download XReasoning datasets to local JSON files.

XReasoning datasets are organized with separate parquet files per language.
This script downloads and converts them to JSON format.

Usage:
    python datasets/download_xreasoning.py
    python datasets/download_xreasoning.py --dataset aime2024
"""

from datasets import load_dataset
import json
import os
import argparse
import requests
import pandas as pd
from io import BytesIO

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

# Dataset HuggingFace paths - using JRQi versions which have proper multilingual support
XREASONING_DATASETS = {
    "aime2024": "JRQi/aime_2024_multilingual",
    "aime2025": "JRQi/aime_2025_multilingual",
    "gpqa": "JRQi/gpqa_diamond_mc_multilingual"
}

# Fallback to shanchen versions
XREASONING_DATASETS_FALLBACK = {
    "aime2024": "shanchen/aime_2024_multilingual",
    "aime2025": "shanchen/aime_2025_multilingual",
    "gpqa": "shanchen/gpqa_diamond_mc_multilingual"
}


def download_parquet_direct(repo_id, lang):
    """Download parquet file directly from HuggingFace."""
    # Try different parquet file naming patterns
    patterns = [
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/{lang}-00000-of-00001.parquet",
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/{lang}/train-00000-of-00001.parquet",
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/{lang}.parquet",
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/{lang}.parquet",
    ]

    for url in patterns:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                df = pd.read_parquet(BytesIO(response.content))
                return df
        except Exception:
            continue
    return None


def process_dataframe(df, lang, dataset_name):
    """Convert dataframe to our standard format."""
    data = []

    for idx, row in df.iterrows():
        # Handle different column naming conventions
        question = None
        answer = None

        # Try different question column names
        for q_col in ["question", "problem", "Question", "Problem", "prompt"]:
            if q_col in row and pd.notna(row[q_col]):
                question = str(row[q_col])
                break

        # Try different answer column names
        for a_col in ["answer", "Answer", "final_answer", "correct_answer", "solution"]:
            if a_col in row and pd.notna(row[a_col]):
                answer = str(row[a_col])
                break

        if question is None:
            continue

        sample = {
            "prompt": [{"from": "user", "value": question}],
            "final_answer": answer if answer else "",
            "question_id": idx,
            "language": lang,
            "language_name": LANGUAGE_NAMES.get(lang, lang)
        }

        # For GPQA, add choices if available
        if dataset_name == "gpqa":
            choices = {}
            for choice_key in ["A", "B", "C", "D"]:
                if choice_key in row and pd.notna(row[choice_key]):
                    choices[choice_key] = str(row[choice_key])
            if choices:
                sample["choices"] = choices

        data.append(sample)

    return data


def download_xreasoning(dataset_name, output_dir=None):
    """Download XReasoning dataset for all languages and save to JSON files."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    all_data = {lang: [] for lang in XREASONING_LANGUAGES}

    # Try primary and fallback dataset sources
    for repo_source, repos in [("JRQi", XREASONING_DATASETS), ("shanchen", XREASONING_DATASETS_FALLBACK)]:
        hf_dataset = repos[dataset_name]

        print(f"\nDownloading {dataset_name} from {hf_dataset}")
        print("=" * 50)

        success_count = 0

        for lang in XREASONING_LANGUAGES:
            if all_data[lang]:  # Already have data for this language
                continue

            print(f"Downloading {lang} ({LANGUAGE_NAMES[lang]})...")

            # Method 1: Try load_dataset with language as config
            try:
                ds = load_dataset(hf_dataset, lang, trust_remote_code=True)
                split_name = "test" if "test" in ds else "train" if "train" in ds else list(ds.keys())[0]

                data = []
                for idx, example in enumerate(ds[split_name]):
                    question = (example.get("question") or example.get("problem") or
                               example.get("Question") or example.get("Problem") or
                               example.get("prompt"))
                    answer = (example.get("answer") or example.get("Answer") or
                             example.get("final_answer") or example.get("correct_answer") or
                             example.get("solution"))

                    sample = {
                        "prompt": [{"from": "user", "value": str(question) if question else ""}],
                        "final_answer": str(answer) if answer else "",
                        "question_id": idx,
                        "language": lang,
                        "language_name": LANGUAGE_NAMES[lang]
                    }

                    if dataset_name == "gpqa":
                        choices = {}
                        for choice_key in ["A", "B", "C", "D"]:
                            if choice_key in example:
                                choices[choice_key] = str(example[choice_key])
                        if choices:
                            sample["choices"] = choices

                    data.append(sample)

                if data:
                    all_data[lang] = data
                    print(f"  Loaded {len(data)} samples via load_dataset")
                    success_count += 1
                    continue

            except Exception as e:
                print(f"  load_dataset failed: {e}")

            # Method 2: Try direct parquet download
            try:
                df = download_parquet_direct(hf_dataset, lang)
                if df is not None and len(df) > 0:
                    data = process_dataframe(df, lang, dataset_name)
                    if data:
                        all_data[lang] = data
                        print(f"  Loaded {len(data)} samples via direct parquet")
                        success_count += 1
                        continue
            except Exception as e:
                print(f"  Direct parquet failed: {e}")

            print(f"  Failed to download {lang}")

        if success_count > 0:
            print(f"\nSuccessfully downloaded {success_count} languages from {repo_source}")
            break

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

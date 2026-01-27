"""
Download MGSM dataset to local JSON files.

Run this script on a machine with internet access before running evaluation
on compute nodes without internet.

Usage:
    python datasets/download_mgsm.py
    python datasets/download_mgsm.py --force  # Force re-download
    python datasets/download_mgsm.py --use-http  # Download via HTTP (bypass HF cache)
"""

from datasets import load_dataset
import json
import os
import argparse
import requests

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

# Direct URLs for MGSM TSV files
MGSM_TSV_URLS = {
    lang: f"https://huggingface.co/datasets/juletxara/mgsm/resolve/main/mgsm_{lang}.tsv"
    for lang in MGSM_LANGUAGES
}


def download_via_http(output_dir):
    """Download MGSM dataset directly via HTTP from TSV files."""
    all_data = {}

    for lang in MGSM_LANGUAGES:
        print(f"Downloading MGSM {lang} ({LANGUAGE_NAMES[lang]}) via HTTP...")
        url = MGSM_TSV_URLS[lang]

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse TSV content
            lines = response.text.strip().split('\n')
            data = []

            for idx, line in enumerate(lines):
                if not line.strip():
                    continue
                # TSV format: question\tanswer (with equation and number embedded)
                parts = line.split('\t')
                if len(parts) >= 2:
                    question = parts[0]
                    answer_text = parts[1]

                    # Extract answer number from answer text
                    # Format is usually: "equation #### number"
                    if '####' in answer_text:
                        answer_parts = answer_text.split('####')
                        answer_number = answer_parts[-1].strip()
                        equation = answer_parts[0].strip() if len(answer_parts) > 1 else ""
                    else:
                        # Try to extract last number
                        import re
                        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
                        answer_number = numbers[-1] if numbers else answer_text
                        equation = ""

                    data.append({
                        "prompt": [{"from": "user", "value": question}],
                        "final_answer": str(answer_number).replace(",", ""),
                        "question_id": idx,
                        "language": lang,
                        "language_name": LANGUAGE_NAMES[lang],
                        "full_answer": answer_text,
                        "equation_solution": equation
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

    return all_data


def download_via_hf(output_dir, force=False):
    """Download MGSM dataset via HuggingFace datasets library."""
    all_data = {}

    download_mode = "force_redownload" if force else None

    for lang in MGSM_LANGUAGES:
        print(f"Downloading MGSM {lang} ({LANGUAGE_NAMES[lang]})...")
        try:
            ds = load_dataset("juletxara/mgsm", lang, download_mode=download_mode)

            data = []
            split_name = "test" if "test" in ds else list(ds.keys())[0]
            for idx, example in enumerate(ds[split_name]):
                data.append({
                    "prompt": [{"from": "user", "value": example["question"]}],
                    "final_answer": str(example["answer_number"]),
                    "question_id": idx,
                    "language": lang,
                    "language_name": LANGUAGE_NAMES[lang],
                    "full_answer": example["answer"],
                    "equation_solution": example.get("equation_solution", "")
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

    return all_data


def download_mgsm(output_dir=None, force=False, use_http=False):
    """Download MGSM dataset for all languages and save to JSON files."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    if use_http:
        print("Using HTTP direct download method...")
        all_data = download_via_http(output_dir)
    else:
        print("Using HuggingFace datasets library...")
        all_data = download_via_hf(output_dir, force)

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

    # Warn if any language failed
    failed = [lang for lang, data in all_data.items() if len(data) == 0]
    if failed:
        print(f"\nWARNING: Failed to download: {', '.join(failed)}")
        if not use_http:
            print("Try running with --use-http flag to download directly via HTTP")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download MGSM dataset')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download (clear cache)')
    parser.add_argument('--use-http', action='store_true',
                        help='Download directly via HTTP (bypass HuggingFace cache)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for JSON files')
    args = parser.parse_args()

    download_mgsm(args.output_dir, args.force, args.use_http)

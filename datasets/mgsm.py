"""
MGSM Dataset Loader

MGSM (Multilingual Grade School Math) is a benchmark containing 250 math problems
from GSM8K, each translated into 10 languages by human annotators.

Languages: en, es, fr, de, ru, zh, ja, th, sw, bn, te (11 total including English)

Dataset source: juletxara/mgsm on HuggingFace
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


def load_mgsm_single_language(lang: str):
    """Load MGSM dataset for a single language.

    Args:
        lang: Language code (e.g., 'en', 'zh', 'ja')

    Returns:
        List of samples with 'prompt', 'final_answer', 'question_id', 'language' keys
    """
    ds = load_dataset("juletxara/mgsm", lang)

    data = []
    for idx, example in enumerate(ds["test"]):
        data.append({
            "prompt": [
                {
                    "from": "user",
                    "value": example["question"]
                }
            ],
            "final_answer": str(example["answer_number"]),
            "question_id": idx,  # Each language has the same 250 questions in the same order
            "language": lang,
            "language_name": LANGUAGE_NAMES[lang],
            "full_answer": example["answer"],
            "equation_solution": example["equation_solution"]
        })
    return data


def load_mgsm_all_languages():
    """Load MGSM dataset for all 11 languages.

    Returns:
        Dict mapping language code to list of samples
    """
    all_data = {}
    for lang in MGSM_LANGUAGES:
        print(f"Loading MGSM {lang} ({LANGUAGE_NAMES[lang]})...")
        all_data[lang] = load_mgsm_single_language(lang)
    return all_data


def create_mgsm_json_files(output_dir: str = "."):
    """Create JSON files for MGSM dataset.

    Creates:
    - mgsm_{lang}.json for each language
    - mgsm_all.json containing all languages (organized by language)
    """
    all_data = load_mgsm_all_languages()

    # Save individual language files
    for lang, data in all_data.items():
        filepath = os.path.join(output_dir, f"mgsm_{lang}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Saved {filepath} ({len(data)} samples)")

    # Save combined file
    combined_filepath = os.path.join(output_dir, "mgsm_all.json")
    with open(combined_filepath, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    print(f"Saved {combined_filepath}")

    return all_data


if __name__ == "__main__":
    create_mgsm_json_files()

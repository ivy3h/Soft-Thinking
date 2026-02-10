"""
XReasoning Dataset Loader

XReasoning is a multilingual reasoning benchmark containing:
- shanchen/aime_2024_multilingual: AIME 2024 problems in multiple languages
- shanchen/aime_2025_multilingual: AIME 2025 problems in multiple languages
- shanchen/gpqa_diamond_mc_multilingual: GPQA Diamond multiple choice in multiple languages

Languages: en, es, fr, de, ru, zh, ja, th, sw, bn, te (11 total, same as MGSM)

Reference: "When Models Reason in Your Language: Controlling Thinking Trace Language
Comes at the Cost of Accuracy" (arXiv:2505.22888)
"""

from datasets import load_dataset
import json
import os

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

# Dataset configurations
XREASONING_DATASETS = {
    "aime2024": "shanchen/aime_2024_multilingual",
    "aime2025": "shanchen/aime_2025_multilingual",
    "gpqa": "shanchen/gpqa_diamond_mc_multilingual"
}


def load_xreasoning_aime(dataset_name: str, lang: str):
    """Load AIME multilingual dataset for a single language.

    Args:
        dataset_name: Either 'aime2024' or 'aime2025'
        lang: Language code (e.g., 'en', 'zh', 'ja')

    Returns:
        List of samples with 'prompt', 'final_answer', 'question_id', 'language' keys
    """
    hf_dataset = XREASONING_DATASETS[dataset_name]
    ds = load_dataset(hf_dataset, lang)

    data = []
    # AIME datasets typically have 'test' or 'train' split
    split_name = "test" if "test" in ds else list(ds.keys())[0]

    for idx, example in enumerate(ds[split_name]):
        # Expected columns: question/problem, answer
        question = example.get("question") or example.get("problem") or example.get("Question")
        answer = example.get("answer") or example.get("Answer") or example.get("final_answer")

        data.append({
            "prompt": [{"from": "user", "value": question}],
            "final_answer": str(answer),
            "question_id": idx,
            "language": lang,
            "language_name": LANGUAGE_NAMES[lang]
        })
    return data


def load_xreasoning_gpqa(lang: str):
    """Load GPQA Diamond multilingual dataset for a single language.

    Args:
        lang: Language code (e.g., 'en', 'zh', 'ja')

    Returns:
        List of samples with 'prompt', 'final_answer', 'choices', 'question_id', 'language' keys
    """
    hf_dataset = XREASONING_DATASETS["gpqa"]
    ds = load_dataset(hf_dataset)

    data = []
    # GPQA uses language codes as split names (e.g., 'en', 'zh'), not configs
    split_name = lang if lang in ds else ("test" if "test" in ds else list(ds.keys())[0])

    for idx, example in enumerate(ds[split_name]):
        # Columns: problem, solution (e.g. "\boxed{D}"), domain
        question = (example.get("problem") or example.get("question") or
                    example.get("Question"))
        answer = (example.get("solution") or example.get("answer") or
                  example.get("Answer") or example.get("correct_answer"))

        # Build choices dict if available
        choices = {}
        for choice_key in ["A", "B", "C", "D"]:
            if choice_key in example:
                choices[choice_key] = example[choice_key]

        # If choices are embedded in question or separate field
        if not choices and "choices" in example:
            choices = example["choices"]

        data.append({
            "prompt": [{"from": "user", "value": question}],
            "final_answer": str(answer),
            "choices": choices,
            "question_id": idx,
            "language": lang,
            "language_name": LANGUAGE_NAMES[lang]
        })
    return data


def load_xreasoning_single_language(dataset_name: str, lang: str):
    """Load XReasoning dataset for a single language.

    Args:
        dataset_name: 'aime2024', 'aime2025', or 'gpqa'
        lang: Language code

    Returns:
        List of samples
    """
    if dataset_name in ["aime2024", "aime2025"]:
        return load_xreasoning_aime(dataset_name, lang)
    elif dataset_name == "gpqa":
        return load_xreasoning_gpqa(lang)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_xreasoning_all_languages(dataset_name: str, languages=None):
    """Load XReasoning dataset for all languages.

    Args:
        dataset_name: 'aime2024', 'aime2025', or 'gpqa'
        languages: List of language codes. If None, loads all languages.

    Returns:
        Dict mapping language code to list of samples
    """
    if languages is None:
        languages = XREASONING_LANGUAGES

    all_data = {}
    for lang in languages:
        print(f"Loading {dataset_name} {lang} ({LANGUAGE_NAMES.get(lang, lang)})...")
        try:
            all_data[lang] = load_xreasoning_single_language(dataset_name, lang)
            print(f"  Loaded {len(all_data[lang])} samples")
        except Exception as e:
            print(f"  Error loading {lang}: {e}")
            all_data[lang] = []

    return all_data


def create_xreasoning_json_files(dataset_name: str, output_dir: str = "."):
    """Create JSON files for XReasoning dataset.

    Creates:
    - {dataset_name}_{lang}.json for each language
    - {dataset_name}_all.json containing all languages
    """
    all_data = load_xreasoning_all_languages(dataset_name)

    # Save individual language files
    for lang, data in all_data.items():
        if data:
            filepath = os.path.join(output_dir, f"xreasoning_{dataset_name}_{lang}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Saved {filepath} ({len(data)} samples)")

    # Save combined file
    combined_filepath = os.path.join(output_dir, f"xreasoning_{dataset_name}_all.json")
    with open(combined_filepath, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    print(f"Saved {combined_filepath}")

    return all_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["aime2024", "aime2025", "gpqa", "all"],
                        default="all", help="Dataset to download")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds_name in ["aime2024", "aime2025", "gpqa"]:
            print(f"\n{'='*60}")
            print(f"Processing {ds_name}")
            print(f"{'='*60}")
            create_xreasoning_json_files(ds_name)
    else:
        create_xreasoning_json_files(args.dataset)

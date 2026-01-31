#!/usr/bin/env python3
"""
Analyze evaluation results and find missing configurations.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# Define the evaluation matrix
MODELS = [
    "Qwen3-4B-Instruct-2507",
    "Qwen3-8B",
    "Qwen3-8B-Base"
]

METHODS = {
    "CoT": False,           # soft_thinking=False
    "Soft-Thinking": True   # soft_thinking=True
}

DATASETS = {
    "MGSM": ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"],
    "XReasoning-AIME2024": ["en"],
    "XReasoning-AIME2025": ["en"],
    "XReasoning-GPQA": ["en"]
}

RESULTS_DIR = Path("/nethome/jhe478/flash/Soft-Thinking/results/results")


def parse_filename(filename):
    """Parse result filename to extract configuration."""
    # Format: {model}_{dataset}_{soft_thinking}_{...}_{lang}.json
    parts = filename.replace(".json", "").split("_")

    # Find model name (can be multi-part like "Qwen3-4B-Instruct-2507")
    model = None
    for m in MODELS:
        if filename.startswith(m.replace("-", "-")):
            model = m
            break

    # Extract soft_thinking flag (True/False)
    soft_thinking = None
    if "_True_" in filename:
        soft_thinking = True
    elif "_False_" in filename:
        soft_thinking = False

    # Extract language (last part before .json)
    lang = parts[-1] if parts else None

    # Extract dataset from directory
    dataset = None

    return model, soft_thinking, lang, dataset


def load_result_file(filepath):
    """Load a result JSON file and extract accuracy."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Handle different result formats
        if isinstance(data, dict):
            if 'accuracy' in data:
                return data['accuracy']
            elif 'results' in data and isinstance(data['results'], list):
                # Calculate accuracy from results list
                correct = sum(1 for r in data['results'] if r.get('correct', False))
                total = len(data['results'])
                return correct / total if total > 0 else 0.0

        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def analyze_results():
    """Analyze all results and create summary tables."""

    # Collect all results
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Scan MGSM results
    mgsm_dir = RESULTS_DIR / "mgsm"
    if mgsm_dir.exists():
        for file in mgsm_dir.glob("*.json"):
            model, soft_thinking, lang, _ = parse_filename(file.name)
            if model and soft_thinking is not None and lang:
                method = "Soft-Thinking" if soft_thinking else "CoT"
                accuracy = load_result_file(file)
                if accuracy is not None:
                    results[model]["MGSM"][method][lang] = accuracy

    # Scan XReasoning results
    for xr_dataset in ["xreasoning_aime2024", "xreasoning_aime2025", "xreasoning_gpqa"]:
        xr_dir = RESULTS_DIR / xr_dataset
        if xr_dir.exists():
            dataset_name = xr_dataset.replace("xreasoning_", "XReasoning-").upper()
            if xr_dataset == "xreasoning_gpqa":
                dataset_name = "XReasoning-GPQA"
            elif xr_dataset == "xreasoning_aime2024":
                dataset_name = "XReasoning-AIME2024"
            elif xr_dataset == "xreasoning_aime2025":
                dataset_name = "XReasoning-AIME2025"

            for file in xr_dir.glob("*.json"):
                model, soft_thinking, _, _ = parse_filename(file.name)
                if model and soft_thinking is not None:
                    method = "Soft-Thinking" if soft_thinking else "CoT"
                    accuracy = load_result_file(file)
                    if accuracy is not None:
                        results[model][dataset_name][method]["en"] = accuracy

    return results


def calculate_metrics(lang_accuracies):
    """Calculate average accuracy and language consistency."""
    if not lang_accuracies:
        return None, None

    accuracies = list(lang_accuracies.values())
    avg_acc = sum(accuracies) / len(accuracies)

    # Language consistency: 1 - std_dev (lower std = higher consistency)
    if len(accuracies) > 1:
        mean = avg_acc
        variance = sum((x - mean) ** 2 for x in accuracies) / len(accuracies)
        std_dev = variance ** 0.5
        consistency = 1 - std_dev  # Simple consistency metric
    else:
        consistency = 1.0

    return avg_acc, consistency


def print_results_table(results):
    """Print formatted results tables."""

    for model in MODELS:
        print(f"\n{'='*100}")
        print(f"Model: {model}")
        print(f"{'='*100}\n")

        if model not in results:
            print(f"  No results found for {model}\n")
            continue

        for dataset in ["MGSM", "XReasoning-AIME2024", "XReasoning-AIME2025", "XReasoning-GPQA"]:
            if dataset not in results[model]:
                continue

            print(f"\n{'-'*100}")
            print(f"Dataset: {dataset}")
            print(f"{'-'*100}")

            # Get expected languages
            if dataset == "MGSM":
                expected_langs = DATASETS["MGSM"]
            else:
                expected_langs = ["en"]

            for method in ["CoT", "Soft-Thinking"]:
                if method not in results[model][dataset]:
                    continue

                lang_accs = results[model][dataset][method]

                print(f"\nMethod: {method}")
                print(f"  {'Language':<10} {'Accuracy':>10}")
                print(f"  {'-'*22}")

                for lang in expected_langs:
                    if lang in lang_accs:
                        acc = lang_accs[lang]
                        print(f"  {lang:<10} {acc:>9.2%}")
                    else:
                        print(f"  {lang:<10} {'MISSING':>10}")

                # Calculate metrics
                avg_acc, consistency = calculate_metrics(lang_accs)
                if avg_acc is not None:
                    print(f"  {'-'*22}")
                    print(f"  {'Average':<10} {avg_acc:>9.2%}")
                    if len(lang_accs) > 1:
                        print(f"  {'Consistency':<10} {consistency:>9.3f}")


def find_missing_configs(results):
    """Find missing evaluation configurations."""

    missing = []

    for model in MODELS:
        for dataset, langs in DATASETS.items():
            for method_name, soft_thinking in METHODS.items():
                # Check if this configuration exists
                has_results = False

                if model in results:
                    if dataset in results[model]:
                        if method_name in results[model][dataset]:
                            lang_results = results[model][dataset][method_name]
                            # Check if all expected languages are present
                            if all(lang in lang_results for lang in langs):
                                has_results = True
                            else:
                                missing_langs = [l for l in langs if l not in lang_results]
                                missing.append({
                                    "model": model,
                                    "dataset": dataset,
                                    "method": method_name,
                                    "status": "partial",
                                    "missing_langs": missing_langs
                                })
                                has_results = True  # Partial results

                if not has_results:
                    missing.append({
                        "model": model,
                        "dataset": dataset,
                        "method": method_name,
                        "status": "missing",
                        "missing_langs": langs
                    })

    return missing


def print_missing_configs(missing):
    """Print missing configurations."""

    print(f"\n\n{'='*100}")
    print("MISSING EVALUATION CONFIGURATIONS")
    print(f"{'='*100}\n")

    # Group by status
    fully_missing = [m for m in missing if m["status"] == "missing"]
    partial_missing = [m for m in missing if m["status"] == "partial"]

    if fully_missing:
        print(f"\nFully Missing ({len(fully_missing)} configurations):")
        print(f"{'-'*100}")
        print(f"{'Model':<30} {'Dataset':<25} {'Method':<15}")
        print(f"{'-'*100}")
        for m in fully_missing:
            print(f"{m['model']:<30} {m['dataset']:<25} {m['method']:<15}")

    if partial_missing:
        print(f"\n\nPartially Complete ({len(partial_missing)} configurations):")
        print(f"{'-'*100}")
        print(f"{'Model':<30} {'Dataset':<25} {'Method':<15} {'Missing Languages'}")
        print(f"{'-'*100}")
        for m in partial_missing:
            langs_str = ", ".join(m["missing_langs"])
            print(f"{m['model']:<30} {m['dataset']:<25} {m['method']:<15} {langs_str}")

    total_missing = len(fully_missing)
    total_partial = len(partial_missing)
    total_expected = len(MODELS) * len(DATASETS) * len(METHODS)
    total_complete = total_expected - total_missing - total_partial

    print(f"\n{'='*100}")
    print(f"Summary:")
    print(f"  Total expected configurations: {total_expected}")
    print(f"  Complete: {total_complete}")
    print(f"  Partially complete: {total_partial}")
    print(f"  Missing: {total_missing}")
    print(f"{'='*100}\n")


def main():
    print("Analyzing evaluation results...")
    print(f"Results directory: {RESULTS_DIR}\n")

    # Analyze results
    results = analyze_results()

    # Print results tables
    print_results_table(results)

    # Find and print missing configurations
    missing = find_missing_configs(results)
    print_missing_configs(missing)


if __name__ == "__main__":
    main()

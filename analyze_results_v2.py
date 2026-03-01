#!/usr/bin/env python3
"""
Analyze evaluation results and find missing configurations.
Uses statistics.json files for MGSM and individual result files for XReasoning.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
import glob

# Define the evaluation matrix
# IMPORTANT: Order by length (longest first) to match correctly
MODELS = [
    "Qwen3-4B-Instruct-2507",
    "Qwen3-8B-Base",  # Must come before Qwen3-8B
    "Qwen3-8B"
]

METHODS = ["CoT", "Soft-Thinking"]

DATASETS = {
    "MGSM": ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"],
    "XReasoning-AIME2024": ["en"],
    "XReasoning-AIME2025": ["en"],
    "XReasoning-GPQA": ["en"]
}

RESULTS_DIR = Path("/nethome/jhe478/flash/Soft-Thinking/results/results")


def parse_statistics_filename(filename):
    """
    Parse statistics filename to extract model and method.
    Format: {model}_{dataset}_{soft_thinking}_..._statistics.json
    """
    # Extract model
    model = None
    for m in MODELS:
        if filename.startswith(m):
            model = m
            break

    if model is None:
        return None, None

    # Extract soft_thinking flag
    if "_True_" in filename:
        method = "Soft-Thinking"
    elif "_False_" in filename:
        method = "CoT"
    else:
        method = None

    return model, method


def load_statistics_file(filepath):
    """Load a statistics.json file and return per-language accuracies and metrics."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        return {
            'per_language_accuracy': data.get('per_language_accuracy', {}),
            'average_accuracy': data.get('average_accuracy', None),
            'consistency': data.get('cross_lingual_consistency', {}).get('average_clc', None)
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_xreasoning_result(filepath):
    """Load XReasoning result file and calculate accuracy."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            # List of results
            correct = sum(1 for r in data if r.get('correct', False))
            total = len(data)
            return correct / total if total > 0 else 0.0
        elif isinstance(data, dict):
            if 'accuracy' in data:
                return data['accuracy']
            elif 'results' in data:
                results = data['results']
                correct = sum(1 for r in results if r.get('correct', False))
                total = len(results)
                return correct / total if total > 0 else 0.0

        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def analyze_results():
    """Analyze all results and create summary tables."""

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    # 1. Analyze MGSM results (from statistics files)
    mgsm_dir = RESULTS_DIR / "mgsm"
    if mgsm_dir.exists():
        for stats_file in mgsm_dir.glob("*_statistics.json"):
            model, method = parse_statistics_filename(stats_file.name)
            if model and method:
                stats = load_statistics_file(stats_file)
                if stats:
                    results[model]["MGSM"][method]['lang_accuracies'] = stats['per_language_accuracy']
                    results[model]["MGSM"][method]['average_accuracy'] = stats['average_accuracy']
                    results[model]["MGSM"][method]['consistency'] = stats['consistency']

    # 2. Analyze XReasoning results
    xr_mappings = {
        "xreasoning_aime2024": "XReasoning-AIME2024",
        "xreasoning_aime2025": "XReasoning-AIME2025",
        "xreasoning_gpqa": "XReasoning-GPQA"
    }

    for xr_dir_name, dataset_name in xr_mappings.items():
        xr_dir = RESULTS_DIR / xr_dir_name
        if xr_dir.exists():
            for result_file in xr_dir.glob("*.json"):
                # Parse filename
                filename = result_file.name
                model = None
                for m in MODELS:
                    if filename.startswith(m):
                        model = m
                        break

                if model:
                    if "_True_" in filename:
                        method = "Soft-Thinking"
                    elif "_False_" in filename:
                        method = "CoT"
                    else:
                        continue

                    accuracy = load_xreasoning_result(result_file)
                    if accuracy is not None:
                        results[model][dataset_name][method]['lang_accuracies'] = {'en': accuracy}
                        results[model][dataset_name][method]['average_accuracy'] = accuracy
                        results[model][dataset_name][method]['consistency'] = 1.0  # Single language

    return results


def print_results_table(results):
    """Print formatted results tables."""

    print("\n" + "="*120)
    print("EVALUATION RESULTS SUMMARY")
    print("="*120)

    for model in MODELS:
        print(f"\n{'='*120}")
        print(f"Model: {model}")
        print(f"{'='*120}")

        if model not in results or not results[model]:
            print(f"  No results found for {model}\n")
            continue

        for dataset in ["MGSM", "XReasoning-AIME2024", "XReasoning-AIME2025", "XReasoning-GPQA"]:
            if dataset not in results[model] or not results[model][dataset]:
                continue

            print(f"\n{'-'*120}")
            print(f"Dataset: {dataset}")
            print(f"{'-'*120}")

            # Get expected languages
            expected_langs = DATASETS[dataset]

            for method in METHODS:
                if method not in results[model][dataset]:
                    continue

                method_data = results[model][dataset][method]
                lang_accs = method_data.get('lang_accuracies', {})
                avg_acc = method_data.get('average_accuracy')
                consistency = method_data.get('consistency')

                print(f"\nMethod: {method}")

                if dataset == "MGSM":
                    # Show all languages for MGSM
                    print(f"  {'Language':<12} {'Accuracy':>10}")
                    print(f"  {'-'*24}")

                    for lang in expected_langs:
                        if lang in lang_accs:
                            acc = lang_accs[lang]
                            print(f"  {lang:<12} {acc:>9.2%}")
                        else:
                            print(f"  {lang:<12} {'MISSING':>10}")

                    print(f"  {'-'*24}")
                    if avg_acc is not None:
                        print(f"  {'Average':<12} {avg_acc:>9.2%}")
                    if consistency is not None:
                        print(f"  {'Consistency':<12} {consistency:>9.3f}")
                else:
                    # For XReasoning, just show accuracy
                    if avg_acc is not None:
                        print(f"  Accuracy: {avg_acc:.2%}")
                    else:
                        print(f"  No results available")


def find_missing_configs(results):
    """Find missing evaluation configurations."""

    missing = []

    for model in MODELS:
        for dataset, expected_langs in DATASETS.items():
            for method in METHODS:
                # Check if this configuration exists
                if model in results and dataset in results[model] and method in results[model][dataset]:
                    method_data = results[model][dataset][method]
                    lang_accs = method_data.get('lang_accuracies', {})

                    # Check if all expected languages are present
                    missing_langs = [lang for lang in expected_langs if lang not in lang_accs]

                    if missing_langs:
                        missing.append({
                            "model": model,
                            "dataset": dataset,
                            "method": method,
                            "status": "partial",
                            "missing_langs": missing_langs
                        })
                else:
                    # Configuration completely missing
                    missing.append({
                        "model": model,
                        "dataset": dataset,
                        "method": method,
                        "status": "missing",
                        "missing_langs": expected_langs
                    })

    return missing


def print_missing_configs(missing):
    """Print missing configurations."""

    print(f"\n\n{'='*120}")
    print("MISSING EVALUATION CONFIGURATIONS")
    print(f"{'='*120}\n")

    # Group by status
    fully_missing = [m for m in missing if m["status"] == "missing"]
    partial_missing = [m for m in missing if m["status"] == "partial"]

    if fully_missing:
        print(f"\nFully Missing ({len(fully_missing)} configurations):")
        print(f"{'-'*120}")
        print(f"{'Model':<35} {'Dataset':<30} {'Method':<20}")
        print(f"{'-'*120}")
        for m in fully_missing:
            print(f"{m['model']:<35} {m['dataset']:<30} {m['method']:<20}")

    if partial_missing:
        print(f"\n\nPartially Complete ({len(partial_missing)} configurations):")
        print(f"{'-'*120}")
        print(f"{'Model':<35} {'Dataset':<30} {'Method':<20} {'Missing Languages'}")
        print(f"{'-'*120}")
        for m in partial_missing:
            langs_str = ", ".join(m["missing_langs"])
            print(f"{m['model']:<35} {m['dataset']:<30} {m['method']:<20} {langs_str}")

    total_missing = len(fully_missing)
    total_partial = len(partial_missing)
    total_expected = len(MODELS) * len(DATASETS) * len(METHODS)
    total_complete = total_expected - total_missing - total_partial

    print(f"\n{'='*120}")
    print(f"Summary:")
    print(f"  Total expected configurations: {total_expected}")
    print(f"  Complete: {total_complete}")
    print(f"  Partially complete: {total_partial}")
    print(f"  Missing: {total_missing}")
    print(f"  Completion rate: {total_complete}/{total_expected} ({100*total_complete/total_expected:.1f}%)")
    print(f"{'='*120}\n")


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

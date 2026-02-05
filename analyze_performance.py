#!/usr/bin/env python3
"""
Analyze model performance across all benchmarks and output a CSV summary.

Produces 4 tables (one per benchmark: MGSM, AIME2024, AIME2025, GPQA)
with per-language accuracy, average accuracy, and cross-lingual consistency.
"""

import json
import os
import csv
import re
from collections import defaultdict
from pathlib import Path
from itertools import combinations

RESULTS_DIR = Path(__file__).parent / "results" / "results"

LANGUAGES = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"]

# Models to detect (order by length descending to avoid prefix collision)
KNOWN_MODELS = [
    "Llama-3.1-8B-Instruct",
    "Qwen3-4B-Instruct-2507",
    "mla_qwen3_0.6b_final",
    "Qwen3-8B-Base",
    "Qwen3-8B",
]


def detect_model(filename):
    """Extract model name from filename (longest prefix match)."""
    for m in KNOWN_MODELS:
        if filename.startswith(m):
            return m
    return None


def detect_setting(filename):
    """Extract True/False (Soft-Thinking / CoT) setting from filename."""
    if "_True_" in filename:
        return "Soft-Thinking"
    elif "_False_" in filename:
        return "CoT"
    return None


def detect_max_tokens(filename):
    """Extract max_generated_tokens from filename pattern like ..._15_16384_0.0_..."""
    m = re.search(r"_(\d+)_(\d{4,6})_0\.0_", filename)
    if m:
        return int(m.group(2))
    return None


# ──────────────────────────────────────────────
#  MGSM: read from statistics files
# ──────────────────────────────────────────────
def load_mgsm_results():
    """Load MGSM results from statistics JSON files."""
    rows = []
    mgsm_dir = RESULTS_DIR / "mgsm"
    if not mgsm_dir.exists():
        return rows

    # Detect duplicates (same model+setting but different max_tokens)
    configs = defaultdict(list)
    for f in sorted(mgsm_dir.glob("*_statistics.json")):
        model = detect_model(f.name)
        setting = detect_setting(f.name)
        if not model or not setting:
            continue
        max_tok = detect_max_tokens(f.name)
        configs[(model, setting)].append((f, max_tok))

    has_dup = {k for k, v in configs.items() if len(v) > 1}

    for (model, setting), entries in configs.items():
        for f, max_tok in entries:
            with open(f) as fp:
                data = json.load(fp)

            lang_acc = data.get("per_language_accuracy", {})
            avg_acc = data.get("average_accuracy")
            consistency = data.get("cross_lingual_consistency", {}).get("average_consistency")
            correct_consistency = data.get("cross_lingual_consistency", {}).get("average_correct_consistency")

            display_setting = setting
            if (model, setting) in has_dup and max_tok:
                display_setting = f"{setting} ({max_tok//1024}K)"

            rows.append({
                "Model": model,
                "Setting": display_setting,
                **{lang: lang_acc.get(lang) for lang in LANGUAGES},
                "Average": avg_acc,
                "Consistency": consistency,
                "Correct_Consistency": correct_consistency,
            })

    return rows


# ──────────────────────────────────────────────
#  AIME / GPQA: compute from per-run result files
# ──────────────────────────────────────────────
def compute_per_language_accuracy(results_list):
    """Given a list of result dicts, compute per-language accuracy."""
    lang_scores = defaultdict(list)
    for r in results_list:
        lang_scores[r["language"]].append(r["passat1"])
    return {lang: sum(v) / len(v) for lang, v in lang_scores.items()}


def compute_consistency(results_list):
    """
    Compute cross-lingual consistency (CO = both_correct / either_correct).
    """
    lang_results = defaultdict(dict)
    for r in results_list:
        lang_results[r["language"]][r["question_id"]] = r["passat1"] > 0

    available = [l for l in LANGUAGES if l in lang_results]
    if len(available) < 2:
        return None

    total_both = 0
    total_either = 0
    for l1, l2 in combinations(available, 2):
        common = set(lang_results[l1]) & set(lang_results[l2])
        for qid in common:
            c1, c2 = lang_results[l1][qid], lang_results[l2][qid]
            if c1 and c2:
                total_both += 1
            if c1 or c2:
                total_either += 1

    return total_both / total_either if total_either > 0 else 0.0


def load_xreasoning_results(benchmark):
    """
    Load xreasoning results for a given benchmark (aime2024 / aime2025 / gpqa).
    Handles both per-run JSON files and statistics-only files.
    """
    rows = []
    xr_dir = RESULTS_DIR / f"xreasoning_{benchmark}"
    if not xr_dir.exists():
        return rows

    # Group run files by (model, setting)
    run_files = defaultdict(list)  # (model, setting) -> [filepath, ...]
    stat_files = {}  # (model, setting) -> filepath

    for f in sorted(xr_dir.iterdir()):
        if not f.name.endswith(".json"):
            continue
        model = detect_model(f.name)
        setting = detect_setting(f.name)
        if not model or not setting:
            continue

        if f.name.endswith("_statistics.json"):
            stat_files[(model, setting)] = f
        else:
            run_files[(model, setting)].append(f)

    # Process configurations that have raw run files
    seen = set()
    for (model, setting), files in run_files.items():
        all_lang_accs = defaultdict(list)  # lang -> [acc_per_run]
        all_results_for_consistency = []

        for fp in files:
            with open(fp) as fh:
                data = json.load(fh)
            lang_acc = compute_per_language_accuracy(data)
            for lang, acc in lang_acc.items():
                all_lang_accs[lang].append(acc)
            all_results_for_consistency = data  # use last run for consistency

        # Average across runs
        avg_lang = {lang: sum(accs) / len(accs) for lang, accs in all_lang_accs.items()}
        overall_avg = sum(avg_lang.values()) / len(avg_lang) if avg_lang else None
        consistency = compute_consistency(all_results_for_consistency)

        rows.append({
            "Model": model,
            "Setting": setting,
            **{lang: avg_lang.get(lang) for lang in LANGUAGES},
            "Average": overall_avg,
            "Consistency": consistency,
            "Correct_Consistency": None,
        })
        seen.add((model, setting))

    # For configs with only statistics files (e.g. GPQA)
    for (model, setting), fp in stat_files.items():
        if (model, setting) in seen:
            continue
        with open(fp) as fh:
            data = json.load(fh)

        lang_acc = data.get("per_language_accuracy", {})
        avg_acc = data.get("average_accuracy")
        consistency = data.get("cross_lingual_consistency", {}).get("average_consistency")

        rows.append({
            "Model": model,
            "Setting": setting,
            **{lang: lang_acc.get(lang) for lang in LANGUAGES},
            "Average": avg_acc,
            "Consistency": consistency,
            "Correct_Consistency": None,
        })

    return rows


def format_val(v):
    """Format a numeric value as percentage string, or '-' if None."""
    if v is None:
        return "-"
    return f"{v * 100:.1f}"


def write_csv(all_tables, output_path):
    """Write all tables to a single CSV file with section separators."""
    columns = ["Model", "Setting"] + LANGUAGES + ["Average", "Consistency", "Correct_Consistency"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for i, (benchmark, rows) in enumerate(all_tables.items()):
            if i > 0:
                writer.writerow([])  # blank separator

            writer.writerow([f"=== {benchmark} ==="])
            writer.writerow(columns)

            # Sort rows: by model name then setting
            rows.sort(key=lambda r: (r["Model"], r["Setting"]))

            for row in rows:
                csv_row = [
                    row["Model"],
                    row["Setting"],
                ] + [
                    format_val(row.get(lang)) for lang in LANGUAGES
                ] + [
                    format_val(row.get("Average")),
                    format_val(row.get("Consistency")),
                    format_val(row.get("Correct_Consistency")),
                ]
                writer.writerow(csv_row)

    print(f"CSV saved to: {output_path}")


def print_table(benchmark, rows):
    """Pretty-print a benchmark table to stdout."""
    if not rows:
        print(f"\n  (no results)\n")
        return

    rows.sort(key=lambda r: (r["Model"], r["Setting"]))

    # Header
    lang_header = "".join(f"{l:>6}" for l in LANGUAGES)
    print(f"  {'Model':<28} {'Setting':<15} {lang_header} {'Avg':>6} {'CO':>6} {'CC':>6}")
    print(f"  {'-'*28} {'-'*15} {'-'*66} {'-'*6} {'-'*6} {'-'*6}")

    for row in rows:
        lang_vals = "".join(f"{format_val(row.get(l)):>6}" for l in LANGUAGES)
        avg_str = f"{format_val(row.get('Average')):>6}"
        co_str = f"{format_val(row.get('Consistency')):>6}"
        cc_str = f"{format_val(row.get('Correct_Consistency')):>6}"
        print(f"  {row['Model']:<28} {row['Setting']:<15} {lang_vals} {avg_str} {co_str} {cc_str}")


def main():
    print("Scanning results directory:", RESULTS_DIR)
    print()

    all_tables = {}

    # MGSM
    mgsm_rows = load_mgsm_results()
    all_tables["MGSM"] = mgsm_rows

    # XReasoning benchmarks
    for bench in ["aime2024", "aime2025", "gpqa"]:
        xr_rows = load_xreasoning_results(bench)
        all_tables[f"XReasoning-{bench.upper()}"] = xr_rows

    # Print to stdout
    for benchmark, rows in all_tables.items():
        print(f"{'=' * 130}")
        print(f"  {benchmark}  (CO=Consistency, CC=Correct Consistency)")
        print(f"{'=' * 130}")
        print_table(benchmark, rows)
        print()

    # Write CSV
    output_path = RESULTS_DIR.parent / "performance_summary.csv"
    write_csv(all_tables, output_path)

    print(f"\nDone. CSV written to: {output_path}")


if __name__ == "__main__":
    main()
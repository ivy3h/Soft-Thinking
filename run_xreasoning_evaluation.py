"""
XReasoning Multilingual Evaluation Script

This script evaluates models on the XReasoning benchmark across all 11 languages
and computes:
1. Per-language accuracy
2. Average accuracy across all languages
3. Cross-lingual consistency (pairwise and overall)

Supports three datasets:
- aime2024: AIME 2024 multilingual (default 5 runs for averaging due to small size)
- aime2025: AIME 2025 multilingual (default 5 runs for averaging due to small size)
- gpqa: GPQA Diamond multilingual
"""

import sglang as sgl
import json
import time
from tqdm import tqdm
import argparse
import os
from transformers import AutoTokenizer
from sglang.srt.sampling.sampling_params import SamplingParams
from matheval import evaluator_map, set_client
import asyncio
import matheval
from huggingface_hub import HfApi
import torch
from itertools import combinations
from collections import defaultdict
from datasets import load_dataset

# Language metadata (same as MGSM)
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

# Default number of evaluation runs (AIME has small dataset, so run multiple times)
DEFAULT_NUM_RUNS = {
    "aime2024": 5,
    "aime2025": 5,
    "gpqa": 1
}


def load_xreasoning_data(dataset_name: str, languages=None, data_dir="./datasets"):
    """Load XReasoning data for specified languages.

    First tries to load from local JSON files, then falls back to HuggingFace Hub.

    Args:
        dataset_name: 'aime2024', 'aime2025', or 'gpqa'
        languages: List of language codes. If None, loads all languages.
        data_dir: Directory containing local JSON files.

    Returns:
        Dict mapping language code to list of samples
    """
    if languages is None:
        languages = XREASONING_LANGUAGES

    all_data = {}

    # Try to load from local combined file first
    combined_file = os.path.join(data_dir, f"xreasoning_{dataset_name}_all.json")
    if os.path.exists(combined_file):
        print(f"Loading {dataset_name} from local file: {combined_file}")
        with open(combined_file, "r", encoding="utf-8") as f:
            local_data = json.load(f)
        for lang in languages:
            if lang in local_data:
                all_data[lang] = local_data[lang]
                print(f"  Loaded {len(all_data[lang])} samples for {lang}")
            else:
                print(f"  Warning: {lang} not found in local data")
        return all_data

    # Try to load from individual language files
    all_local = True
    for lang in languages:
        lang_file = os.path.join(data_dir, f"xreasoning_{dataset_name}_{lang}.json")
        if os.path.exists(lang_file):
            print(f"Loading {dataset_name} {lang} from local file...")
            with open(lang_file, "r", encoding="utf-8") as f:
                all_data[lang] = json.load(f)
            print(f"  Loaded {len(all_data[lang])} samples for {lang}")
        else:
            all_local = False
            break

    if all_local:
        return all_data

    # Fall back to HuggingFace Hub
    print("Local files not found, loading from HuggingFace Hub...")
    hf_dataset = XREASONING_DATASETS[dataset_name]
    all_data = {}

    for lang in languages:
        print(f"Loading {dataset_name} {lang} ({LANGUAGE_NAMES.get(lang, lang)})...")
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
                    "language_name": LANGUAGE_NAMES.get(lang, lang)
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
            print(f"  Loaded {len(data)} samples for {lang}")
        except Exception as e:
            print(f"  Error loading {lang}: {e}")
            all_data[lang] = []

    return all_data


def calculate_cross_lingual_consistency(results_by_lang, languages):
    """Calculate cross-lingual consistency metrics.

    Consistency (CO) is calculated as:
    CO = both_correct / either_correct

    This matches the tinker-cookbook implementation.
    """
    lang_results = {}
    for lang in languages:
        if lang not in results_by_lang:
            continue
        lang_results[lang] = {}
        for r in results_by_lang[lang]:
            qid = r["question_id"]
            lang_results[lang][qid] = {
                "correct": r["judge_info"][0]["finally_judge_result"] if r["judge_info"] else False,
                "passat1": r["passat1"]
            }

    available_langs = [l for l in languages if l in lang_results]

    pairwise_consistency = {}

    for lang1, lang2 in combinations(available_langs, 2):
        both_correct_count = 0
        either_correct_count = 0

        common_qids = set(lang_results[lang1].keys()) & set(lang_results[lang2].keys())

        for qid in common_qids:
            r1 = lang_results[lang1][qid]
            r2 = lang_results[lang2][qid]

            # Count both correct
            if r1["correct"] and r2["correct"]:
                both_correct_count += 1

            # Count either correct (at least one correct)
            if r1["correct"] or r2["correct"]:
                either_correct_count += 1

        # Calculate consistency as both_correct / either_correct
        if either_correct_count > 0:
            consistency = both_correct_count / either_correct_count
        else:
            consistency = 0.0

        pairwise_consistency[(lang1, lang2)] = {
            "both_correct": both_correct_count,
            "either_correct": either_correct_count,
            "consistency": consistency,
            "total_questions": len(common_qids)
        }

    # Calculate average consistency
    if pairwise_consistency:
        average_consistency = sum(v["consistency"] for v in pairwise_consistency.values()) / len(pairwise_consistency)
    else:
        average_consistency = 0.0

    return {
        "pairwise_consistency": {f"{k[0]}-{k[1]}": v for k, v in pairwise_consistency.items()},
        "average_consistency": average_consistency,
        "num_language_pairs": len(pairwise_consistency)
    }


def main():
    parser = argparse.ArgumentParser(description='XReasoning Multilingual Evaluation')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['aime2024', 'aime2025', 'gpqa'],
                        help='XReasoning dataset to evaluate')
    parser.add_argument('--languages', type=str, nargs='+', default=None,
                        help='Languages to evaluate (default: all 11 languages)')
    parser.add_argument('--num_runs', type=int, default=None,
                        help='Number of evaluation runs (default: 5 for AIME, 1 for GPQA)')

    # Model parameters
    parser.add_argument('--sampling_backend', type=str, choices=["pytorch", "flashinfer"],
                        default="flashinfer", help='Sampling backend')
    parser.add_argument('--attention_backend', type=str,
                        choices=["triton", "flashinfer", "torch_native", "fa3"],
                        default="triton", help='Attention backend (default: triton, disables flash-attn)')
    parser.add_argument('--model_name', type=str, required=True, default="Qwen/QwQ-32B",
                        help='Model name or path')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='GPU number (tensor parallel size)')
    parser.add_argument('--cuda_graph_max_bs', type=int, default=None,
                        help='Max batch size for CUDA graph')
    parser.add_argument('--max_running_requests', type=int, default=None,
                        help='Max concurrent requests')
    parser.add_argument('--max_batch', type=int, default=1000000,
                        help='Max batch size')
    parser.add_argument('--mem_fraction_static', type=float, default=0.5,
                        help='Static memory fraction per GPU')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--output_dir', type=str, default="results",
                        help='Output directory')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index for samples')
    parser.add_argument('--end_idx', type=int, default=10000,
                        help='End index for samples')

    # Sampling parameters
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples per prompt')
    parser.add_argument('--max_generated_tokens', type=int, default=32768,
                        help='Max generated tokens')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p')
    parser.add_argument('--top_k', type=int, default=30, help='Top-k')
    parser.add_argument('--min_p', type=float, default=0.0, help='Min-p')
    parser.add_argument('--after_thinking_temperature', type=float, default=0.6)
    parser.add_argument('--after_thinking_top_p', type=float, default=0.95)
    parser.add_argument('--after_thinking_top_k', type=int, default=30)
    parser.add_argument('--after_thinking_min_p', type=float, default=0.0)
    parser.add_argument('--early_stopping_entropy_threshold', type=float, default=0.0)
    parser.add_argument('--early_stopping_length_threshold', type=int, default=256)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    # Noise parameters
    parser.add_argument('--dirichlet_alpha', type=float, default=1.0)
    parser.add_argument('--gumbel_softmax_temperature', type=float, default=1.0)
    parser.add_argument('--add_noise_dirichlet', action='store_true')
    parser.add_argument('--add_noise_gumbel_softmax', action='store_true')

    # Evaluation parameters
    parser.add_argument('--use_llm_judge', action='store_true', help='Enable LLM judge')
    parser.add_argument('--api_base', type=str, default=None)
    parser.add_argument('--deployment_name', type=str, default=None)
    parser.add_argument('--api_version', type=str, default=None)
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--judge_model_name', type=str, default="gpt-4.1-2025-04-14")
    parser.add_argument('--push_results_to_hf', action='store_true')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--hf_repo_id', type=str, default=None)

    # Soft thinking parameters
    parser.add_argument("--enable_soft_thinking", action="store_true")
    parser.add_argument("--think_end_str", type=str, default="</think>")
    parser.add_argument("--max_topk", type=int, default=15)

    args = parser.parse_args()

    dataset_name = args.dataset
    languages = args.languages if args.languages else XREASONING_LANGUAGES
    num_runs = args.num_runs if args.num_runs else DEFAULT_NUM_RUNS[dataset_name]

    print(f"Dataset: {dataset_name}")
    print(f"Evaluating on languages: {languages}")
    print(f"Number of evaluation runs: {num_runs}")

    # Setup LLM judge if needed
    matheval.set_client(args.api_base, args.deployment_name, args.api_version,
                        args.api_key, args.judge_model_name)

    # Load dataset
    xreasoning_data = load_xreasoning_data(dataset_name, languages)

    # Select appropriate evaluator
    if dataset_name in ["aime2024", "aime2025"]:
        evaluator_key = dataset_name  # Uses AIMEEvaluator
    else:
        evaluator_key = "gpqa_diamond"  # Uses GPQAEvaluator

    # Prompt templates
    MATH_QUERY_TEMPLATE = """
Please reason step by step, and put your final answer within \\boxed{{}}.

{Question}
""".strip()

    GPQA_QUERY_TEMPLATE = """
Please solve the following multiple-choice question. Please show your choice in the answer field with only the choice letter, e.g.,"answer": "C".

{Question}
""".strip()

    query_template = GPQA_QUERY_TEMPLATE if dataset_name == "gpqa" else MATH_QUERY_TEMPLATE

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Sampling params
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "after_thinking_temperature": args.after_thinking_temperature,
        "after_thinking_top_p": args.after_thinking_top_p,
        "after_thinking_top_k": args.after_thinking_top_k,
        "after_thinking_min_p": args.after_thinking_min_p,
        "n": 1,
        "gumbel_softmax_temperature": args.gumbel_softmax_temperature,
        "dirichlet_alpha": args.dirichlet_alpha,
        "max_new_tokens": args.max_generated_tokens,
        "think_end_str": args.think_end_str,
        "early_stopping_entropy_threshold": args.early_stopping_entropy_threshold,
        "early_stopping_length_threshold": args.early_stopping_length_threshold
    }

    # Output directory setup
    os.makedirs(f"{args.output_dir}/results/xreasoning_{dataset_name}", exist_ok=True)

    noise_suffix = (
        (f"_gumbel_{args.gumbel_softmax_temperature}" if args.add_noise_gumbel_softmax else "")
        + (f"_dirichlet_{args.dirichlet_alpha}" if args.add_noise_dirichlet else "")
    )
    base_filename = (
        f"{args.model_name.split('/')[-1]}_xreasoning_{dataset_name}_{args.enable_soft_thinking}_"
        f"{args.num_samples}_{args.temperature}_{args.top_p}_{args.top_k}_{args.min_p}_"
        f"{args.repetition_penalty}_{args.dirichlet_alpha}_{args.max_topk}_{args.max_generated_tokens}_"
        f"{args.early_stopping_entropy_threshold}_{args.early_stopping_length_threshold}{noise_suffix}"
    )

    print("=" * 60)
    print(f"Starting XReasoning Evaluation: {dataset_name}")
    print(f"Number of runs: {num_runs}")
    print("=" * 60)

    # Check if all runs already exist (complete resume)
    stats_file = f"{args.output_dir}/results/xreasoning_{dataset_name}/{base_filename}_statistics.json"
    if os.path.exists(stats_file):
        print(f"\n✓ Found existing complete evaluation results!")
        print(f"  Loading from: {stats_file}")
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                results_statistics = json.load(f)

            print(f"\n✓ Successfully loaded existing results:")
            print(f"  Dataset: {results_statistics.get('dataset', 'N/A')}")
            print(f"  Number of runs: {results_statistics.get('num_runs', 'N/A')}")
            print(f"  Average accuracy: {results_statistics.get('average_accuracy', 0):.4f}")
            print(f"\n✓ All evaluation already completed. Exiting.")
            return results_statistics
        except Exception as e:
            print(f"⚠ Failed to load existing statistics: {e}")
            print(f"⚠ Will re-run evaluation")

    start_time = time.time()

    # Store results across all runs
    all_runs_results = {lang: [] for lang in languages}
    all_runs_accuracies = {lang: [] for lang in languages}

    for run_idx in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{num_runs}")
        print(f"{'='*60}")

        # Check if this run already exists (resume functionality)
        if num_runs > 1:
            run_results_file = f"{args.output_dir}/results/xreasoning_{dataset_name}/{base_filename}_run{run_idx + 1}.json"
            if os.path.exists(run_results_file):
                print(f"✓ Run {run_idx + 1} already completed, loading existing results...")
                try:
                    with open(run_results_file, "r", encoding="utf-8") as f:
                        existing_results = json.load(f)

                    # Group results by language and calculate accuracies
                    results_by_lang = {}
                    for result in existing_results:
                        lang = result.get("language")
                        if lang not in results_by_lang:
                            results_by_lang[lang] = []
                        results_by_lang[lang].append(result)

                    # Calculate accuracies for each language
                    for lang, results in results_by_lang.items():
                        lang_accuracy = sum([r["passat1"] for r in results]) / len(results) if results else 0
                        all_runs_accuracies[lang].append(lang_accuracy)
                        all_runs_results[lang].append(results)
                        print(f"  {lang}: {lang_accuracy:.4f} ({lang_accuracy * 100:.2f}%)")

                    print(f"✓ Successfully loaded results from {run_results_file}")
                    continue  # Skip to next run
                except Exception as e:
                    print(f"⚠ Failed to load existing results: {e}")
                    print(f"⚠ Will re-run this iteration")

        results_by_lang = {}
        language_accuracies = {}

        # Evaluate each language
        for lang in languages:
            if not xreasoning_data.get(lang):
                print(f"Skipping {lang} - no data available")
                continue

            print(f"\n{'-'*40}")
            print(f"Evaluating {lang} ({LANGUAGE_NAMES.get(lang, lang)})")
            print(f"{'-'*40}")

            samples = xreasoning_data[lang]
            samples = samples[args.start_idx:min(args.end_idx, len(samples))]

            if not samples:
                print(f"No samples for {lang}")
                continue

            # Prepare prompts
            prompt_list = []
            idx_list = []

            for idx, sample in enumerate(samples):
                chat = [{"role": "user", "content": query_template.format(
                    Question=sample["prompt"][0]["value"]
                )}]
                prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True,
                                                       tokenize=False)

                for _ in range(args.num_samples):
                    prompt_list.append(prompt)
                idx_list.append(idx)

            # Generate responses
            decoded_text_list = []
            finish_generation_list = []
            generated_tokens_list = []

            batch_idx = 0
            while batch_idx < len(prompt_list):
                print(f"Processing batch {batch_idx // args.max_batch + 1}...")

                llm = sgl.Engine(
                    model_path=args.model_name,
                    tp_size=args.num_gpus,
                    log_level="info",
                    trust_remote_code=True,
                    random_seed=args.random_seed + run_idx,  # Different seed per run
                    max_running_requests=args.max_running_requests,
                    mem_fraction_static=args.mem_fraction_static,
                    disable_cuda_graph=True,
                    disable_overlap_schedule=True,
                    enable_soft_thinking=args.enable_soft_thinking,
                    add_noise_dirichlet=args.add_noise_dirichlet,
                    add_noise_gumbel_softmax=args.add_noise_gumbel_softmax,
                    max_topk=args.max_topk,
                    cuda_graph_max_bs=args.cuda_graph_max_bs,
                    sampling_backend=args.sampling_backend,
                    attention_backend=args.attention_backend
                )

                outputs = llm.generate(
                    prompt_list[batch_idx:batch_idx + args.max_batch],
                    sampling_params
                )

                decoded_text_list.extend([o["text"] for o in outputs])
                finish_generation_list.extend([
                    o["meta_info"]["finish_reason"]["type"] == "stop" and not args.enable_soft_thinking
                    for o in outputs
                ])
                generated_tokens_list.extend([
                    o["meta_info"]["completion_tokens"] for o in outputs
                ])

                batch_idx += args.max_batch
                llm.shutdown()
                torch.cuda.empty_cache()

            # Evaluate results
            results = []
            for i, idx in enumerate(idx_list):
                sample = samples[idx]
                judge_info = []
                passat1_list = []
                decoded_text = decoded_text_list[i * args.num_samples:(i + 1) * args.num_samples]
                finish_generation = finish_generation_list[i * args.num_samples:(i + 1) * args.num_samples]

                for j in range(args.num_samples):
                    for _ in range(5):  # Retry logic
                        try:
                            rule_judge_result, extracted_answer = evaluator_map[evaluator_key].rule_judge(
                                decoded_text[j], sample["final_answer"], finish_generation[j]
                            )

                            llm_judge_result = None
                            if not rule_judge_result and args.use_llm_judge:
                                llm_judge_result = evaluator_map[evaluator_key].llm_judge(
                                    decoded_text[j], sample["final_answer"],
                                    extracted_answer, finish_generation[j]
                                )

                            finally_judge_result = rule_judge_result or llm_judge_result

                            judge_info.append({
                                "rule_judge_result": rule_judge_result,
                                "llm_judge_result": llm_judge_result,
                                "finally_judge_result": finally_judge_result
                            })
                            passat1_list.append(1.0 if finally_judge_result else 0.0)
                            break
                        except Exception as e:
                            print(f"Error: {e}", flush=True)
                            time.sleep(0.5)

                result = {
                    "hyperparams": str(args),
                    "prompt": sample["prompt"][0]["value"],
                    "completion": decoded_text,
                    "ground_truth": sample["final_answer"],
                    "generated_tokens": generated_tokens_list[i * args.num_samples:(i + 1) * args.num_samples],
                    "avg_generated_tokens": sum(generated_tokens_list[i * args.num_samples:(i + 1) * args.num_samples]) / args.num_samples,
                    "time": 0,
                    "idx": idx,
                    "question_id": sample["question_id"],
                    "language": lang,
                    "run_idx": run_idx,
                    "n": args.num_samples,
                    "finish_generation": finish_generation_list[i * args.num_samples:(i + 1) * args.num_samples],
                    "judge_info": judge_info,
                    "passat1": sum(passat1_list) / len(passat1_list) if passat1_list else 0,
                    "passat1_list": passat1_list
                }
                results.append(result)

            # Calculate language accuracy for this run
            lang_accuracy = sum([r["passat1"] for r in results]) / len(results) if results else 0
            language_accuracies[lang] = lang_accuracy
            results_by_lang[lang] = results
            all_runs_accuracies[lang].append(lang_accuracy)
            all_runs_results[lang].append(results)

            print(f"{lang} Run {run_idx + 1} Accuracy: {lang_accuracy:.4f} ({lang_accuracy * 100:.2f}%)")

        # Save per-run results
        if num_runs > 1:
            run_results_file = f"{args.output_dir}/results/xreasoning_{dataset_name}/{base_filename}_run{run_idx + 1}.json"
            all_run_data = []
            for lang, res_list in results_by_lang.items():
                all_run_data.extend(res_list)
            with open(run_results_file, "w", encoding="utf-8") as f:
                json.dump(all_run_data, f, indent=4, ensure_ascii=False)

    # Calculate final averaged results
    final_language_accuracies = {}
    final_language_std = {}
    for lang in languages:
        if all_runs_accuracies[lang]:
            accs = all_runs_accuracies[lang]
            final_language_accuracies[lang] = sum(accs) / len(accs)
            if len(accs) > 1:
                mean = final_language_accuracies[lang]
                variance = sum((x - mean) ** 2 for x in accs) / len(accs)
                final_language_std[lang] = variance ** 0.5
            else:
                final_language_std[lang] = 0.0

    # Use last run's results for consistency calculation
    consistency_metrics = calculate_cross_lingual_consistency(results_by_lang, languages)

    end_time = time.time()

    # Calculate overall average
    average_accuracy = sum(final_language_accuracies.values()) / len(final_language_accuracies) if final_language_accuracies else 0

    # Compile final statistics
    results_statistics = {
        "dataset": dataset_name,
        "num_runs": num_runs,
        "languages_evaluated": languages,
        "num_languages": len(languages),
        "per_language_accuracy": {lang: acc for lang, acc in final_language_accuracies.items()},
        "per_language_std": {lang: std for lang, std in final_language_std.items()},
        "per_language_all_runs": {lang: accs for lang, accs in all_runs_accuracies.items()},
        "average_accuracy": average_accuracy,
        "cross_lingual_consistency": consistency_metrics,
        "time_taken_hours": (end_time - start_time) / 3600
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"XReasoning {dataset_name} Evaluation Summary")
    print(f"Number of runs: {num_runs}")
    print("=" * 60)

    print("\nPer-Language Accuracy (averaged over runs):")
    for lang, acc in sorted(final_language_accuracies.items(), key=lambda x: -x[1]):
        std = final_language_std.get(lang, 0)
        if num_runs > 1:
            print(f"  {lang} ({LANGUAGE_NAMES.get(lang, lang):10s}): {acc:.4f} ± {std:.4f} ({acc * 100:.2f}%)")
        else:
            print(f"  {lang} ({LANGUAGE_NAMES.get(lang, lang):10s}): {acc:.4f} ({acc * 100:.2f}%)")

    print(f"\nAverage Accuracy: {average_accuracy:.4f} ({average_accuracy * 100:.2f}%)")
    print(f"Average Cross-Lingual Consistency (CO): {consistency_metrics['average_consistency']:.4f}")
    print(f"  (CO = both_correct / either_correct, matching tinker-cookbook)")
    print(f"Time taken: {(end_time - start_time) / 3600:.2f} hours")

    # Save overall statistics
    stats_file = f"{args.output_dir}/results/xreasoning_{dataset_name}/{base_filename}_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(results_statistics, f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {args.output_dir}/results/xreasoning_{dataset_name}/")
    print(f"Statistics file: {stats_file}")

    # Push to HuggingFace if requested
    if args.push_results_to_hf:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=stats_file,
            path_in_repo=stats_file,
            repo_id=args.hf_repo_id,
            token=args.hf_token
        )
        print(f"Results pushed to HuggingFace: {args.hf_repo_id}")

    return results_statistics


if __name__ == "__main__":
    main()

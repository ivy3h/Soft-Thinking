"""
MGSM (Multilingual Grade School Math) Evaluation Script

This script evaluates models on the MGSM benchmark across all 11 languages
and computes:
1. Per-language accuracy
2. Average accuracy across all languages
3. Cross-lingual consistency (pairwise and overall)
"""

import sglang as sgl
import json
import time
from tqdm import tqdm
import argparse
import os
from transformers import AutoTokenizer
from sglang.srt.sampling.sampling_params import SamplingParams
from matheval import evaluator_map, set_client, MGSM_LANGUAGES
import asyncio
import matheval
from huggingface_hub import HfApi
import torch
from itertools import combinations
from collections import defaultdict
from datasets import load_dataset
from timing_utils import EvalTimer

def is_gemma_model(model_name: str) -> bool:
    """Check if the model is a Gemma 3 model which needs special handling."""
    model_name_lower = model_name.lower()
    return "gemma-3" in model_name_lower or "gemma3" in model_name_lower


def get_gemma_engine_kwargs(model_name: str, base_kwargs: dict) -> dict:
    """
    Get engine kwargs adapted for Gemma 3 models.

    Gemma 3 models use head_dim=256 which requires special flashinfer kernels.
    The cascade JIT compilation can take extremely long (>30 mins) or hang.

    Solution: Disable radix cache to avoid cascade kernel compilation.
    This trades off some KV cache efficiency for reliable startup.
    """
    kwargs = base_kwargs.copy()

    if is_gemma_model(model_name):
        print(f"[Gemma Adapter] Detected Gemma 3 model: {model_name}")
        print(f"[Gemma Adapter] Disabling radix cache to avoid cascade JIT compilation issues")
        kwargs["disable_radix_cache"] = True
        # Gemma 3 requires flashinfer attention backend for sliding window attention
        if kwargs.get("attention_backend") != "flashinfer":
            print(f"[Gemma Adapter] Setting attention_backend to flashinfer (required for sliding window attention)")
            kwargs["attention_backend"] = "flashinfer"

    return kwargs


# Language metadata
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


def load_mgsm_data(languages=None, data_dir="./datasets"):
    """Load MGSM data for specified languages.

    First tries to load from local JSON files, then falls back to HuggingFace Hub.

    Args:
        languages: List of language codes. If None, loads all languages.
        data_dir: Directory containing local JSON files.

    Returns:
        Dict mapping language code to list of samples
    """
    if languages is None:
        languages = MGSM_LANGUAGES

    all_data = {}

    # Try to load from local combined file first
    combined_file = os.path.join(data_dir, "mgsm_all.json")
    if os.path.exists(combined_file):
        print(f"Loading MGSM from local file: {combined_file}")
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
        lang_file = os.path.join(data_dir, f"mgsm_{lang}.json")
        if os.path.exists(lang_file):
            print(f"Loading MGSM {lang} from local file...")
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
    all_data = {}
    for lang in languages:
        print(f"Loading MGSM {lang} ({LANGUAGE_NAMES.get(lang, lang)})...")
        try:
            ds = load_dataset("juletxara/mgsm", lang)

            data = []
            for idx, example in enumerate(ds["test"]):
                data.append({
                    "prompt": [{"from": "user", "value": example["question"]}],
                    "final_answer": str(example["answer_number"]),
                    "question_id": idx,
                    "language": lang,
                    "language_name": LANGUAGE_NAMES.get(lang, lang),
                    "full_answer": example["answer"],
                    "equation_solution": example["equation_solution"]
                })
            all_data[lang] = data
            print(f"  Loaded {len(data)} samples for {lang}")
        except Exception as e:
            print(f"  Error loading {lang}: {e}")
            all_data[lang] = []

    return all_data


def calculate_cross_lingual_consistency(results_by_lang, languages):
    """Calculate cross-lingual consistency metrics.

    For each pair of languages, compute the fraction of questions where
    the model gives the same answer (both correct or both incorrect with
    same extracted answer).

    Args:
        results_by_lang: Dict mapping language to list of result dicts
        languages: List of language codes to consider

    Returns:
        Dict containing:
        - pairwise_consistency: Dict[(lang1, lang2)] -> consistency score
        - average_consistency: Overall average consistency
        - pairwise_correct_consistency: Same answer AND both correct
    """
    # Build question_id -> results mapping for each language
    lang_results = {}
    for lang in languages:
        if lang not in results_by_lang:
            continue
        lang_results[lang] = {}
        for r in results_by_lang[lang]:
            qid = r["question_id"]
            lang_results[lang][qid] = {
                "correct": r["judge_info"][0]["finally_judge_result"] if r["judge_info"] else False,
                "extracted_answer": r.get("extracted_answer", ""),
                "passat1": r["passat1"]
            }

    available_langs = [l for l in languages if l in lang_results]

    pairwise_consistency = {}
    pairwise_correct_consistency = {}

    # Calculate pairwise consistency
    for lang1, lang2 in combinations(available_langs, 2):
        same_answer_count = 0
        both_correct_count = 0
        total_questions = 0

        # Get common question IDs
        common_qids = set(lang_results[lang1].keys()) & set(lang_results[lang2].keys())

        for qid in common_qids:
            r1 = lang_results[lang1][qid]
            r2 = lang_results[lang2][qid]

            # Consistency: both correct or both incorrect
            if r1["correct"] == r2["correct"]:
                same_answer_count += 1

            # Strict consistency: both correct
            if r1["correct"] and r2["correct"]:
                both_correct_count += 1

            total_questions += 1

        if total_questions > 0:
            pairwise_consistency[(lang1, lang2)] = same_answer_count / total_questions
            pairwise_correct_consistency[(lang1, lang2)] = both_correct_count / total_questions
        else:
            pairwise_consistency[(lang1, lang2)] = 0.0
            pairwise_correct_consistency[(lang1, lang2)] = 0.0

    # Calculate average consistency
    if pairwise_consistency:
        average_consistency = sum(pairwise_consistency.values()) / len(pairwise_consistency)
        average_correct_consistency = sum(pairwise_correct_consistency.values()) / len(pairwise_correct_consistency)
    else:
        average_consistency = 0.0
        average_correct_consistency = 0.0

    return {
        "pairwise_consistency": {f"{k[0]}-{k[1]}": v for k, v in pairwise_consistency.items()},
        "pairwise_correct_consistency": {f"{k[0]}-{k[1]}": v for k, v in pairwise_correct_consistency.items()},
        "average_consistency": average_consistency,
        "average_correct_consistency": average_correct_consistency,
        "num_language_pairs": len(pairwise_consistency)
    }


def main():
    parser = argparse.ArgumentParser(description='MGSM Multilingual Math Evaluation')

    # Dataset parameters
    parser.add_argument('--languages', type=str, nargs='+', default=None,
                        help='Languages to evaluate (default: all 11 languages)')

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
    parser.add_argument('--end_idx', type=int, default=250,
                        help='End index for samples')

    # Sampling parameters
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples')
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

    # Reeval mode
    parser.add_argument('--reeval', action='store_true',
                        help='Re-evaluate from existing results file')
    parser.add_argument('--reeval_file', type=str, default=None,
                        help='Path to existing results file for re-evaluation')

    # Engine mode
    parser.add_argument('--single_engine', action='store_true',
                        help='Use single engine for all languages (reduces memory overhead)')
    parser.add_argument('--engine_startup_delay', type=float, default=2.0,
                        help='Delay in seconds before starting new engine (for memory cleanup)')

    # Multi-run evaluation
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of evaluation runs (default: 1, use 5 for averaging)')

    # Resume mode
    parser.add_argument('--no_chat_template', action='store_true',
                        help='Do not use chat template (for base models without chat template)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing results (skip completed languages/runs)')

    # Engine parameters
    parser.add_argument('--watchdog_timeout', type=float, default=300,
                        help='Watchdog timeout in seconds. Set higher for slow models.')

    args = parser.parse_args()

    # Set languages
    languages = args.languages if args.languages else MGSM_LANGUAGES
    print(f"Evaluating on languages: {languages}")

    # Setup LLM judge if needed
    matheval.set_client(args.api_base, args.deployment_name, args.api_version,
                        args.api_key, args.judge_model_name)

    # Load dataset
    mgsm_data = load_mgsm_data(languages)

    # Prompt template (same as GSM8K)
    MATH_QUERY_TEMPLATE = """
Please reason step by step, and put your final answer within \\boxed{{}}.

{Question}
""".strip()

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
    os.makedirs(f"{args.output_dir}/results/mgsm", exist_ok=True)

    noise_suffix = (
        (f"_gumbel_{args.gumbel_softmax_temperature}" if args.add_noise_gumbel_softmax else "")
        + (f"_dirichlet_{args.dirichlet_alpha}" if args.add_noise_dirichlet else "")
    )
    base_filename = (
        f"{args.model_name.split('/')[-1]}_mgsm_{args.enable_soft_thinking}_{args.num_samples}_"
        f"{args.temperature}_{args.top_p}_{args.top_k}_{args.min_p}_{args.repetition_penalty}_"
        f"{args.dirichlet_alpha}_{args.max_topk}_{args.max_generated_tokens}_"
        f"{args.early_stopping_entropy_threshold}_{args.early_stopping_length_threshold}{noise_suffix}"
    )

    num_runs = args.num_runs

    print("=" * 60)
    print("Starting MGSM Evaluation")
    print(f"Number of runs: {num_runs}")
    print("=" * 60)

    # Check if statistics file already exists (complete resume)
    stats_file = f"{args.output_dir}/results/mgsm/{base_filename}_statistics.json"
    if args.resume and os.path.exists(stats_file):
        print(f"\n[OK] Found existing statistics file!")
        print(f"  Loading from: {stats_file}")
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                results_statistics = json.load(f)
            existing_num_runs = results_statistics.get('num_runs', 1)
            if existing_num_runs == num_runs:
                print(f"  num_runs matches ({num_runs}), average accuracy: {results_statistics.get('average_accuracy', 0):.4f}")
                print(f"\n[OK] All evaluation already completed. Exiting.")
                return results_statistics
            else:
                print(f"  num_runs mismatch: existing={existing_num_runs}, requested={num_runs}. Re-running.")
        except Exception as e:
            print(f"[!] Failed to load existing statistics: {e}")
            print(f"[!] Will re-run evaluation")

    start_time = time.time()
    timer = EvalTimer(total_languages=len(languages), total_runs=num_runs)

    # Store results across all runs
    all_runs_results = {lang: [] for lang in languages}
    all_runs_accuracies = {lang: [] for lang in languages}

    for run_idx in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{num_runs}")
        print(f"{'='*60}")

        timer.start_run(run_idx)

        # Check if this run already exists (resume functionality)
        if num_runs > 1:
            run_results_file = f"{args.output_dir}/results/mgsm/{base_filename}_run{run_idx + 1}.json"
            if args.resume and os.path.exists(run_results_file):
                print(f"[OK] Run {run_idx + 1} already completed, loading existing results...")
                try:
                    with open(run_results_file, "r", encoding="utf-8") as f:
                        existing_results = json.load(f)

                    # Group results by language and calculate accuracies
                    run_results_by_lang = {}
                    for result in existing_results:
                        lang = result.get("language")
                        if lang not in run_results_by_lang:
                            run_results_by_lang[lang] = []
                        run_results_by_lang[lang].append(result)

                    for lang, results in run_results_by_lang.items():
                        lang_accuracy = sum([r["passat1"] for r in results]) / len(results) if results else 0
                        all_runs_accuracies[lang].append(lang_accuracy)
                        all_runs_results[lang].append(results)
                        print(f"  {lang}: {lang_accuracy:.4f} ({lang_accuracy * 100:.2f}%)")

                    print(f"[OK] Successfully loaded results from {run_results_file}")
                    timer.end_run(skipped=True)
                    continue
                except Exception as e:
                    print(f"[!] Failed to load existing results: {e}")
                    print(f"[!] Will re-run this iteration")

        results_by_lang = {}
        language_accuracies = {}

        # Single engine mode - create engine once per run (different seed per run)
        shared_engine = None
        if args.single_engine:
            print("Using single engine mode - creating shared engine...")
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            base_engine_kwargs = {
                "model_path": args.model_name,
                "tp_size": args.num_gpus,
                "log_level": "info",
                "trust_remote_code": True,
                "random_seed": args.random_seed + run_idx,
                "max_running_requests": args.max_running_requests,
                "mem_fraction_static": args.mem_fraction_static,
                "disable_cuda_graph": args.enable_soft_thinking,
                "disable_overlap_schedule": args.enable_soft_thinking,
                "enable_soft_thinking": args.enable_soft_thinking,
                "add_noise_dirichlet": args.add_noise_dirichlet,
                "add_noise_gumbel_softmax": args.add_noise_gumbel_softmax,
                "max_topk": args.max_topk,
                "cuda_graph_max_bs": args.cuda_graph_max_bs,
                "sampling_backend": args.sampling_backend,
                "attention_backend": args.attention_backend,
                "watchdog_timeout": args.watchdog_timeout
            }
            engine_kwargs = get_gemma_engine_kwargs(args.model_name, base_engine_kwargs)
            shared_engine = sgl.Engine(**engine_kwargs)
            print("Shared engine created successfully.")

        # Evaluate each language
        for lang in languages:
            print(f"\n{'-' * 40}")
            print(f"Evaluating {lang} ({LANGUAGE_NAMES.get(lang, lang)})")
            print(f"{'-' * 40}")

            timer.start_language(lang)

            # Determine result file path
            if num_runs > 1:
                lang_results_file = f"{args.output_dir}/results/mgsm/{base_filename}_run{run_idx + 1}_{lang}.json"
            else:
                lang_results_file = f"{args.output_dir}/results/mgsm/{base_filename}_{lang}.json"

            # Check for existing results if resume mode is enabled
            if args.resume and os.path.exists(lang_results_file):
                try:
                    with open(lang_results_file, "r", encoding="utf-8") as f:
                        cached_results = json.load(f)
                    expected_samples = min(args.end_idx, len(mgsm_data[lang])) - args.start_idx
                    if len(cached_results) >= expected_samples:
                        print(f"  Found existing results for {lang} with {len(cached_results)} samples, skipping...")
                        lang_accuracy = sum([r["passat1"] for r in cached_results]) / len(cached_results) if cached_results else 0
                        language_accuracies[lang] = lang_accuracy
                        results_by_lang[lang] = cached_results
                        all_runs_accuracies[lang].append(lang_accuracy)
                        all_runs_results[lang].append(cached_results)
                        print(f"  {lang} ({LANGUAGE_NAMES.get(lang, lang)}) Accuracy: {lang_accuracy:.4f} ({lang_accuracy * 100:.2f}%)")
                        timer.end_language(skipped=True)
                        continue
                    else:
                        print(f"  Found partial results for {lang} ({len(cached_results)}/{expected_samples} samples), re-evaluating...")
                except Exception as e:
                    print(f"  Error loading cached results for {lang}: {e}, re-evaluating...")

            samples = mgsm_data[lang]
            samples = samples[args.start_idx:min(args.end_idx, len(samples))]

            # Prepare prompts
            prompt_list = []
            idx_list = []

            for idx, sample in enumerate(samples):
                question_text = MATH_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])
                if args.no_chat_template:
                    prompt = question_text
                else:
                    chat = [{"role": "user", "content": question_text}]
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

                if args.single_engine and shared_engine is not None:
                    llm = shared_engine
                else:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(args.engine_startup_delay)

                    base_engine_kwargs = {
                        "model_path": args.model_name,
                        "tp_size": args.num_gpus,
                        "log_level": "info",
                        "trust_remote_code": True,
                        "random_seed": args.random_seed + run_idx,
                        "max_running_requests": args.max_running_requests,
                        "mem_fraction_static": args.mem_fraction_static,
                        "disable_cuda_graph": True,
                        "disable_overlap_schedule": True,
                        "enable_soft_thinking": args.enable_soft_thinking,
                        "add_noise_dirichlet": args.add_noise_dirichlet,
                        "add_noise_gumbel_softmax": args.add_noise_gumbel_softmax,
                        "max_topk": args.max_topk,
                        "cuda_graph_max_bs": args.cuda_graph_max_bs,
                        "sampling_backend": args.sampling_backend,
                        "attention_backend": args.attention_backend,
                        "watchdog_timeout": args.watchdog_timeout
                    }
                    engine_kwargs = get_gemma_engine_kwargs(args.model_name, base_engine_kwargs)
                    llm = sgl.Engine(**engine_kwargs)

                batch_start = time.time()
                outputs = llm.generate(
                    prompt_list[batch_idx:batch_idx + args.max_batch],
                    sampling_params
                )
                batch_duration = time.time() - batch_start

                decoded_text_list.extend([o["text"] for o in outputs])
                finish_generation_list.extend([
                    o["meta_info"]["finish_reason"]["type"] == "stop" and not args.enable_soft_thinking
                    for o in outputs
                ])
                batch_tokens = [o["meta_info"]["completion_tokens"] for o in outputs]
                generated_tokens_list.extend(batch_tokens)

                batch_samples = len(outputs) // args.num_samples
                timer.record_batch(tokens=sum(batch_tokens), samples=batch_samples, duration=batch_duration)
                print(f"  Batch done: {len(outputs)} outputs, {sum(batch_tokens)} tokens in {batch_duration:.1f}s "
                      f"({sum(batch_tokens)/batch_duration:.1f} tok/s)")

                batch_idx += args.max_batch

                if not args.single_engine:
                    llm.shutdown()
                    del llm
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(args.engine_startup_delay)

            # Calculate per-sample inference time
            total_inference_sec = sum(b["duration_sec"] for b in timer._lang_batches)
            per_sample_time = total_inference_sec / len(idx_list) if idx_list else 0

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
                            rule_judge_result, extracted_answer = evaluator_map["mgsm"].rule_judge(
                                decoded_text[j], sample["final_answer"], finish_generation[j]
                            )

                            llm_judge_result = None
                            if not rule_judge_result and args.use_llm_judge:
                                llm_judge_result = evaluator_map["mgsm"].llm_judge(
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
                    "time": round(per_sample_time, 3),
                    "idx": idx,
                    "question_id": sample["question_id"],
                    "language": lang,
                    "run_idx": run_idx,
                    "n": args.num_samples,
                    "finish_generation": finish_generation_list[i * args.num_samples:(i + 1) * args.num_samples],
                    "judge_info": judge_info,
                    "passat1": sum(passat1_list) / len(passat1_list) if passat1_list else 0,
                    "passat1_list": passat1_list,
                    "extracted_answer": extracted_answer if 'extracted_answer' in dir() else ""
                }
                results.append(result)

            # Save language-specific results
            with open(lang_results_file, "w", encoding="utf-8") as f:
                results.sort(key=lambda x: x["idx"])
                json.dump(results, f, indent=4, ensure_ascii=False)

            # Calculate language accuracy
            lang_accuracy = sum([r["passat1"] for r in results]) / len(results) if results else 0
            language_accuracies[lang] = lang_accuracy
            results_by_lang[lang] = results
            all_runs_accuracies[lang].append(lang_accuracy)
            all_runs_results[lang].append(results)

            print(f"\n{lang} Run {run_idx + 1} Accuracy: {lang_accuracy:.4f} ({lang_accuracy * 100:.2f}%)")
            timer.end_language()

        timer.end_run()

        # Save per-run combined results (for resume support)
        if num_runs > 1:
            run_results_file = f"{args.output_dir}/results/mgsm/{base_filename}_run{run_idx + 1}.json"
            all_run_data = []
            for lang, res_list in results_by_lang.items():
                all_run_data.extend(res_list)
            with open(run_results_file, "w", encoding="utf-8") as f:
                json.dump(all_run_data, f, indent=4, ensure_ascii=False)
            print(f"Run {run_idx + 1} results saved to: {run_results_file}")

        # Shutdown shared engine after each run (will recreate with new seed)
        if args.single_engine and shared_engine is not None:
            print("Shutting down shared engine...")
            shared_engine.shutdown()
            del shared_engine
            shared_engine = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

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

    # Print pairwise consistency
    print("\n" + "=" * 60)
    print("Cross-Lingual Consistency (from last run)")
    print("=" * 60)

    print("\nPairwise Consistency (same correctness):")
    for pair, score in sorted(consistency_metrics["pairwise_consistency"].items()):
        print(f"  {pair}: {score:.4f} ({score * 100:.2f}%)")

    print(f"\nPairwise Correct Consistency (both correct):")
    for pair, score in sorted(consistency_metrics["pairwise_correct_consistency"].items()):
        print(f"  {pair}: {score:.4f} ({score * 100:.2f}%)")

    end_time = time.time()

    # Calculate overall average
    average_accuracy = sum(final_language_accuracies.values()) / len(final_language_accuracies) if final_language_accuracies else 0

    # Compile final statistics
    timing_stats = timer.get_stats()
    results_statistics = {
        "num_runs": num_runs,
        "languages_evaluated": languages,
        "num_languages": len(languages),
        "per_language_accuracy": {lang: acc for lang, acc in final_language_accuracies.items()},
        "per_language_std": {lang: std for lang, std in final_language_std.items()},
        "per_language_all_runs": {lang: accs for lang, accs in all_runs_accuracies.items()},
        "average_accuracy": average_accuracy,
        "cross_lingual_consistency": consistency_metrics,
        "total_samples_per_language": len(results_by_lang.get(languages[0], [])) if languages and results_by_lang else 0,
        "time_taken_hours": (end_time - start_time) / 3600,
        "timing": timing_stats
    }

    # Print summary
    print("\n" + "=" * 60)
    print("MGSM Evaluation Summary")
    if num_runs > 1:
        print(f"Number of runs: {num_runs}")
    print("=" * 60)

    print("\nPer-Language Accuracy" + (" (averaged over runs):" if num_runs > 1 else ":"))
    for lang, acc in sorted(final_language_accuracies.items(), key=lambda x: -x[1]):
        std = final_language_std.get(lang, 0)
        if num_runs > 1:
            print(f"  {lang} ({LANGUAGE_NAMES.get(lang, lang):10s}): {acc:.4f} ± {std:.4f} ({acc * 100:.2f}%)")
        else:
            print(f"  {lang} ({LANGUAGE_NAMES.get(lang, lang):10s}): {acc:.4f} ({acc * 100:.2f}%)")

    print(f"\nAverage Accuracy: {average_accuracy:.4f} ({average_accuracy * 100:.2f}%)")
    print(f"Average Cross-Lingual Consistency: {consistency_metrics['average_consistency']:.4f} ({consistency_metrics['average_consistency'] * 100:.2f}%)")
    print(f"Average Correct Consistency: {consistency_metrics['average_correct_consistency']:.4f} ({consistency_metrics['average_correct_consistency'] * 100:.2f}%)")
    print(f"Time taken: {(end_time - start_time) / 3600:.2f} hours")

    # Save overall statistics
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(results_statistics, f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {args.output_dir}/results/mgsm/")
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

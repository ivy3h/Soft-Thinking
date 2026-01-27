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

    results_by_lang = {}
    language_accuracies = {}

    print("=" * 60)
    print("Starting MGSM Evaluation")
    print("=" * 60)

    start_time = time.time()

    # Single engine mode - create engine once and reuse
    shared_engine = None
    if args.single_engine:
        print("Using single engine mode - creating shared engine...")
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        shared_engine = sgl.Engine(
            model_path=args.model_name,
            tp_size=args.num_gpus,
            log_level="info",
            trust_remote_code=True,
            random_seed=args.random_seed,
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
        print("Shared engine created successfully.")

    # Evaluate each language
    for lang in languages:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {lang} ({LANGUAGE_NAMES.get(lang, lang)})")
        print(f"{'=' * 60}")

        samples = mgsm_data[lang]
        samples = samples[args.start_idx:min(args.end_idx, len(samples))]

        # Prepare prompts
        prompt_list = []
        idx_list = []

        for idx, sample in enumerate(samples):
            chat = [{"role": "user", "content": MATH_QUERY_TEMPLATE.format(
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

            if args.single_engine and shared_engine is not None:
                # Use shared engine
                llm = shared_engine
            else:
                # Force garbage collection before starting new engine
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(args.engine_startup_delay)  # Wait for memory to be released

                llm = sgl.Engine(
                    model_path=args.model_name,
                    tp_size=args.num_gpus,
                    log_level="info",
                    trust_remote_code=True,
                    random_seed=args.random_seed,
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

            # Only shutdown if not using shared engine
            if not args.single_engine:
                llm.shutdown()
                del llm
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(args.engine_startup_delay)  # Wait for memory to be released

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
                "time": 0,
                "idx": idx,
                "question_id": sample["question_id"],
                "language": lang,
                "n": args.num_samples,
                "finish_generation": finish_generation_list[i * args.num_samples:(i + 1) * args.num_samples],
                "judge_info": judge_info,
                "passat1": sum(passat1_list) / len(passat1_list) if passat1_list else 0,
                "passat1_list": passat1_list,
                "extracted_answer": extracted_answer if 'extracted_answer' in dir() else ""
            }
            results.append(result)

        # Save language-specific results
        lang_results_file = f"{args.output_dir}/results/mgsm/{base_filename}_{lang}.json"
        with open(lang_results_file, "w", encoding="utf-8") as f:
            results.sort(key=lambda x: x["idx"])
            json.dump(results, f, indent=4, ensure_ascii=False)

        # Calculate language accuracy
        lang_accuracy = sum([r["passat1"] for r in results]) / len(results) if results else 0
        language_accuracies[lang] = lang_accuracy
        results_by_lang[lang] = results

        print(f"\n{lang} ({LANGUAGE_NAMES.get(lang, lang)}) Accuracy: {lang_accuracy:.4f} ({lang_accuracy * 100:.2f}%)")

    # Calculate cross-lingual consistency
    print("\n" + "=" * 60)
    print("Calculating Cross-Lingual Consistency")
    print("=" * 60)

    consistency_metrics = calculate_cross_lingual_consistency(results_by_lang, languages)

    # Print pairwise consistency
    print("\nPairwise Consistency (same correctness):")
    for pair, score in sorted(consistency_metrics["pairwise_consistency"].items()):
        print(f"  {pair}: {score:.4f} ({score * 100:.2f}%)")

    print(f"\nPairwise Correct Consistency (both correct):")
    for pair, score in sorted(consistency_metrics["pairwise_correct_consistency"].items()):
        print(f"  {pair}: {score:.4f} ({score * 100:.2f}%)")

    end_time = time.time()

    # Final statistics
    average_accuracy = sum(language_accuracies.values()) / len(language_accuracies) if language_accuracies else 0

    # Compile final statistics
    results_statistics = {
        "languages_evaluated": languages,
        "num_languages": len(languages),
        "per_language_accuracy": {lang: acc for lang, acc in language_accuracies.items()},
        "average_accuracy": average_accuracy,
        "cross_lingual_consistency": consistency_metrics,
        "total_samples_per_language": len(results_by_lang.get(languages[0], [])) if languages else 0,
        "time_taken_hours": (end_time - start_time) / 3600
    }

    # Print summary
    print("\n" + "=" * 60)
    print("MGSM Evaluation Summary")
    print("=" * 60)

    print("\nPer-Language Accuracy:")
    for lang, acc in sorted(language_accuracies.items(), key=lambda x: -x[1]):
        print(f"  {lang} ({LANGUAGE_NAMES.get(lang, lang):10s}): {acc:.4f} ({acc * 100:.2f}%)")

    print(f"\nAverage Accuracy: {average_accuracy:.4f} ({average_accuracy * 100:.2f}%)")
    print(f"Average Cross-Lingual Consistency: {consistency_metrics['average_consistency']:.4f} ({consistency_metrics['average_consistency'] * 100:.2f}%)")
    print(f"Average Correct Consistency: {consistency_metrics['average_correct_consistency']:.4f} ({consistency_metrics['average_correct_consistency'] * 100:.2f}%)")
    print(f"Time taken: {(end_time - start_time) / 3600:.2f} hours")

    # Save overall statistics
    stats_file = f"{args.output_dir}/results/mgsm/{base_filename}_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(results_statistics, f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {args.output_dir}/results/mgsm/")
    print(f"Statistics file: {stats_file}")

    # Shutdown shared engine if used
    if args.single_engine and shared_engine is not None:
        print("Shutting down shared engine...")
        shared_engine.shutdown()
        del shared_engine
        import gc
        gc.collect()
        torch.cuda.empty_cache()

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

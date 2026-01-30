"""
Merge LoRA adapter into base model and prepare for evaluation.

This script:
1. Loads the base model
2. Loads the LoRA adapter from a checkpoint
3. Merges the adapter into the base model
4. Saves the merged model to a specified output directory
5. (Optionally) runs evaluation using the merged model

Usage:
    python merge_lora_and_eval.py \
        --adapter_path /path/to/checkpoint-50 \
        --output_dir /path/to/merged_model \
        --base_model Qwen/Qwen3-4B-Instruct-2507

    # Or auto-detect base model from adapter_config.json:
    python merge_lora_and_eval.py \
        --adapter_path /path/to/checkpoint-50 \
        --output_dir /path/to/merged_model
"""

import argparse
import json
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(adapter_path: str, output_dir: str, base_model: str | None = None,
               cache_dir: str | None = None):
    """Merge LoRA adapter into base model and save.

    Args:
        adapter_path: Path to LoRA adapter directory (with adapter_config.json)
        output_dir: Where to save the merged model
        base_model: Base model name/path. If None, read from adapter_config.json
        cache_dir: HuggingFace cache directory
    """
    # Auto-detect base model from adapter_config.json
    if base_model is None:
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No adapter_config.json found at {adapter_path}")
        with open(config_path, "r") as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError("Could not determine base model from adapter_config.json")

    print(f"Base model: {base_model}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output dir: {output_dir}")

    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"Merged model already exists at {output_dir}, skipping merge.")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,  # Adapter dir contains tokenizer files
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    # Load and merge LoRA
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Merge complete!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and prepare for evaluation")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to LoRA adapter checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save merged model")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model name/path (auto-detected from adapter_config.json if not specified)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory")
    args = parser.parse_args()

    merge_lora(args.adapter_path, args.output_dir, args.base_model, args.cache_dir)


if __name__ == "__main__":
    main()

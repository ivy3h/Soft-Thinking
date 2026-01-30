#!/usr/bin/env python3
"""
Re-merge LoRA adapter with fp16 on GPU instead of bf16 on CPU.
"""
import argparse
import json
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def remerge_lora_fp16(adapter_path, output_dir, cache_dir=None):
    """Re-merge LoRA adapter with fp16 on GPU."""

    # Auto-detect base model
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path, "r") as f:
        adapter_config = json.load(f)
    base_model = adapter_config["base_model_name_or_path"]

    print(f"Base model: {base_model}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output dir: {output_dir}")
    print(f"\nUsing fp16 on GPU (cuda:0)\n")

    if os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"Output directory already exists, removing...")
        import shutil
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Load base model with fp16 on GPU
    print("Loading base model (fp16, cuda:0)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,  # fp16 instead of bf16
        device_map="cuda:0",         # GPU instead of CPU
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    remerge_lora_fp16(args.adapter_path, args.output_dir, args.cache_dir)

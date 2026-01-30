#!/usr/bin/env python3
"""
Generate SLURM scripts for missing evaluation configurations.
"""

import os
from pathlib import Path

# Missing configurations
MISSING_CONFIGS = [
    # Qwen3-4B-Instruct-2507
    ("Qwen3-4B-Instruct-2507", "XReasoning-AIME2024", "Soft-Thinking"),
    ("Qwen3-4B-Instruct-2507", "XReasoning-AIME2025", "CoT"),
    ("Qwen3-4B-Instruct-2507", "XReasoning-AIME2025", "Soft-Thinking"),
    ("Qwen3-4B-Instruct-2507", "XReasoning-GPQA", "CoT"),
    ("Qwen3-4B-Instruct-2507", "XReasoning-GPQA", "Soft-Thinking"),

    # Qwen3-8B-Base
    ("Qwen3-8B-Base", "MGSM", "Soft-Thinking"),
    ("Qwen3-8B-Base", "XReasoning-AIME2024", "Soft-Thinking"),
    ("Qwen3-8B-Base", "XReasoning-AIME2025", "CoT"),
    ("Qwen3-8B-Base", "XReasoning-AIME2025", "Soft-Thinking"),
    ("Qwen3-8B-Base", "XReasoning-GPQA", "CoT"),
    ("Qwen3-8B-Base", "XReasoning-GPQA", "Soft-Thinking"),

    # Qwen3-8B
    ("Qwen3-8B", "XReasoning-AIME2024", "CoT"),
    ("Qwen3-8B", "XReasoning-AIME2024", "Soft-Thinking"),
    ("Qwen3-8B", "XReasoning-AIME2025", "Soft-Thinking"),
    ("Qwen3-8B", "XReasoning-GPQA", "CoT"),
    ("Qwen3-8B", "XReasoning-GPQA", "Soft-Thinking"),
]

# Model name mappings
MODEL_NAMES = {
    "Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen3-8B-Base": "Qwen/Qwen3-8B-Base",
    "Qwen3-8B": "Qwen/Qwen3-8B",
}

# XReasoning dataset mappings
XR_DATASETS = {
    "XReasoning-AIME2024": "aime2024",
    "XReasoning-AIME2025": "aime2025",
    "XReasoning-GPQA": "gpqa_diamond",
}

def generate_mgsm_script(model, model_name, method, output_dir):
    """Generate MGSM evaluation script."""

    soft_thinking_flag = "--enable_soft_thinking \\" if method == "Soft-Thinking" else ""
    method_suffix = "st" if method == "Soft-Thinking" else "cot"
    job_name = f"mgsm_{model.lower().replace('-', '_')}_{method_suffix}"

    script = f"""#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J {job_name}
#SBATCH -o /nethome/jhe478/flash/Soft-Thinking/logs/{job_name}_%j.log

# Evaluate {model} on MGSM ({method})

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

export TRANSFORMERS_CACHE=/coc/pskynet6/jhe478/huggingface/transformers
export HF_HOME=/coc/pskynet6/jhe478/huggingface
export HF_DATASETS_CACHE=/coc/pskynet6/jhe478/huggingface/datasets

python run_mgsm_evaluation.py \\
    --model_name "{model_name}" \\
    --max_generated_tokens 16384 \\
    --temperature 0.6 \\
    --top_p 0.95 \\
    --top_k 30 \\
    --min_p 0.001 \\
    --mem_fraction_static 0.6 \\
    --start_idx 0 \\
    --end_idx 250 \\
    --num_gpus 1 \\
    --num_samples 1 \\
    --single_engine \\
    {soft_thinking_flag}--resume

echo "Evaluation completed!"
"""

    filename = f"{job_name}.sh"
    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        f.write(script)

    os.chmod(filepath, 0o755)
    return filepath


def generate_xreasoning_script(model, model_name, dataset, method, output_dir):
    """Generate XReasoning evaluation script."""

    xr_dataset = XR_DATASETS[dataset]
    soft_thinking_flag = "--enable_soft_thinking \\" if method == "Soft-Thinking" else ""
    method_suffix = "st" if method == "Soft-Thinking" else "cot"

    # Determine runs based on dataset
    if "AIME" in dataset:
        num_runs = 5
    else:  # GPQA
        num_runs = 1

    job_name = f"xr_{xr_dataset}_{model.lower().replace('-', '_')}_{method_suffix}"

    script = f"""#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J {job_name}
#SBATCH -o /nethome/jhe478/flash/Soft-Thinking/logs/{job_name}_%j.log

# Evaluate {model} on {dataset} ({method})

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

export TRANSFORMERS_CACHE=/coc/pskynet6/jhe478/huggingface/transformers
export HF_HOME=/coc/pskynet6/jhe478/huggingface
export HF_DATASETS_CACHE=/coc/pskynet6/jhe478/huggingface/datasets

python run_xreasoning_evaluation.py \\
    --model_name "{model_name}" \\
    --dataset {xr_dataset} \\
    --num_runs {num_runs} \\
    --max_generated_tokens 16384 \\
    --temperature 0.6 \\
    --top_p 0.95 \\
    --top_k 30 \\
    --min_p 0.001 \\
    --mem_fraction_static 0.6 \\
    --num_gpus 1 \\
    --num_samples 1 \\
    {soft_thinking_flag.rstrip()}

echo "Evaluation completed!"
"""

    filename = f"{job_name}.sh"
    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        f.write(script)

    os.chmod(filepath, 0o755)
    return filepath


def main():
    output_dir = Path("/nethome/jhe478/flash/Soft-Thinking/scripts/missing_evals")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating scripts in: {output_dir}\n")

    generated_scripts = []

    for model, dataset, method in MISSING_CONFIGS:
        model_name = MODEL_NAMES[model]

        if dataset == "MGSM":
            filepath = generate_mgsm_script(model, model_name, method, output_dir)
        else:
            filepath = generate_xreasoning_script(model, model_name, dataset, method, output_dir)

        generated_scripts.append(filepath)
        print(f"[OK] Generated: {filepath.name}")

    print(f"\n[OK] Generated {len(generated_scripts)} scripts")
    print(f"\nTo submit all jobs, run:")
    print(f"  cd {output_dir}")
    print(f"  for script in *.sh; do sbatch $script; done")

    # Also create a submit_all script
    submit_script = output_dir / "submit_all.sh"
    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all missing evaluation jobs\n\n")
        f.write(f"cd {output_dir}\n\n")
        for script in sorted(generated_scripts):
            f.write(f"echo \"Submitting {script.name}...\"\n")
            f.write(f"sbatch {script.name}\n")
            f.write("sleep 1\n\n")

    os.chmod(submit_script, 0o755)
    print(f"\nOr run the batch submit script:")
    print(f"  bash {submit_script}")


if __name__ == "__main__":
    main()

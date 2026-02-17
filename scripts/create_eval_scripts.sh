#!/bin/bash
# Create eval scripts for all training models
cd /coc/pskynet6/jhe478/Soft-Thinking/scripts

EXCLUDE_4B="starrysky,heistotron,deebot,nestor,cheetah,chitti,tachikoma,optimistprime,uniblab,puma,perseverance,clippy,xaea-12,megazord,trublu"
EXCLUDE_8B="$EXCLUDE_4B,baymax"

# Function to create MGSM 5-run eval script
create_mgsm_5runs() {
    local name=$1   # e.g., q3_4b_ms1k_solar
    local model=$2  # full model path
    local gpus=$3   # 1 or 2
    local mem=$4    # 64G or 96G
    local cpus=$5   # 4 or 8
    local exclude=$6
    local jobname=$7

    cat > "eval_${name}_mgsm_5runs.sh" << EOF
#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a40:${gpus}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH -x ${exclude}
#SBATCH -J ${jobname}
#SBATCH -o logs/eval_${name}_mgsm_5runs_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

MODEL="${model}"

python run_mgsm_evaluation.py \\
    --model_name "\$MODEL" \\
    --max_generated_tokens 32768 \\
    --temperature 0.6 \\
    --top_p 0.95 \\
    --top_k 30 \\
    --min_p 0.001 \\
    --mem_fraction_static 0.8 \\
    --num_gpus ${gpus} \\
    --num_samples 1 \\
    --num_runs 5 \\
    --resume
EOF
}

# Function to create xreasoning eval script
create_xreasoning() {
    local name=$1
    local model=$2
    local dataset=$3  # aime2024, aime2025, gpqa, math500, mmlu_prox
    local gpus=$4
    local mem=$5
    local cpus=$6
    local exclude=$7
    local jobname=$8

    local shortname=$(echo $dataset | sed 's/aime2024/aime24/;s/aime2025/aime25/')

    cat > "eval_${name}_${shortname}.sh" << EOF
#!/bin/bash
#SBATCH -p overcap
#SBATCH --account=overcap
#SBATCH --qos short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a40:${gpus}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH -x ${exclude}
#SBATCH -J ${jobname}
#SBATCH -o logs/eval_${name}_${shortname}_%j.log

source ~/.bashrc
conda activate st

cd /coc/pskynet6/jhe478/Soft-Thinking

MODEL="${model}"

python run_xreasoning_evaluation.py \\
    --dataset "${dataset}" \\
    --model_name "\$MODEL" \\
    --max_generated_tokens 32768 \\
    --temperature 0.6 \\
    --top_p 0.95 \\
    --top_k 30 \\
    --min_p 0.001 \\
    --mem_fraction_static 0.8 \\
    --num_gpus ${gpus} \\
    --num_samples 1
EOF
}

echo "Creating eval scripts..."

# === 4B SOLAR 10lang ===
M4B_SOLAR="/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-4b/full/sft_ms1k_solar_val"
create_mgsm_5runs "q3_4b_ms1k_solar" "$M4B_SOLAR" 1 64G 4 "$EXCLUDE_4B" "mgsm_4bsol_5r"
for ds in aime2024 aime2025 gpqa math500 mmlu_prox; do
    sn=$(echo $ds | sed 's/aime2024/aime24/;s/aime2025/aime25/;s/mmlu_prox/mmlu/')
    create_xreasoning "q3_4b_ms1k_solar" "$M4B_SOLAR" "$ds" 1 64G 4 "$EXCLUDE_4B" "${sn}_4bsol"
done

# === 8B non-SOLAR ===
M8B="/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-8b/full/sft_ms1k_val"
create_mgsm_5runs "q3_8b_ms1k" "$M8B" 2 96G 8 "$EXCLUDE_8B" "mgsm_8bms1k_5r"
for ds in aime2024 aime2025 gpqa math500 mmlu_prox; do
    sn=$(echo $ds | sed 's/aime2024/aime24/;s/aime2025/aime25/;s/mmlu_prox/mmlu/')
    create_xreasoning "q3_8b_ms1k" "$M8B" "$ds" 2 96G 8 "$EXCLUDE_8B" "${sn}_8bms1k"
done

# === 8B SOLAR ===
M8B_SOLAR="/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-8b/full/sft_ms1k_solar_val"
create_mgsm_5runs "q3_8b_ms1k_solar" "$M8B_SOLAR" 2 96G 8 "$EXCLUDE_8B" "mgsm_8bsol_5r"
for ds in aime2024 aime2025 gpqa math500 mmlu_prox; do
    sn=$(echo $ds | sed 's/aime2024/aime24/;s/aime2025/aime25/;s/mmlu_prox/mmlu/')
    create_xreasoning "q3_8b_ms1k_solar" "$M8B_SOLAR" "$ds" 2 96G 8 "$EXCLUDE_8B" "${sn}_8bsol"
done

# === 4B MGSM-7 non-SOLAR ===
M4B_MGSM7="/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-4b/full/sft_ms1k_mgsm7_val"
create_mgsm_5runs "q3_4b_mgsm7" "$M4B_MGSM7" 1 64G 4 "$EXCLUDE_4B" "mgsm_4bm7_5r"

# === 4B MGSM-7 SOLAR ===
M4B_MGSM7_SOLAR="/coc/pskynet6/jhe478/LlamaFactory/saves/qwen3-4b/full/sft_ms1k_mgsm7_solar_val"
create_mgsm_5runs "q3_4b_mgsm7_solar" "$M4B_MGSM7_SOLAR" 1 64G 4 "$EXCLUDE_4B" "mgsm_4bm7s_5r"

echo "Done! Created eval scripts."
ls -la eval_q3_4b_ms1k_solar_*.sh eval_q3_8b_ms1k_*.sh eval_q3_4b_mgsm7_*.sh eval_q3_4b_mgsm7_solar_*.sh 2>/dev/null

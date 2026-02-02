#!/bin/bash
# Submit all Gemma-3 evaluation jobs
# Models: gemma-3-4b-it, gemma-3-12b-it
# Datasets: MGSM (nlprx-lab, 4h), xreasoning/AIME2024+2025+GPQA (overcap, 12h)
# Methods: baseline, soft-thinking

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Ensure logs directory exists
mkdir -p logs

echo "=========================================="
echo "Submitting Gemma-3 evaluation jobs"
echo "=========================================="

# --- MGSM jobs (nlprx-lab partition, A40, 4h) ---
echo ""
echo "--- MGSM (nlprx-lab, 4h) ---"

echo "Submitting: gemma-3-4b-it MGSM baseline"
sbatch scripts/mgsm_gemma3_4b_baseline.sh

echo "Submitting: gemma-3-4b-it MGSM soft-thinking"
sbatch scripts/mgsm_gemma3_4b_st.sh

echo "Submitting: gemma-3-12b-it MGSM baseline"
sbatch scripts/mgsm_gemma3_12b_baseline.sh

echo "Submitting: gemma-3-12b-it MGSM soft-thinking"
sbatch scripts/mgsm_gemma3_12b_st.sh

# --- xreasoning jobs (overcap partition, 12h) ---
echo ""
echo "--- xreasoning: AIME2024 + AIME2025 + GPQA (overcap) ---"

echo "Submitting: gemma-3-4b-it xreasoning baseline"
sbatch scripts/xreasoning_gemma3_4b_baseline.sh

echo "Submitting: gemma-3-4b-it xreasoning soft-thinking"
sbatch scripts/xreasoning_gemma3_4b_st.sh

echo "Submitting: gemma-3-12b-it xreasoning baseline"
sbatch scripts/xreasoning_gemma3_12b_baseline.sh

echo "Submitting: gemma-3-12b-it xreasoning soft-thinking"
sbatch scripts/xreasoning_gemma3_12b_st.sh

echo ""
echo "=========================================="
echo "All 8 jobs submitted!"
echo "  4 MGSM jobs    (nlprx-lab, A40, 4h)"
echo "  4 xreasoning jobs (overcap, A40, 12h)"
echo "=========================================="
echo ""
echo "Use 'squeue -u \$USER' to check job status"

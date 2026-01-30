#!/bin/bash
# Submit all missing evaluation jobs

cd /nethome/jhe478/flash/Soft-Thinking/scripts/missing_evals

echo "Submitting mgsm_qwen3_8b_base_st.sh..."
sbatch mgsm_qwen3_8b_base_st.sh
sleep 1

echo "Submitting xr_aime2024_qwen3_4b_instruct_2507_st.sh..."
sbatch xr_aime2024_qwen3_4b_instruct_2507_st.sh
sleep 1

echo "Submitting xr_aime2024_qwen3_8b_base_st.sh..."
sbatch xr_aime2024_qwen3_8b_base_st.sh
sleep 1

echo "Submitting xr_aime2024_qwen3_8b_cot.sh..."
sbatch xr_aime2024_qwen3_8b_cot.sh
sleep 1

echo "Submitting xr_aime2024_qwen3_8b_st.sh..."
sbatch xr_aime2024_qwen3_8b_st.sh
sleep 1

echo "Submitting xr_aime2025_qwen3_4b_instruct_2507_cot.sh..."
sbatch xr_aime2025_qwen3_4b_instruct_2507_cot.sh
sleep 1

echo "Submitting xr_aime2025_qwen3_4b_instruct_2507_st.sh..."
sbatch xr_aime2025_qwen3_4b_instruct_2507_st.sh
sleep 1

echo "Submitting xr_aime2025_qwen3_8b_base_cot.sh..."
sbatch xr_aime2025_qwen3_8b_base_cot.sh
sleep 1

echo "Submitting xr_aime2025_qwen3_8b_base_st.sh..."
sbatch xr_aime2025_qwen3_8b_base_st.sh
sleep 1

echo "Submitting xr_aime2025_qwen3_8b_st.sh..."
sbatch xr_aime2025_qwen3_8b_st.sh
sleep 1

echo "Submitting xr_gpqa_diamond_qwen3_4b_instruct_2507_cot.sh..."
sbatch xr_gpqa_diamond_qwen3_4b_instruct_2507_cot.sh
sleep 1

echo "Submitting xr_gpqa_diamond_qwen3_4b_instruct_2507_st.sh..."
sbatch xr_gpqa_diamond_qwen3_4b_instruct_2507_st.sh
sleep 1

echo "Submitting xr_gpqa_diamond_qwen3_8b_base_cot.sh..."
sbatch xr_gpqa_diamond_qwen3_8b_base_cot.sh
sleep 1

echo "Submitting xr_gpqa_diamond_qwen3_8b_base_st.sh..."
sbatch xr_gpqa_diamond_qwen3_8b_base_st.sh
sleep 1

echo "Submitting xr_gpqa_diamond_qwen3_8b_cot.sh..."
sbatch xr_gpqa_diamond_qwen3_8b_cot.sh
sleep 1

echo "Submitting xr_gpqa_diamond_qwen3_8b_st.sh..."
sbatch xr_gpqa_diamond_qwen3_8b_st.sh
sleep 1


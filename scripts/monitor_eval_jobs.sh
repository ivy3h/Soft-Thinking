#!/bin/bash
# Monitor evaluation jobs, detect preemptions, and auto-resubmit
# Usage: bash scripts/monitor_eval_jobs.sh

LOG="/coc/pskynet6/jhe478/Soft-Thinking/scripts/logs/monitor_eval.log"
mkdir -p "$(dirname "$LOG")"

# Define all jobs to monitor: "JOB_NAME SCRIPT_PATH"
declare -A JOB_SCRIPTS
JOB_SCRIPTS["eval_q3_4b_ms1k"]="scripts/merge_and_eval_qwen3_4b_sft_ms1k.sh"
JOB_SCRIPTS["eval_q3_4b_ms1k_abl"]="scripts/merge_and_eval_qwen3_4b_sft_ms1k_ablation.sh"
JOB_SCRIPTS["eval_q25_sft_gsm8k"]="scripts/merge_and_eval_qwen25_3b_sft.sh"
JOB_SCRIPTS["eval_q25_sft_abl"]="scripts/merge_and_eval_qwen25_3b_sft_ablation.sh"
JOB_SCRIPTS["xr_ms1k_gpqa"]="scripts/eval_q3_4b_ms1k_gpqa.sh"
JOB_SCRIPTS["xr_ms1ka_gpqa"]="scripts/eval_q3_4b_ms1k_ablation_gpqa.sh"
JOB_SCRIPTS["xr_ms1k_a24"]="scripts/eval_q3_4b_ms1k_aime24.sh"
JOB_SCRIPTS["xr_ms1ka_a24"]="scripts/eval_q3_4b_ms1k_ablation_aime24.sh"
JOB_SCRIPTS["xr_ms1k_a25"]="scripts/eval_q3_4b_ms1k_aime25.sh"
JOB_SCRIPTS["xr_ms1ka_a25"]="scripts/eval_q3_4b_ms1k_ablation_aime25.sh"
JOB_SCRIPTS["xr_q25_3b_a24"]="scripts/eval_qwen25_3b_aime24.sh"
JOB_SCRIPTS["xr_q25_3b_a25"]="scripts/eval_qwen25_3b_aime25.sh"
JOB_SCRIPTS["xr_g3_4b_base"]=""  # Don't resubmit, just track

# Track which jobs have completed successfully
declare -A COMPLETED

cd /coc/pskynet6/jhe478/Soft-Thinking

echo "$(date): Monitor started" | tee -a "$LOG"
echo "Tracking ${#JOB_SCRIPTS[@]} job types" | tee -a "$LOG"

while true; do
    # Get current running jobs
    RUNNING_JOBS=$(squeue -u jhe478 -o "%j %T %i" --noheader 2>/dev/null)

    # Count how many eval jobs are still active
    ACTIVE_EVAL=0

    for JOB_NAME in "${!JOB_SCRIPTS[@]}"; do
        SCRIPT="${JOB_SCRIPTS[$JOB_NAME]}"

        # Skip if already marked completed
        if [ "${COMPLETED[$JOB_NAME]}" == "done" ]; then
            continue
        fi

        # Check if job is currently running or pending
        if echo "$RUNNING_JOBS" | grep -q "$JOB_NAME"; then
            ACTIVE_EVAL=$((ACTIVE_EVAL + 1))
            continue
        fi

        # Job not in queue - check if it was preempted recently
        RECENT=$(sacct -u jhe478 --starttime=$(date -d '30 minutes ago' +%Y-%m-%dT%H:%M:%S) \
            --format=JobName%30,State%15,JobID%12 --noheader 2>/dev/null | grep "$JOB_NAME")

        if echo "$RECENT" | grep -qi "PREEMPTED\|FAILED\|CANCELLED\|NODE_FAIL"; then
            if [ -n "$SCRIPT" ]; then
                echo "$(date): $JOB_NAME was preempted/failed, resubmitting..." | tee -a "$LOG"
                RESULT=$(sbatch "$SCRIPT" 2>&1)
                echo "$(date): Resubmitted: $RESULT" | tee -a "$LOG"
                ACTIVE_EVAL=$((ACTIVE_EVAL + 1))
            fi
        elif echo "$RECENT" | grep -qi "COMPLETED"; then
            COMPLETED[$JOB_NAME]="done"
            echo "$(date): $JOB_NAME COMPLETED successfully" | tee -a "$LOG"
        fi
    done

    # Check completion status
    DONE_COUNT=0
    for JOB_NAME in "${!JOB_SCRIPTS[@]}"; do
        if [ "${COMPLETED[$JOB_NAME]}" == "done" ]; then
            DONE_COUNT=$((DONE_COUNT + 1))
        fi
    done

    echo "$(date): Active=$ACTIVE_EVAL, Completed=$DONE_COUNT/${#JOB_SCRIPTS[@]}" >> "$LOG"

    # If no active eval jobs and we've seen some complete, we might be done
    if [ "$ACTIVE_EVAL" -eq 0 ] && [ "$DONE_COUNT" -gt 0 ]; then
        echo "$(date): All tracked eval jobs appear complete or not running." | tee -a "$LOG"
        echo "$(date): Final status:" | tee -a "$LOG"
        for JOB_NAME in "${!JOB_SCRIPTS[@]}"; do
            echo "  $JOB_NAME: ${COMPLETED[$JOB_NAME]:-unknown}" | tee -a "$LOG"
        done
        break
    fi

    # Check every 5 minutes
    sleep 300
done

echo "$(date): Monitor finished" | tee -a "$LOG"

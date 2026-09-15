#!/usr/bin/env bash
# Runs on: Slurm (wrapper, submits sbatch jobs)
# src/ts2/scripts/finetuning/submit_all_finetune.sh
# Launch one sbatch job per recording_id.
#
# Usage:
#   ./submit_all_finetune.sh trainer=mtm_finetune wandb.project=ts2-mtm_finetune-ft \
#                       ckpt.load_from='n9e7a3bx/best.pt' num_epochs=500 \
#                       +sweep.base_lr='[1e-5,2e-5,5e-5,1e-4]' \
#                       num_workers=3 +ray.gpu=0.125 +ray.cpu=3
#
# One task only (default is both):        +tasks='[co_smoothing]'
# A grid for one task, not for the other: +task_sweep.forecasting.base_lr='[1e-5,2e-5]'
#
# Env knobs: EIDS_FILE, REPO_DIR, LOG_DIR, DATA_ROOT, BWB_CKPT_DIR, WALLTIME,
#            JOB_PREFIX (set it per model so concurrent sweeps stay tellable apart).
# Partition, account, qos and reservation come from SBATCH_* in .env.

# sbatch takes partition/account/qos/reservation from SBATCH_* in the submitting
# environment, so lift those out of .env here; an already-set value wins.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ -f "$HERE/.env" ]]; then
    while IFS='=' read -r k v; do [[ -n "${!k:-}" ]] || export "$k=$v"; done \
        < <(set -a; source "$HERE/.env"; set +a; env | grep '^SBATCH_')
fi

REPO_DIR="${REPO_DIR:-$HERE}"
EIDS_FILE="${EIDS_FILE:-$REPO_DIR/src/ibl_bwb_eval/data/eval_recording_ids.txt}"
SBATCH_SCRIPT="$REPO_DIR/src/ts2/scripts/finetuning/launch_finetune.sbatch"
LOG_DIR="${LOG_DIR:-}"
# Shorter than the sbatch default so the jobs stay backfill-friendly on a busy partition.
WALLTIME="${WALLTIME:-06:00:00}"
JOB_PREFIX="${JOB_PREFIX:-ts2-ft}"

extra=(--time="$WALLTIME")
[[ -n "$LOG_DIR" ]] && mkdir -p "$LOG_DIR" && extra+=(--output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err")

export REPO_DIR

while IFS= read -r eid || [[ -n "$eid" ]]; do
    [[ -z "$eid" || "$eid" =~ ^# ]] && continue
    echo "submitting eid=$eid"
    sbatch \
        --job-name="${JOB_PREFIX}-${eid:0:8}" \
        "${extra[@]}" \
        "$SBATCH_SCRIPT" \
        "$@" \
        recording_id="$eid"
done < "$EIDS_FILE"

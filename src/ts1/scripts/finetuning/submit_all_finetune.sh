#!/usr/bin/env bash
# Runs on: Slurm (wrapper, submits sbatch jobs)
# src/ts1/scripts/finetuning/submit_all_finetune.sh
# Launch one sbatch job per recording_id.
#
# Usage:
#   REPO_DIR=/path/to/worktree ./submit_all_finetune.sh trainer=poyo_plus_gradual_unfreezing \
#                       wandb.project=ts1-poyo_plus-ft-repro "ckpt.load_from='g8bows70/epoch_32.pt'" \
#                       num_epochs=300 num_workers=3 +ray.gpu=0.125 +ray.cpu=3

# sbatch takes partition/account/qos/reservation from SBATCH_* in the submitting
# environment, so lift those out of .env here; an already-set value wins.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ -f "$HERE/.env" ]]; then
    while IFS='=' read -r k v; do [[ -n "${!k:-}" ]] || export "$k=$v"; done \
        < <(set -a; source "$HERE/.env"; set +a; env | grep '^SBATCH_')
fi

REPO_DIR="${REPO_DIR:-$HERE}"
EIDS_FILE="${EIDS_FILE:-$REPO_DIR/src/ibl_bwb_eval/data/eval_recording_ids.txt}"
SBATCH_SCRIPT="$REPO_DIR/src/ts1/scripts/finetuning/launch_finetune.sbatch"
LOG_DIR="${LOG_DIR:-}"

extra=()
[[ -n "$LOG_DIR" ]] && mkdir -p "$LOG_DIR" && extra+=(--output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err")

export REPO_DIR

while IFS= read -r eid || [[ -n "$eid" ]]; do
    [[ -z "$eid" || "$eid" =~ ^# ]] && continue
    echo "submitting eid=$eid"
    sbatch \
        --job-name="ts1-ft-${eid:0:8}" \
        "${extra[@]}" \
        "$SBATCH_SCRIPT" \
        "$@" \
        recording_id="$eid"
done < "$EIDS_FILE"

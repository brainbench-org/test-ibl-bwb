#!/usr/bin/env bash
# Runs on: Slurm (wrapper, submits sbatch jobs)
# src/ts1/scripts/finetuning/submit_possm_repro_sweep.sh
#
# TS1 finetune sweep for the POSSM repro pretrain, the POSSM twin of
# submit_poyo_plus_repro_sweep.sh: finetune.py sweeps base_lr per (recording_id, task) at
# seed 42, selects on best/val/avg, then reruns each winner on seeds 43-47 into a
# "<project>-seeds" project. The grid, num_epochs=300 and the gradual unfreezing all live
# in possm_gradual_unfreezing.yaml, not here.
#
# This re-sweeps base_lr from scratch rather than replaying a saved per-pair lr table.
#
# Set up to be comparable with poyo_plus rather than with May:
#   - unfreeze_at_epoch is a flat 40, matching poyo_plus; the May sweep's per-task
#     schedule is described in possm_gradual_unfreezing.yaml and is deliberately not used
#   - base_lr grid is poyo_plus's [1e-5..5e-4]. May saturated its top rung (5e-4 won 6 of
#     8 tasks), so expect the same ceiling here; that is the price of an equal grid
#   - loads the v0.0.9 repro pretrain, not $BWB_NEURIPS_CKPT_DIR/possm.pt, because
#     poyo_plus's sweep likewise loaded its own repro pretrain
#
# One sbatch per recording_id. Each job runs 8 tasks x 5 LRs = 40 sweep finetunes then
# 8 x 5 = 40 seed finetunes, 8 at a time on its GPU (ray.gpu=0.125); poyo_plus's
# equivalent took 33 min - 1h28.
#
# CONCURRENCY caps how many run at once by chaining the jobs into that many
# afterany-dependency chains, which Slurm enforces without a babysitting process. Size it
# as `<GPUs in the pool> - <other users' GPUs> - <GPUs you already hold> - 2`, leaving 2
# free for everyone else.
#
# Usage:
#   DRY_RUN=1 ./src/ts1/scripts/finetuning/submit_possm_repro_sweep.sh        # print only
#   ./src/ts1/scripts/finetuning/submit_possm_repro_sweep.sh
#   CONCURRENCY=5 ./src/ts1/scripts/finetuning/submit_possm_repro_sweep.sh
#   EIDS="0802ced5-..." EXTRA="+tasks=[choice] num_epochs=3" \
#       ./src/ts1/scripts/finetuning/submit_possm_repro_sweep.sh              # smoke test

set -euo pipefail

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
# Resolved against BWB_CKPT_DIR by core.trainer.load_ckpt when it is not an absolute path.
CKPT="${CKPT:-possm_multitask_pretrain_hy9q0pwk/best.pt}"
PROJECT="${PROJECT:-ts1-possm-ft-hy9q0pwk-repro}"
WALLTIME="${WALLTIME:-06:00:00}"
CONCURRENCY="${CONCURRENCY:-3}"
LOG_DIR="${LOG_DIR:-}"
export REPO_DIR

extra=()
[[ -n "$LOG_DIR" ]] && mkdir -p "$LOG_DIR" && extra+=(--output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err")

if [[ -n "${EIDS:-}" ]]; then
    read -r -a eids <<< "$EIDS"
else
    mapfile -t eids < <(grep -vE '^\s*(#|$)' "$EIDS_FILE")
fi

# chain[i] holds the job id most recently submitted to chain i; the next job assigned
# there waits on it. afterany, so a failed job still releases its successor.
declare -a chain=()

n=0
for eid in "${eids[@]}"; do
    slot=$((n % CONCURRENCY))
    args=(
        --job-name="ts1-possm-ft-${eid:0:8}"
        --time="$WALLTIME"
    )
    args+=("${extra[@]}")
    [[ -n "${chain[slot]:-}" ]] && args+=(--dependency="afterany:${chain[slot]}")
    args+=(
        "$SBATCH_SCRIPT"
        trainer=possm_gradual_unfreezing
        # quoted: hydra reads the dots in the path as config nesting otherwise
        "ckpt.load_from='${CKPT}'"
        recording_id="$eid"
        wandb.project="$PROJECT"
        num_workers=3
        +ray.gpu=0.125
        +ray.cpu=3
    )
    [[ -n "${EXTRA:-}" ]] && read -r -a extra <<< "$EXTRA" && args+=("${extra[@]}")

    n=$((n + 1))
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "sbatch ${args[*]}"
        chain[slot]="<job${n}>"
    else
        jid=$(sbatch --parsable "${args[@]}")
        echo "submitted eid=$eid jid=$jid chain=$slot${chain[slot]:+ after ${chain[slot]}}"
        chain[slot]="$jid"
    fi
done

echo "${DRY_RUN:+[dry run] }$n jobs in $CONCURRENCY chains -> wandb $PROJECT (seeds land in ${PROJECT}-seeds)"

#!/usr/bin/env bash
# Runs on: Slurm (wrapper, submits sbatch jobs)
# src/ts1/scripts/finetuning/submit_poyo_plus_repro_sweep.sh
#
# TS1 finetune sweep for the POYO+ attention-backend pretrains, reproducing the recipe
# behind wandb ts1-poyo_plus-ft-neurips2026: finetune.py sweeps base_lr per
# (recording_id, task) at seed 42, selects on best/val/avg, then reruns each winner on
# seeds 43-47 into a "<project>-seeds" project. The grid, num_epochs=300 and the gradual
# unfreezing all live in poyo_plus_gradual_unfreezing.yaml, not here.
#
# One sbatch per (pretrain, recording_id). Each job runs 8 tasks x 5 LRs = 40 sweep
# finetunes then 8 x 5 = 40 seed finetunes, 8 at a time on its GPU (ray.gpu=0.125).
#
# Usage:
#   DRY_RUN=1 ./src/ts1/scripts/finetuning/submit_poyo_plus_repro_sweep.sh   # print only
#   ./src/ts1/scripts/finetuning/submit_poyo_plus_repro_sweep.sh
#   PRETRAINS=nested ./src/ts1/scripts/finetuning/submit_poyo_plus_repro_sweep.sh

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
LOG_DIR="${LOG_DIR:-}"
export REPO_DIR

extra=()
[[ -n "$LOG_DIR" ]] && mkdir -p "$LOG_DIR" && extra+=(--output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err")

# The two 40-epoch pretrains from wandb poyo-plus-pretrain-repro. best.pt, not a fixed
# epoch_N.pt: they peaked at different epochs (xformers 24, nested 35).
declare -A CKPT=(
    [xformers]="poyo_plus_multitask_pretrain_nzc9chpl/best.pt"
    [nested]="poyo_plus_multitask_pretrain_sqgo1e6b/best.pt"
)

PRETRAINS="${PRETRAINS:-xformers nested}"
n=0
for name in $PRETRAINS; do
    ckpt="${CKPT[$name]:-}"
    [[ -n "$ckpt" ]] || { echo "unknown pretrain '$name'; have: ${!CKPT[*]}" >&2; exit 1; }

    while IFS= read -r eid || [[ -n "$eid" ]]; do
        [[ -z "$eid" || "$eid" =~ ^# ]] && continue
        n=$((n + 1))
        args=(
            --job-name="ts1-ft-${name}-${eid:0:8}"
            "${extra[@]}"
            "$SBATCH_SCRIPT"
            trainer=poyo_plus_gradual_unfreezing
            # quoted: hydra reads the dots in the path as config nesting otherwise
            "ckpt.load_from='${ckpt}'"
            recording_id="$eid"
            wandb.project="ts1-poyo_plus-ft-${name}-repro"
            num_workers=3
            +ray.gpu=0.125
            +ray.cpu=3
        )
        if [[ -n "${DRY_RUN:-}" ]]; then
            echo "sbatch ${args[*]}"
        else
            echo "submitting pretrain=$name eid=$eid"
            sbatch "${args[@]}" >/dev/null
        fi
    done < "$EIDS_FILE"
done

echo "${DRY_RUN:+[dry run] }$n jobs for: $PRETRAINS"

#!/bin/bash
#SBATCH --job-name=rrr-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --gpus=1
#SBATCH --time=04:00:00

# Runs on: Slurm (batch script, submit with sbatch)
# src/pretrain/scripts/rrr/launch_rrr_pretrain.sbatch
# One array task per behavior. V is task-specific (its last axis is the task's
# output dim), so each behavior needs its own pretrain run and checkpoint.
#
# lr and epochs are per-task and come from the July 2026 sweep in
# wandb:ibl-benchmark/rrr-ts1-pretrain, not from the config defaults.
#
# Usage:
#   sbatch --array=0-2 src/pretrain/scripts/rrr/launch_rrr_pretrain.sbatch
#   sbatch --array=0 --time=00:20:00 ... num_epochs=1 'recording_ids=[<eid>,<eid>]'   # smoke

set -euo pipefail

# $0 is a node-local spool copy in a batch job, so fall back to the submit dir.
REPO="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# (task, base_lr, num_epochs) reproducing the best run per behavior
TASKS=(choice whisker_motion_energy licking_rate)
LRS=(1e-2 1e-3 1e-3)
EPOCHS=(30 10 10)

I="${SLURM_ARRAY_TASK_ID:-0}"
TASK="${TASKS[$I]}"
LR="${LRS[$I]}"
NEPOCHS="${EPOCHS[$I]}"

SEED="${SEED:-42}"
PROJECT="${PROJECT:-rrr-pretrain-repro}"

# Prebuilt shared venv rather than src/setup.sh: array tasks can land on the same
# node and would otherwise race on the same node-local uv install. .env still has to
# be loaded by hand, since that is where data_root comes from.
source "$REPO/venv/bin/activate"
if [[ -f .env ]]; then set -a; source .env; set +a; fi

# BWB_CKPT_DIR comes from .env (default ./ckpt); point it at shared storage there.
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300

echo "=== rrr pretrain | task=$TASK lr=$LR epochs=$NEPOCHS seed=$SEED | node=$(hostname) ==="

# data_root comes from the hydra config, which reads BWB_DATA_ROOT_PRETRAIN -> BWB_DATA_ROOT
# out of .env; DATA_ROOT here overrides both.
DATA_ROOT_ARGS=()
if [[ -n "${DATA_ROOT:-}" ]]; then DATA_ROOT_ARGS=(data_root="$DATA_ROOT"); fi

python src/pretrain/train.py \
    trainer=rrr_pretrain \
    task="$TASK" \
    "${DATA_ROOT_ARGS[@]}" \
    seed="$SEED" \
    base_lr="$LR" \
    num_epochs="$NEPOCHS" \
    num_workers=24 \
    batch_size=32 \
    ckpt.enable=true \
    log_train_step=false \
    val.patience=999 \
    wandb.mode=online \
    wandb.project="$PROJECT" \
    wandb.name="rrr-${TASK}-r10-lr${LR}-e${NEPOCHS}-repro" \
    +wandb.group="rrr-pretrain-repro" \
    "+wandb.tags=[rrr,pretrain,repro,${TASK}]" \
    hydra.run.dir="outputs/rrr-pretrain/${TASK}/${SLURM_ARRAY_JOB_ID:-local}_${I}" \
    "$@"

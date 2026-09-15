#!/bin/bash
#SBATCH --job-name=poyo-plus-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=448G
#SBATCH --gpus=2
#SBATCH --time=24:00:00

# Runs on: Slurm (batch script, submit with sbatch)
# src/pretrain/scripts/poyo_plus/launch_poyo_plus_pretrain.sbatch
#
# Re-derivation of wandb run v5xiizn7, the POYO+ pretrain every
# ts1-poyo_plus-ft-neurips2026 finetune loads (as v5xiizn7/epoch_32.pt). Its weights are
# gone and the code that produced them is not in git history, so this reproduces the
# recipe on today's pretrain path rather than replaying the run. The one forced deviation
# is the data: that run's build exposed `train_domain`, where
# src/pretrain/datasets/multi_task_behavior.py now reads `pretrain_{split}_domain`. Only
# the all_units build carries those keys, over the same 423 pretrain sessions -- point
# BWB_DATA_ROOT_PRETRAIN (or BWB_DATA_ROOT) at it in .env.
#
# 2 GPUs mirrors the original's DDP shape; batch_size is the global batch, so changing
# --gpus changes the per-device batch and is not a free knob.
#
# ATTN_IMPL is nested or xformers; the latter needs the optional `xformers` extra.
#
# Usage:
#   sbatch src/pretrain/scripts/poyo_plus/launch_poyo_plus_pretrain.sbatch
#   SMOKE=1 sbatch --time=01:00:00 \
#       src/pretrain/scripts/poyo_plus/launch_poyo_plus_pretrain.sbatch
#   EPOCHS=40 ATTN_IMPL=xformers sbatch --gpus=1 --cpus-per-task=28 --mem=224G \
#       src/pretrain/scripts/poyo_plus/launch_poyo_plus_pretrain.sbatch

set -euo pipefail

# $0 is a node-local spool copy in a batch job, so fall back to the submit dir.
REPO="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

ATTN_IMPL="${ATTN_IMPL:-nested}"
PROJECT="${PROJECT:-poyo-plus-pretrain-repro}"
SEED="${SEED:-42}"
# batch_size is the global batch: the trainer feeds batch_size // world_size per device
# and sets max_lr = base_lr * sqrt(batch_size), so the LR is unchanged by --gpus while
# per-device memory is not. EPOCHS drives the OneCycle schedule, not just its length.
EPOCHS="${EPOCHS:-100}"
# appended to the wandb run name, for legs that deviate from the reference recipe
NAME_SUFFIX="${NAME_SUFFIX:-}"

# Prebuilt shared venv rather than src/setup.sh: jobs can land on the same node and would
# otherwise race on the same node-local uv install. .env still has to be loaded by hand,
# since that is where data_root comes from.
source "$REPO/venv/bin/activate"
if [[ -f .env ]]; then set -a; source .env; set +a; fi

# BWB_CKPT_DIR comes from .env (default ./ckpt); point it at shared storage there.
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300

# A smoke run: 1 epoch over 4 sessions, just enough to prove the build loads and to time
# the attention backend. Everything else stays identical to the full run.
SMOKE_ARGS=()
if [[ -n "${SMOKE:-}" ]]; then
    EIDS=$(head -4 src/core/data/pretrain_recording_ids.txt | paste -sd,)
    SMOKE_ARGS=(num_epochs=1 "recording_ids=[${EIDS}]" ckpt.enable=false)
fi

echo "=== poyo+ pretrain | attn_impl=$ATTN_IMPL seed=$SEED smoke=${SMOKE:-0} | node=$(hostname) ==="

# data_root comes from the hydra config, which reads BWB_DATA_ROOT_PRETRAIN -> BWB_DATA_ROOT
# out of .env; DATA_ROOT here overrides both.
DATA_ROOT_ARGS=()
if [[ -n "${DATA_ROOT:-}" ]]; then DATA_ROOT_ARGS=(data_root="$DATA_ROOT"); fi

python src/pretrain/train.py \
    trainer=poyo_plus_multitask_pretrain \
    "${DATA_ROOT_ARGS[@]}" \
    model.attn_impl="$ATTN_IMPL" \
    num_epochs="$EPOCHS" \
    batch_size=512 \
    base_lr=5e-5 \
    num_workers=24 \
    seed="$SEED" \
    ckpt.every_n_epochs=1 \
    +ddp.find_unused_parameters=true \
    log_train_step=false \
    wandb.mode=online \
    wandb.project="$PROJECT" \
    wandb.name="poyo_plus-repro-${ATTN_IMPL}-e${EPOCHS}-seed${SEED}${NAME_SUFFIX}" \
    "+wandb.tags=[poyo_plus,pretrain,repro,${ATTN_IMPL}]" \
    hydra.run.dir="outputs/poyo-plus-pretrain/${SLURM_JOB_ID:-local}" \
    "${SMOKE_ARGS[@]}" \
    "$@"

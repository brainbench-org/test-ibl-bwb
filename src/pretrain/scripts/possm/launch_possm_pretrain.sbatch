#!/bin/bash
#SBATCH --job-name=possm-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=448G
#SBATCH --gpus=2
#SBATCH --time=24:00:00

# Runs on: Slurm (batch script, submit with sbatch)
# src/pretrain/scripts/possm/launch_possm_pretrain.sbatch
#
# Re-derivation of wandb run 7ju4n1wd (project possm-pt_05_01_26), the POSSM pretrain
# every ts1-possm-ft-neurips2026 finetune loads. That run is archived as
# $BWB_NEURIPS_CKPT_DIR/possm.pt; this reproduces its recipe on today's pretrain path
# rather than replaying it. Deviations from that run, all deliberate:
#   - data is v0.0.9/all_units, not v0.0.5/nonfiltered (the pretrain dataset now reads
#     pretrain_{split}_domain, which only the all_units build carries)
#   - grad_clip defaults to 1.0 repo-wide since #398; 7ju4n1wd ran unclipped
#   - no_weight_decay is a shared config default since #396
#   - ckpt.every_n_epochs=1 keeps the per-epoch series; 7ju4n1wd saved only best/last
#
# POSSM has no attn_impl knob (GRU backbone + a 1-latent perceiver encoder), so unlike
# the POYO+ launcher there is no backend to choose.
#
# 2 GPUs: batch_size is the global batch, fed as batch_size // world_size per device,
# and max_lr = base_lr * sqrt(batch_size), so --gpus changes per-device memory and
# wall-clock but not the effective batch or the LR.
#
# Usage:
#   sbatch src/pretrain/scripts/possm/launch_possm_pretrain.sbatch
#   SMOKE=1 sbatch --gpus=1 --cpus-per-task=28 --mem=224G --time=01:00:00 \
#       src/pretrain/scripts/possm/launch_possm_pretrain.sbatch

set -euo pipefail

# $0 is a node-local spool copy in a batch job, so fall back to the submit dir.
REPO="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

PROJECT="${PROJECT:-possm-pretrain-repro}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
NAME_SUFFIX="${NAME_SUFFIX:-}"

# Prebuilt shared venv rather than src/setup.sh: concurrent jobs would otherwise race on
# the same node-local uv install. .env still has to be loaded by hand, since that is
# where data_root comes from.
source "$REPO/venv/bin/activate"
if [[ -f .env ]]; then set -a; source .env; set +a; fi

# BWB_CKPT_DIR comes from .env (default ./ckpt); point it at shared storage there.
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300

# 1 epoch over 4 sessions, just enough to prove the build loads and the multitask readout
# builds -- POSSM's pretrain path has not run since the readout fix in #386.
SMOKE_ARGS=()
if [[ -n "${SMOKE:-}" ]]; then
    EIDS=$(head -4 src/core/data/pretrain_recording_ids.txt | paste -sd,)
    SMOKE_ARGS=(num_epochs=1 "recording_ids=[${EIDS}]" ckpt.enable=false)
fi

echo "=== possm pretrain | seed=$SEED epochs=$EPOCHS smoke=${SMOKE:-0} | node=$(hostname) ==="

# data_root comes from the hydra config, which reads BWB_DATA_ROOT_PRETRAIN -> BWB_DATA_ROOT
# out of .env; DATA_ROOT here overrides both.
DATA_ROOT_ARGS=()
if [[ -n "${DATA_ROOT:-}" ]]; then DATA_ROOT_ARGS=(data_root="$DATA_ROOT"); fi

python src/pretrain/train.py \
    trainer=possm_multitask_pretrain \
    "${DATA_ROOT_ARGS[@]}" \
    num_epochs="$EPOCHS" \
    num_workers=24 \
    seed="$SEED" \
    ckpt.every_n_epochs=1 \
    +ddp.find_unused_parameters=true \
    log_train_step=false \
    wandb.mode=online \
    wandb.project="$PROJECT" \
    wandb.name="possm-repro-e${EPOCHS}-seed${SEED}${NAME_SUFFIX}" \
    "+wandb.tags=[possm,pretrain,repro]" \
    hydra.run.dir="outputs/possm-pretrain/${SLURM_JOB_ID:-local}" \
    "${SMOKE_ARGS[@]}" \
    "$@"

#!/bin/bash
#SBATCH --job-name=ts2-ft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --gpus=1
#SBATCH --time=1-12:00:00

# Runs on: Slurm (batch script, submit with sbatch)
# src/ts2/scripts/finetuning/launch_finetune.sbatch
# One job = one GPU = one recording_id; finetune.py fans out both tasks x the sweep
# grid x the eval seeds over that GPU with Ray.
#
# Env knobs: REPO_DIR, DATA_ROOT, BWB_CKPT_DIR (LOG_DIR is handled by submit_all_finetune.sh).
#
# Usage:
#   sbatch src/ts2/scripts/finetuning/launch_finetune.sbatch \
#       trainer=mtm_finetune wandb.project=ts2-mtm_finetune-ft \
#       ckpt.load_from='n9e7a3bx/best.pt' num_epochs=500 \
#       num_workers=3 +ray.gpu=0.125 +ray.cpu=3 recording_id=<eid>

# $0 is a node-local spool copy in a batch job, so fall back to the submit dir.
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1

# Stash the Hydra args, source setup with only the venv path, then restore them.
# The venv is keyed to REPO_DIR: `setup.sh` does an editable install, so sharing the
# default venv with a job running from another checkout would repoint it mid-run.
ARGS=("$@")
set --
VENV_PATH="${VENV_DIR:-/tmp/$(id -un)/$(basename "$PWD")/venv}"
# setup.sh runs `uv pip install -e` unconditionally, and a whole-suite launch puts several
# jobs on one node sharing this path -- concurrent installs into one venv corrupt it. Hold
# an flock across setup so they queue instead; after the first job it is a fast no-op.
mkdir -p "$(dirname "$VENV_PATH")"
exec 9>"$(dirname "$VENV_PATH")/.setup.lock"
flock 9
source src/setup.sh "$VENV_PATH"
flock -u 9
exec 9>&-
set -- "${ARGS[@]}"

# BWB_CKPT_DIR comes from .env (default ./ckpt); point it at shared storage there.
# data_root comes from the hydra config, which reads BWB_DATA_ROOT_SELECTED_UNITS -> BWB_DATA_ROOT
# out of .env; DATA_ROOT here overrides both.
DATA_ROOT_ARGS=()
if [[ -n "${DATA_ROOT:-}" ]]; then DATA_ROOT_ARGS=(data_root="$DATA_ROOT"); fi

python src/ts2/scripts/finetuning/finetune.py \
    "${DATA_ROOT_ARGS[@]}" \
    "$@"

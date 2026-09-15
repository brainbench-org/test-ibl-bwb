#!/bin/bash
#SBATCH --job-name=lolcat-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --gpus=1
#SBATCH --time=12:00:00

# Runs on: Slurm (batch script, submit with sbatch)
# src/ts3/scripts/lolcat/launch_lolcat_train.sbatch
#
# First LOLCAT run on the current data build (v0.0.9/selected_units). Unlike nemo and
# nuclr this is not a re-derivation -- the baseline landed in #364 with no reference run
# behind it, so this run is what the reference becomes. Hyperparameters are exactly
# lolcat/configs/trainer/lolcat.yaml; nothing is overridden but seed, project and the run name.
#
# LOLCAT is supervised, so it reports test F1 itself and there is no eval_probes.sbatch
# follow-up: test() runs the best-val checkpoint over the eval regime and prints both the
# single-unit and the multi-unit (5-nearest-units-within-60um probability averaging)
# classification reports. Those two reports are the deliverable; they go through the
# logger, so Hydra writes them to $OUT_DIR/train.log alongside the slurm log.
#
# Deviations from a plain `python train.py`:
#   - disable_pbar=true. Under WANDB_MODE=online the captured stdout is non-blocking and
#     tqdm's write rate outruns it; that killed the nuclr job 7920717 at epoch 6 with a
#     BlockingIOError. Nothing to do with the model, but it ends the run.
#   - the histogram cache is expected to exist already (build_lolcat_cache.sbatch).
#     setup() calls update_lolcat_cache regardless, but on a warm cache that is a params
#     check and two dataset opens, not a ~420-recording ray build on the GPU.
#
# Usage:
#   sbatch src/ts3/scripts/lolcat/launch_lolcat_train.sbatch
#   SMOKE=1 sbatch --time=02:00:00 \
#       src/ts3/scripts/lolcat/launch_lolcat_train.sbatch

set -euo pipefail

# $0 is a node-local spool copy in a batch job, so fall back to the submit dir.
REPO="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

PROJECT="${PROJECT:-ts3-lolcat}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-500}"
NAME_SUFFIX="${NAME_SUFFIX:-}"

# Prebuilt shared venv rather than src/setup.sh: concurrent jobs would otherwise race on
# the same node-local uv install. .env still has to be loaded by hand, since that is
# where data_root comes from.
source "$REPO/venv/bin/activate"
if [[ -f .env ]]; then set -a; source .env; set +a; fi

# BWB_CKPT_DIR comes from .env (default ./ckpt); point it at shared storage there.
export BWB_CACHE_DIR="${BWB_CACHE_DIR:-$REPO/cache}"
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300

RUN_NAME="lolcat-v009-seed${SEED}${NAME_SUFFIX}"
OUT_DIR="${OUT_DIR:-$REPO/outputs/ts3-lolcat/${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT_DIR"

# 30 epochs, not 2: val runs every 5 epochs and the balanced sampler only retunes once it
# has both a train and a val loss, so a shorter smoke test never exercises the step it is
# most likely to break in. Test still runs, which is the other half of the point.
SMOKE_ARGS=()
if [[ -n "${SMOKE:-}" ]]; then
    EPOCHS=30
    SMOKE_ARGS=(ckpt.enable=false)
fi

echo "=== lolcat train | seed=$SEED epochs=$EPOCHS smoke=${SMOKE:-0} | node=$(hostname) ==="
echo "out_dir=$OUT_DIR"

# data_root comes from the hydra config, which reads BWB_DATA_ROOT_SELECTED_UNITS ->
# BWB_DATA_ROOT out of .env; DATA_ROOT here overrides both.
DATA_ROOT_ARGS=()
if [[ -n "${DATA_ROOT:-}" ]]; then DATA_ROOT_ARGS=(data_root="$DATA_ROOT"); fi

python src/ts3/train.py \
    trainer=lolcat \
    "${DATA_ROOT_ARGS[@]}" \
    num_epochs="$EPOCHS" \
    seed="$SEED" \
    wandb.mode=online \
    disable_pbar=true \
    wandb.project="$PROJECT" \
    wandb.name="$RUN_NAME" \
    "+wandb.tags=[lolcat,ts3,supervised]" \
    "+slurm_jobid=${SLURM_JOB_ID:-0}" \
    hydra.run.dir="outputs/ts3-lolcat/${SLURM_JOB_ID:-local}" \
    "${SMOKE_ARGS[@]}" \
    "$@"

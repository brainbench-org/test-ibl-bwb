#!/bin/bash
#SBATCH --job-name=rrr-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --gpus=1
#SBATCH --time=06:00:00

# Runs on: Slurm (batch script, submit with sbatch)
# src/pretrain/scripts/rrr/launch_rrr_probe.sbatch
# Phase 2 of the reduced-rank protocol: transfer each pretrained V to every eval
# recording, fitting only that recording's U/b (core.finetuning.Probe).
#
# One array task per behavior, because finetune.py reads cfg.sweep once in main()
# before any job exists -- the grid is global to an invocation, and task_overrides
# (applied per job, after the grid is expanded) cannot make it task-specific. A
# single multi-task invocation would force one grid on all three; licking_rate then
# selects an lr it diverges at. Each behavior also has its own V, since V's last
# axis is the task's output dim.
#
# Usage:
#   sbatch --array=0-2 src/pretrain/scripts/rrr/launch_rrr_probe.sbatch [hydra overrides]

set -euo pipefail

# $0 is a node-local spool copy in a batch job, so fall back to the submit dir.
REPO="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# (task, pretrained V, base_lr grid). Grids reproduce the July 2026 sweeps: licking
# is capped at 1e-3 because Poisson diverges above it, costing a whole recording.
TASKS=(choice whisker_motion_energy licking_rate)
CKPTS=(0qdaroqh/best.pt ugmtqt3p/best.pt axii8o7o/best.pt)
GRIDS=("[1e-4,3e-4,1e-3,3e-3,1e-2]" "[1e-4,3e-4,1e-3,3e-3,1e-2]" "[3e-5,1e-4,3e-4,1e-3]")

I="${SLURM_ARRAY_TASK_ID:-0}"
TASK="${TASKS[$I]}"
CKPT="${CKPTS[$I]}"
GRID="${GRIDS[$I]}"

PROJECT="${PROJECT:-rrr-probe-repro}"

# Shared venv rather than src/setup.sh, so .env has to be loaded by hand.
source "$REPO/venv/bin/activate"
if [[ -f .env ]]; then set -a; source .env; set +a; fi

# BWB_CKPT_DIR comes from .env (default ./ckpt); point it at shared storage there.
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300

# Dozens of ray workers share this allocation, and each torch process otherwise
# sizes its thread pool from the machine's core count, not the cgroup's. Left
# unset, load average hit ~3x the allocated CPUs and per-run time inflated 60%
# (529s vs July's 322s).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== rrr probe | task=$TASK ckpt=$CKPT grid=$GRID | $(hostname) ==="

# data_root comes from the hydra config, which reads BWB_DATA_ROOT_ALL_UNITS -> BWB_DATA_ROOT
# out of .env; DATA_ROOT here overrides both.
DATA_ROOT_ARGS=()
if [[ -n "${DATA_ROOT:-}" ]]; then DATA_ROOT_ARGS=(data_root="$DATA_ROOT"); fi

python src/ts1/scripts/finetuning/finetune.py \
    trainer=rrr_probe \
    "${DATA_ROOT_ARGS[@]}" \
    "+tasks=[$TASK]" \
    "ckpt.load_from='$CKPT'" \
    "+sweep.base_lr=$GRID" \
    num_workers=1 \
    +ray.gpu=0.05 \
    +ray.cpu=1.5 \
    save_preds.enable=true \
    save_preds.path=predictions/ \
    save_preds.label=rrr \
    wandb.mode=online \
    wandb.project="${PROJECT}-${TASK}" \
    hydra.run.dir="outputs/rrr-probe/${TASK}/${SLURM_ARRAY_JOB_ID:-local}_${I}" \
    "$@"

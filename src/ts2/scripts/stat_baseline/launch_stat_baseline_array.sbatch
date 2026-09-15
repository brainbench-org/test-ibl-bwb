#!/bin/bash
#SBATCH --job-name=ts2-stat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# No --mem: a partition that allocates memory per CPU rejects a --mem that is not an
# exact multiple of its DefMemPerCPU, so let the 16 CPUs above carry their own default
# allocation (~88G where this was measured), which is plenty here.
#SBATCH --time=06:00:00
#SBATCH --array=0-7

# Runs on: Slurm (batch script, submit with sbatch)
# Full TS2 statistical-baseline sweep as a single slurm job array, on CPU.
#
# The baselines are zero-parameter (everything is fit in closed form inside
# link_datasets), so they hold a GPU without using it. Send them to a CPU
# partition instead (SBATCH_PARTITION in .env), which is typically far less
# contended than a GPU one.
# CPU support comes from core.utils.util.get_device; nothing here is GPU-specific.
#
# The array index selects one (task, method) pair from COMBOS; each array task
# then loops over every eval recording sequentially inside its own allocation.
# 8 array tasks x 28 sessions = the whole 224-run grid, submitted once. Measured
# on a login node, mean_rate and readout_rrr are ~2 min/session, so a combo is ~1h.
#
# Predictions are registered under
#   $PRED_DIR/$LABEL_PREFIX-<method>/ts2-<task>/<eid>/seed_<seed>.safetensors
# which is exactly the layout ibl_bwb_eval.scoring.ts2 expects.
#
# Usage:
#   # whole grid
#   sbatch src/ts2/scripts/stat_baseline/launch_stat_baseline_array.sbatch
#
#   # co_smoothing only (indices 0-3), at most 2 at a time
#   sbatch --array=0-3%2 src/ts2/scripts/stat_baseline/launch_stat_baseline_array.sbatch
#
#   # one combo, extra hydra overrides passed through
#   sbatch --array=2 src/ts2/scripts/stat_baseline/launch_stat_baseline_array.sbatch model.rank=8
#
#   # re-run only what is missing (skips sessions whose .safetensors exists)
#   RESUME=1 sbatch src/ts2/scripts/stat_baseline/launch_stat_baseline_array.sbatch
#
# Then score:
#   python -m ibl_bwb_eval.scoring.ts2 --pred_dir $PRED_DIR --gt_dir $GT_DIR
#
# Env overrides: SEEDS PRED_DIR RUN_DIR LABEL_PREFIX PROJECT DATA_ROOT EIDS_FILE RESUME VENV
# Set WANDB_MODE=offline if the CPU nodes cannot reach the wandb server.

# Repo root. Not derivable from BASH_SOURCE: slurm copies the batch script to a
# node-local spool file, so $0 does not point into the checkout. SLURM_SUBMIT_DIR
# is where sbatch was invoked, which is the repo root in normal use.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || { echo "cannot cd to REPO=$REPO" >&2; exit 2; }
if [[ ! -f src/ts2/train.py ]]; then
    echo "REPO=$REPO is not an ibl-bwb checkout (no src/ts2/train.py)" >&2
    exit 2
fi

# Stash the Hydra args, source setup with its own args, then restore them.
ARGS=("$@")
set -- ${VENV:+"$VENV"}
source src/setup.sh
set -- "${ARGS[@]}"

# index -> "task:method"; keep in sync with --array above
COMBOS=(
    "co_smoothing:mean_rate"
    "co_smoothing:pop_coupling"
    "co_smoothing:readout_rrr"
    "co_smoothing:readout_isi"
    "forecasting:mean_rate"
    "forecasting:trailing_mean"
    "forecasting:shrinkage"
    "forecasting:ridge_ar"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#COMBOS[@]} )); then
    echo "array index $IDX out of range (0-$(( ${#COMBOS[@]} - 1 )))" >&2
    exit 2
fi
TASK="${COMBOS[$IDX]%%:*}"
METHOD="${COMBOS[$IDX]##*:}"

# the linear algebra is the whole workload here, so give BLAS the allocation
NTHREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"

# data_root comes from the hydra config, which reads BWB_DATA_ROOT_SELECTED_UNITS -> BWB_DATA_ROOT
# out of .env; DATA_ROOT here overrides both.
# TS2 rejects any build that is not selected_units (ts2_dataset.require_unit_filtering).
DATA_ROOT_ARGS=()
if [[ -n "${DATA_ROOT:-}" ]]; then DATA_ROOT_ARGS=(data_root="$DATA_ROOT"); fi
EIDS_FILE="${EIDS_FILE:-$REPO/src/ibl_bwb_eval/data/eval_recording_ids.txt}"
PRED_DIR="${PRED_DIR:-$REPO/predictions}"
LABEL_PREFIX="${LABEL_PREFIX:-stat}"
# One hydra output dir per run. Without this the array tasks run concurrently into
# the default outputs/<date>/<time> and overwrite each other's train.log, which is
# where the selected hyperparameters are recorded.
RUN_DIR="${RUN_DIR:-$REPO/outputs/${SLURM_ARRAY_JOB_ID:-local}}"
PROJECT="${PROJECT:-ts2-stat-baseline}"
# The baselines are closed-form and deterministic, so extra seeds only duplicate
# the same numbers. Add more here if the submission needs n > 1.
SEEDS="${SEEDS:-43}"
RESUME="${RESUME:-0}"

LABEL="${LABEL_PREFIX}-${METHOD}"

if [[ ! -f "$EIDS_FILE" ]]; then
    echo "EIDS_FILE not found: $EIDS_FILE" >&2
    exit 2
fi
mapfile -t EIDS < <(grep -vE '^[[:space:]]*(#|$)' "$EIDS_FILE")
n=${#EIDS[@]}
# An empty session list must not look like a clean sweep: without this the loop
# body never runs and the array task exits 0 having done nothing.
if (( n == 0 )); then
    echo "no recording ids in $EIDS_FILE" >&2
    exit 2
fi

echo "=== array $IDX | task=$TASK method=$METHOD | $n sessions | seeds: $SEEDS ==="
echo "    node=$(hostname) threads=$NTHREADS"
echo "    data_root=${DATA_ROOT:-${BWB_DATA_ROOT_SELECTED_UNITS:-${BWB_DATA_ROOT:-<unset>}}}"
echo "    predictions -> $PRED_DIR/$LABEL/ts2-$TASK/<eid>/seed_<seed>.safetensors"
echo "    extra hydra args: $*"

pass=0; fail=0; skip=0; failed=()
i=0
for eid in "${EIDS[@]}"; do
    i=$((i+1))
    for seed in $SEEDS; do
        out="$PRED_DIR/$LABEL/ts2-$TASK/$eid/seed_${seed}.safetensors"
        if [[ "$RESUME" == "1" && -f "$out" ]]; then
            skip=$((skip+1)); echo "--- [$i/$n] skip (exists) eid=$eid seed=$seed"
            continue
        fi
        echo "=== [$i/$n] $(date +%H:%M:%S) eid=$eid seed=$seed ==="
        if python src/ts2/train.py \
                hydra.run.dir="$RUN_DIR/$LABEL/$TASK/$eid" \
                trainer="$METHOD" \
                task="$TASK" \
                "${DATA_ROOT_ARGS[@]}" \
                recording_id="$eid" \
                seed="$seed" \
                num_workers=4 \
                pin_memory=false \
                persistent_workers=false \
                disable_pbar=true \
                wandb.project="$PROJECT" \
                save_preds.enable=true \
                save_preds.path="$PRED_DIR" \
                save_preds.label="$LABEL" \
                "$@"; then
            pass=$((pass+1))
        else
            fail=$((fail+1)); failed+=("$eid:$seed"); echo "FAILED eid=$eid seed=$seed"
        fi
    done
done

echo "=== done [$TASK/$METHOD]: $pass ok, $fail failed, $skip skipped ==="
if (( fail )); then printf 'FAILED %s\n' "${failed[@]}"; fi
exit $(( fail > 0 ))

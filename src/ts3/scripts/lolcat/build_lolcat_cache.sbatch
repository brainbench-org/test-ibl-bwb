#!/bin/bash
#SBATCH --job-name=lolcat-cache
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00

# Runs on: Slurm (batch script, submit with sbatch)
# src/ts3/scripts/lolcat/build_lolcat_cache.sbatch
#
# Builds the LOLCAT per-trial ISI histogram cache on the current data build. Pure CPU
# work (ray over recordings, numpy histograms per unit per trial), so it runs off the
# GPU reservation and the training job only has to train.
#
# One call covers both regimes: update_lolcat_cache loops pretrain then eval into the
# same .pt, and LOLCATTrainer reads that one file for all three of its datasets. Writes
# are flock-guarded and sharded every 25 recordings, so rerunning resumes a partial
# build; changing n_bins/t_min/t_max/log_bins invalidates the whole cache and rebuilds.
#
# The defaults here must match configs/train.yaml (n_bins=128 via model.n_bins, t_min=1e-3,
# t_max=3.0, log_bins=true) -- a mismatch is not an error, it silently throws the cache
# away and rebuilds on the first training epoch, on the GPU.
#
# Sizing: ~51k units over ~420 sessions at ~500 trials x 128 float32 bins is a ~13GB .pt.
# The merge step loads every shard at once, which is why this wants the std-mem node.
#
# Usage:
#   sbatch src/ts3/scripts/lolcat/build_lolcat_cache.sbatch

set -euo pipefail

# $0 is a node-local spool copy in a batch job, so fall back to the submit dir.
REPO="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

source "$REPO/venv/bin/activate"
if [[ -f .env ]]; then set -a; source .env; set +a; fi

DATA_ROOT="${DATA_ROOT:-$BWB_DATA_ROOT_SELECTED_UNITS}"
CACHE_PATH="${CACHE_PATH:-$REPO/cache/lolcat_isi_hists.pt}"
mkdir -p "$(dirname "$CACHE_PATH")"

echo "=== lolcat cache | data_root=$DATA_ROOT cache=$CACHE_PATH | node=$(hostname) ==="

python src/ts3/models/supervised/lolcat/cache.py \
    --data_root "$DATA_ROOT" \
    --cache_path "$CACHE_PATH" \
    --n_bins 128 \
    --t_min 1e-3 \
    --t_max 3.0 \
    "$@"

python - "$CACHE_PATH" <<'PY'
import sys
import torch
d = torch.load(sys.argv[1], weights_only=False)
print("params:", d["params"])
print("recordings:", len(d["rids"]), "units:", len(d["uids"]))
print("trials/unit (first 5):", [h.shape for h in d["isi_hists"][:5]])
PY

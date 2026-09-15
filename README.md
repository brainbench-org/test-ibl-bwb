<p align="center">
  <img src="docs/source/_static/logo_transparent.png" alt="IBL BrainWideBench" width="200">
</p>

<p align="center">
  <b>IBL BrainWideBench</b> benchmarks large-scale pretraining and across-animal transfer
  in multi-region neural recordings from the
  <a href="https://www.internationalbrainlab.com/">International Brain Laboratory</a>.
  <br>&nbsp;
</p>

<p align="center">
  <a href="https://brainbench-org.github.io/ibl-bwb/"><img alt="Documentation" src=".github/assets/buttons/documentation.svg"></a>
  <a href="https://brainwidebench.iblcore.org/index.html"><img alt="Leaderboard" src=".github/assets/buttons/leaderboard.svg"></a>
  <br><br>
  <!-- TODO: swap in the paper URL (openreview or arXiv) once it is public -->
  <a href="TODO"><img alt="Paper" src=".github/assets/buttons/paper.svg"></a>
  <a href="https://brainbench-org.github.io/ibl-bwb/guides/dataset.html"><img alt="Dataset" src=".github/assets/buttons/dataset.svg"></a>
  <a href="https://huggingface.co/NerDSLab?search=ibl-bwb"><img alt="Pretrained checkpoints on Hugging Face" src=".github/assets/buttons/checkpoints.svg"></a>
  <a href="https://wandb.ai/ibl-benchmark/projects"><img alt="Training runs on Weights & Biases" src=".github/assets/buttons/runs.svg"></a>
  <a href="#citation"><img alt="Cite" src=".github/assets/buttons/cite.svg"></a>
</p>

## Quickstart

The following shows how to download a single session and run the MLP baseline on Task Suite 1.

**1. Download**
```bash
EID="0802ced5-33a3-405e-8336-b65ebc5cb07c"
DATA_ROOT="${HOME}/scratch/data"
TASK="choice"

# creates a directory and downloads the brainset for this session. It may take a couple of minutes (a few Gb).
mkdir -p "${DATA_ROOT}/ibl_brain_wide_bench_2026/eval"

curl -L -o "${DATA_ROOT}/ibl_brain_wide_bench_2026/eval/${EID}.h5" \
    "https://brain-wide-bench.s3.amazonaws.com/brainsets/all_units/ibl_brain_wide_bench_2026/eval/${EID}.h5"
```

**2. Train**
From the root of the repository:
```bash
python src/ts1/train.py \
    trainer=mlp \
    data_root="${DATA_ROOT}" \
    task="${TASK}" \
    recording_id="${EID}" \
    ckpt.enable=true
```

**3. Eval**: runs automatically at the end of training. To run eval only on a saved checkpoint:
```bash
python src/ts1/train.py \
    trainer=mlp \
    data_root="${DATA_ROOT}" \
    task="${TASK}" \
    recording_id="${EID}" \
    ckpt.load_from=ckpt/mlp_<run_id>/best.pt \
    num_epochs=0
```
Checkpointing is off by default, hence `ckpt.enable=true` in step 2: it writes `best.pt` under the
checkpoint directory printed at startup. `ckpt.load_from` takes that path, or one relative to `ckpt.dir`.

Available tasks: `choice`, `reward`, `stimulus_contrast`, `whisker_motion_energy`, `wheel_speed`, `right_paw_speed`, `left_paw_speed`, `licking_rate`.

## Setup
Make sure you are using `python>=3.10` and you have `uv` installed (https://docs.astral.sh/uv/getting-started/installation/).

After cloning the repository, you can setup your environment using the `setup.sh` script.

```bash
cd ibl-bwb
source src/setup.sh <path/to/venv>
```

This will automatically install the required dependencies and activate a virtual environment in the path of your choice. If no path is provided, it will default to `/tmp/<username>/<checkout-dir-name>/venv`, or to `VENV_DIR` from `.env` if that is set.

Or if you'd like to setup your environment manually, inside an active environment you can install required dependencies:

```bash
uv pip install -e ".[train]"
```

POYO+ runs on a stock-torch attention kernel by default. `xformers` gives it a leaner one
-- mostly a lower peak memory, plus a smaller speedup -- but pins a wheel to an exact torch
build, so it is a separate extra:

```bash
uv pip install -e ".[train,xformers]"
```

Then pass `model.attn_impl=xformers`. Checkpoints are interchangeable between the two.

The TS1 `cebra` baseline is a separate extra for the same reason, that it constrains the
rest of the environment (numpy below 2.0 on Windows):

```bash
uv pip install -e ".[train,cebra]"
```

You can avoid caching packages during `uv` install using `--no-cache` or setting the environment variable `UV_NO_CACHE=1`. This matters on a cluster with a small home quota.

### Environment variables

Nothing in the benchmark needs credentials out of the box: W&B logging is
disabled, and the brainsets on S3 are public.

Optional settings live in a `.env` at the repo root, which is gitignored and which
every training entry point loads. Start from the annotated template:

```bash
cp .env.example .env
```

The ones you are most likely to want:

| Variable | Effect |
|---|---|
| `BWB_DATA_ROOT` | Sets `data_root` for every training script, so you stop passing it on the command line |
| `BWB_DATA_ROOT_ALL_UNITS`, `BWB_DATA_ROOT_SELECTED_UNITS`, `BWB_DATA_ROOT_PRETRAIN` | Per-build roots, each falling back to `BWB_DATA_ROOT`. Needed only if you keep more than one build, since the two share a directory name and cannot share a root ([dataset guide](https://brainbench-org.github.io/ibl-bwb/guides/dataset.html)) |
| `BWB_CKPT_DIR` | Where checkpoints go (default `./ckpt`) |
| `WANDB_MODE=online` | Turns on W&B logging, which then also needs `WANDB_API_KEY` |

Every config value read from the environment uses `${oc.env:VAR,default}`, so a
command-line override always wins over `.env`. `.env.example` documents the rest
(output directories, Ray, the data-building pipeline, cluster launch scripts).

Each entry point reads its own root and falls back to `BWB_DATA_ROOT`, so holding
both builds does not mean re-exporting a path between runs:

| Entry point | Variable | Build it reads |
|---|---|---|
| `src/ts1` | `BWB_DATA_ROOT_ALL_UNITS` | `all_units`; another build runs but warns |
| `src/ts2`, `src/ts3` | `BWB_DATA_ROOT_SELECTED_UNITS` | `selected_units`; anything else is refused |
| `src/pretrain` | `BWB_DATA_ROOT_PRETRAIN` | unconstrained, whichever you pretrain on |
| `src/pretrain`, `trainer=nuclr_pretrain` or `nemo_pretrain` | `BWB_DATA_ROOT_SELECTED_UNITS` | `selected_units`; these two score TS3 as they train |

`BWB_DATA_ROOT_PRETRAIN` covers `src/pretrain` only, and the two TS3 trainers that
live there pin themselves to `BWB_DATA_ROOT_SELECTED_UNITS` instead. They and
`src/ts3/train.py` read both regimes from one `data_root`, so their pretrain
sessions have to sit under the `selected_units` root next to the eval ones.

A `custom` build, or any other directory, goes in the variable for the suite you
are running, or on the command line:

```bash
python src/ts1/train.py ... data_root=$PWD/data/custom
```

W&B specifics:

- `online` uploads as the run goes and needs `WANDB_API_KEY`. `offline` keeps the
  full run locally under `WANDB_DIR`, ready to push later with `wandb sync
  wandb/offline-run-*`, which is the useful middle ground if you want your own
  metrics but no account. The default `disabled` records nothing.
- `WANDB_ENTITY` is unset by default, so runs go to your own default entity.
- Every run logs the global pre-clip gradient norm as `train/grad_norm`. Per
  parameter, `log_grads=true` adds `grad_norm/` and `grad_weight_ratio/`,
  `log_weights=true` adds `weight_norm/`. Both are off by default and log every
  step once on; they cost one key per parameter, so raise
  `log_stats_every_n_steps` on a large model.
- `log_code=true` uploads a snapshot of your working tree to the run. It is off by
  default.

### Development

Install dev dependencies (formatters, linters, test runner):
```bash
uv pip install -e ".[train,dev]"
```

To build the documentation locally:
```bash
cd docs
make html
# open build/html/index.html in your browser
```

Then set up pre-commit hooks (`ruff`) before pushing changes:
```bash
pre-commit install
```

## Downloading the data

The benchmark data has been staged in `brainsets` using [torch_brain](https://github.com/neuro-galaxy/torch_brain) where each recording session corresponds to an HDF5 file.
The data is stored on AWS on this bucket [s3://brain-wide-bench/brainsets/](s3://brain-wide-bench/brainsets/), and is browsable at [this address](https://brain-wide-bench.s3.amazonaws.com/index.html).

The bucket is organized as follows:
```
s3://brain-wide-bench/brainsets/
├── all_units/ibl_brain_wide_bench_2026/
│   ├── eval/       # 29 sessions from 13 held-out mice
│   └── pretrain/   # 423 sessions from 126 mice
└── selected_units/ibl_brain_wide_bench_2026/
    ├── eval/       # same 29 sessions, units filtered to label==1 and firing rate >= 1 Hz
    └── pretrain/   # same 423 sessions, units filtered to label==1 and firing rate >= 1 Hz
```

### Quick start: inspect a single file

Before downloading the full dataset, you may want to inspect a single session file. Using `boto3` (no credentials required for public buckets):

```python
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import h5py
import io

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

bucket = "brain-wide-bench"
key = "brainsets/all_units/ibl_brain_wide_bench_2026/pretrain/004d8fd5-41e7-4f1b-a45b-0d4ad76fe446.h5"

obj = s3.get_object(Bucket=bucket, Key=key)
data = io.BytesIO(obj["Body"].read())

with h5py.File(data, "r") as f:
    print("Units:", f["units/id"].shape[0])
    print("Trials:", f["trials/start"].shape[0])
    print("Brain regions (Beryl):", sorted(set(f["units/region_beryl"].asstr())))
```

Or to download the file to disk first:

```python
s3.download_file(bucket, key, key.split("/")[-1])
```

### Downloading the full dataset

**Via AWS CLI**: install instructions [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). No credentials are needed for public buckets; pass `--no-sign-request`.

Download all splits:
```bash
aws s3 sync s3://brain-wide-bench/brainsets/ /your/local/path/ --no-sign-request
```

Download a single split (e.g. `all_units` eval):
```bash
aws s3 sync s3://brain-wide-bench/brainsets/all_units/ibl_brain_wide_bench_2026/eval/ \
    /your/local/path/all_units/ibl_brain_wide_bench_2026/eval/ --no-sign-request
```

**Via boto3**: install with `pip install boto3`:

```python
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

def download_split(unit_filter, split, local_root):
    prefix = f"brainsets/{unit_filter}/ibl_brain_wide_bench_2026/{split}/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket="brain-wide-bench", Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            dest = Path(local_root) / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file("brain-wide-bench", key, str(dest))
            print(f"Downloaded {key}")

download_split("all_units", "pretrain", "/your/local/path/")
```


## Pretrained checkpoints

The pretrained checkpoint behind each reported model is on the
[Hugging Face Hub](https://huggingface.co/NerDSLab?search=ibl-bwb), so the pretraining
stage is optional:

```bash
mkdir -p ckpt
curl -L https://huggingface.co/NerDSLab/ibl-bwb-mtm/resolve/main/mtm.pt -o ckpt/mtm.pt

python src/ts2/train.py trainer=mtm_finetune task=co_smoothing \
    recording_id=<session_id> ckpt.load_from=$PWD/ckpt/mtm.pt
```

A file is named after its model, with a seed suffix for the five-seed models (NuCLR and
NEMO), which also ship the unit embeddings each checkpoint produced. See the
[pretraining guide](https://brainbench-org.github.io/ibl-bwb/guides/pretraining.html) for the full table.

## Running the baselines

*TODO: move to TS-specific directories*

For Task Suite 1 (Decoding), you can run the baselines using the common training script:

```bash
python src/ts1/train.py trainer=... data_root=</path/to/processed> recording_id=... task=...
```

`trainer` is the only required model-selection argument: each trainer config pins its own default `model` config, which you can override separately with `model=...`.

To tune the model instead, you can just change the script to `tune.py` after launching a Ray cluster:
```bash
ray start --head
python src/ts1/tune.py ...
```

Note that the interface for tuning is the same as the training script, so you can override the same configs as well as any additional ones for tuning (e.g. ray configs, sweep seeds, etc.).

For tuning with a little more infrastructure, you can use `tune.sh`:
```bash
bash src/ts1/scripts/tune.sh ...
```

The interface is again the same as the training script, but this will automatically launch a Ray head, tune on all tasks at once if no task is specified, cleanup after tuning is complete, and use a "GPU padder" to ensure GPU utilization is above a target threshold.

### Which scripts need Slurm

Nothing here requires a scheduler: any model runs with `python src/<suite>/train.py`. The
shell scripts are launch convenience only, and every one carries a `# Runs on:` header line.

| Pattern | Slurm | What it is |
|---|---|---|
| `*.sbatch` | yes | A job body. `sbatch <file> [hydra overrides]`, not a direct run. |
| `submit_*.sh` | yes | Fans out, one `sbatch` job per recording or pretrain. |
| `setup.sh`, `tune.sh`, `run_*.sh` | no | Plain bash, any machine. |

Without Slurm, use the last row and read the rest: each `.sbatch` ends in one
`python src/.../train.py` call you can copy, under `#SBATCH` headers that only ask for
CPUs, memory, GPUs and wall time.

<details>
<summary>Documentation TODO</summary>

## Developing your own models

TODO

### Overview of the codebase

TODO

### Using the existing infrastructure

TODO

### Writing your own Trainer

TODO

### Writing your own data pipeline for pretraining

TODO

### Modifications to the evaluation pipeline

TODO (add warnings about changes to the evaluation pipeline)

## Benchmarking compute requirements

TODO (table/figure with compute requirements of the different baselines?)

</details>

---

## Preparing the brainsets from IBL dataset
Please follow instructions in the [README](/ibl_brain_wide_bench_2026) under `ibl_brain_wide_bench_2026/` to download and process the data for the benchmark.


### Acknowledgments
> [!NOTE]
> This project utilizes code from or was influenced by [torch_brain](https://github.com/neuro-galaxy/torch_brain) and [nuclr](https://github.com/nerdslab/nuclr), as well as private code by Vinam Arora and Divyansha Lachi.

## Citation

If you use IBL BrainWideBench, please cite the paper:

```bibtex
@inproceedings{ibl_brain_wide_bench_2026,
  title     = {TODO paper title},
  author    = {TODO author list, in the paper's order},
  booktitle = {TODO proceedings title},
  year      = {TODO},
  url       = {TODO openreview or arXiv URL},
}
```

The benchmark is built on the IBL Brain Wide Map, so please cite the dataset as well:

```bibtex
@misc{ibl_brain_wide_map,
  title  = {TODO IBL Brain Wide Map dataset citation},
  author = {{International Brain Laboratory}},
  year   = {TODO},
}
```

The paper is not public yet, so both entries are placeholders. They mirror
[`CITATION.cff`](CITATION.cff), which is what GitHub's "Cite this repository" button and
citation tooling read, and [the docs front page](docs/source/index.rst); update all three
together.

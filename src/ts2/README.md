# Task Suite 2: Neural Activity Prediction

TS2 trains one model per session. Each run requires three required arguments:
- `recording_id`: session UUID (see `src/ibl_bwb_eval/data/eval_recording_ids.txt`)
- `task`: `co_smoothing` or `forecasting`
- `data_root`: path to the dataset root

All commands below are run from the repo root.

---

## Autoencoder MLP

Trained from scratch, no masking, no pretrained checkpoint. Single call per session.

```bash
python src/ts2/train.py \
    trainer=autoencoder \
    recording_id=<EID> \
    task=co_smoothing \
    data_root=<PATH>
```

Multi-seed (seeds 43-47, sequential):

```bash
python src/ts2/scripts/train_seeds.py \
    trainer=autoencoder \
    recording_id=<EID> \
    task=co_smoothing \
    data_root=<PATH>
```

Config location: `src/ts2/models/single_session/autoencoder/configs/trainer/autoencoder.yaml`

---

## Single-session NDT

Transformer trained from scratch with random block masking. Same interface as the autoencoder.

```bash
python src/ts2/train.py \
    trainer=ndt \
    recording_id=<EID> \
    task=co_smoothing \
    data_root=<PATH>
```

Multi-seed:

```bash
python src/ts2/scripts/train_seeds.py \
    trainer=ndt \
    recording_id=<EID> \
    task=co_smoothing \
    data_root=<PATH>
```

Config location: `src/ts2/models/single_session/ndt/configs/trainer/ndt.yaml`

---

## LFADS

Sequential VAE trained from scratch, with KL and L2 regularizers. Same interface as the autoencoder.

```bash
python src/ts2/train.py \
    trainer=lfads \
    recording_id=<EID> \
    task=co_smoothing \
    data_root=<PATH>
```

Multi-seed:

```bash
python src/ts2/scripts/train_seeds.py \
    trainer=lfads \
    recording_id=<EID> \
    task=co_smoothing \
    data_root=<PATH>
```

During training LFADS corrupts its own encoder input in the shape of the task's hold-out and
takes the reconstruction gradient through the corrupted entries, so it needs no `masker`. Val
and test arrive already stripped by the dataset. Co-smoothing hides single entries (`cd_rate`,
what keeps the model off the identity solution) and whole unit rows (`unit_cd_rate`, the shape
val and test use); forecasting zeroes the trailing bins, with `forecast_grad_scope` taking the
gradient from the held-out `tail` (the default, and what val scores) or the whole `window`.

The regularizers are best treated as search parameters rather than constants, so the config defaults are a starting point and the KL/L2 scales are meant to be tuned:

```bash
python src/ts2/tune.py \
    trainer=lfads \
    recording_id=<EID> \
    task=co_smoothing \
    data_root=<PATH>
```

Config location: `src/ts2/models/single_session/lfads/configs/trainer/lfads.yaml`

---

## Pretrained NDT Stitch

Requires a pretrained NDT Stitch checkpoint. The trainer runs full finetuning: all parameters are trained from the first epoch.

Single session, single seed:

```bash
python src/ts2/train.py \
    trainer=ndt_stitch_finetune \
    recording_id=<EID> \
    task=co_smoothing \
    data_root=<PATH>
```

`ndt_stitch_finetune` and `mtm_finetune` default `ckpt.load_from` to the pretrain their
`ts2-*-ft-neurips2026` sweep used -- `ndt_stitch.pt` / `mtm.pt` under
`BWB_NEURIPS_CKPT_DIR` (set in `.env`; see that archive's `MANIFEST.md`). The ts1
suites finetune the same two pretrains. Loading is all it affects -- runs still save
to `ckpt.dir`/`BWB_CKPT_DIR`.

To finetune a different pretrain, pass `ckpt.load_from=<CKPT_PATH>`, either a path like
`<wandb_run_id>/last.pt` relative to `ckpt.dir` or an absolute path to the checkpoint file.

Full sweep across all sessions and both tasks using Ray (runs one job per session/task/seed in parallel):

```bash
python src/ts2/scripts/finetuning/finetune.py \
    trainer=ndt_stitch_finetune \
    data_root=<PATH> \
    +ray.gpu=0.125 +ray.cpu=3
```

This runs a two-phase sweep: hyperparameter selection (seed 42), then multi-seed evaluation
(seeds 43-47, logged to `<wandb.project>-seeds`). To override the session list pass
`recording_id=<EID>` (single) or point to a file via the `EIDS_FILE` env var.

The grid comes from the trainer config's `sweep` block, one key per Hydra path, expanded as a
cartesian product. Pass `'sweep.base_lr=[...]'` to replace it, or
`'+sweep.<hydra.path>=[...]'` to tune a second key jointly.

Config location: `src/ts2/models/pretrained/ndt_stitch/configs/trainer/ndt_stitch_finetune.yaml`

---

## MtM (Masked Token Model)

Requires a pretrained MtM checkpoint. All parameters are trained from the first epoch.

```bash
python src/ts2/train.py \
    trainer=mtm_finetune \
    recording_id=<EID> \
    task=co_smoothing \
    data_root=<PATH>
```

Full sweep across all sessions:

```bash
python src/ts2/scripts/finetuning/finetune.py \
    trainer=mtm_finetune \
    data_root=<PATH> \
    +ray.gpu=0.125 +ray.cpu=3
```

### Inference prompt token

MtM is prompted: every pretraining forward pass prepends the token of the mask mode that
produced the input, so inference has to carry one too. `predict` always passes the token
matching how the dataset corrupts the input, `neuron` for `co_smoothing`, `causal` for
`forecasting`, set in `TASK_MASK_MODE` (`mtm_eval_trainer.py`). It is not a config knob:
any other token is a mismatch with the data.

The training-time masker still samples all four modes by default. To align finetuning with
the evaluated mode, override it, e.g. for forecasting:

```bash
masker.mask_types='[causal]' masker.causal_ratio=0.1
```

Config locations:
- `src/ts2/models/pretrained/mtm/configs/trainer/mtm_finetune.yaml`

---

## Statistical baselines

Zero-parameter methods: everything is fit in closed form in `link_datasets` (hyperparameters
selected on val), so there is no optimizer and no GPU work. Run them on CPU nodes with
`num_epochs=1`: the config already sets that, and the trainer then reports val and runs
test.

Each method applies to one task and degenerates to the per-unit mean rate on the other.
`val/*` is the score the fit already computed while selecting, not a separate val pass,
so there is no val bps. Forecasting fits take one example per window, so they fit on
overlapping windows (stride `forecast_ind`) to cover every phase of a trial; co-smoothing
fits are per bin, where overlap would only repeat rows.

| `trainer=` | Task | Selected on val |
|---|---|---|
| `mean_rate` | either | nothing |
| `trailing_mean` | `forecasting` | `k` |
| `shrinkage` | `forecasting` | `alpha` |
| `ridge_ar` | `forecasting` | `window_length`, `ridge_lambda` |
| `pop_coupling` | `co_smoothing` | `sigma`, `gamma` |
| `readout_rrr` | `co_smoothing` | `rank`, `ridge_lambda` |
| `readout_isi` | `co_smoothing` | `rank`, `ridge_lambda` |

The reduced-rank readouts select on the val hold-out units (`is_held_out_val`, an independent
draw) and refit for the test units, so the scored units take no part in selection.

Single session:

```bash
python src/ts2/train.py \
    trainer=ridge_ar \
    recording_id=<EID> \
    task=forecasting \
    data_root=<PATH>
```

Full sweep, one Slurm job array: 8 array tasks, one per (task, method) pair, each looping
every eval recording inside its own allocation. 8 x 28 = the whole 224-run grid from one
submit. CPU only (no `--gpus`, BLAS gets the 16 cores), so point `SBATCH_PARTITION` at a
CPU partition.

```bash
sbatch src/ts2/scripts/stat_baseline/launch_stat_baseline_array.sbatch
sbatch --array=0-3%2 src/ts2/scripts/stat_baseline/launch_stat_baseline_array.sbatch  # co_smoothing, 2 at a time
sbatch --array=2 src/ts2/scripts/stat_baseline/launch_stat_baseline_array.sbatch model.rank=8
RESUME=1 sbatch src/ts2/scripts/stat_baseline/launch_stat_baseline_array.sbatch  # only what is missing
```

Env knobs: `SEEDS` (default 43), `PRED_DIR` (default `<repo>/predictions`), `RUN_DIR`,
`LABEL_PREFIX` (default `stat`), `PROJECT` (default `ts2-stat-baseline`), `DATA_ROOT`,
`EIDS_FILE`, `RESUME`, `VENV`. Trailing arguments are forwarded as Hydra overrides.
Predictions land in `$PRED_DIR/<prefix>-<method>/ts2-<task>/<eid>/seed_<seed>.safetensors`,
the layout `ibl_bwb_eval.scoring.ts2` expects.

Without Slurm, [run_all_stat_baseline.sh](scripts/stat_baseline/run_all_stat_baseline.sh)
does one `METHOD`/`TASK` at a time, sequentially on the current machine (`PROJECT`,
`DATA_ROOT`, `EIDS_FILE`, `LOGDIR`). It does not save predictions unless you pass
`save_preds.enable=true save_preds.path=<dir> save_preds.label=stat-<method>`.

Config locations: `src/ts2/models/single_session/stat_baseline/configs/trainer/{mean_rate,trailing_mean,shrinkage,ridge_ar,pop_coupling,readout_rrr,readout_isi}.yaml`

---

## Common overrides

These apply to any model:

| Override | Default | Notes |
|---|---|---|
| `task=forecasting` | `co_smoothing` | Switch task |
| `num_epochs=200` | 100 (300 for pretrained) | Training budget |
| `base_lr=1e-4` | model-dependent | Learning rate |
| `seed=42` | 42 | Reproducibility seed |
| `wandb.project=my-project` | `ts2-<trainer>` | W&B project name |
| `wandb.mode=disabled` | env-dependent | Disable W&B logging |
| `save_preds.enable=true` | false | Save test predictions to disk |
| `ckpt.enable=true` | false | Save checkpoints |
| `ckpt.dir=<PATH>` | `ckpt/` | Checkpoint directory |

## Session list

All eval session IDs are in `src/ibl_bwb_eval/data/eval_recording_ids.txt`. A quick bash loop for any single-session model:

```bash
while IFS= read -r eid; do
    python src/ts2/train.py trainer=ndt recording_id="$eid" task=co_smoothing data_root=<PATH>
done < src/ibl_bwb_eval/data/eval_recording_ids.txt
```

For pretrained models, prefer the `finetune.py` script which parallelises across sessions with Ray.

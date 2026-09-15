# Task Suite 1: Decoding of behavior and stimulus

TS1 trains one model per session. Each run requires three arguments:
- `recording_id`: session UUID (see `src/ibl_bwb_eval/data/eval_recording_ids.txt`)
- `task`: one of `choice`, `reward`, `stimulus_contrast`, `whisker_motion_energy`,
  `wheel_speed`, `right_paw_speed`, `left_paw_speed`, `licking_rate`
- `data_root`: path to the dataset root (defaults to `BWB_DATA_ROOT_ALL_UNITS`, then
  `BWB_DATA_ROOT`)

All commands below are run from the repo root. The full baseline table, with the model
config and eval trainer each `trainer=` resolves to, is in `docs/source/guides/ts1.rst`.

---

## Single session

Trained from scratch on one session, no checkpoint needed.

| `trainer=` | Model |
|---|---|
| `linear` | Single linear layer on binned spikes |
| `mlp` | Multi-layer perceptron on binned spikes |
| `tcn` | Temporal convolutional network |
| `gru` | Gated recurrent unit network |
| `cebra` | CEBRA encoder fit in-run, then an MLP readout |
| `ndt_superv` | Supervised single-session Transformer |
| `poyo` | POYO 1M, no pretraining |
| `possm` | POSSM 10M, no pretraining |

```bash
python src/ts1/train.py \
    trainer=linear \
    recording_id=<EID> \
    task=choice \
    data_root=<PATH>
```

Every `trainer=` above is a model config plus its trainer, so the same run can be spelled
with the generic eval trainer:

```bash
python src/ts1/train.py trainer=eval model=linear recording_id=<EID> task=choice data_root=<PATH>
```

Multi-seed (seeds 43-47, sequential):

```bash
python src/ts1/scripts/train_seeds.py \
    trainer=linear \
    recording_id=<EID> \
    task=choice \
    data_root=<PATH>
```

Config locations: `src/ts1/models/single_session/<model>/configs/trainer/<trainer>.yaml`

---

## Pretrained

Each of these adapts a checkpoint pretrained in `src/pretrain`, and differs only in which
parameters are trained and when.

| `trainer=` | Model | Strategy |
|---|---|---|
| `ndt_stitch_finetune` | `ndt_stitch_10M` | Full finetuning |
| `mtm_finetune` | `mtm_10M` | Full finetuning |
| `ndt2_finetune` | `ndt2_10M` | Full finetuning |
| `ndt2_linear_probe` | `ndt2_10M` | Readout only |
| `ndt2_decoder_probe` | `ndt2_10M` | Decoder and readout |
| `ndt2_gradual_unfreezing` | `ndt2_10M` | Unfreeze at `finetuning.strategy.unfreeze_at_epoch` |
| `neds_gradual_unfreezing` | `neds_10M` | Gradual unfreezing |
| `possm_gradual_unfreezing` | `possm_10M` | Gradual unfreezing |
| `poyo_gradual_unfreezing` | `poyo_10M` | Gradual unfreezing |
| `poyo_plus_gradual_unfreezing` | `poyo_plus_10M` | Gradual unfreezing |
| `rrr_probe` | `rrr` | Per-session `U`/`b` on a frozen shared basis `V` |

```bash
python src/ts1/train.py \
    trainer=ndt_stitch_finetune \
    recording_id=<EID> \
    task=choice \
    data_root=<PATH>
```

`ndt_stitch_finetune`, `mtm_finetune`, `possm_gradual_unfreezing` and
`poyo_plus_gradual_unfreezing` default `ckpt.load_from` to the pretrain their
`ts1-*-ft-neurips2026` sweep used, resolved under `BWB_NEURIPS_CKPT_DIR` (set in `.env`;
see that archive's `MANIFEST.md`). Loading is all it affects: runs still save to
`ckpt.dir`/`BWB_CKPT_DIR`. The rest leave `ckpt.load_from` unset, so pass it as either a
path like `<wandb_run_id>/last.pt` relative to `ckpt.dir` or an absolute path:

```bash
python src/ts1/train.py \
    trainer=ndt2_finetune \
    ckpt.load_from=<CKPT_PATH> \
    recording_id=<EID> \
    task=choice \
    data_root=<PATH>
```

Full sweep across all sessions and tasks using Ray (one job per session/task/seed in
parallel):

```bash
python src/ts1/scripts/finetuning/finetune.py \
    trainer=ndt_stitch_finetune \
    data_root=<PATH> \
    +ray.gpu=0.125 +ray.cpu=3
```

This runs a two-phase sweep: hyperparameter selection (seed 42), then multi-seed
evaluation (seeds 43-47, logged to `<wandb.project>-seeds`). The grid comes from the
trainer config's `sweep` block, one key per Hydra path, expanded as a cartesian product.
Pass `'sweep.base_lr=[...]'` to replace it, `'+sweep.<hydra.path>=[...]'` to tune a second
key jointly, or `'task_sweep.<task>.<key>=[...]'` for one task only. Restrict the run with
`recording_id=<EID>` or `tasks=[choice,reward]`.

Config locations: `src/ts1/models/pretrained/<model>/configs/trainer/<trainer>.yaml`

### NDT2 SSL calibration

`ndt2_calibrate` is not itself a TS1 baseline: it is an optional self-supervised pass over
the eval session, run before `ndt2_finetune` to adapt the checkpoint to that session. Its
trainer is the pretrain-side `NDT2Pretrain`, not a `TS1EvalTrainer`.

```bash
python src/ts1/train.py \
    trainer=ndt2_calibrate \
    ckpt.load_from=<CKPT_PATH> \
    recording_id=<EID> \
    task=choice \
    data_root=<PATH>
```

---

## Tuning

`tune.py` runs the model's Optuna search space (`create_search_space`) over Ray:

```bash
python src/ts1/tune.py \
    trainer=mlp \
    recording_id=<EID> \
    task=choice \
    data_root=<PATH>
```

---

## Common overrides

These apply to any model:

| Override | Default | Notes |
|---|---|---|
| `num_epochs=200` | 100, raised to 200-300 by several pretrained trainers | Training budget |
| `base_lr=1e-4` | 1e-3 | Learning rate |
| `batch_size=64` | 32 | Batch size |
| `seed=42` | 42 | Reproducibility seed |
| `precision=bf16` | fp32, set to bf16 by the pretrained trainers and `ndt_superv` | Autocast precision |
| `wandb.project=my-project` | `ts1-<trainer>` | W&B project name |
| `wandb.mode=disabled` | env-dependent | Disable W&B logging |
| `save_preds.enable=true` | false | Save test predictions to disk |
| `ckpt.enable=true` | false | Save checkpoints |
| `ckpt.dir=<PATH>` | `ckpt/` | Checkpoint directory |

## Session list

All eval session IDs are in `src/ibl_bwb_eval/data/eval_recording_ids.txt`. A quick bash
loop for any single-session model:

```bash
while IFS= read -r eid; do
    python src/ts1/train.py trainer=mlp recording_id="$eid" task=choice data_root=<PATH>
done < src/ibl_bwb_eval/data/eval_recording_ids.txt
```

For pretrained models, prefer `finetuning/finetune.py`, which parallelises across sessions
with Ray.

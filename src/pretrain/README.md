# Pretraining

Most of these models are pretrained on the full multi-session dataset before
being finetuned on a single session for TS1 and TS2 evaluation. NuCLR and NEMO
instead learn unit-level embeddings that TS3 probes for brain region.

All commands below are run from the repo root. The only required argument is
`data_root`; everything else has a default in the trainer config.

## Released checkpoints

The checkpoint behind every model in the paper's reported results is published on the
[Hugging Face Hub](https://huggingface.co/NerDSLab) under `NerDSLab/ibl-bwb-*`, so you can
skip pretraining. Point a downstream trainer's `ckpt.load_from` at the file:

```bash
curl -L https://huggingface.co/NerDSLab/ibl-bwb-mtm/resolve/main/best.pt -o ckpt/mtm.pt
```

The file name differs per repository. The table, with the W&B run behind each checkpoint,
is in the [pretraining guide](../../docs/source/guides/pretraining.rst).

---

## Every model owns its trainer

A model directory holds its model, trainer and configs, and no model's trainer inherits
another's. To add one, copy `my_model/`, which subclasses `core.trainer.BaseTrainer`.

The duplication that follows is deliberate: a training protocol is part of a model's
reported result, so tuning one model must not move another's numbers. Shared code is only
machinery that does not know its caller (`BaseTrainer`, `core/` helpers,
`pretrain/datasets/`). When two trainers converge, promote the stateless part, not the loop.

---

## NDT Stitch

Masked-autoencoder Transformer with block masking (50% mask ratio). Three model
configs: `ndt_stitch_10M` (default, 256-wide x 13 layers), `ndt_stitch_10M_wide`
(same budget reshaped to 512-wide x 5 layers), and `ndt_stitch_20M`
(256-wide x 25 layers).

```bash
python src/pretrain/train.py \
    trainer=ndt_stitch_pretrain \
    data_root=<PATH>
```

Other sizes, via `model=`:

```bash
python src/pretrain/train.py \
    trainer=ndt_stitch_pretrain \
    model=ndt_stitch_20M \
    data_root=<PATH>
```

Config location: `src/pretrain/models/ndt_stitch/configs/trainer/ndt_stitch_pretrain.yaml`

---

## MtM (multi-task masking)

Region-aware masked autoencoder with mixed masking strategies (neuron, causal,
inter-region, intra-region). Three model configs: `mtm_10M` (default, 256-wide x
12 layers), `mtm_10M_wide` (same budget reshaped to 512-wide x 5 layers), and
`mtm_20M` (256-wide x 25 layers).

```bash
python src/pretrain/train.py \
    trainer=mtm_pretrain \
    data_root=<PATH>
```

Other sizes, via `model=`:

```bash
python src/pretrain/train.py \
    trainer=mtm_pretrain \
    model=mtm_20M \
    data_root=<PATH>
```

Config location: `src/pretrain/models/mtm/configs/trainer/mtm_pretrain.yaml`

---

## NDT2

NDT with a simpler masking scheme and gradient accumulation (32 steps by
default). Two sizes: `ndt2_10M` (default) and `ndt2_20M`.

```bash
python src/pretrain/train.py \
    trainer=ndt2_pretrain \
    data_root=<PATH>
```

20M variant:

```bash
python src/pretrain/train.py \
    trainer=ndt2_pretrain \
    model=ndt2_20M \
    data_root=<PATH>
```

Config location: `src/pretrain/models/ndt2/configs/trainer/ndt2_pretrain.yaml`

---

## NEDS

Jointly pretrains on all eight TS1 behavioral tasks with a 10% mask ratio.
Single size (10M).

```bash
python src/pretrain/train.py \
    trainer=neds_pretrain \
    data_root=<PATH>
```

Config location: `src/pretrain/models/neds/configs/trainer/neds_pretrain.yaml`

---

## POYO

Token-based model pretrained on a single behavioral task. Three sizes:
`poyo_10M` (default), `poyo_15M`, and `poyo_20M`.

```bash
python src/pretrain/train.py \
    trainer=poyo_pretrain \
    task=<TASK> \
    data_root=<PATH>
```

Other sizes, via `model=`:

```bash
python src/pretrain/train.py \
    trainer=poyo_pretrain \
    model=poyo_20M \
    task=<TASK> \
    data_root=<PATH>
```

Available tasks: `choice`, `reward`, `stimulus_contrast`, `whisker_motion_energy`,
`wheel_speed`, `right_paw_speed`, `left_paw_speed`, `licking_rate`.

Config location: `src/pretrain/models/poyo/configs/trainer/poyo_pretrain.yaml`

---

## POYO+

POYO pretrained on all eight TS1 tasks at once, with per-task loss weighting and
unit dropout over the input units. Single size (10M).

```bash
python src/pretrain/train.py \
    trainer=poyo_plus_multitask_pretrain \
    data_root=<PATH>
```

To pretrain on a subset of tasks:

```bash
python src/pretrain/train.py \
    trainer=poyo_plus_multitask_pretrain \
    data_root=<PATH> \
    "tasks=[choice,reward,whisker_motion_energy]"
```

The task embedding sizes from the TS1 task vocabulary rather than from `tasks=`, so
a subset run stays checkpoint-compatible with a full one. Changing the TS1 task set
itself reshapes that table and invalidates every earlier checkpoint.

`model.attn_impl` picks the attention backend, `nested` (default) or `xformers`; both
compute the same attention, so checkpoints carry over either way.

Config location: `src/pretrain/models/poyo_plus/configs/trainer/poyo_plus_multitask_pretrain.yaml`

---

## POSSM

Multi-task pretraining on all eight TS1 tasks with per-task loss weighting.
Single size (10M).

```bash
python src/pretrain/train.py \
    trainer=possm_multitask_pretrain \
    data_root=<PATH>
```

To pretrain on a subset of tasks:

```bash
python src/pretrain/train.py \
    trainer=possm_multitask_pretrain \
    data_root=<PATH> \
    "tasks=[choice,reward,whisker_motion_energy]"
```

Config location: `src/pretrain/models/possm/configs/trainer/possm_multitask_pretrain.yaml`

---

## RRR

Reduced-rank linear decoder, and phase 1 of a two-phase protocol: this run fits the
shared temporal basis `V` on the pretrain sessions, and `src/ts1` then transfers `V`
to each eval recording, fitting only that recording's `U`/`b` (`trainer=rrr_probe`).
`best.pt` is the entire output of this phase, so `ckpt.enable` defaults to true.

`task` is required and takes one behavior: `V`'s last axis is the task's output
dimension, so each behavior needs its own run and its own checkpoint.

```bash
python src/pretrain/train.py \
    trainer=rrr_pretrain \
    task=<TASK> \
    data_root=<PATH>
```

Available tasks: `choice`, `reward`, `stimulus_contrast`, `whisker_motion_energy`,
`wheel_speed`, `right_paw_speed`, `left_paw_speed`, `licking_rate`.

The rank is the model's main regularizer, via `model.temporal_rank` (default 10):

```bash
python src/pretrain/train.py \
    trainer=rrr_pretrain \
    task=<TASK> \
    model.temporal_rank=20 \
    data_root=<PATH>
```

Config location: `src/pretrain/models/rrr/configs/trainer/rrr_pretrain.yaml`

---

## NuCLR

Contrastive pretraining on single-unit features, probed by TS3. Reads the
`selected_units` build from `BWB_DATA_ROOT_SELECTED_UNITS`, not
`BWB_DATA_ROOT_PRETRAIN`.

Training reads pretrain sessions only and writes checkpoints, nothing else. It
monitors itself with a probe over pretrain units (`train/l1subout_br`), which
`monitor=null` turns off:

```bash
python src/pretrain/train.py \
    trainer=nuclr_pretrain \
    data_root=<PATH>
```

Turning a checkpoint into unit embeddings is TS3's step, not a pretraining one:

```bash
python src/ts3/extract.py extractor=nuclr extractor.ckpt=<CKPT> data_root=<PATH>
```

Nothing under `src/pretrain/` names the eval regime: the extractor lives in
`ts3/models/inductive/nuclr/` and is what chooses one. See `src/ts3/README.md`.

Config location: `src/pretrain/models/nuclr/configs/trainer/nuclr_pretrain.yaml`,
with the deviations from the official implementation in
`src/pretrain/models/nuclr/README.md`.

---

## NEMO

Aligns waveform and autocorrelogram views of a unit with a CLIP loss, probed by
TS3. Same data root and the same split of duties as NuCLR. The first run caches
waveforms to `cache_path`; pre-build it with
`python -m pretrain.models.nemo.cache`. Early stopping reads
`train/l1subout_br/f1_macro`, so `best.pt` is the checkpoint to hand
`src/ts3/extract.py extractor=nemo`, and `monitor=null` gives up both along with the
probe. NEMO's encoding contract (peak-normalisation, the ACG scale factor, the concat
order) stays in `pretrain/models/nemo/encoding.py`, shared with the online monitor; the
extractor around it is in `ts3/models/inductive/nemo/`.

```bash
python src/pretrain/train.py \
    trainer=nemo_pretrain \
    data_root=<PATH>
```

Config location: `src/pretrain/models/nemo/configs/trainer/nemo_pretrain.yaml`

---

## Common overrides

These apply to any pretrain run:

| Override | Default | Notes |
|---|---|---|
| `num_epochs=200` | 100 | Training budget |
| `base_lr=1e-4` | model-dependent | Learning rate |
| `batch_size=64` | model-dependent | Samples per batch |
| `seed=42` | 42 | Reproducibility seed |
| `ckpt.dir=<PATH>` | `ckpt/` | Checkpoint directory |
| `wandb.project=my-project` | `pretrain-<trainer>` | W&B project name |
| `wandb.mode=disabled` | env-dependent | Disable W&B logging |

DDP is enabled automatically when multiple GPUs are detected.

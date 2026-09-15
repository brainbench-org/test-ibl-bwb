# NuCLR

Implements [NuCLR](https://arxiv.org/abs/2512.01199) based on the [official implementation](https://github.com/nerdslab/nuclr).

## Usage

Run from the repo root:

```bash
python src/pretrain/train.py trainer=nuclr_pretrain data_root=<data-root>
```

By default, this train script:
- Saves checkpoints to `./ckpt/` (configured with `ckpt.dir`)
- Reads pretrain sessions only, and writes no embeddings

Unit embeddings come from TS3, over a checkpoint you name. That is the only step
that reads the eval sessions, and it encodes both regimes in one pass so the two
halves of the file share one set of weights:
```bash
python src/ts3/extract.py extractor=nuclr extractor.ckpt=/path/to/checkpoint.pt data_root=<data-root>
```

## Differences from official implementation

### 1. No probe-level data file splitting

**Original:** The official implementation first splits the data files into probe-level files.
This is done as a way to avoid performing across-probe contrast in the loss.

**This:** Here we do not split the files. We handle the within-probe-only contrast
by passing ``probe_id`` to the loss which applies appropriate masks such that across-probe contrast
is not performed.

**Effect:** This will change the composition of batches during training, leading to slight differences
in training dynamics. This should only have a minor effect.

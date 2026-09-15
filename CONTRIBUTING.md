# Contributing

There are three ways to contribute: submit a model to the benchmark, report a
problem, or change the code. Each is covered below.

> **STATUS: TODO / draft.** Commands and conventions here were read off the repo and
> are accurate. Sections marked TODO are decisions that have not been made yet.

## 1. Submitting a model to the benchmark

You do not need to open a pull request to appear on the benchmark. You train your
model however you like, produce prediction files in the format below, and upload them.

The benchmark website and the leaderboard are at
<https://brainwidebench.iblcore.org/index.html>. Scores are published there rather
than in this repository, so nothing here needs updating when the ranking changes.

TODO the upload path itself is undecided, and the site still lists its docs as
coming soon. Open questions:

- Is submission a web form on the site, an API, or a repository dispatch?
- Do submitters need an account, and can a submission be anonymous during review?
- Is there a submission rate limit, and are the eval labels held out server-side?
- What metadata does a submission carry (model name, parameter count, whether it
  used the pretrain split, a paper or code link)?
- Is scoring run server-side, or does the submitter upload scores as well?

**Producing a submission (TS1 and TS2).** Training writes predictions when you
enable `save_preds` and give the run a label, which is the submission id:

```bash
python src/ts1/train.py \
    trainer=<your_model> \
    data_root=<path> \
    save_preds.enable=true \
    save_preds.label=<submission-id>
```

That writes `.safetensors` files under
`predictions/<label>/<task>/<recording_id>/seed_<seed>.safetensors`
(the root is `BWB_PREDICTIONS_DIR`, default `predictions/`).

**Producing a submission (TS3).** TS3 scores unit embeddings, so which entry point
writes the predictions depends on what your model produces.

A model that emits *embeddings* (any `extractor=` under `ts3/models/inductive/` or
`ts3/models/transductive/`) is probed after the fact, and the probe writes the submission:

```bash
python src/ts3/eval.py probe=linear emb_path=<emb_path> \
    save_preds.enable=true save_preds.label=<submission-id>
```

The submission is filed under the seed of the run behind the embeddings, which
`src/ts3/extract.py` records in the file. Add `seed=<seed>` only for an embeddings file you
built yourself, which carries no such record.

A model that emits *probabilities* directly (LOLCAT) writes them from its own run, using
the same `save_preds` block as TS1 and TS2:

```bash
python src/ts3/train.py trainer=lolcat \
    save_preds.enable=true \
    save_preds.label=<submission-id>
```

Either way TS3 emits one label per scored variant, since single-unit and multi-unit
predictions are scored separately: `<label>_single` and `<label>_multi` from LOLCAT,
and `<label>_linear_single`, `<label>_linear_multi`, `<label>_mlp_single`,
`<label>_mlp_multi` from the probes.

Note TS3 files predictions directly under `<label>/<task>/seed_<seed>.safetensors`, with
no recording-id segment: its units are scored as one pooled set across sessions.

**Scoring locally first.** Each suite has a scorer you can run against your own
ground truth before uploading. The scorers ship as `ibl-bwb-eval` on PyPI, so this
needs neither a checkout nor the training stack:

```bash
pip install ibl-bwb-eval --extra-index-url https://download.pytorch.org/whl/cpu
```

```bash
python -m ibl_bwb_eval.scoring.ts1 --pred_dir predictions/<label> --gt_dir ground_truth/ts1
python -m ibl_bwb_eval.scoring.ts2 --pred_dir predictions/<label> --gt_dir ground_truth/ts2
python -m ibl_bwb_eval.scoring.ts3 --pred_dir predictions/<label> --gt_dir ground_truth/
# all three, aggregated and ranked
python -m ibl_bwb_eval.scoring.aggregation --pred_dir predictions/ --gt_dir ground_truth/
```

TODO: state which seeds a valid submission must cover, and whether a partial
submission (one task suite, or a subset of sessions) is accepted or rejected.

## 2. Reporting issues

Issues are filed on this repository's
[GitHub Issues page](https://github.com/brainbench-org/ibl-bwb/issues). A good report
includes:

- What you ran: the full command, including every Hydra override.
- The `recording_id`, `task`, and seed involved.
- The commit you are on (`git rev-parse --short HEAD`) and your `pip freeze`, or at
  least the torch and `torch-brain` versions.
- The full traceback, not just the last line.

Two things are worth reporting even though they are not crashes: a number you
cannot reproduce from a documented command, and documentation that told you to do
something that does not work.

TODO: issue templates? Labels? Who triages?

## 3. Contributing code

### Setup

```bash
source src/setup.sh <path/to/venv>     # creates the venv and installs [train]
uv pip install -e ".[train,dev]"       # or do it manually inside an active env
pre-commit install
```

See the [README](README.md#setup) for the full setup, including the optional
`xformers` extra and the `.env` variables.

### Before you push

```bash
ruff check . && ruff format --check .   # pinned to 0.15.9, also run by pre-commit
pytest src/tests -q
```

CI currently runs [ruff](.github/workflows/ruff.yml) on every PR, and
[test_pipeline](.github/workflows/test_pipeline.yml) only on changes under
`ibl_brain_wide_bench_2026/`.

TODO `src/tests` is not run by any workflow, so a broken test only shows up if you
run it locally. This should be a CI job.

### Adding a model

Adding a model does not touch any central file. Model configs are auto-discovered by
[model_config_discovery.py](src/hydra_plugins/model_config_discovery.py), so you just
create the directory:

```
src/pretrain/models/<name>/          src/<suite>/models/<single_session|pretrained>/<name>/
├── __init__.py                      the package, and the only thing a _target_ names
├── <name>.py                        the model
├── <name>_<role>.py                 its trainer: <name>_pretrain.py under pretrain/,
└── configs/                         <name>_eval_trainer.py in a suite
    ├── model/<name>.yaml
    └── trainer/<name>_<role>.yaml
```

and it becomes available as `trainer=<name>`. There is no registry to edit.
[src/pretrain/models/my_model/](src/pretrain/models/my_model/) is the scaffold
to copy. Anything else the model needs sits alongside (`masker.py`, `dataset.py`,
`monitor.py`, `cache.py` are all in use), and a second training regime is a second
`<name>_<role>.py` rather than a subdirectory: POSSM keeps
`possm_multitask_pretrain.py` and `possm_single_task_pretrain.py` side by side, and TS1's
NDT2 serves five trainer configs from one `ndt2_eval_trainer.py`.

**A `_target_` names a package, never a module.** Write
`pretrain.models.<name>.<Class>`, not `pretrain.models.<name>.<module>.<Class>`, and
export the class from the package's `__init__.py`. This is the one identifier in the
repo that outlives the checkout: a checkpoint stores the config it trained with, and
[ts3/models/base.py](src/ts3/models/base.py) rebuilds the model from that stored
`model._target_`. A target that reaches into a module pins a filename forever, so
renaming the file strands every checkpoint written before the rename. Targeting the
package leaves everything inside it free to move.
[src/tests/test_config_targets.py](src/tests/test_config_targets.py) checks this.

**Inside a model directory, import relatively.** `from .cache import ...`, not
`from pretrain.models.nemo.cache import ...`. A model directory is a unit meant to be
copied and renamed, which is what the scaffold is for, and relative imports survive that
without editing any file inside it. This is the only place the repo uses them:
`core/`, `ibl_bwb_eval/` and the suite top levels stay absolute, since nothing copies
those. It is a separate rule from the `_target_` one above and does not interact with it:
a target is a string frozen into checkpoints, an import is not.

TODO: does a new baseline need docs and a multi-seed run to be merged, or is code enough?

### Task suites are meant to be self-contained

Someone who only cares about Task Suite 2 should find everything that governs TS2 in
`src/ts2/`, without tracing into a shared base class that also has to serve TS1 and TS3.

Duplication across `ts1/`, `ts2/` and `ts3/` is therefore accepted on purpose. Do not
factor parallel code out into `core/` just because it looks similar. `core/` is for
machinery that is genuinely suite-agnostic (the trainer loop, checkpointing, batching,
samplers); a per-suite trainer is not that, even when two of them currently agree.

The cost is real: a fix in one suite's eval trainer usually needs applying by hand to
the others. When you make one, say so in the PR description so the other suites get it.

### Changes that can move a reported number

TODO. This needs a real policy. Anything touching `src/ibl_bwb_eval/`, an
`*_eval_trainer.py`, the dataset splits, the metrics, or seed handling can change a
published benchmark result while CI stays green. Open questions:

- Who has to review such a PR?
- Do we require a before/after run on a fixed set of sessions, and which ones?
- How do we record that a number in the paper came from a specific commit?

### Releasing `ibl-bwb-eval`

`src/ibl_bwb_eval/` is published to PyPI as a second distribution built from this same
checkout, with its own metadata in [packaging/ibl-bwb-eval/](packaging/ibl-bwb-eval/).
The repo distribution itself is never published: training runs from a checkout.

1. Bump `version` in `packaging/ibl-bwb-eval/pyproject.toml`.
2. `./packaging/ibl-bwb-eval/build.sh && uvx twine check dist/*`. Both already run on
   every PR, so neither should surprise you here.
3. Upload with `uv publish`, then tag the commit you built from as `v<version>`.

Three things about PyPI are worth knowing before step 1. A version number can never be
reused, even after deleting that release, so a bad upload costs the number. Everything
in the metadata is frozen per version, including the project URLs and the dependency
pins, and can only be corrected by releasing again. And yanking hides a release from
resolution without removing it, so it does not undo a publish.

What forces a release is the section above: a change that moves a reported number, or
one that alters the submission format or the public API, has to be reachable by version,
because the leaderboard scores against a pinned one.

Step 3 is manual today, with an account-scoped token. It becomes a GitHub Actions
workflow using PyPI Trusted Publishing once the repository move lands, at which point
there is no token to leak.

### Code style

`ruff` with the config in [pyproject.toml](pyproject.toml) is the whole style guide
(line length 100, isort, pyupgrade, bugbear, simplify). Beyond that:

- Comments explain *why*, not *what*.
- Keep comment density proportional to the surrounding code.

### Naming

The benchmark has one name, "IBL BrainWideBench", rendered in whatever casing the layer
it appears in requires. Pick the rendering from the layer, not from taste:

| Layer | Rendering | Example |
| --- | --- | --- |
| Prose, titles, the paper | `IBL BrainWideBench` | the docs body |
| PyPI, repo, S3 buckets, URLs | hyphens | `brain-wide-bench` |
| Python packages and modules | underscores | `ibl_brain_wide_bench_2026` |
| Python classes | CamelCase, initialisms stay capital | `IBLBrainWideBenchTS1` |

Four rules on top of that:

**Names that cross the repo boundary carry the `ibl` prefix.** True of the repo, the
brainset, every dataset class and the published `ibl-bwb-eval`. A name a stranger types
or imports needs to say whose benchmark it is; an internal one does not.

**A distribution name and its import name differ only in `-` versus `_`.** We control
both, so there is no reason to ship a `scikit-learn`/`sklearn` split.

**An initialism has to be total within its tier, and defined here.** The two in use:

- `IBL`, International Brain Laboratory. Everywhere, never spelled out.
- `BWB`, BrainWideBench. The repo, the distributions and the package namespaces
  (`ibl-bwb`, `ibl-bwb-eval`, `ibl_bwb_eval`), where the name is typed by hand
  constantly. Data identifiers, class names and prose spell it out.

**A model has one rendering per layer, and it comes from the paper.** The paper's casing
in prose and in class names (`NEMO`, `POSSM`, `MtM`, `NuCLR`, `NDT2`, `POYOPlus`), and
lowercase snake_case wherever the name is typed as an argument or written to disk
(`model=nemo`, `trainer=possm_pretrain`, `extractor=poyo`, the model directory, the
checkpoint and embedding filenames). A wrapper keeps the model's rendering and adds its
role: pretrainers end in `Pretrain`, suite eval trainers in `EvalTrainer`, TS3 extractors
in `Extractor`. `src/tests/test_naming.py` checks the class names against the registry in
`src/tests/test_citations.py`.

One name here outlives a rename: a pretrain model class. TS3 rebuilds a checkpoint's model
from the `_target_` stored inside it ([ts3/models/base.py](src/ts3/models/base.py)), and
NuCLR's trainer restores `cfg.model` the same way, so renaming a model class invalidates
every existing checkpoint of that model. Nothing else is read back out of a stored config:
TS1 and TS2 instantiate the model from the live config, and an extractor's embedding
filename comes from the checkpoint path, not the class.

`ts1`, `ts2` and `ts3` work the same way: they are the only spelling of Task Suite N, and
`task_id()` makes them load-bearing. What breaks a name is using the short and long forms
as synonyms in the same tier, so if you introduce a short form, make it the only one.

Some names cannot be changed by renaming them here, because something outside the repo
already refers to them: the `brain-wide-bench` bucket and its
`brainsets/<build>/ibl_brain_wide_bench_2026/<regime>/<eid>.h5` keys, which are cached by
ETag and which users already have on disk; the `tsN-<task>` directory names from
`task_id()`, which name every prediction file already written; and the `COSMOS_LABELS`
order. Treat these as fixed vocabulary that new names conform to.

### Documentation

Docs are Sphinx, under [docs/](docs/):

```bash
uv pip install -e ".[docs]"
cd docs && make html        # or: make html-live, which serves and rebuilds on save
```

The API reference is generated from `__api_ref__` blocks; see
[docs/source/api_reference.py](docs/source/api_reference.py) for the format.

#### Citing a model's paper

One entry per method in [docs/source/refs.bib](docs/source/refs.bib), keyed by the model
directory name (`ndt2`, `poyo_plus`, `lolcat`), cited from the first line of the model
class docstring:

```python
class NDT2(BaseModel):
    """Multi-context masked autoencoder over patched spike tokens :cite:`ndt2`.

    Reference implementation: `context_general_bci
    <https://github.com/joel99/context_general_bci>`_.
    """
```

That first line is what the API tables show, and the role keeps the reference itself in
one place: [docs/source/references.rst](docs/source/references.rst) renders the whole bib.
A reference implementation is a link and not a bib entry, and goes in the paragraph below
it. Trainers, extractors and eval wrappers do not repeat the citation, they point at the
model class; markdown READMEs link the paper URL directly, since roles do not render on
GitHub. Generic baselines (`Linear`, `MLP`, `GRU`, `TCN`, the statistical baselines) carry no
citation.

`src/tests/test_citations.py` holds the registry of which class cites which key, and fails
on a key that is cited but undefined, or defined but never cited.

### Pull requests

TODO. Nothing decided. Open questions: branch naming (history uses `fix/`,
`refactor/`), squash vs merge commits, required reviewers, PR template, and whether
outside contributors need a CLA or DCO.

Contributions are made under the repository's [MIT license](LICENSE). Whether a CLA
or DCO is required on top of that is still open, above.

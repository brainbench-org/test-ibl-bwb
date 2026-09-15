# Brainsets generation

## Commands to generate the brainsets

From the root of the repository, I have launched the following commands

1) all units for the first benchmark (neural decoding)
```shell
brainsets prepare ./ibl_brain_wide_bench_2026 --local --processed-dir <data_root>/all_units
```

2) only selected units for the second and third benchmark.
```shell
brainsets prepare ./ibl_brain_wide_bench_2026 --local --processed-dir <data_root>/selected_units  --unit-filter all
```

## Commands to upload to AWS

```shell
aws s3 --profile ibl sync <data_root> s3://brain-wide-bench/brainsets/
```

```shell
aws s3 --profile ibl cp <codebase_root>/ibl_brain_wide_bench_2026/generation_logs/s3_sample_README.md s3://brain-wide-bench/brainsets/README.md
```

## What produced a given file

Every generated `.h5` carries its own provenance, written by `get_provenance()` in
`pipeline.py`: the pipeline commit, whether the tree was dirty and a hash of the diff if
it was, the pipeline arguments, the build timestamp, `ONE_REVISION_LAST_BEFORE`, and the
ONE-api, ibllib, iblatlas, numpy and scipy versions. Read those fields rather than
trusting anything checked in here, which cannot track a rebuild.

A `uv.lock` used to sit in this directory. It described the environment behind pipeline
`derived_version` 0.0.4 and was never updated, so by 0.0.9 it pointed at an environment
that produced data no longer in the bucket. Recover it from `4598ea8c` if the April 2026
build ever needs reconstructing.


"""Run a local processing smoke test for the IBL brain-wide map pipeline.

Builds a minimal set of pipeline arguments, loads the session manifest, and processes a
single evaluation session twice: once unfiltered and once with the full unit filter set.
Intended for interactive/local debugging of the preprocessing pipeline rather than as a
reusable command-line entry point. Edit RAW_DIR and ROOT_DIR to match your machine.
"""

# %%
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline  # importable because of the sys.path insert above

RAW_DIR = Path("/mnt/s0/BrainSets/raw/ibl_brain_wide_bench_2026")
ROOT_DIR = Path("/mnt/s0/BrainSets/proc_test")

EID = "0a018f12-ee06-4b11-97aa-bbbff5448e9f"  # from data/eval_eids_small.txt


@dataclass
class Args:
    reprocess: bool = True
    list_sessions: str = False
    download_first: bool = True
    unit_filter: tuple = ()
    small: bool = False
    example_frame: bool = False
    api_retries: int = 3


def run(args: Args, label: str):
    this_pipeline = pipeline.Pipeline(
        raw_dir=RAW_DIR,
        processed_dir=ROOT_DIR / label / "ibl_brain_wide_bench_2026",
        args=args,
    )
    manifest = this_pipeline.get_manifest(RAW_DIR, args)
    this_pipeline.process(manifest.loc[EID])


run(Args(), "all_units")
run(Args(unit_filter=["all"]), "selected_units")

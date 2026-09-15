"""Extractors whose unit embedding is a free parameter indexed by unit identity.

A held-out unit has no row until a run creates one, so the eval half of the embeddings file
comes from per-session finetuning checkpoints rather than from the pretrain checkpoint. See
``ts3.models.transductive.base`` for what that costs, and ``ts3.models.inductive`` for the
models it does not apply to.
"""

from .base import TransductiveExtractor
from .ndt_stitch import NDTStitchExtractor
from .poyo import POYOExtractor

__all__ = ["NDTStitchExtractor", "POYOExtractor", "TransductiveExtractor"]

__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": None,
            "autosummary": ["TransductiveExtractor", "POYOExtractor", "NDTStitchExtractor"],
        },
    ],
}

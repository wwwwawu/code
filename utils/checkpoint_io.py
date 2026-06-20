"""Checkpoint loading helpers for ProtoWD.

PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
ProtoWD checkpoints store training metadata, optimizer state, scheduler state,
and RNG state, so trusted project checkpoints must be loaded with
``weights_only=False``.
"""

from __future__ import annotations

import torch


def load_trusted_checkpoint(path, map_location=None):
    """Load a trusted ProtoWD checkpoint across old and new PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not have the weights_only argument.
        return torch.load(path, map_location=map_location)

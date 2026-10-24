"""Reserved implementation module for the Global Magnitude Pruning owner."""
from __future__ import annotations

from torch import nn


def prune_global_magnitude(model: nn.Module, ratio: float, *, exclude_layers=None, inplace: bool = False):
    """Implement the common ``(model, masks, metadata)`` contract here."""
    raise NotImplementedError("implemented on feat/global-pruning; integrate via this contract")

"""Reserved implementation module for the Layer-wise Magnitude Pruning owner."""
from __future__ import annotations

from torch import nn


def prune_layerwise_magnitude(model: nn.Module, ratio: float, *, exclude_layers=None, inplace: bool = False):
    """Implement the common ``(model, masks, metadata)`` contract here."""
    raise NotImplementedError("implemented on feat/layerwise-pruning; integrate via this contract")

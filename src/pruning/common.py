"""Stable entry point for pruning implementations.

Algorithms live in their owner modules; this file is the integration contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class PruningResult:
    """The common return value of a pruning operation."""

    model: nn.Module
    masks: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any]


def prune_model(
    model: nn.Module,
    method: str,
    ratio: float,
    *,
    exclude_layers: tuple[str, ...] | list[str] | None = None,
    inplace: bool = False,
) -> tuple[nn.Module, dict[str, torch.Tensor], dict[str, Any]]:
    """Prune Conv2d/Linear weights and return ``(model, masks, metadata)``.

    ``method`` is ``global_magnitude`` or ``layerwise_magnitude``.  Masks are
    boolean tensors keyed by parameter name, where True means retained.
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio must be in [0, 1)")
    implementations = {
        "global_magnitude": (".global_magnitude", "prune_global_magnitude"),
        "layerwise_magnitude": (".layerwise_magnitude", "prune_layerwise_magnitude"),
    }
    if method not in implementations:
        raise ValueError(f"unknown pruning method: {method}")
    module_name, function_name = implementations[method]
    try:
        module = __import__(f"{__package__}{module_name}", fromlist=[function_name])
    except ModuleNotFoundError as error:
        raise NotImplementedError(f"{method} is not yet available on this branch") from error
    function = getattr(module, function_name)
    return function(model, ratio, exclude_layers=exclude_layers, inplace=inplace)

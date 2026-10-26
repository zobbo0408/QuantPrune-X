"""Stand-in pruning used only to wire the pipeline before owner modules merge.

Results produced with this stub are tagged ``pruning_backend = "stub"`` and are
not experiment results. It follows the ``prune_model`` contract in
``pruning/common.py`` so it can be swapped out without pipeline changes.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from .transforms import PRUNING_METHODS


def _targets(model: nn.Module, exclude_layers) -> dict[str, torch.Tensor]:
    excluded = set(exclude_layers or ())
    return {f"{name}.weight": module.weight for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name not in excluded}


def _keep_largest(values: torch.Tensor, ratio: float) -> torch.Tensor:
    """Exactly ``round(ratio * n)`` smallest magnitudes removed; ties broken by position."""
    flat = values.detach().abs().flatten()
    removed = round(ratio * flat.numel())
    mask = torch.ones_like(flat, dtype=torch.bool)
    if removed:
        mask[torch.argsort(flat, stable=True)[:removed]] = False
    return mask.reshape(values.shape)


@torch.no_grad()
def stub_prune_model(model: nn.Module, method: str, ratio: float, *, exclude_layers=None,
                     inplace: bool = False) -> tuple[nn.Module, dict[str, torch.Tensor], dict[str, Any]]:
    if method not in PRUNING_METHODS:
        raise ValueError(f"unknown pruning method: {method}")
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio must be in [0, 1)")
    result = model if inplace else copy.deepcopy(model)
    targets = _targets(result, exclude_layers)
    if method == "global_magnitude":
        names = list(targets)
        joined = _keep_largest(torch.cat([targets[name].flatten() for name in names]), ratio)
        masks, offset = {}, 0
        for name in names:
            count = targets[name].numel()
            masks[name] = joined[offset:offset + count].reshape(targets[name].shape)
            offset += count
    else:
        masks = {name: _keep_largest(weight, ratio) for name, weight in targets.items()}
    for name, weight in targets.items():
        weight.mul_(masks[name].to(dtype=weight.dtype))
    return result, masks, {"backend": "stub", "method": method, "target_ratio": ratio}

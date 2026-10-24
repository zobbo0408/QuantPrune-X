"""Persistence and application of pruning masks."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import nn


def _normalise(masks: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(mask, dtype=torch.bool).detach().cpu() for name, mask in masks.items()}


def save_masks(path: str | Path, masks: Mapping[str, torch.Tensor]) -> None:
    """Save CPU boolean masks using PyTorch's portable checkpoint format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "masks": _normalise(masks)}, path)


def load_masks(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a mask artifact created by :func:`save_masks`."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "masks" not in payload:
        raise ValueError("invalid mask artifact")
    return _normalise(payload["masks"])


@torch.no_grad()
def apply_masks(model: nn.Module, masks: Mapping[str, torch.Tensor], *, strict: bool = True) -> nn.Module:
    """Apply retained-weight masks in place; masked weights become exact zero."""
    parameters = dict(model.named_parameters())
    unknown = set(masks) - set(parameters)
    if strict and unknown:
        raise KeyError(f"mask keys not found in model: {sorted(unknown)}")
    for name, mask in masks.items():
        if name not in parameters:
            continue
        parameter = parameters[name]
        if tuple(parameter.shape) != tuple(mask.shape):
            raise ValueError(f"mask shape mismatch for {name}")
        parameter.mul_(mask.to(device=parameter.device, dtype=parameter.dtype))
    return model

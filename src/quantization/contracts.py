"""Shared Q-DQ/materialized quantization interface contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


SUPPORTED_BITS = (4, 8)


@dataclass(frozen=True)
class QuantizationResult:
    model: nn.Module
    metadata: Mapping[str, Any]


def validate_bits(bits: int) -> int:
    if type(bits) is not int or bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}")
    return bits


def validate_masks(
    model: nn.Module, mask: Mapping[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Validate all masks and return detached bool tensors on parameter devices."""
    if mask is None:
        return {}
    if not isinstance(mask, Mapping):
        raise TypeError("mask must map parameter names to tensors")
    parameters = dict(model.named_parameters())
    unknown = set(mask) - set(parameters)
    if unknown:
        raise KeyError(f"mask keys not found in model: {sorted(unknown)}")
    weight_ids = {id(layer.weight) for layer in model.modules()
                  if isinstance(layer, (nn.Conv2d, nn.Linear))}
    validated = {}
    for name, retained in mask.items():
        parameter = parameters[name]
        if id(parameter) not in weight_ids:
            raise ValueError(f"mask must name a Conv2d/Linear weight: {name}")
        if not isinstance(retained, torch.Tensor):
            raise TypeError(f"mask must be a tensor for {name}")
        if tuple(retained.shape) != tuple(parameter.shape):
            raise ValueError(f"mask shape mismatch for {name}")
        if not torch.all((retained == 0) | (retained == 1)).item():
            raise ValueError(f"mask must contain only 0 and 1 for {name}")
        validated[name] = retained.detach().to(device=parameter.device, dtype=torch.bool)
    return validated


def quantize_signature() -> str:
    """Canonical algorithm-owner API: quantize(model, bits, mask=None, inplace=False)."""
    return "quantize(model, bits, mask=None, inplace=False) -> (model, metadata)"

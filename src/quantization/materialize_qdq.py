"""Materialize fake-QDQ weights for a reproducible floating-point evaluation."""
from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
from torch import nn

from .contracts import validate_bits


@torch.no_grad()
def quantize(
    model: nn.Module,
    bits: int,
    mask: Mapping[str, torch.Tensor] | None = None,
    *,
    inplace: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Return a Q-DQ materialized copy and metadata.

    This is fake quantization: weights are dequantized back to their original
    floating dtype.  If a pruning mask is supplied, its removed entries stay
    exactly zero, which makes P→Q comparable with Q→P.
    """
    validate_bits(bits)
    result = model if inplace else copy.deepcopy(model)
    parameters = dict(result.named_parameters())
    if mask is not None:
        unknown = set(mask) - set(parameters)
        if unknown:
            raise KeyError(f"mask keys not found in model: {sorted(unknown)}")
    qmax = (1 << (bits - 1)) - 1
    quantized = []
    for name, parameter in parameters.items():
        if parameter.ndim < 2:
            continue
        retained = None if mask is None else mask.get(name)
        values = parameter.detach()
        if retained is not None:
            if tuple(retained.shape) != tuple(values.shape):
                raise ValueError(f"mask shape mismatch for {name}")
            retained = retained.to(device=values.device, dtype=torch.bool)
            values = values * retained
        maximum = values.abs().max()
        if maximum.item() != 0:
            scale = maximum / qmax
            parameter.copy_((values / scale).round().clamp(-qmax, qmax) * scale)
        else:
            parameter.zero_()
        if retained is not None:
            parameter.mul_(retained.to(dtype=parameter.dtype))
        quantized.append(name)
    return result, {
        "representation": "qdq",
        "bit_width": bits,
        "weight_mode": "per_tensor_symmetric",
        "activation_mode": None,
        "mask_applied": mask is not None,
        "quantized_parameters": quantized,
    }

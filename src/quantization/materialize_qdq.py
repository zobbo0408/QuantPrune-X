"""Materialize fake-QDQ weights for a reproducible floating-point evaluation."""
from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
from torch import nn

from .contracts import validate_bits, validate_masks
from .layers import validate_quantized_layers
from .quantizer import Quantizer


def _quantizer_metadata(quantizer: Quantizer, *, enabled: bool) -> dict[str, Any]:
    """Describe quantizer settings; numerical buffers belong in state_dict."""
    return {
        "enabled": bool(enabled),
        "bit_width": quantizer.bits,
        "symmetric": quantizer.symmetric,
        "range_update": "every_forward",
    }


@torch.no_grad()
def quantize(
    model: nn.Module,
    bits: int,
    mask: Mapping[str, torch.Tensor] | None = None,
    *,
    inplace: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Return a Q-DQ materialized copy and metadata.

    QuantizedConv2d/QuantizedLinear weights use the baseline Quantizer once,
    then stay materialized. Ordinary Conv2d/Linear layers are not converted.
    Existing activation settings and ranges are preserved; unconfigured layers
    enable asymmetric activation Q-DQ at ``bits``. Masks are validated before
    mutation and their removed weights stay exactly zero.
    """
    validate_bits(bits)
    validate_quantized_layers(model)
    masks = validate_masks(model, mask)
    result = model if inplace else copy.deepcopy(model)
    parameter_names = {id(parameter): name for name, parameter in result.named_parameters()}
    weight_quantizers = {}
    quantized = []
    layers = {}
    # Snapshot traversal before attaching quantizers, which are also modules.
    for module_name, layer in list(result.named_modules()):
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            continue
        if not hasattr(layer, "activation_quantizer"):
            layer.activation_quantizer = Quantizer(bits, layer.weight.ndim, False).to(layer.weight.device)
            layer.activation_quantizer.train(layer.training)
            layer.quantize_activation = True

        parameter = layer.weight
        name = parameter_names[id(parameter)]
        if id(parameter) not in weight_quantizers:
            retained = masks.get(name)
            values = parameter.detach()
            if retained is not None:
                values = values.masked_fill(~retained, 0)
            quantizer = Quantizer(bits, parameter.ndim, True).to(parameter.device)
            quantizer.train(layer.training)
            materialized = quantizer(values)
            if retained is not None:
                materialized.masked_fill_(~retained, 0)
            parameter.copy_(materialized)
            weight_quantizers[id(parameter)] = quantizer
            quantized.append(name)
        else:
            quantizer = copy.deepcopy(weight_quantizers[id(parameter)])
        layer.weight_quantizer = quantizer
        layer.quantize_weight = False
        layers[module_name] = {
            "weight": _quantizer_metadata(quantizer, enabled=False),
            "activation": _quantizer_metadata(layer.activation_quantizer, enabled=layer.quantize_activation),
        }
    return result, {
        "representation": "qdq",
        "bit_width": bits,
        "weight_mode": "per_tensor_symmetric",
        "activation_mode": "per_tensor_asymmetric",
        "mask_applied": mask is not None,
        "quantized_parameters": quantized,
        "layers": layers,
    }

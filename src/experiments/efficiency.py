"""Consistent static efficiency metrics used by experiment result rows."""
from __future__ import annotations

from torch import nn

from pruning.sparsity import model_sparsity


def static_efficiency(model: nn.Module) -> dict[str, float | int]:
    """Report parameter count, retained count, and serialized FP32-equivalent size."""
    sparsity = model_sparsity(model)
    return {
        "parameter_count": sparsity["parameter_count"],
        "nonzero_count": sparsity["nonzero_count"],
        "model_size_mb": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / (1024 * 1024),
    }

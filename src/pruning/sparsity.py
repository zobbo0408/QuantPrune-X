"""Sparsity accounting shared by all experiment paths."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def model_sparsity(model: nn.Module) -> dict[str, Any]:
    """Return parameter-level zero counts; buffers are deliberately excluded."""
    total = 0
    nonzero = 0
    per_parameter: dict[str, dict[str, int | float]] = {}
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        kept = int(torch.count_nonzero(parameter.detach()).item())
        total += count
        nonzero += kept
        per_parameter[name] = {"total": count, "nonzero": kept, "sparsity": 0.0 if count == 0 else 1.0 - kept / count}
    return {"parameter_count": total, "nonzero_count": nonzero, "actual_sparsity": 0.0 if total == 0 else 1.0 - nonzero / total, "per_parameter": per_parameter}

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
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}")
    return bits


def quantize_signature() -> str:
    """Canonical algorithm-owner API: quantize(model, bits, mask=None, inplace=False)."""
    return "quantize(model, bits, mask=None, inplace=False) -> (model, metadata)"

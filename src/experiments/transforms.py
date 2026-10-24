"""Canonical labels for evaluation transformation order."""
from __future__ import annotations

from enum import StrEnum


class ProcessingOrder(StrEnum):
    FP32 = "FP32"
    QUANTIZED = "Q"
    PRUNED = "P"
    QUANTIZE_THEN_PRUNE = "Q→P"
    PRUNE_THEN_QUANTIZE = "P→Q"

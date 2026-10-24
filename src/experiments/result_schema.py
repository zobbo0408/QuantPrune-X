"""Validation for one JSON-serialisable experiment result row."""
from __future__ import annotations

from typing import Any, Mapping

REQUIRED_RESULT_FIELDS = frozenset({
    "run_id", "model", "checkpoint_source", "source_checkpoint", "representation",
    "bit_width", "pruning_method", "target_ratio", "actual_sparsity", "processing_order",
    "seed", "clean_accuracy", "asr_all", "asr_non_target", "parameter_count",
    "nonzero_count", "model_size_mb", "peak_memory_mb", "latency_ms", "mask_path",
    "output_checkpoint",
})


def validate_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate required columns and return a plain dict suitable for JSONL/CSV."""
    missing = REQUIRED_RESULT_FIELDS - set(result)
    if missing:
        raise ValueError(f"missing result fields: {sorted(missing)}")
    row = dict(result)
    if row["processing_order"] not in {"FP32", "Q", "P", "Q→P", "P→Q"}:
        raise ValueError("invalid processing_order")
    for key in ("clean_accuracy", "asr_all", "asr_non_target", "actual_sparsity"):
        value = row[key]
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{key} must be in [0, 1]")
    if row["bit_width"] is not None and int(row["bit_width"]) not in (4, 8):
        raise ValueError("bit_width must be 4, 8, or null")
    return row

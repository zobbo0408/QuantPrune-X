"""Portable checkpoint envelope for Q-DQ export artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


def save_quantized_checkpoint(path: str | Path, model: nn.Module, metadata: Mapping[str, Any]) -> None:
    """Save model weights plus Q-DQ provenance; never serialize a module object."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "state_dict": model.state_dict(), "quantization": dict(metadata)}, path)


def load_quantized_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read the envelope; caller chooses the architecture before loading weights."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload or "quantization" not in payload:
        raise ValueError("invalid quantized checkpoint")
    return payload

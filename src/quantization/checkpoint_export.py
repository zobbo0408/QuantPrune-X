"""Q-DQ (v1) and packed INT4/INT8 (v2) checkpoints; independent encoding."""

from __future__ import annotations

from enum import StrEnum
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
from torch import nn

from .contracts import validate_bits


class Layout(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"


def _pack_codes(codes: torch.Tensor, width: int) -> torch.Tensor:
    """Pack 1-bit bitmap or 4-bit integer codes, lowest bits first."""
    per_byte = 8 // width
    padded = torch.zeros(
        (codes.numel() + per_byte - 1) // per_byte * per_byte, dtype=torch.uint8
    )
    padded[: codes.numel()] = codes
    shifts = torch.arange(per_byte, dtype=torch.uint8) * width
    return (padded.reshape(-1, per_byte) << shifts).sum(dim=1).to(torch.uint8)


def _unpack_codes(data: torch.Tensor, width: int, count: int) -> torch.Tensor:
    per_byte = 8 // width
    shifts = torch.arange(per_byte, dtype=torch.uint8) * width
    codes = ((data[:, None] >> shifts) & ((1 << width) - 1)).flatten()
    return codes[:count]


def _pack_weight(
    weight: torch.Tensor, scale: float, bits: int, layout: Layout
) -> dict[str, Any]:
    # Recover integers from materialized Q-DQ weights using their original scale.
    integers = (weight.flatten() / scale).round().to(torch.int8)
    entry = {"shape": list(weight.shape), "dtype": str(weight.dtype), "scale": scale}
    if layout is Layout.SPARSE:
        nonzero = integers.ne(0)
        entry["bitmap"] = _pack_codes(nonzero, 1)
        integers = integers[nonzero]
    entry["values"] = (
        _pack_codes(integers.to(torch.uint8) & 0xF, 4) if bits == 4 else integers
    )
    return entry


def _unpack_weight(entry: Mapping[str, Any], bits: int, layout: Layout) -> torch.Tensor:
    shape = entry["shape"]
    count = math.prod(shape)
    scale = entry["scale"]
    dtype = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
    }[entry["dtype"]]
    nonzero = None
    value_count = count
    if layout is Layout.SPARSE:
        nonzero = _unpack_codes(entry["bitmap"], 1, count).bool()
        value_count = int(nonzero.sum().item())
    values = entry["values"]
    if bits == 4:
        codes = _unpack_codes(values, 4, value_count).to(torch.int8)
        values = torch.where(codes >= 8, codes - 16, codes)
    if nonzero is not None:
        dense = torch.zeros(count, dtype=torch.int8)
        dense[nonzero] = values
        values = dense
    # Match the materializer's arithmetic before exposing FP32 inference weights.
    return (values.to(dtype) * scale).reshape(shape).float()


def save_quantized_checkpoint(
    path: str | Path,
    model: nn.Module,
    metadata: Mapping[str, Any],
    *,
    layout: Layout | None = None,
) -> None:
    """Atomically save weights and metadata, retaining Q-DQ as the default.

    ``layout=None`` saves Q-DQ; DENSE or SPARSE saves packed integer weights.
    Packed saves require the materializer's parameter names, bit width and
    original scales. Callers may include
    ``model`` (a registry name), source provenance and transform conditions in
    metadata. Inputs are never modified; no activation quantizers are enabled.
    """
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    payload = {"format_version": 1, "state_dict": state, "quantization": dict(metadata)}
    if layout is not None:
        layout = Layout(layout)
        bits = validate_bits(metadata["bit_width"])
        packed_weights = {}
        for name in metadata["quantized_parameters"]:
            packed_weights[name] = _pack_weight(
                state.pop(name), metadata["scales"][name], bits, layout
            )
        payload.update(format_version=2, layout=layout.value, packed_weights=packed_weights)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def load_quantized_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load v1 Q-DQ or decode v2 packed weights into an FP32 state dictionary.

    Non-quantized parameters and buffers retain their stored dtype. Model
    architecture and activation quantization are not inferred from filenames.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or "state_dict" not in payload
        or "quantization" not in payload
    ):
        raise ValueError("invalid quantized checkpoint")
    version = payload.get("format_version", 1)
    if version == 1:
        return payload
    if version != 2:
        raise ValueError(f"unsupported quantized checkpoint version: {version}")
    layout = Layout(payload["layout"])
    bits = validate_bits(payload["quantization"]["bit_width"])
    state = dict(payload["state_dict"])
    for name, entry in payload["packed_weights"].items():
        state[name] = _unpack_weight(entry, bits, layout)
    payload["state_dict"] = state
    return payload

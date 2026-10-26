"""Portable checkpoint envelope for Q-DQ export artifacts."""
from __future__ import annotations

import copy
from pathlib import Path
import re
from typing import Any, Mapping

import torch
from torch import nn

from evaluation.runtime import resolve_device
from models.registry import create_model
from .contracts import validate_bits
from .layers import QuantizedLayer, validate_quantized_layers
from .materialize_qdq import _quantizer_metadata
from .quantizer import Quantizer


SOURCE_FIELDS = ("source_checkpoint", "source_sha256", "checkpoint_source", "source_url", "weight_license")
OBSERVATION_FIELDS = {
    "min_val": "range_tracker.min_val", "max_val": "range_tracker.max_val",
    "scale": "scale", "zero_point": "zero_point",
}


def _cpu_metadata(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("checkpoint metadata keys must be strings")
        return {key: _cpu_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_metadata(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError(f"unsupported checkpoint metadata type: {type(value).__name__}")


def _validate_metadata(metadata: dict[str, Any]) -> None:
    validate_bits(metadata.get("bit_width"))
    for key, expected in (("representation", "qdq"), ("weight_mode", "per_tensor_symmetric"),
                          ("activation_mode", "per_tensor_asymmetric")):
        if metadata.get(key) != expected:
            raise ValueError(f"invalid quantization metadata: {key}")
    if type(metadata.get("mask_applied")) is not bool:
        raise ValueError("invalid mask_applied metadata")
    names = metadata.get("quantized_parameters")
    if (not isinstance(names, list) or not all(isinstance(name, str) for name in names)
            or len(names) != len(set(names))):
        raise ValueError("invalid quantized_parameters metadata")
    layers = metadata.get("layers")
    if not isinstance(layers, dict) or not all(isinstance(name, str) for name in layers):
        raise ValueError("invalid layer metadata")
    for key in ("model_name", *SOURCE_FIELDS):
        if key not in metadata or (metadata[key] is not None
                                   and (not isinstance(metadata[key], str) or not metadata[key])):
            raise ValueError(f"invalid checkpoint metadata: {key}")
    digest = metadata["source_sha256"]
    if digest is not None and re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ValueError("invalid source SHA-256")
    if metadata["checkpoint_source"] not in (None, "public", "self_trained"):
        raise ValueError("invalid checkpoint_source")


def _restore_quantizer(record: Any, ndim: int, *, weight: bool,
                       state: Mapping[str, torch.Tensor], prefix: str) -> Quantizer:
    if not isinstance(record, dict):
        raise ValueError(f"missing quantizer metadata: {prefix}")
    validate_bits(record.get("bit_width"))
    if type(record.get("enabled")) is not bool or (weight and record["enabled"]):
        raise ValueError(f"invalid quantizer enabled flag: {prefix}")
    if record.get("symmetric") is not weight or record.get("range_update") != "every_forward":
        raise ValueError(f"invalid quantizer mode or range update: {prefix}")
    with torch.device("cpu"):
        quantizer = Quantizer(record["bit_width"], ndim, weight)
    for suffix in ("min_val", "max_val"):
        expected = getattr(quantizer, suffix)
        actual = state.get(f"{prefix}.{suffix}")
        if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype or not torch.equal(actual, expected):
            raise ValueError(f"invalid quantizer integer range buffer: {prefix}.{suffix}")
    keys = [f"{prefix}.{suffix}" for suffix in OBSERVATION_FIELDS.values()]
    present = [key in state for key in keys]
    if any(present) and not all(present):
        raise ValueError(f"incomplete quantizer observation: {prefix}")
    if not any(present):
        return quantizer
    for name, state_key in zip(OBSERVATION_FIELDS, keys):
        value = state[state_key]
        if (not isinstance(value, torch.Tensor) or value.shape != (1,) * ndim
                or not value.is_floating_point() or not torch.isfinite(value).all().item()):
            raise ValueError(f"invalid observation tensor: {state_key}")
        if name in ("min_val", "max_val"):
            setattr(quantizer.range_tracker, name, torch.empty_like(value))
        else:
            setattr(quantizer, name, torch.empty_like(value))
    lo, hi, scale, _ = (state[key] for key in keys)
    if torch.any(lo > hi).item() or torch.any(scale <= 0).item():
        raise ValueError(f"invalid observed range or scale: {prefix}")
    return quantizer


def save_quantized_checkpoint(path: str | Path, model: nn.Module, metadata: Mapping[str, Any]) -> None:
    """Save CPU weights, current Q-DQ state and source metadata in v2 format.

    Set metadata["model_name"] for registered models. Omitted source fields
    are None; unobserved buffers remain unobserved.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata must be a mapping")
    metadata = _cpu_metadata(metadata)
    metadata.setdefault("model_name", None)
    for key in SOURCE_FIELDS:
        metadata.setdefault(key, None)
    layers = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            continue
        if (not isinstance(layer, QuantizedLayer) or layer.quantize_weight
                or not hasattr(layer, "weight_quantizer") or not hasattr(layer, "activation_quantizer")):
            raise ValueError(f"expected a materialized Q-DQ layer: {name}")
        layers[name] = {
            "weight": _quantizer_metadata(layer.weight_quantizer, enabled=layer.quantize_weight),
            "activation": _quantizer_metadata(layer.activation_quantizer, enabled=layer.quantize_activation),
        }
    metadata["layers"] = layers
    _validate_metadata(metadata)
    state = model.state_dict()
    for key, value in state.items():
        state[key] = value.detach().cpu() if isinstance(value, torch.Tensor) else _cpu_metadata(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 2, "state_dict": state, "quantization": metadata}, path)


def load_quantized_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read a checkpoint dictionary on CPU using weights-only deserialization."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or type(payload.get("format_version")) is not int
            or payload["format_version"] not in (1, 2)
            or not isinstance(payload.get("state_dict"), dict)
            or not isinstance(payload.get("quantization"), dict)):
        raise ValueError("invalid quantized checkpoint")
    if payload["format_version"] == 2:
        _validate_metadata(payload["quantization"])
    return payload


def load_quantized_model(
    path: str | Path, *, model: nn.Module | None = None, device: str = "auto",
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore an eval model and metadata from v2, preserving activation updates.

    Use a registered model_name or a copy of model with Q-DQ-capable layers.
    Ordinary Conv2d/Linear templates are rejected. Load strictly on CPU,
    then move to device (auto selects CUDA when available, otherwise CPU).
    """
    destination = resolve_device(device)
    payload = load_quantized_checkpoint(path)
    if payload["format_version"] != 2:
        raise ValueError("full model restoration requires format_version=2; re-export the checkpoint")
    metadata, state = payload["quantization"], payload["state_dict"]
    if model is None:
        if metadata["model_name"] is None:
            raise ValueError("a model template is required for an unregistered model")
        with torch.device("cpu"):
            result = create_model(metadata["model_name"])
    else:
        result = copy.deepcopy(model).cpu()
    validate_quantized_layers(result)
    layers = {name: layer for name, layer in result.named_modules()
              if isinstance(layer, (nn.Conv2d, nn.Linear))}
    if set(layers) != set(metadata["layers"]):
        raise ValueError("checkpoint layer metadata does not match the model")
    weight_ids = {id(layer.weight) for layer in layers.values()}
    expected_names = {name for name, parameter in result.named_parameters() if id(parameter) in weight_ids}
    if expected_names != set(metadata["quantized_parameters"]):
        raise ValueError("quantized parameter metadata does not match the model")
    for name, layer in layers.items():
        record = metadata["layers"][name]
        if not isinstance(record, dict) or not {"weight", "activation"} <= record.keys():
            raise ValueError(f"missing layer quantization settings: {name}")
        for kind in ("weight", "activation"):
            prefix = f"{name}." if name else ""
            quantizer = _restore_quantizer(record[kind], layer.weight.ndim, weight=kind == "weight",
                                           state=state, prefix=f"{prefix}{kind}_quantizer")
            setattr(layer, f"{kind}_quantizer", quantizer)
        layer.quantize_weight = False
        layer.quantize_activation = record["activation"]["enabled"]
    expected_state = result.state_dict()
    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError("state dictionary keys must be strings")
        if key in expected_state and isinstance(expected_state[key], torch.Tensor):
            if not isinstance(value, torch.Tensor) or value.dtype != expected_state[key].dtype:
                raise ValueError(f"state dtype mismatch: {key}")
    result.load_state_dict(state, strict=True)
    return result.eval().requires_grad_(False).to(destination), metadata

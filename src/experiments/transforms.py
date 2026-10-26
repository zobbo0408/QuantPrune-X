"""Canonical labels, run specifications and transformation order for experiments."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from pruning.masks import apply_masks
from quantization.contracts import SUPPORTED_BITS

PRUNING_METHODS = ("global_magnitude", "layerwise_magnitude")


class ProcessingOrder(StrEnum):
    FP32 = "FP32"
    QUANTIZED = "Q"
    PRUNED = "P"
    QUANTIZE_THEN_PRUNE = "Q→P"
    PRUNE_THEN_QUANTIZE = "P→Q"


# ASCII directory-safe names; the arrow labels stay in result rows only.
ORDER_SLUGS = {
    ProcessingOrder.FP32: "fp32",
    ProcessingOrder.QUANTIZED: "q",
    ProcessingOrder.PRUNED: "p",
    ProcessingOrder.QUANTIZE_THEN_PRUNE: "q_then_p",
    ProcessingOrder.PRUNE_THEN_QUANTIZE: "p_then_q",
}
QUANTIZED_ORDERS = frozenset({ProcessingOrder.QUANTIZED, ProcessingOrder.QUANTIZE_THEN_PRUNE,
                              ProcessingOrder.PRUNE_THEN_QUANTIZE})
PRUNED_ORDERS = frozenset({ProcessingOrder.PRUNED, ProcessingOrder.QUANTIZE_THEN_PRUNE,
                           ProcessingOrder.PRUNE_THEN_QUANTIZE})


@dataclass(frozen=True)
class RunSpec:
    """One experiment: a model, a processing order and the parameters that order needs."""

    model: str
    order: ProcessingOrder
    bits: int | None = None
    method: str | None = None
    ratio: float | None = None
    seed: int = 0

    def validate(self) -> RunSpec:
        order = ProcessingOrder(self.order)
        if order in QUANTIZED_ORDERS:
            if self.bits not in SUPPORTED_BITS:
                raise ValueError(f"{order} requires bits in {SUPPORTED_BITS}, got {self.bits}")
        elif self.bits is not None:
            raise ValueError(f"{order} must not set bits")
        if order in PRUNED_ORDERS:
            if self.method not in PRUNING_METHODS:
                raise ValueError(f"{order} requires method in {PRUNING_METHODS}, got {self.method}")
            # Same range as prune_model(): 0.0 is a valid no-op control, 1.0 is rejected.
            if (self.ratio is None or isinstance(self.ratio, bool) or not isinstance(self.ratio, (int, float))
                    or not 0.0 <= self.ratio < 1.0):
                raise ValueError(f"{order} requires ratio in [0, 1), got {self.ratio}")
        elif self.method is not None or self.ratio is not None:
            raise ValueError(f"{order} must not set method or ratio")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError(f"seed must be a nonnegative integer, got {self.seed}")
        return self


def run_id(spec: RunSpec) -> str:
    """Stable, human-readable identifier; identical specs always map to the same directory."""
    spec.validate()
    parts = [spec.model, ORDER_SLUGS[ProcessingOrder(spec.order)]]
    if spec.bits is not None:
        parts.append(f"int{spec.bits}")
    if spec.method is not None:
        parts.extend([spec.method, f"r{round(spec.ratio * 100):02d}"])
    parts.append(f"s{spec.seed}")
    return "__".join(parts)


def expand_matrix(
    models: Sequence[str],
    orders: Sequence[str],
    bit_widths: Sequence[int],
    methods: Sequence[str],
    ratios: Sequence[float],
    *,
    seed: int,
) -> list[RunSpec]:
    """Expand the agreed comparison matrix in a deterministic order."""
    selected = [ProcessingOrder(order) for order in orders]
    specs = []
    for model in models:
        for order in selected:
            bit_options = bit_widths if order in QUANTIZED_ORDERS else [None]
            prune_options = [(m, r) for m in methods for r in ratios] if order in PRUNED_ORDERS else [(None, None)]
            for bits in bit_options:
                for method, ratio in prune_options:
                    specs.append(RunSpec(model, order, bits=bits, method=method, ratio=ratio, seed=seed).validate())
    identifiers = [run_id(spec) for spec in specs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("matrix contains duplicate runs")
    return specs


@dataclass(frozen=True)
class Operations:
    """Owner-module entry points; injected so stubs and real modules are interchangeable."""

    quantize: Callable[..., tuple[nn.Module, Mapping[str, Any]]]
    prune: Callable[..., tuple[nn.Module, Mapping[str, torch.Tensor], Mapping[str, Any]]]


def default_operations() -> Operations:
    from pruning.common import prune_model
    from quantization.materialize_qdq import quantize

    return Operations(quantize=quantize, prune=prune_model)


@dataclass(frozen=True)
class TransformResult:
    model: nn.Module
    masks: Mapping[str, torch.Tensor] | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def apply_transforms(spec: RunSpec, fp32_model: nn.Module, operations: Operations) -> TransformResult:
    """Apply the spec's processing order to a copy of ``fp32_model``."""
    order = ProcessingOrder(spec.validate().order)
    model = copy.deepcopy(fp32_model)
    masks = None
    metadata: dict[str, Any] = {"steps": []}
    if order is ProcessingOrder.QUANTIZE_THEN_PRUNE or order is ProcessingOrder.QUANTIZED:
        model, metadata["quantization"] = operations.quantize(model, spec.bits, inplace=False)
        metadata["steps"].append("quantize")
    if order in PRUNED_ORDERS:
        model, masks, metadata["pruning"] = operations.prune(model, spec.method, spec.ratio, inplace=False)
        metadata["steps"].append("prune")
    if order is ProcessingOrder.PRUNE_THEN_QUANTIZE:
        model, metadata["quantization"] = operations.quantize(model, spec.bits, mask=masks, inplace=False)
        # Re-apply in case a quantizer maps removed weights away from exact zero.
        apply_masks(model, masks)
        metadata["steps"].extend(["quantize", "reapply_mask"])
        metadata["mask_reapplied"] = True
    if masks is not None:
        assert_masked_zero(model, masks)
    return TransformResult(model=model, masks=masks, metadata=metadata)


@torch.no_grad()
def assert_masked_zero(model: nn.Module, masks: Mapping[str, torch.Tensor]) -> None:
    """Fail loudly if any removed weight is not exactly zero (not stripped by ``python -O``)."""
    parameters = dict(model.named_parameters())
    for name, mask in masks.items():
        if name not in parameters:
            raise RuntimeError(f"mask refers to a missing parameter: {name}")
        removed = ~mask.to(device=parameters[name].device, dtype=torch.bool)
        survivors = int(torch.count_nonzero(parameters[name][removed]).item())
        if survivors:
            raise RuntimeError(f"{survivors} pruned weights are nonzero in {name}")


def target_layer_sparsity(masks: Mapping[str, torch.Tensor] | None) -> float | None:
    """Removed fraction over the pruned tensors only, comparable with ``target_ratio``."""
    if masks is None:
        return None
    total = sum(mask.numel() for mask in masks.values())
    kept = sum(int(torch.count_nonzero(mask).item()) for mask in masks.values())
    return 0.0 if total == 0 else 1.0 - kept / total

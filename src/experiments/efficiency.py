"""Consistent static efficiency metrics used by experiment result rows."""
from __future__ import annotations

from contextlib import contextmanager
import statistics
import time
from typing import Iterator

import torch
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


class PeakMemory:
    """Peak CUDA allocation in MiB; ``None`` on CPU where no comparable counter exists."""

    value_mb: float | None = None


@contextmanager
def track_peak_memory(device: torch.device) -> Iterator[PeakMemory]:
    peak = PeakMemory()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    yield peak
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak.value_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)


@torch.inference_mode()
def measure_latency(model: nn.Module, input_shape: tuple[int, ...], device: torch.device, *,
                    batch_size: int = 1, warmup: int = 10, iterations: int = 100) -> float:
    """Mean forward time in milliseconds (schema: 평균 추론 시간) for a fixed random batch."""
    if batch_size < 1 or warmup < 0 or iterations < 1:
        raise ValueError("batch_size and iterations must be positive and warmup nonnegative")
    model.eval()
    generator = torch.Generator().manual_seed(0)
    batch = torch.randn((batch_size, *input_shape), generator=generator).to(device)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else (lambda *_: None)
    for _ in range(warmup):
        model(batch)
    synchronize(device)
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        model(batch)
        synchronize(device)
        timings.append((time.perf_counter() - started) * 1000.0)
    return statistics.fmean(timings)

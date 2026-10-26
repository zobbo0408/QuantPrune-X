"""One experiment run: load → transform → verify masks → evaluate → measure → save."""
from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Mapping

import torch
from torch import nn

from checkpoints import ROOT, sha256
from evaluation.metrics import evaluate_model
from evaluation.runtime import code_sha256, environment, resolve_device, seed_everything, utc_now
from pruning.masks import save_masks
from pruning.sparsity import model_sparsity
from quantization.layers import configure_model

from .efficiency import measure_latency, static_efficiency, track_peak_memory
from .result_schema import validate_result
from .transforms import Operations, RunSpec, apply_transforms, run_id, target_layer_sparsity

RESULT_FILE = "results.json"
FAILURE_FILE = "failure.json"


@dataclass(frozen=True)
class EvaluationSettings:
    output: Path
    device: str = "auto"
    activation_quantization: bool = True
    save_masks: bool = True
    progress: bool = False
    checkpoint_source: str = "public"
    pruning_backend: str = "common"
    checkpoints: Mapping[str, Path] = field(default_factory=dict)
    input_shape: tuple[int, ...] = (3, 32, 32)
    latency_batch_size: int = 1
    latency_warmup: int = 10
    latency_iterations: int = 100


def _write_atomic(path: Path, text: str) -> None:
    """Write via a sibling temp file so an interrupted run never leaves a half-written artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_result(run_dir: str | Path, result: Mapping[str, Any]) -> Path:
    """Validate and write the canonical ``results.json`` artifact for one run."""
    output = Path(run_dir) / RESULT_FILE
    _write_atomic(output, json.dumps(validate_result(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return output


def is_completed(run_dir: str | Path) -> bool:
    """A run counts as done only when it saved a successful, readable result."""
    path = Path(run_dir) / RESULT_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "success"
    except (OSError, ValueError):
        return False


def record_failure(run_dir: str | Path, spec: RunSpec, error: BaseException) -> Path:
    """Keep failures beside the run so a rerun retries them; results.json is never written."""
    output = Path(run_dir) / FAILURE_FILE
    record = {"status": "failed", "run_id": run_id(spec), "failed_at": utc_now(),
              "error": f"{type(error).__name__}: {error}",
              "traceback": "".join(traceback.format_exception(error))}
    _write_atomic(output, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    _log(Path(run_dir), f"FAILED {record['error']}")
    return output


def _log(run_dir: Path, message: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{utc_now()} {message}\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _csv_text(row: Mapping[str, Any]) -> str:
    scalars = {key: value for key, value in row.items() if not isinstance(value, (Mapping, list))}
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(scalars))
    writer.writeheader()
    writer.writerow(scalars)
    return stream.getvalue()


def _relative(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    for root in (base, ROOT):
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return path.name


def run_one(spec: RunSpec, settings: EvaluationSettings, *, load_fp32: Callable[[str], nn.Module],
            loader, operations: Operations) -> dict[str, Any]:
    """Execute one spec and persist ``results.json``, ``metrics.csv``, ``run.log`` (and masks)."""
    identifier = run_id(spec)
    run_dir = settings.output / identifier
    _log(run_dir, f"START {identifier}")
    seed_everything(spec.seed)
    device = resolve_device(settings.device)
    started, timer = utc_now(), time.perf_counter()

    transformed = apply_transforms(spec, load_fp32(spec.model), operations)
    model = transformed.model
    activation_bits = spec.bits if spec.bits is not None and settings.activation_quantization else None
    if activation_bits is not None:
        # Weights are already Q-DQ from the shared quantize(); add activation fake quantization only,
        # so the runtime weight quantizer never re-quantizes them.
        configure_model(model, activation_bits, weight=False, activation=True)
    model = model.to(device)
    _log(run_dir, f"TRANSFORMED steps={transformed.metadata['steps']} device={device}")

    with track_peak_memory(device) as peak:
        metrics = evaluate_model(model, loader, device, progress=settings.progress)
    latency = measure_latency(model, settings.input_shape, device, batch_size=settings.latency_batch_size,
                              warmup=settings.latency_warmup, iterations=settings.latency_iterations)

    mask_file = None
    if transformed.masks is not None and settings.save_masks:
        mask_file = run_dir / "masks.pt"
        save_masks(mask_file, transformed.masks)

    row = _result_row(spec, settings, model, transformed, metrics, device,
                      activation_bits=activation_bits, peak_mb=peak.value_mb, latency=latency, mask_file=mask_file,
                      started=started, elapsed=time.perf_counter() - timer)
    write_result(run_dir, row)
    _write_atomic(run_dir / "metrics.csv", _csv_text(row))
    (run_dir / FAILURE_FILE).unlink(missing_ok=True)
    _log(run_dir, f"DONE clean={row['clean_accuracy']:.4f} asr_all={row['asr_all']:.4f}")
    return json.loads((run_dir / RESULT_FILE).read_text(encoding="utf-8"))


def _result_row(spec: RunSpec, settings: EvaluationSettings, model: nn.Module, transformed, metrics, device, *,
                activation_bits, peak_mb, latency, mask_file, started, elapsed) -> dict[str, Any]:
    """Required schema fields first, then provenance extras (allowed by ``validate_result``)."""
    checkpoint = settings.checkpoints.get(spec.model)
    return {
        "run_id": run_id(spec), "status": "success", "model": spec.model,
        "checkpoint_source": settings.checkpoint_source,
        "source_checkpoint": _relative(Path(checkpoint), settings.output) if checkpoint else None,
        "representation": "fp32" if spec.bits is None else "qdq",
        "bit_width": spec.bits, "activation_bit_width": activation_bits,
        "pruning_method": spec.method or "none", "pruning_backend": settings.pruning_backend if spec.method else None,
        "target_ratio": spec.ratio, "actual_sparsity": model_sparsity(model)["actual_sparsity"],
        "target_layer_sparsity": target_layer_sparsity(transformed.masks),
        "processing_order": str(spec.order), "seed": spec.seed,
        "clean_accuracy": metrics["clean_accuracy"], "asr_all": metrics["asr_all"],
        "asr_non_target": metrics["asr_non_target"],
        "clean_loss": metrics["clean_loss"], "triggered_loss": metrics["triggered_loss"],
        "samples": metrics["samples"], "asr_non_target_denominator": metrics["asr_non_target_denominator"],
        **static_efficiency(model),
        "peak_memory_mb": peak_mb, "latency_ms": latency,
        "latency_batch_size": settings.latency_batch_size,
        "mask_path": _relative(mask_file, settings.output), "output_checkpoint": None,
        "device": str(device),
        "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
        "code_sha256": code_sha256(), "lock_sha256": sha256(ROOT / "uv.lock"),
        "started_at": started, "finished_at": utc_now(), "elapsed_seconds": elapsed,
        "environment": environment(device), "transform": _jsonable(transformed.metadata),
    }

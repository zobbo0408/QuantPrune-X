"""Verify exported checkpoints against runtime and materialized quantize results."""
from __future__ import annotations

import time
import torch

from checkpoints import load_checkpoint, sha256
from quantization.checkpoint_export import OBSERVATION_FIELDS, load_quantized_checkpoint, load_quantized_model
from quantization.layers import configure_model
from quantization.materialize_qdq import quantize
from quantization.quantizer import Quantizer
from .metrics import evaluate_model
from .quantized_checkpoints import make_report, project_path, read_config, run_jobs
from .runtime import seed_everything, utc_now


RTOL, ATOL = 1e-5, 1e-6
METRIC_FIELDS = (
    "samples", "clean_correct", "clean_denominator", "asr_all_successes", "asr_all_denominator",
    "asr_non_target_successes", "asr_non_target_denominator", "clean_accuracy", "asr_all", "asr_non_target",
)


def evaluate_stage(model, loader, device, *, reference=None, progress=True):
    """Capture CPU logits or compare every clean/trigger forward to a reference."""
    captured = []
    forwards = 0
    maximum_error = 0.0

    def inspect_logits(module, inputs, output):
        nonlocal forwards, maximum_error
        actual = output.detach().cpu()
        if not torch.isfinite(actual).all().item():
            raise AssertionError("nonfinite evaluation logits")
        if reference is None:
            captured.append(actual.clone())
        else:
            if forwards >= len(reference):
                raise AssertionError("evaluation produced more forwards than the runtime reference")
            expected = reference[forwards]
            torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
            if not torch.equal(actual.argmax(1), expected.argmax(1)):
                raise AssertionError("predictions differ despite logits passing tolerance")
            maximum_error = max(maximum_error, (actual - expected).abs().max().item())
        forwards += 1

    with model.register_forward_hook(inspect_logits):
        metrics = evaluate_model(model, loader, device, progress=progress)
    if reference is not None and forwards != len(reference):
        raise AssertionError("evaluation produced fewer forwards than the runtime reference")
    return metrics, captured, {"forward_calls": forwards, "max_absolute_error": maximum_error}


def compare_metrics(actual, expected):
    for name in METRIC_FIELDS:
        if actual[name] != expected[name]:
            raise AssertionError(f"evaluation metric differs: {name}: {actual[name]} != {expected[name]}")


def compare_checkpoint_state(model, payload, *, observations=True):
    """Compare restored state to the saved file; observations may evolve after forward."""
    state, reference = model.state_dict(), payload["state_dict"]

    def include(name):
        parts = name.split(".")
        if observations or "activation_quantizer" not in parts:
            return True
        index = parts.index("activation_quantizer")
        name = ".".join(parts[:index])
        if not payload["quantization"]["layers"][name]["activation"]["enabled"]:
            return True
        suffix = ".".join(parts[index + 1:])
        return suffix not in OBSERVATION_FIELDS.values()

    state = {key: value for key, value in state.items() if include(key)}
    reference = {key: value for key, value in reference.items() if include(key)}
    if set(state) != set(reference):
        raise AssertionError("restored state dictionary names differ")
    for name, value in reference.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(state[name].detach().cpu(), value.detach().cpu(), rtol=0, atol=0,
                                       msg=lambda message: f"{name}: {message}")
        elif state[name] != value:
            raise AssertionError(f"restored extra state differs: {name}")
    for name, settings in payload["quantization"]["layers"].items():
        layer = model.get_submodule(name)
        for kind in ("weight", "activation"):
            quantizer = getattr(layer, f"{kind}_quantizer")
            record = settings[kind]
            if (getattr(layer, f"quantize_{kind}"), quantizer.bits, quantizer.symmetric) != (
                    record["enabled"], record["bit_width"], record["symmetric"]):
                raise AssertionError(f"restored quantization settings differ: {name}.{kind}")


def reference_source(job):
    path, digest = job.metadata.get("source_checkpoint"), job.metadata.get("source_sha256")
    if not path or not digest:
        raise ValueError("verification requires source_checkpoint and source_sha256 metadata")
    path = project_path(path)
    if sha256(path) != digest:
        raise ValueError(f"reference checkpoint SHA-256 mismatch: {path}")
    return path, digest


def preflight(job):
    path, digest = reference_source(job)
    load_checkpoint(job.model, path, expected_hash=digest)


def verify_checkpoint(job, config, loader, device, *, progress):
    started, timer = utc_now(), time.perf_counter()
    payload = load_quantized_checkpoint(job.path)
    path, digest = reference_source(job)
    seed_everything(config.evaluation["seed"])
    model = configure_model(load_checkpoint(job.model, path, expected_hash=digest).to(device), job.bits)
    for name, settings in job.metadata["layers"].items():
        layer = model.get_submodule(name)
        activation = settings["activation"]
        layer.quantize_activation = activation["enabled"]
        layer.activation_quantizer = Quantizer(activation["bit_width"], layer.weight.ndim,
                                              activation["symmetric"]).to(layer.weight.device)
    runtime_metrics, reference, runtime_check = evaluate_stage(model, loader, device, progress=progress)
    model, _ = quantize(model, job.bits, inplace=True)
    seed_everything(config.evaluation["seed"])
    materialized_metrics, _, materialized_check = evaluate_stage(model, loader, device, reference=reference, progress=progress)
    compare_metrics(materialized_metrics, runtime_metrics)
    del model
    restored, _ = load_quantized_model(job.path, device=str(device))
    compare_checkpoint_state(restored, payload)
    seed_everything(config.evaluation["seed"])
    restored_metrics, _, restored_check = evaluate_stage(restored, loader, device, reference=reference, progress=progress)
    compare_metrics(restored_metrics, runtime_metrics)
    compare_checkpoint_state(restored, payload, observations=False)
    report = make_report(job, config, device,
                         [("runtime", runtime_metrics), ("materialized", materialized_metrics), ("restored", restored_metrics)],
                         operation="verification", started=started, elapsed=time.perf_counter() - timer)
    report["verification"] = {"rtol": RTOL, "atol": ATOL, "runtime": runtime_check,
                              "materialized": materialized_check, "restored": restored_check,
                              "predictions_equal": True, "metric_counts_equal": True,
                              "state_equal_before_forward": True, "stable_state_equal_after_forward": True}
    return report


def run_verification(path, *, output=None, checkpoints=None, run_id=None, progress=True):
    return run_jobs(read_config(path, output=output, checkpoints=checkpoints, run_id=run_id,
                                default_output="artifacts/runs/quantized-verification"),
                    verify_checkpoint, progress=progress, preflight=preflight)

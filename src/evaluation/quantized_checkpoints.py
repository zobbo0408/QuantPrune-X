"""Batch evaluation of saved quantized checkpoints with independent settings."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import time
import tomllib

from torch.utils.data import DataLoader, Subset

from checkpoints import ROOT, sha256
from data.cifar10 import MEAN, STD, test_loader
from evaluate import RunConfig
from quantization.checkpoint_export import load_quantized_checkpoint, load_quantized_model
from .metrics import evaluate_model
from .results import save_report
from .runtime import environment, provenance, resolve_device, seed_everything, utc_now


EVALUATION_DEFAULTS = {
    "data_dir": "datasets/cifar10", "device": "auto", "batch_size": 128, "workers": 0,
    "seed": 225, "target_class": 0, "normalize": True, "download_data": False, "limit": None,
}


@dataclass(frozen=True)
class CheckpointJob:
    index: int
    path: Path
    metadata: dict

    @property
    def model(self):
        return self.metadata["model_name"]

    @property
    def bits(self):
        return self.metadata["bit_width"]


@dataclass(frozen=True)
class AnalysisConfig:
    path: Path
    output: Path
    run_id: str
    evaluation: dict
    jobs: tuple[CheckpointJob, ...]

    @property
    def destination(self):
        return self.output / self.run_id

    def job_output(self, job):
        return self.destination / f"{job.index:03d}-{job.model}-int{job.bits}"


def project_path(value):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("expected a nonempty path")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def path_text(path):
    path, root = path.resolve(), ROOT.resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def read_config(path, *, output=None, checkpoints=None, run_id=None,
                default_output="artifacts/runs/quantized-evaluation"):
    path = project_path(path)
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if set(data) - {"checkpoints", "output", "run_id", "evaluation"}:
        raise ValueError("unknown analysis settings")
    settings = data.get("evaluation", {})
    if not isinstance(settings, dict) or set(settings) - EVALUATION_DEFAULTS.keys():
        raise ValueError("invalid evaluation settings")
    settings = {**EVALUATION_DEFAULTS, **settings}
    settings["data_dir"] = project_path(settings["data_dir"])
    inputs = checkpoints if checkpoints is not None else data.get("checkpoints", [])
    if not isinstance(inputs, (list, tuple)) or not inputs:
        raise ValueError("checkpoints must list exported files; set TOML checkpoints or --checkpoints")
    paths = [project_path(value).resolve() for value in inputs]
    if len(paths) != len(set(paths)):
        raise ValueError("checkpoint paths must be distinct")
    identifier = run_id if run_id is not None else data.get("run_id", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if not isinstance(identifier, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", identifier) is None:
        raise ValueError("run_id must be a single path component starting with a letter or digit")
    destination = project_path(output if output is not None else data.get("output", default_output))
    jobs = []
    for index, checkpoint in enumerate(paths):
        payload = load_quantized_checkpoint(checkpoint)
        if payload["format_version"] != 2:
            raise ValueError("analysis requires format_version=2")
        metadata = payload["quantization"]
        RunConfig(model=metadata["model_name"], checkpoint=checkpoint, output=destination,
                  precisions=(f"int{metadata['bit_width']}",), **settings).validate()
        jobs.append(CheckpointJob(index, checkpoint, metadata))
    return AnalysisConfig(path, destination, identifier, settings, tuple(jobs))


def make_report(job, config, device, results, *, operation, started, elapsed):
    settings = config.evaluation
    return {
        "schema_version": 1, "operation": operation, "run_id": config.run_id,
        "model": job.model, "dataset": "cifar10-test",
        "checkpoint": {"path": path_text(job.path), **{key: job.metadata.get(key) for key in
                       ("source_checkpoint", "source_sha256", "checkpoint_source", "source_url", "weight_license")}},
        "conditions": {**settings, "data_dir": path_text(settings["data_dir"]), "shuffle": False,
                       "requested_device": settings["device"]},
        "quantization": {"representation": "qdq", "bit_width": job.bits, "range_update": "every_forward",
                         "weight_mode": "per_tensor_symmetric", "activation_mode": "per_tensor_asymmetric", "calibration": False},
        "preprocessing": {"mean": MEAN if settings["normalize"] else None,
                          "std": STD if settings["normalize"] else None,
                          "trigger": {"shape": "square", "slice": [27, 31], "channels": "all",
                                      "value": "image maximum before normalization"}},
        "environment": environment(device), "provenance": provenance(job.path),
        "experiment": {"path": path_text(config.path), "sha256": sha256(config.path)},
        "output_directory": path_text(config.job_output(job)),
        "started_at": started, "finished_at": utc_now(), "elapsed_seconds": elapsed,
        "results": [{"stage": stage, "precision": f"INT{job.bits}", "bits": job.bits, **metrics}
                    for stage, metrics in results],
    }


def evaluate_checkpoint(job, config, loader, device, *, progress):
    started, timer = utc_now(), time.perf_counter()
    model, _ = load_quantized_model(job.path, device=str(device))
    seed_everything(config.evaluation["seed"])
    metrics = evaluate_model(model, loader, device, progress=progress)
    return make_report(job, config, device, [("quantized", metrics)], operation="evaluation",
                       started=started, elapsed=time.perf_counter() - timer)


def run_jobs(config, action, *, progress=True, preflight=None):
    device = resolve_device(config.evaluation["device"])
    if config.destination.exists():
        raise FileExistsError(f"output run already exists: {config.destination}")
    if preflight is not None:
        for job in config.jobs:
            preflight(job)
    created = []
    config.destination.mkdir(parents=True, exist_ok=False)
    try:
        settings = config.evaluation
        seed_everything(settings["seed"])
        loader = test_loader(str(settings["data_dir"]), download=settings["download_data"],
                             batch_size=settings["batch_size"], workers=settings["workers"],
                             normalize=settings["normalize"], target=settings["target_class"])
        if settings["limit"] is not None:
            if settings["limit"] > len(loader.dataset):
                raise ValueError("limit exceeds the available test samples")
            loader = DataLoader(Subset(loader.dataset, range(settings["limit"])), batch_size=settings["batch_size"],
                                num_workers=settings["workers"], shuffle=False)
        reports = []
        for job in config.jobs:
            destination = config.job_output(job)
            destination.mkdir(exist_ok=False)
            created.append(destination)
            report = action(job, config, loader, device, progress=progress)
            save_report(report, destination)
            reports.append(report)
        return reports
    finally:
        for directory in reversed([config.destination, *created]):
            try:
                directory.rmdir()
            except OSError:
                pass


def run_evaluation(path, *, output=None, checkpoints=None, run_id=None, progress=True):
    return run_jobs(read_config(path, output=output, checkpoints=checkpoints, run_id=run_id),
                    evaluate_checkpoint, progress=progress)


def argument_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", type=Path, help="TOML evaluation or verification settings")
    parser.add_argument("--checkpoints", type=Path, nargs="+", help="override exported checkpoint paths")
    parser.add_argument("--output", type=Path, help="override the report root")
    parser.add_argument("--run-id", help="override the report run identifier")
    parser.add_argument("--quiet", action="store_true", help="disable progress bars")
    return parser

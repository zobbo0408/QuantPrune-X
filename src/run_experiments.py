"""Run the QuantPrune-X experiment matrix from a TOML file; completed runs are skipped."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import io
import json
from pathlib import Path
import tomllib
from typing import Any, Callable

from torch import nn
from torch.utils.data import DataLoader, Subset

from checkpoints import ROOT, load_checkpoint, read_manifest
from data.cifar10 import test_loader
from experiments.pipeline import RESULT_FILE, EvaluationSettings, is_completed, record_failure, run_one
from experiments.stubs import stub_prune_model
from experiments.transforms import Operations, RunSpec, default_operations, expand_matrix, run_id
from models.registry import MODEL_NAMES

SECTIONS = {
    "experiment": {"name", "seed", "output"},
    "matrix": {"models", "orders", "bit_widths", "pruning_methods", "ratios"},
    "evaluation": {"data_dir", "device", "batch_size", "workers", "target_class", "normalize",
                   "download_data", "max_batches", "activation_quantization", "save_masks"},
    "efficiency": {"latency_batch_size", "latency_warmup", "latency_iterations"},
    "pipeline": {"pruning_backend"},
    "checkpoints": None,  # model name → path
}
BACKENDS = ("common", "stub")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    models: tuple[str, ...]
    specs: tuple[RunSpec, ...]
    settings: EvaluationSettings
    data_dir: Path
    batch_size: int
    workers: int
    target_class: int
    normalize: bool
    download_data: bool
    max_batches: int | None


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _check_keys(config: dict[str, Any]) -> None:
    unknown_sections = set(config) - set(SECTIONS)
    if unknown_sections:
        raise ValueError(f"unknown TOML sections: {sorted(unknown_sections)}")
    for section, allowed in SECTIONS.items():
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"[{section}] must be a table")
        if allowed is not None and set(values) - allowed:
            raise ValueError(f"unknown keys in [{section}]: {sorted(set(values) - allowed)}")


def read_config(path: str | Path, output: str | Path | None = None) -> ExperimentConfig:
    path = Path(path)
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    _check_keys(config)
    experiment, matrix = config.get("experiment", {}), config.get("matrix", {})
    evaluation, efficiency = config.get("evaluation", {}), config.get("efficiency", {})
    backend = config.get("pipeline", {}).get("pruning_backend", "common")
    if backend not in BACKENDS:
        raise ValueError(f"pruning_backend must be one of {BACKENDS}")
    models = tuple(matrix.get("models", MODEL_NAMES))
    specs = tuple(expand_matrix(models, matrix.get("orders", ["FP32", "Q", "P", "Q→P", "P→Q"]),
                                matrix.get("bit_widths", [8, 4]),
                                matrix.get("pruning_methods", ["global_magnitude", "layerwise_magnitude"]),
                                matrix.get("ratios", [0.3, 0.5, 0.7, 0.9]), seed=experiment.get("seed", 0)))
    name = experiment.get("name", path.stem)
    checkpoints = {model: project_path(config.get("checkpoints", {}).get(model, read_manifest(model)["local_path"]))
                   for model in models}
    settings = EvaluationSettings(
        output=project_path(output or experiment.get("output", f"artifacts/runs/{name}")),
        device=evaluation.get("device", "auto"),
        activation_quantization=evaluation.get("activation_quantization", True),
        save_masks=evaluation.get("save_masks", True), pruning_backend=backend, checkpoints=checkpoints,
        latency_batch_size=efficiency.get("latency_batch_size", 1),
        latency_warmup=efficiency.get("latency_warmup", 10),
        latency_iterations=efficiency.get("latency_iterations", 100))
    max_batches = evaluation.get("max_batches")
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError("max_batches must be a positive integer")
    return ExperimentConfig(name=name, models=models, specs=specs, settings=settings,
                            data_dir=project_path(evaluation.get("data_dir", "datasets/cifar10")),
                            batch_size=evaluation.get("batch_size", 128), workers=evaluation.get("workers", 0),
                            target_class=evaluation.get("target_class", 0),
                            normalize=evaluation.get("normalize", True),
                            download_data=evaluation.get("download_data", False), max_batches=max_batches)


def build_loader(config: ExperimentConfig) -> DataLoader:
    loader = test_loader(str(config.data_dir), download=config.download_data, batch_size=config.batch_size,
                         workers=config.workers, normalize=config.normalize, target=config.target_class)
    if config.max_batches is None:
        return loader
    limit = min(len(loader.dataset), config.max_batches * config.batch_size)
    return DataLoader(Subset(loader.dataset, range(limit)), batch_size=config.batch_size,
                      num_workers=config.workers, shuffle=False)


def build_operations(backend: str) -> Operations:
    operations = default_operations()
    return replace(operations, prune=stub_prune_model) if backend == "stub" else operations


def write_summary(output: Path, specs) -> Path:
    rows = []
    for spec in specs:
        path = output / run_id(spec) / RESULT_FILE
        if is_completed(path.parent):
            row = json.loads(path.read_text(encoding="utf-8"))
            rows.append({key: value for key, value in row.items() if not isinstance(value, (dict, list))})
    stream = io.StringIO(newline="")
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = output / "summary.csv"
    output.mkdir(parents=True, exist_ok=True)
    summary.write_text(stream.getvalue(), encoding="utf-8")
    return summary


def run_matrix(specs, settings: EvaluationSettings, *, load_fp32: Callable[[str], nn.Module], loader,
               operations: Operations, log: Callable[[str], None] = lambda _message: None) -> dict[str, int]:
    """Run every spec once; reruns skip successes and retry failures."""
    counts = {"total": len(specs), "completed": 0, "skipped": 0, "failed": 0}
    for index, spec in enumerate(specs, start=1):
        identifier = run_id(spec)
        run_dir = settings.output / identifier
        if is_completed(run_dir):
            counts["skipped"] += 1
            log(f"[{index}/{len(specs)}] skip {identifier}")
            continue
        try:
            row = run_one(spec, settings, load_fp32=load_fp32, loader=loader, operations=operations)
        except Exception as error:  # one broken run must not stop the matrix
            record_failure(run_dir, spec, error)
            counts["failed"] += 1
            log(f"[{index}/{len(specs)}] FAILED {identifier}: {type(error).__name__}: {error}")
            continue
        counts["completed"] += 1
        log(f"[{index}/{len(specs)}] done {identifier} clean={row['clean_accuracy']:.4f} asr={row['asr_all']:.4f}")
    write_summary(settings.output, specs)
    return counts


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="TOML experiment configuration")
    parser.add_argument("--output", type=Path, help="override the run artifact directory")
    parser.add_argument("--dry-run", action="store_true", help="list run ids without executing")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = read_config(args.config, args.output)
    if args.dry_run:
        for spec in config.specs:
            print(run_id(spec))
        print(f"{len(config.specs)} runs → {config.settings.output}")
        return
    checkpoints = config.settings.checkpoints
    counts = run_matrix(config.specs, config.settings, load_fp32=lambda name: load_checkpoint(name, checkpoints[name]),
                        loader=build_loader(config), operations=build_operations(config.settings.pruning_backend),
                        log=print)
    print(f"completed={counts['completed']} skipped={counts['skipped']} failed={counts['failed']} "
          f"total={counts['total']} → {config.settings.output}")
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

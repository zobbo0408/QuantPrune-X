"""Export packed INT4/INT8 checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tomllib

import torch

from checkpoints import ROOT, load_checkpoint, read_manifest, sha256
from experiments.transforms import (
    PRUNING_METHODS,
    RunSpec,
    apply_transforms,
    default_operations,
    expand_matrix,
    run_id as combination_id,
)
from models.registry import MODEL_NAMES
from quantization.checkpoint_export import Layout, save_quantized_checkpoint

DEFAULT_CONFIG = ROOT / "configs/quantization/packed_export.toml"
SECTIONS = {
    "experiment": {"seed", "output"},
    "matrix": {"models", "orders", "bit_widths", "pruning_methods", "ratios"},
    "storage": {"layouts"},
    "checkpoints": set(MODEL_NAMES),
}
CSV_FIELDS = (
    "run_id",
    "combination_id",
    "export_id",
    "checkpoint_path",
    "model",
    "processing_order",
    "bit_width",
    "pruning_method",
    "target_ratio",
    "seed",
    "layout",
    "source_checkpoint",
    "checkpoint_sha256",
    "status",
    "error",
    "fp32_file_size_bytes",
    "export_file_size_bytes",
    "size_reduction_percent",
)


@dataclass(frozen=True)
class ExportConfig:
    specs: tuple[RunSpec, ...]
    layouts: tuple[Layout, ...]
    checkpoints: dict[str, Path]
    output: Path


def read_config(path: Path, output: Path | None = None) -> ExportConfig:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if set(config) - set(SECTIONS):
        raise ValueError(
            f"unknown TOML sections: {sorted(set(config) - set(SECTIONS))}"
        )
    for section, allowed in SECTIONS.items():
        values = config.get(section, {})
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError(f"invalid or unknown settings in [{section}]")

    experiment = config.get("experiment", {})
    matrix = config.get("matrix", {})
    defaults = {
        "models": list(MODEL_NAMES),
        "orders": ["P→Q", "Q→P"],
        "bit_widths": [4, 8],
        "pruning_methods": list(PRUNING_METHODS),
        "ratios": [0.3, 0.5, 0.7, 0.9],
    }
    axes = {name: matrix.get(name, default) for name, default in defaults.items()}
    layouts = config.get("storage", {}).get("layouts", ["dense", "sparse"])
    for name, values in {**axes, "layouts": layouts}.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a nonempty array")
    for name in ("models", "orders", "pruning_methods"):
        if any(value not in defaults[name] for value in axes[name]):
            raise ValueError(f"{name} must contain only values from {defaults[name]}")
    if any(type(bits) is not int or bits not in (4, 8) for bits in axes["bit_widths"]):
        raise ValueError("bit_widths must contain only integers 4 and 8")
    if any(layout not in ("dense", "sparse") for layout in layouts):
        raise ValueError("layouts must contain only dense or sparse")
    if len(set(layouts)) != len(layouts):
        raise ValueError("layouts contains duplicate outputs")

    # Shared validation also rejects ratios/seeds and colliding combination IDs.
    specs = expand_matrix(
        axes["models"],
        axes["orders"],
        axes["bit_widths"],
        axes["pruning_methods"],
        axes["ratios"],
        seed=experiment.get("seed", 0),
    )
    overrides = config.get("checkpoints", {})
    checkpoints = {
        model: ROOT / overrides.get(model, read_manifest(model)["local_path"])
        for model in axes["models"]
    }
    destination = (
        output if output is not None else experiment.get("output", "artifacts/exports")
    )
    return ExportConfig(
        tuple(specs),
        tuple(Layout(value) for value in layouts),
        checkpoints,
        ROOT / destination,
    )


def run_directory(output: Path, name: str | None) -> Path:
    if name is None:
        name = datetime.now(timezone.utc).strftime("export-%Y%m%dT-%H%M%S-%fZ")
    if (
        not name.strip()
        or name in (".", "..")
        or any(char in name for char in ("/", "\\", "\0"))
    ):
        raise ValueError("run ID must be a single directory name, excluding . and ..")
    destination = output / name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"run directory already exists: {destination}")
    return destination


def export_id(spec: RunSpec, layout: Layout) -> str:
    return f"{combination_id(spec)}__{layout.value}"


def result_rows(
    config: ExportConfig, spec: RunSpec, destination: Path, digest: str
) -> list[dict]:
    return [
        {
            "run_id": destination.name,
            "combination_id": combination_id(spec),
            "export_id": export_id(spec, layout),
            "checkpoint_path": f"models/{export_id(spec, layout)}.pt",
            "model": spec.model,
            "processing_order": spec.order.value,
            "bit_width": spec.bits,
            "pruning_method": spec.method,
            "target_ratio": spec.ratio,
            "seed": spec.seed,
            "layout": layout.value,
            "source_checkpoint": str(config.checkpoints[spec.model]),
            "checkpoint_sha256": digest,
            "status": "failed",
            "error": "",
        }
        for layout in config.layouts
    ]


def export_combination(
    spec, model, manifest, fp32_bytes, destination, rows, operations
):
    """Transform once, then save each layout independently using the original scales."""
    torch.manual_seed(spec.seed)
    transformed = apply_transforms(spec, model, operations)
    metadata = dict(transformed.metadata["quantization"])
    metadata.pop("activation_mode", None)
    metadata.update(
        representation="packed",
        model=spec.model,
        processing_order=spec.order.value,
        pruning_method=spec.method,
        target_ratio=spec.ratio,
        seed=spec.seed,
        run_id=destination.name,
        combination_id=combination_id(spec),
        source_checkpoint=rows[0]["source_checkpoint"],
        checkpoint_sha256=rows[0]["checkpoint_sha256"],
        checkpoint_provenance=manifest,
        transform_steps=transformed.metadata["steps"],
    )
    for row in rows:
        path = destination / row["checkpoint_path"]
        try:
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"checkpoint already exists: {path}")
            save_quantized_checkpoint(
                path, transformed.model, metadata, layout=Layout(row["layout"])
            )
            size = path.stat().st_size
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        else:
            row.update(
                status="success",
                fp32_file_size_bytes=fp32_bytes,
                export_file_size_bytes=size,
                size_reduction_percent=100 * (1 - size / fp32_bytes),
            )


def export_model(
    config: ExportConfig, model_name: str, destination: Path
) -> Iterator[dict]:
    """Load once and yield results as combinations finish, including failures."""
    checkpoint = config.checkpoints[model_name]
    specs = [spec for spec in config.specs if spec.model == model_name]
    digest = ""
    try:
        manifest = read_manifest(model_name)
        digest = sha256(checkpoint)
        model = load_checkpoint(model_name, checkpoint)
        # Use the actual source checkpoint size, including serialization overhead.
        fp32_bytes = checkpoint.stat().st_size
    except Exception as error:
        for spec in specs:
            for row in result_rows(config, spec, destination, digest):
                row["error"] = f"{type(error).__name__}: {error}"
                yield row
        return

    operations = default_operations()
    for spec in specs:
        rows = result_rows(config, spec, destination, digest)
        try:
            export_combination(
                spec, model, manifest, fp32_bytes, destination, rows, operations
            )
        except Exception as error:
            for row in rows:
                row["error"] = f"{type(error).__name__}: {error}"
        yield from rows


def run_export(config: ExportConfig, destination: Path) -> dict[str, int]:
    """Collect model exports into one summary and report progress."""
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "models").mkdir()
    counts = {"success": 0, "failed": 0}
    with (destination / "summary.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for model_name in config.checkpoints:
            for row in export_model(config, model_name, destination):
                writer.writerow(row)
                stream.flush()
                counts[row["status"]] += 1
                detail = f": {row['error']}" if row["error"] else ""
                print(f"{row['status']} {row['export_id']}{detail}", flush=True)
    return counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG,
        help="TOML export configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, help="override the parent directory for export runs"
    )
    parser.add_argument(
        "--run-id", help="name of this whole export run; defaults to a UTC timestamp"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list planned outputs without creating files",
    )
    args = parser.parse_args(argv)
    try:
        config = read_config(args.config, args.output)
        destination = run_directory(config.output, args.run_id)
        total = len(config.specs) * len(config.layouts)
        if args.dry_run:
            for spec in config.specs:
                for layout in config.layouts:
                    identifier = export_id(spec, layout)
                    print(
                        f"{identifier} → {destination / 'models' / (identifier + '.pt')}"
                    )
            print(
                f"{total} checkpoints → {destination}\nsummary → {destination / 'summary.csv'}"
            )
            return
        counts = run_export(config, destination)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(
        f"success={counts['success']} failed={counts['failed']} total={total} → {destination}"
    )
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

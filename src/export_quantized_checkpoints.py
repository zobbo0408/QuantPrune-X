"""Quantize and save checkpoints from a project-relative TOML configuration."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import tomllib


from checkpoints import ROOT, load_checkpoint, read_manifest, sha256
from evaluation.runtime import code_sha256, resolve_device
from quantization.checkpoint_export import save_quantized_checkpoint
from quantization.contracts import validate_bits
from quantization.materialize_qdq import quantize


MODELS = ("resnet18", "alexnet")
QUANTIZATION_DEFAULTS = {
    "representation": "qdq", "weight_mode": "per_tensor_symmetric",
    "activation_mode": "per_tensor_asymmetric", "preserve_pruned_zeros": True,
}


@dataclass(frozen=True)
class CheckpointInput:
    model: str
    path: Path
    sha256: str
    checkpoint_source: str
    source_url: str
    weight_license: str


@dataclass(frozen=True)
class ExportConfig:
    path: Path
    output: Path
    run_id: str
    bits: tuple[int, ...]
    device: str
    inputs: tuple[CheckpointInput, ...]

    def destination(self, model, bits):
        return self.output / f"qdq_int{bits}" / model / self.run_id


def project_path(value):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("expected a nonempty path")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _path_text(path):
    path = path.resolve()
    return path.relative_to(ROOT.resolve()).as_posix() if path.is_relative_to(ROOT.resolve()) else str(path)


def _table(config, name, allowed):
    table = config.get(name, {})
    if not isinstance(table, dict) or set(table) - set(allowed):
        raise ValueError(f"invalid or unknown {name} settings")
    return table


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("checkpoint SHA-256 must be 64 hexadecimal characters")
    return value.lower()


def _source_input(name, options):
    if not isinstance(options, dict) or set(options) - {"path", "sha256", "checkpoint_source", "source_url", "weight_license"}:
        raise ValueError(f"invalid checkpoint settings: {name}")
    source = options.get("checkpoint_source", "public")
    if source == "public":
        manifest = read_manifest(name)
        digest = _digest(manifest.get("sha256"))
        if "sha256" in options and _digest(options["sha256"]) != digest:
            raise ValueError(f"public checkpoint SHA-256 differs from manifest: {name}")
        path = project_path(options.get("path", manifest["local_path"]))
        if not path.is_file() and "path" not in options:
            legacy = ROOT / "artifacts/checkpoints" / name / "qcb.pth"
            if legacy.is_file():
                path = legacy
        url, license = manifest.get("source_url"), manifest.get("license")
        for key, value in (("source_url", url), ("weight_license", license)):
            if key in options and options[key] != value:
                raise ValueError(f"public {key} must match the manifest: {name}")
    elif source == "self_trained":
        if not {"path", "sha256", "source_url", "weight_license"} <= options.keys():
            raise ValueError(f"self_trained input requires path, SHA-256, source URL and weight license: {name}")
        path, digest = project_path(options["path"]), _digest(options["sha256"])
        url, license = options["source_url"], options["weight_license"]
    else:
        raise ValueError(f"unknown checkpoint_source: {source}")
    if not all(isinstance(value, str) and value for value in (url, license)):
        raise ValueError(f"checkpoint source URL and weight license are required: {name}")
    return CheckpointInput(name, path, digest, source, url, license)


def read_export(path, output=None, *, run_id=None, device=None):
    path = project_path(path)
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if set(config) - {"models", "run_id", "output", "device", "quantization", "checkpoints"}:
        raise ValueError("unknown top-level export settings")
    names = config.get("models", list(MODELS))
    if (not isinstance(names, list) or not names or any(name not in MODELS for name in names)
            or len(names) != len(set(names))):
        raise ValueError("models must list distinct resnet18/alexnet names")
    quantization = _table(config, "quantization", {*QUANTIZATION_DEFAULTS, "bit_widths"})
    for key, value in QUANTIZATION_DEFAULTS.items():
        if key in quantization and (type(quantization[key]) is not type(value) or quantization[key] != value):
            raise ValueError(f"unsupported quantization setting: {key}")
    bits = quantization.get("bit_widths", [4, 8])
    if not isinstance(bits, list) or not bits:
        raise ValueError("bit_widths must be a nonempty list")
    for bit in bits:
        validate_bits(bit)
    if len(bits) != len(set(bits)):
        raise ValueError("bit_widths must be distinct")
    run_id = run_id if run_id is not None else config.get("run_id", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
        raise ValueError("run_id must be a single path component starting with a letter or digit")
    device = device if device is not None else config.get("device", "auto")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a nonempty string")
    destination = project_path(output if output is not None else config.get("output", "artifacts/checkpoints"))
    checkpoints = _table(config, "checkpoints", MODELS)
    inputs = tuple(_source_input(name, checkpoints.get(name, {})) for name in names)
    for item in inputs:
        if not item.path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {item.path}")
    for item in inputs:
        if sha256(item.path) != item.sha256:
            raise ValueError(f"checkpoint SHA-256 mismatch: {item.path}")
    return ExportConfig(path, destination, run_id, tuple(bits), device, inputs)


def _export_one(config, item, bits, device):
    output = config.destination(item.model, bits) / "model.pt"
    model = load_checkpoint(item.model, item.path, expected_hash=item.sha256).to(device)
    model, metadata = quantize(model, bits, inplace=True)
    metadata.update(model_name=item.model, source_checkpoint=_path_text(item.path), source_sha256=item.sha256,
                    checkpoint_source=item.checkpoint_source, source_url=item.source_url, weight_license=item.weight_license,
                    export_device=str(device), export_config_sha256=sha256(config.path), code_sha256=code_sha256())
    save_quantized_checkpoint(output, model.cpu(), metadata)
    return {"model": item.model, "bit_width": bits, "output_checkpoint": _path_text(output)}


def run_export(path, *, output=None, run_id=None, device=None):
    config = read_export(path, output, run_id=run_id, device=device)
    device = resolve_device(config.device)
    destinations = [config.destination(item.model, bits) for item in config.inputs for bits in config.bits]
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(f"output run already exists: {destination}")
    reserved = []
    try:
        for destination in destinations:
            destination.mkdir(parents=True, exist_ok=False)
            reserved.append(destination)
        for item in config.inputs:
            load_checkpoint(item.model, item.path, expected_hash=item.sha256)
        return [_export_one(config, item, bits, device)
                for item in config.inputs for bits in config.bits]
    finally:
        for destination in reserved:
            try:
                destination.rmdir()
            except OSError:
                pass


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and save checkpoints")
    parser.add_argument("config", help="TOML export configuration")
    parser.add_argument("--output", type=Path, help="override the TOML output checkpoint root")
    parser.add_argument("--run-id", help="override the TOML run identifier")
    parser.add_argument("--device", help="override the conversion device")
    parser.add_argument("--quiet", action="store_true", help="suppress exported checkpoint paths")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        reports = run_export(args.config, output=args.output, run_id=args.run_id, device=args.device)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AssertionError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    if not args.quiet:
        for report in reports:
            print(f"{report['model']} INT{report['bit_width']}: exported {report['output_checkpoint']}")


if __name__ == "__main__":
    main()

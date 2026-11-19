"""ResNet-18 defense search (slide item 2-①): prune ONLY layer2 (layer-wise) and quantize.

Stage sensitivity showed layer-wise 70% pruning of layer2 alone keeps INT4 ASR near the
FP32 level. This sweeps the ratio and both processing orders for INT8 and INT4.
Run: PYTHONPATH=src uv run --locked python analysis/resnet18/layer2_defense.py
"""
from __future__ import annotations

import copy
import csv
import time

from torch import nn

from checkpoints import ROOT, load_checkpoint, read_manifest
from data.cifar10 import test_loader
from evaluation.metrics import evaluate_model
from evaluation.runtime import resolve_device, seed_everything
from experiments.transforms import assert_masked_zero
from pruning.common import prune_model
from pruning.masks import apply_masks
from quantization.layers import configure_model
from quantization.materialize_qdq import quantize

RATIOS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TARGET = "layer2"
OUTPUT = ROOT / "artifacts/analysis/resnet18/layer2_defense.csv"


def run(model, bits, order, ratio, excluded, loader, device):
    model = copy.deepcopy(model)
    masks = None
    if order == "Q→P":
        model, _ = quantize(model, bits)
    if order in ("P", "Q→P", "P→Q"):
        model, masks, _ = prune_model(model, "layerwise_magnitude", ratio, exclude_layers=excluded)
    if order == "P→Q":
        model, _ = quantize(model, bits, mask=masks)
        apply_masks(model, masks)
    if order == "Q":
        model, _ = quantize(model, bits)
    if masks is not None:
        assert_masked_zero(model, masks)
    if bits is not None:
        configure_model(model, bits, weight=False, activation=True)
    return evaluate_model(model.to(device), loader, device, progress=False)


def main() -> None:
    seed_everything(0)
    device = resolve_device("cuda")
    loader = test_loader(str(ROOT / "datasets/cifar10"), batch_size=128, workers=0, normalize=True, target=0)
    model = load_checkpoint("resnet18", ROOT / read_manifest("resnet18")["local_path"])
    excluded = [name for name, module in model.named_modules()
                if isinstance(module, (nn.Conv2d, nn.Linear)) and not name.startswith(TARGET)]
    plan = [(None, "FP32", 0.0), (8, "Q", 0.0), (4, "Q", 0.0)]
    plan += [(bits, order, ratio) for ratio in RATIOS for bits, order in
             ((None, "P"), (8, "Q→P"), (8, "P→Q"), (4, "Q→P"), (4, "P→Q"))]
    rows, started = [], time.perf_counter()
    for bits, order, ratio in plan:
        metrics = run(model, bits, order, ratio, excluded, loader, device)
        row = {"target": TARGET, "order": order, "bits": bits or 32, "ratio": ratio,
               "clean_accuracy": metrics["clean_accuracy"], "asr_all": metrics["asr_all"],
               "asr_non_target": metrics["asr_non_target"]}
        rows.append(row)
        print(f"{order:<5} {('INT' + str(bits)) if bits else 'FP32':<5} r={ratio:.1f}  "
              f"CA={row['clean_accuracy'] * 100:6.2f} ASR={row['asr_all'] * 100:6.2f}", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} evaluations in {time.perf_counter() - started:.0f}s → {OUTPUT}")


if __name__ == "__main__":
    main()

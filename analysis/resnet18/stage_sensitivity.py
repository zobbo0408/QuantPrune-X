"""ResNet-18 stage sensitivity (slide item 1-③): which stage's pruning moves CA and ASR.

Groups: stem (conv1), layer1..layer4 (incl. shortcut convs), fc (linear).
  A. global-70% mask applied to ONE group only          → is that group sufficient for the ASR jump?
  B. global-70% mask applied to all groups EXCEPT one    → is that group necessary?
  C. layer-wise pruning of ONE group at 30/50/70%       → per-stage CA/ASR sensitivity
  D. C at 70% followed by INT4 Q-DQ (P→Q4)              → which stage drives the INT4 ASR collapse
Full CIFAR-10 test set; evaluation uses the same functions as the experiment pipeline.
Run: PYTHONPATH=src uv run --locked python analysis/resnet18/stage_sensitivity.py
"""
from __future__ import annotations

import copy
import csv
import time

import torch
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

GROUPS = ("stem", "layer1", "layer2", "layer3", "layer4", "fc")
OUTPUT = ROOT / "artifacts/analysis/resnet18/stage_sensitivity.csv"


def group_of(module_name: str) -> str:
    if module_name.startswith("layer"):
        return module_name.split(".")[0]
    return "stem" if module_name == "conv1" else "fc"


def evaluate(model: nn.Module, masks, loader, device, bits=None) -> dict:
    model = copy.deepcopy(model)
    if bits is not None:
        model, _ = quantize(model, bits, mask=masks)
        if masks is not None:
            apply_masks(model, masks)
        configure_model(model, bits, weight=False, activation=True)
    if masks is not None:
        assert_masked_zero(model, masks)
    metrics = evaluate_model(model.to(device), loader, device, progress=False)
    return {"clean_accuracy": metrics["clean_accuracy"], "asr_all": metrics["asr_all"],
            "asr_non_target": metrics["asr_non_target"]}


def restrict(model: nn.Module, full_masks: dict, keep_groups: set[str]):
    """Apply only the parts of ``full_masks`` that belong to ``keep_groups``."""
    masks = {name: mask for name, mask in full_masks.items() if group_of(name.rsplit(".", 1)[0]) in keep_groups}
    pruned = copy.deepcopy(model)
    apply_masks(pruned, masks)
    return pruned, masks


def main() -> None:
    seed_everything(0)
    device = resolve_device("cuda")
    loader = test_loader(str(ROOT / "datasets/cifar10"), batch_size=128, workers=0, normalize=True, target=0)
    model = load_checkpoint("resnet18", ROOT / read_manifest("resnet18")["local_path"])
    module_names = [name for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))]
    rows = []

    def record(experiment, group, ratio, order, masks, metrics):
        removed = sum(int((~m).sum()) for m in masks.values()) if masks else 0
        rows.append({"experiment": experiment, "group": group, "ratio": ratio, "order": order,
                     "removed_weights": removed, **metrics})
        print(f"{experiment:<26}{group:<8}{ratio:>5} {order:<5} removed={removed:>9,} "
              f"CA={metrics['clean_accuracy'] * 100:6.2f} ASR={metrics['asr_all'] * 100:6.2f}", flush=True)

    started = time.perf_counter()
    record("reference", "-", 0.0, "FP32", None, evaluate(model, None, loader, device))
    _, global_masks, _ = prune_model(model, "global_magnitude", 0.7)
    pruned, masks = restrict(model, global_masks, set(GROUPS))
    record("global70_all", "all", 0.7, "P", masks, evaluate(pruned, masks, loader, device))
    for group in GROUPS:
        pruned, masks = restrict(model, global_masks, {group})
        record("A_global70_only", group, 0.7, "P", masks, evaluate(pruned, masks, loader, device))
    for group in GROUPS:
        pruned, masks = restrict(model, global_masks, set(GROUPS) - {group})
        record("B_global70_except", group, 0.7, "P", masks, evaluate(pruned, masks, loader, device))
    for group in GROUPS:
        excluded = [name for name in module_names if group_of(name) != group]
        for ratio in (0.3, 0.5, 0.7):
            pruned, masks, _ = prune_model(model, "layerwise_magnitude", ratio, exclude_layers=excluded)
            record("C_layerwise_only", group, ratio, "P", masks, evaluate(pruned, masks, loader, device))
            if ratio == 0.7:
                record("D_layerwise_only_int4", group, ratio, "P→Q4", masks,
                       evaluate(pruned, masks, loader, device, bits=4))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} evaluations in {time.perf_counter() - started:.0f}s → {OUTPUT}")


if __name__ == "__main__":
    main()

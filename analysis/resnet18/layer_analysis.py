"""ResNet-18 layer-level analysis (slide items 1-② and 1-④).

For every Conv2d/Linear weight of the public QCB ResNet-18:
  * fraction removed by global / layer-wise magnitude pruning at each ratio,
  * fraction that INT8 / INT4 Q-DQ maps to exactly zero,
  * overlap between pruned weights and quantization-zeroed weights.

Writes CSVs under artifacts/analysis/resnet18/ and prints stage-level summaries.
Run: PYTHONPATH=src uv run --locked python analysis/resnet18/layer_analysis.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn

from checkpoints import ROOT, load_checkpoint, read_manifest
from pruning.common import prune_model
from quantization.materialize_qdq import quantize

RATIOS = (0.3, 0.5, 0.7, 0.9)
METHODS = ("global_magnitude", "layerwise_magnitude")
BITS = (8, 4)
OUTPUT = ROOT / "artifacts/analysis/resnet18"


def stage_of(name: str) -> str:
    if name.startswith("layer"):
        stage = name.split(".")[0]
        return f"{stage}.shortcut" if "shortcut" in name else stage
    return "stem" if name.startswith("conv1") else "fc"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    model = load_checkpoint("resnet18", ROOT / read_manifest("resnet18")["local_path"])
    weights = {f"{name}.weight": module.weight.detach().clone() for name, module in model.named_modules()
               if isinstance(module, (nn.Conv2d, nn.Linear))}
    total = sum(w.numel() for w in weights.values())

    quant_zero = {bits: {name: (param == 0) for name, param in quantize(model, bits)[0].named_parameters()
                         if name in weights} for bits in BITS}
    removed = {(method, ratio): {name: ~mask for name, mask in prune_model(model, method, ratio)[1].items()}
               for method in METHODS for ratio in RATIOS}

    layer_rows = []
    for name, weight in weights.items():
        n = weight.numel()
        row = {"layer": name, "stage": stage_of(name), "numel": n, "share_of_model": n / total,
               "abs_max": float(weight.abs().max()), "abs_median": float(weight.abs().median()),
               "dead_lt_1e-8": float((weight.abs() < 1e-8).float().mean())}
        for bits in BITS:
            row[f"int{bits}_zero"] = float(quant_zero[bits][name].float().mean())
        for (method, ratio), masks in removed.items():
            row[f"{method.split('_')[0]}_{int(ratio * 100)}_removed"] = float(masks[name].float().mean())
        layer_rows.append(row)
    write_csv(OUTPUT / "layer_removal.csv", layer_rows)

    overlap_rows = []
    for (method, ratio), masks in removed.items():
        for bits in BITS:
            pruned = torch.cat([masks[name].flatten() for name in weights])
            zeroed = torch.cat([quant_zero[bits][name].flatten() for name in weights])
            both = int((pruned & zeroed).sum())
            overlap_rows.append({
                "method": method, "ratio": ratio, "bits": bits,
                "pruned": int(pruned.sum()), "quant_zero": int(zeroed.sum()), "both": both,
                "pruned_that_quant_also_zeroes": both / max(int(pruned.sum()), 1),
                "quant_zeroes_covered_by_pruning": both / max(int(zeroed.sum()), 1),
                "jaccard": both / max(int((pruned | zeroed).sum()), 1),
            })
    write_csv(OUTPUT / "prune_quant_overlap.csv", overlap_rows)
    report(layer_rows, overlap_rows)


def report(layer_rows: list[dict], overlap_rows: list[dict]) -> None:
    stages = list(dict.fromkeys(row["stage"] for row in layer_rows))
    columns = ["dead_lt_1e-8", "int8_zero", "int4_zero"] + [f"{m}_{r}_removed" for m in ("global", "layerwise")
                                                             for r in (30, 50, 70, 90)]
    print("== stage-level fraction (weighted by numel)")
    print(f"{'stage':<17}{'share':>6} " + " ".join(f"{c.replace('_removed', '').replace('_zero', '0'):>12}" for c in columns))
    for stage in stages:
        rows = [row for row in layer_rows if row["stage"] == stage]
        n = sum(row["numel"] for row in rows)
        share = sum(row["share_of_model"] for row in rows)
        values = [sum(row[c] * row["numel"] for row in rows) / n for c in columns]
        print(f"{stage:<17}{share * 100:5.1f}% " + " ".join(f"{v * 100:11.1f}%" for v in values))
    print("\n== overlap: pruned weights vs weights Q-DQ maps to zero")
    print(f"{'method':<22}{'ratio':>6}{'bits':>5}{'pruned∩Q0/pruned':>18}{'pruned∩Q0/Q0':>15}{'jaccard':>9}")
    for row in overlap_rows:
        print(f"{row['method']:<22}{row['ratio']:>6}{row['bits']:>5}{row['pruned_that_quant_also_zeroes'] * 100:17.1f}%"
              f"{row['quant_zeroes_covered_by_pruning'] * 100:14.1f}%{row['jaccard']:9.3f}")


if __name__ == "__main__":
    main()

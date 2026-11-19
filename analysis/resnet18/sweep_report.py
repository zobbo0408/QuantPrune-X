"""Summarize the ResNet-18 ratio sweep (slide items 1-①, 2-①, 2-②).

* Trajectory table: CA/ASR per ratio for every order and method.
* Defense candidates: conditions whose CA stays within MAX_CA_DROP of the same-bit Q baseline,
  ranked by ASR (lower is better).
Run: python3 analysis/resnet18/sweep_report.py  (stdlib only)
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "artifacts/runs/resnet18-ratio-sweep/summary.csv"
OUTPUT = ROOT / "artifacts/analysis/resnet18"
MAX_CA_DROP = 3.0  # %p versus the quantized-only baseline with the same bit width
COLUMNS = (("P", None), ("Q→P", 8), ("P→Q", 8), ("Q→P", 4), ("P→Q", 4))


def pct(row: dict, key: str) -> float:
    return float(row[key]) * 100


def main() -> None:
    rows = list(csv.DictReader(SUMMARY.open(encoding="utf-8")))
    baseline = {row["bit_width"] or "32": row for row in rows if row["processing_order"] in ("FP32", "Q")}
    cell = {(row["pruning_method"], row["processing_order"], row["bit_width"] or "32", row["target_ratio"]): row
            for row in rows if row["pruning_method"] != "none"}
    ratios = sorted({row["target_ratio"] for row in rows if row["target_ratio"]}, key=float)
    for bits, label in (("32", "FP32"), ("8", "INT8"), ("4", "INT4")):
        print(f"{label:<5} baseline  CA {pct(baseline[bits], 'clean_accuracy'):5.1f}  ASR {pct(baseline[bits], 'asr_all'):5.1f}")

    for method in ("global_magnitude", "layerwise_magnitude"):
        print(f"\n== {method}  (CA / ASR %)")
        print("ratio | " + " | ".join(f"{o}{'' if b is None else b:<4}" .ljust(11) for o, b in COLUMNS))
        for ratio in ratios:
            values = []
            for order, bits in COLUMNS:
                row = cell[(method, order, str(bits or 32), ratio)]
                values.append(f"{pct(row, 'clean_accuracy'):5.1f}/{pct(row, 'asr_all'):5.1f}")
            print(f"{float(ratio) * 100:4.0f}% | " + " | ".join(values))

    candidates = []
    for (method, order, bits, ratio), row in cell.items():
        if bits == "32":
            continue
        drop = pct(baseline[bits], "clean_accuracy") - pct(row, "clean_accuracy")
        if drop <= MAX_CA_DROP:
            candidates.append({"bits": bits, "method": method, "order": order, "ratio": float(ratio),
                               "clean_accuracy": pct(row, "clean_accuracy"), "ca_drop_vs_q": drop,
                               "asr_all": pct(row, "asr_all"), "asr_non_target": pct(row, "asr_non_target")})
    candidates.sort(key=lambda c: (c["bits"], c["asr_all"]))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "defense_candidates_full_model.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    print(f"\n== whole-model defense candidates (CA drop ≤ {MAX_CA_DROP}%p vs Q), lowest ASR first")
    for bits in ("8", "4"):
        print(f"-- INT{bits}")
        for c in [c for c in candidates if c["bits"] == bits][:5]:
            print(f"   {c['method']:<20} {c['order']:<4} {c['ratio'] * 100:3.0f}%  CA {c['clean_accuracy']:5.1f} "
                  f"(−{c['ca_drop_vs_q']:.1f})  ASR {c['asr_all']:5.1f}")


if __name__ == "__main__":
    main()

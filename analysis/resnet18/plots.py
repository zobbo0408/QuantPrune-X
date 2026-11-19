"""Figures for the ResNet-18 analysis report.

Reads the CSVs written by the other analysis scripts and the ratio sweep; writes PNGs to
artifacts/analysis/resnet18/figures/. matplotlib is not a project dependency, so run it in
a throwaway environment without touching uv.lock:
  uv run --no-project --python 3.12 --with matplotlib==3.9.2 python analysis/resnet18/plots.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "artifacts/analysis/resnet18"
SWEEP = ROOT / "artifacts/runs/resnet18-ratio-sweep/summary.csv"
FIGURES = DATA / "figures"

# Reference categorical palette (validated: adjacent CVD ΔE ≥ 9.1, light surface).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED, GRID, AXIS, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#c3c2b7"
MARKERS = {BLUE: "o", ORANGE: "s", AQUA: "^", YELLOW: "D"}
STAGES = ("stem", "layer1", "layer2", "layer3", "layer4", "fc")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
    "font.family": "DejaVu Sans", "text.color": INK,
})


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def line(ax, xs, ys, color, label, **kwargs):
    ax.plot(xs, ys, color=color, linewidth=2, marker=MARKERS[color], markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3, **kwargs)


def reference(ax, y, text, *, below=False):
    ax.axhline(y, color=MUTED, linewidth=1, zorder=1)
    ax.annotate(text, xy=(1, y), xycoords=("axes fraction", "data"), xytext=(-2, -3 if below else 3),
                textcoords="offset points", ha="right", va="top" if below else "bottom", fontsize=8, color=MUTED)


def note(ax, x, y, text, **kwargs):
    ax.text(x, y, text, **{"fontsize": 8.5, "color": INK2, **kwargs})


def save(fig, name: str, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.945, subtitle, ha="left", va="top", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)
    print("wrote", FIGURES / name)


def sweep_table():
    rows = read(SWEEP)
    base = {(r["bit_width"] or "32"): r for r in rows if r["processing_order"] in ("FP32", "Q")}
    cells = defaultdict(list)
    for r in rows:
        if r["pruning_method"] != "none":
            cells[(r["pruning_method"], r["processing_order"], r["bit_width"] or "32")].append(r)
    for key in cells:
        cells[key].sort(key=lambda r: float(r["target_ratio"]))
    return base, cells


def series(rows, key):
    return [float(r["target_ratio"]) * 100 for r in rows], [float(r[key]) * 100 for r in rows]


def fig1_prune_only(base, cells):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, label in ((axes[0], "clean_accuracy", "Clean accuracy (%)"), (axes[1], "asr_all", "ASR (%)")):
        for method, color, name in (("global_magnitude", BLUE, "Global"), ("layerwise_magnitude", ORANGE, "Layer-wise")):
            line(ax, *series(cells[(method, "P", "32")], key), color, name)
        reference(ax, float(base["32"][key]) * 100, f"FP32 {float(base['32'][key]) * 100:.1f}",
                  below=key == "asr_all")
        ax.set(xlabel="Pruning ratio (%)", ylabel=label, ylim=(0, 102), xlim=(0, 100))
        ax.legend(loc="lower left" if key == "clean_accuracy" else "upper left")
    note(axes[1], 64, 60, "Global 65→70%:\nASR 27.5 → 95.5", ha="right", va="center")
    note(axes[1], 93.5, 18, "95%: 14.3", ha="right", va="bottom")
    save(fig, "fig1_prune_only_trajectory.png", "Pruning alone switches the backdoor on (FP32, no quantization)",
         "ResNet-18 QCB, CIFAR-10 test set (10,000). Global pruning keeps accuracy to 95% but ASR jumps between 65% and 70%.")


def fig2_3_quantized(base, cells, key, name, title, subtitle):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for row, (method, method_name) in enumerate((("global_magnitude", "Global"), ("layerwise_magnitude", "Layer-wise"))):
        for col, bits in enumerate(("8", "4")):
            ax = axes[row][col]
            line(ax, *series(cells[(method, "Q→P", bits)], key), BLUE, "Q→P (quantize, then prune)")
            line(ax, *series(cells[(method, "P→Q", bits)], key), ORANGE, "P→Q (prune, then quantize)")
            reference(ax, float(base[bits][key]) * 100, f"INT{bits} only {float(base[bits][key]) * 100:.1f}")
            ax.set_title(f"{method_name} · INT{bits}", loc="left")
            ax.set(ylim=(0, 102), xlim=(0, 100))
            if row == 1:
                ax.set_xlabel("Pruning ratio (%)")
            if col == 0:
                ax.set_ylabel("ASR (%)" if key == "asr_all" else "Clean accuracy (%)")
    axes[0][0].legend(loc="lower left")
    save(fig, name, title, subtitle)


def fig4_tradeoff(base, cells):
    defense = {(r["order"], r["bits"]): [] for r in read(DATA / "layer2_defense.csv")}
    for r in read(DATA / "layer2_defense.csv"):
        defense[(r["order"], r["bits"])].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, bits in zip(axes, ("8", "4")):
        q_ca, q_asr = float(base[bits]["clean_accuracy"]) * 100, float(base[bits]["asr_all"]) * 100
        ax.axvspan(q_ca - 3, 100, ymax=20 / 105, color=AQUA, alpha=0.12, linewidth=0, zorder=0)
        note(ax, 41, 97, f"shaded = target zone\n(CA drop ≤ 3%p vs INT{bits} only, ASR ≤ 20%)", fontsize=8, va="top")
        groups = ((cells[("global_magnitude", "P→Q", bits)], BLUE, "Global (whole model) P→Q"),
                  (cells[("layerwise_magnitude", "P→Q", bits)], ORANGE, "Layer-wise (whole model) P→Q"),
                  (sorted(defense[("P→Q", bits)], key=lambda r: float(r["ratio"])), AQUA, "Layer-wise, layer2 only, P→Q"))
        for rows, color, label in groups:
            line(ax, [float(r["clean_accuracy"]) * 100 for r in rows], [float(r["asr_all"]) * 100 for r in rows],
                 color, label)
        ax.plot([q_ca], [q_asr], marker="*", markersize=13, color=INK, zorder=4, linestyle="none", label=f"INT{bits} only")
        best = {"8": [(91.6, 14.3, "Global 95%", (76, 31)), (89.3, 13.6, "layer2 80%", (76, 19))],
                "4": [(89.7, 26.7, "Global 95%", (73, 45)), (88.7, 12.3, "layer2 70%", (73, 33))]}[bits]
        for x, y, text, where in best:
            ax.annotate(text, xy=(x, y), xytext=where, fontsize=8.5, color=INK2,
                        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
        ax.set(title=f"INT{bits}", xlabel="Clean accuracy (%)", xlim=(40, 95), ylim=(0, 105))
    axes[0].set_ylabel("ASR (%)")
    axes[0].legend(loc="center left")
    save(fig, "fig4_asr_ca_tradeoff.png", "Where pruning defends the quantized model: ASR vs accuracy",
         "Each line walks the pruning ratio (5% steps; layer2-only 30–90%). Lower-right = accurate and backdoor suppressed.")


def stage_means(rows, column):
    totals, weights = defaultdict(float), defaultdict(int)
    for r in rows:
        stage = r["stage"].split(".")[0]
        totals[stage] += float(r[column]) * int(r["numel"])
        weights[stage] += int(r["numel"])
    return [totals[s] / weights[s] * 100 for s in STAGES], weights


def fig5_stage_removal():
    rows = read(DATA / "layer_removal.csv")
    columns = (("global_70_removed", BLUE, "Global 70% pruning"), ("global_90_removed", ORANGE, "Global 90% pruning"),
               ("int8_zero", AQUA, "INT8 Q-DQ → exactly 0"), ("int4_zero", YELLOW, "INT4 Q-DQ → exactly 0"))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.2
    _, weights = stage_means(rows, columns[0][0])
    total = sum(weights.values())
    for index, (column, color, label) in enumerate(columns):
        values, _ = stage_means(rows, column)
        xs = [i + (index - 1.5) * width for i in range(len(STAGES))]
        ax.bar(xs, values, width=width - 0.03, color=color, label=label, zorder=3)
    ax.set_xticks(range(len(STAGES)), [f"{s}\n({weights[s] / total * 100:.1f}% of weights)" for s in STAGES])
    ax.set(ylabel="Weights removed / zeroed (%)", ylim=(0, 100))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2)
    save(fig, "fig5_stage_removal.png", "Global pruning removes the same weights INT8 rounds to zero",
         "Share of each stage's Conv/Linear weights removed (layer-wise always removes exactly the target ratio per stage).\n"
         "Global 70% vs INT8 zero set over the whole model: Jaccard 0.938.")


def fig6_sensitivity():
    rows = read(DATA / "stage_sensitivity.csv")
    pick = lambda experiment, ratio="0.7": {r["group"]: float(r["asr_all"]) * 100 for r in rows
                                            if r["experiment"] == experiment and r["ratio"] == ratio}
    panels = ((pick("A_global70_only"), "layer3", "Global-70% mask on ONE stage only", 11.27, "FP32 11.3"),
              (pick("B_global70_except"), "layer3", "Global-70% mask on all stages EXCEPT one", 95.50, "all stages 95.5"),
              (pick("D_layerwise_only_int4"), "layer2", "Layer-wise 70% on ONE stage, then INT4", 99.99, "INT4 only 100"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, (values, focus, title, ref, ref_text) in zip(axes, panels):
        colors = [BLUE if s == focus else GRAY for s in STAGES]
        bars = ax.bar(STAGES, [values[s] for s in STAGES], color=colors, width=0.62, zorder=3)
        reference(ax, ref, ref_text)
        ax.bar_label(bars, labels=[f"{values[s]:.1f}" if s == focus else "" for s in STAGES],
                     padding=2, fontsize=8.5, color=INK)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set(ylim=(0, 110))
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelrotation=0)
    axes[0].set_ylabel("ASR (%)")
    save(fig, "fig6_stage_sensitivity.png", "layer3 holds the switch; pruning layer2 breaks the INT4 trigger",
         "Full test set. Left/middle: layer3 is sufficient and necessary for the pruning-induced ASR jump. Right: only layer2 keeps INT4 ASR low.")


def fig7_layer2_defense():
    rows = read(DATA / "layer2_defense.csv")
    base = {r["bits"]: r for r in rows if r["order"] in ("FP32", "Q")}
    groups = ((("Q→P", "8"), BLUE, "Q→P INT8"), (("P→Q", "8"), ORANGE, "P→Q INT8"),
              (("Q→P", "4"), AQUA, "Q→P INT4"), (("P→Q", "4"), YELLOW, "P→Q INT4"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, label in ((axes[0], "asr_all", "ASR (%)"), (axes[1], "clean_accuracy", "Clean accuracy (%)")):
        for (order, bits), color, name in groups:
            data = sorted((r for r in rows if r["order"] == order and r["bits"] == bits), key=lambda r: float(r["ratio"]))
            line(ax, [float(r["ratio"]) * 100 for r in data], [float(r[key]) * 100 for r in data], color, name)
        ax.set(xlabel="Layer-wise pruning ratio inside layer2 (%)", ylabel=label, xlim=(25, 95),
               ylim=(0, 102) if key == "asr_all" else (45, 95))
    for bits, below in (("8", False), ("4", True)):
        reference(axes[1], float(base[bits]["clean_accuracy"]) * 100,
                  f"INT{bits} only {float(base[bits]['clean_accuracy']) * 100:.1f}", below=below)
    axes[0].annotate("70%, INT4: P→Q 12.3 vs Q→P 64.0", xy=(70, 12.3), xytext=(32, 45), fontsize=8.5, color=INK2,
                     arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    axes[0].legend(loc="lower left")
    save(fig, "fig7_layer2_defense.png", "Pruning only layer2 (≈4% of weights) disarms the quantized backdoor",
         "Layer-wise pruning restricted to layer2, then INT8/INT4 Q-DQ in both orders. Full test set.")


def main() -> None:
    base, cells = sweep_table()
    fig1_prune_only(base, cells)
    fig2_3_quantized(base, cells, "asr_all", "fig2_quantized_asr.png", "Quantized paths: ASR by pruning ratio",
                     "Global pruning ≤ 90% never lowers the quantized ASR; layer-wise lowers it only from ~65%, mostly for INT4.")
    fig2_3_quantized(base, cells, "clean_accuracy", "fig3_quantized_ca.png", "Quantized paths: clean accuracy by pruning ratio",
                     "Global holds accuracy through 95%; layer-wise collapses past ~75%. Order matters most for INT4 (Q→P drops faster).")
    fig4_tradeoff(base, cells)
    fig5_stage_removal()
    fig6_sensitivity()
    fig7_layer2_defense()


if __name__ == "__main__":
    main()

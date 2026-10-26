"""Figures for E1 (tuned baselines on Task A). Plotting is kept separate from training.

Usage:  python code/plot_e1.py
Reads results/e1_baselines/*.json; writes report/figures/e1_*.pdf and .png.

Figure 1: best tail excess risk vs batch size, each optimiser tuned per batch size.
Figure 2: excess-risk curves at the tuned learning rate, B = 16 and B = 1024 (small multiples).
Colours: validated categorical slots 1-4 (dataviz reference palette, light mode). Aqua and
yellow are below 3:1 contrast, so every line also gets a direct text label.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "e1_baselines"
FIGURES = REPO / "report" / "figures"
OPTIMISERS = [  # fixed order = fixed colour slot
    ("adamw", "AdamW", "#2a78d6"),
    ("sgd", "SGD + Nesterov", "#eb6834"),
    ("muon-ns5", "Muon (NS5)", "#1baf7a"),
    ("muon-exact", "Muon (exact SVD)", "#eda100"),
]
LINESTYLE = {"muon-exact": (0, (4, 2))}  # dashed: the two Muon lines nearly coincide


def spread_labels(ys: list[float], min_gap: float) -> list[float]:
    """Nudge label positions (in log10 units) apart so no two are closer than min_gap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = list(ys)
    for a, b in zip(order, order[1:]):
        if placed[b] - placed[a] < min_gap:
            placed[b] = placed[a] + min_gap
    return placed
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})


def load():
    runs = defaultdict(list)  # (label, B, lr) -> records
    for path in sorted(RESULTS.glob("*.json")):
        record = json.loads(path.read_text())
        cfg = record["config"]
        runs[(cfg["name"].split("_")[0], cfg["batch_size"], cfg["optimizer"]["lr"])].append(record)
    return runs


def tuned(runs):
    """(label, B) -> (best lr, list of records at that lr), by mean tail excess risk over seeds."""
    best = {}
    for (label, B, lr), records in runs.items():
        if any(r["summary"]["diverged"] for r in records):
            continue
        score = statistics.fmean(r["summary"]["tail_mean_excess_risk"] for r in records)
        if (label, B) not in best or score < best[(label, B)][0]:
            best[(label, B)] = (score, lr, records)
    return {k: (lr, recs) for k, (score, lr, recs) in best.items()}


def figure_vs_batch(best):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ends = []
    for label, name, colour in OPTIMISERS:
        Bs = sorted(B for (lab, B) in best if lab == label)
        if not Bs:
            continue
        values = [[r["summary"]["tail_mean_excess_risk"] for r in best[(label, B)][1]] for B in Bs]
        means = [statistics.fmean(v) for v in values]
        ax.plot(Bs, means, color=colour, linewidth=1.5, marker="o", markersize=5, label=name,
                linestyle=LINESTYLE.get(label, "-"), markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        for B, v in zip(Bs, values):  # each seed as a small dot: n = 2, so no error bar
            ax.scatter([B] * len(v), v, s=6, color=colour, alpha=0.6, zorder=2, linewidths=0)
        ends.append((name, Bs[-1], means[-1]))
    import math
    label_y = spread_labels([math.log10(y) for _, _, y in ends], min_gap=0.07)
    for (name, B, _), y in zip(ends, label_y):
        ax.annotate(name, (B, 10 ** y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([16, 64, 256, 1024], labels=["16", "64", "256", "1024"])
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Excess risk (mean of last 10% of evaluations)")
    ax.set_title("Task A, each optimiser tuned per batch size\n(6000 steps, constant learning rate, 2 seeds)",
                 fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_xlim(12, 3000)
    return fig


def figure_curves(best):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, B in zip(axes, (16, 1024)):
        for label, name, colour in OPTIMISERS:
            if (label, B) not in best:
                continue
            lr, records = best[(label, B)]
            steps = [h["step"] for h in records[0]["history"]]
            curve = [statistics.fmean(r["history"][i]["excess_risk"] for r in records) for i in range(len(steps))]
            ax.plot(steps, curve, color=colour, linewidth=1.3, label=f"{name} (lr {lr:g})",
                    linestyle=LINESTYLE.get(label, "-"))
        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_title(f"B = {B}", fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Excess risk (evaluation set)")
    fig.suptitle("Excess risk during training at each optimiser's tuned learning rate (mean of 2 seeds)",
                 fontsize=8.5, color=INK, x=0.0, ha="left", y=1.05)
    return fig


BATCH_COLOURS = {16: "#86b6ef", 64: "#3987e5", 256: "#1c5cab", 1024: "#0d366b"}  # ordinal ramp, validated


def figure_per_example(best):
    """Small multiples, one panel per optimiser: excess risk vs examples processed, one line per
    batch size. Equal-steps runs give larger batches more data; this view shows the data cost."""
    fig, axes = plt.subplots(1, len(OPTIMISERS), figsize=(9.0, 2.6), sharey=True, sharex=True)
    for ax, (label, name, _) in zip(axes, OPTIMISERS):
        for B, colour in BATCH_COLOURS.items():
            if (label, B) not in best:
                continue
            records = best[(label, B)][1]
            history = records[0]["history"][1:]
            examples = [h["examples"] for h in history]
            curve = [statistics.fmean(r["history"][i + 1]["excess_risk"] for r in records) for i in range(len(history))]
            ax.plot(examples, curve, color=colour, linewidth=1.3, label=f"B = {B}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Examples processed")
    axes[0].set_ylabel("Excess risk")
    axes[-1].legend(fontsize=7, title="batch size", title_fontsize=7)
    fig.suptitle("Data efficiency at each optimiser's tuned learning rate (mean of 2 seeds)",
                 fontsize=8.5, color=INK, x=0.0, ha="left", y=1.07)
    return fig


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    best = tuned(load())
    for name, fig in (("e1_tuned_vs_batch", figure_vs_batch(best)), ("e1_curves", figure_curves(best)),
                      ("e1_per_example", figure_per_example(best))):
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES / f"{name}.{ext}")
        print(f"wrote report/figures/{name}.pdf/.png")
    print("\nTuned learning rates and tail excess risk (mean over seeds):")
    for label, name, _ in OPTIMISERS:
        cells = sorted((B, lr, statistics.fmean(r["summary"]["tail_mean_excess_risk"] for r in recs))
                       for (lab, B), (lr, recs) in best.items() if lab == label)
        print(f"  {name:18s} " + "  ".join(f"B{B}: lr {lr:g} -> {v:.5f}" for B, lr, v in cells))


if __name__ == "__main__":
    main()

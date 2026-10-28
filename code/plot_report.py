"""Figures 1 and 3 of the report (see report/outline.md). Experiment-level figures come from plot_e1.py and plot_e3.py.

Usage:  python code/plot_report.py
Reads results/e1_baselines/*.json, results/timing/per_second_summary.json (from analyze_time.py) and
results/e3_branches/evaluation_summary_followup.json (from `analyze_branches.py evaluate --followup`).
Writes report/figures/fig1_step_vs_time.pdf/.png and fig3_f1_exponent_shift.pdf/.png.

Figure 1: the tuned E1 baselines against batch size, (a) at equal steps: tail excess risk over the last 10% of
6000 steps, the protocol's outcome, mean of 2 seeds; (b) at equal wall-clock time: the seed-averaged curve read
at the time SGD needs for 6000 steps. The y-axis is shared, so the shift between panels is the cost of slower steps.
Figure 3 (exploratory, F1): (a, b) change against continuing Muon for p <= 0, one panel per momentum, each batch
size's best p enlarged; (c) the paired difference p = -0.5 minus p = -0.25 against batch size.
Colours: categorical slots 1-4 (optimisers) and 1-2 (momenta), and the ordinal blue ramp (batch size), all passing
the dataviz validator. Aqua and yellow are below 3:1 contrast, so identity also comes from the legend and, for
exact Muon, a dashed line.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import plot_e1  # noqa: E402  (its rcParams and optimiser colours)
from plot_e3 import BATCH_COLOURS, INK, INK_2, MOMENTUM_COLOURS, relative_change  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "report" / "figures"
BATCHES = [16, 64, 256, 1024]
WIDTH = 6.3  # inches: A4 text width with 2.5 cm margins


def batch_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCHES, labels=[str(B) for B in BATCHES])
    ax.minorticks_off()
    ax.set_xlabel("Batch size B")


def figure_step_vs_time():
    best = plot_e1.tuned(plot_e1.load())
    rows = json.loads((REPO / "results" / "timing" / "per_second_summary.json").read_text())["rows"]
    equal_time = {(r["optimiser"], r["B"]): r["excess_at_equal_time"] for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), sharey=True)
    for label, name, colour in plot_e1.OPTIMISERS:
        per_step = [np.mean([r["summary"]["tail_mean_excess_risk"] for r in best[(label, B)][1]]) for B in BATCHES]
        per_time = [equal_time[(name, B)] for B in BATCHES]
        for ax, ys in zip(axes, (per_step, per_time)):
            ax.plot(BATCHES, ys, color=colour, linewidth=1.5, linestyle=plot_e1.LINESTYLE.get(label, "-"),
                    marker="o", markersize=5.5, markeredgecolor="white", markeredgewidth=0.9, label=name, zorder=3)
    for ax, title in zip(axes, ("(a) Equal steps: 6000 per run", "(b) Equal time: SGD's 6000-step time")):
        batch_axis(ax)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    axes[0].set_yticks(ticks, labels=[f"{t:g}" for t in ticks])
    axes[0].minorticks_off()
    axes[0].set_ylim(0.003, 0.4)
    axes[0].set_ylabel("Excess risk (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=7.5)
    return fig


def paired_difference(cond: dict) -> tuple[float, float]:
    """Mean and 95% t half-width of p = -0.5 minus p = -0.25 over seeds, in % of p = -0.25's mean."""
    a, b = np.asarray(cond["arms"]["p-0.5"]["values"]), np.asarray(cond["arms"]["p-0.25"]["values"])
    d = 100 * (a - b) / b.mean()
    return float(d.mean()), float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))


def figure_exponent_shift():
    summary = json.loads((REPO / "results" / "e3_branches" / "evaluation_summary_followup.json").read_text())
    exponents = [("p-0.75", -0.75), ("p-0.5", -0.5), ("p-0.25", -0.25), ("p0", 0.0)]
    fig = plt.figure(figsize=(WIDTH, 2.5))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.42)
    first = fig.add_subplot(grid[0])
    panels = [first, fig.add_subplot(grid[1], sharey=first), fig.add_subplot(grid[2])]
    xs = [p for _, p in exponents]
    for ax, m in zip(panels, (0.9, 0.95)):
        ax.axhline(0, color=INK_2, linewidth=0.8, zorder=1)
        for cond in sorted((c for c in summary["conditions"] if c["m"] == m), key=lambda c: c["B"]):
            mean, lo, hi = zip(*(relative_change(cond, arm) for arm, _ in exponents))
            colour = BATCH_COLOURS[cond["B"]]
            ax.errorbar(xs, mean, yerr=[np.subtract(mean, lo), np.subtract(hi, mean)], color=colour, linewidth=1.5,
                        marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=0.8, capsize=2,
                        elinewidth=0.8, label=f"B = {cond['B']}", zorder=3)
            k = xs.index(cond["p_star"])  # this batch size's best p over all six exponents, enlarged
            ax.plot(xs[k], mean[k], "o", color=colour, markersize=8.5, markeredgecolor="white",
                    markeredgewidth=1.2, zorder=4)
        ax.set_xticks(xs, labels=["−0.75", "−0.5", "−0.25", "0\n(Muon)"])
        ax.tick_params(axis="x", labelsize=7.5)  # the negative labels sit close
        ax.set_xlabel("Late-phase exponent p")
        ax.set_title(f"({'a' if m == 0.9 else 'b'}) β = {m:g}", fontsize=8.5, color=INK, loc="left")
    panels[0].set_ylabel("Change vs continuing Muon (%)")
    plt.setp(panels[1].get_yticklabels(), visible=False)

    ax = panels[2]
    ax.axhline(0, color=INK_2, linewidth=0.8, zorder=1)
    for m, colour in MOMENTUM_COLOURS.items():
        conds = sorted((c for c in summary["conditions"] if c["m"] == m), key=lambda c: c["B"])
        mean, half = zip(*(paired_difference(c) for c in conds))
        dodge = 0.92 if m == 0.9 else 1.087  # side by side, so the intervals do not overlap
        ax.errorbar([c["B"] * dodge for c in conds], mean, yerr=half, color=colour, linewidth=1.5, marker="o",
                    markersize=4.5, markeredgecolor="white", markeredgewidth=0.8, capsize=2, elinewidth=0.8,
                    label=f"β = {m:g}", zorder=3)
    batch_axis(ax)
    ax.set_xlim(11, 1500)
    ax.set_ylim(-5.2, 6.8)  # headroom so the notes below clear the B = 64 interval
    ax.set_ylabel("Paired difference (%)")
    ax.set_title("(c) p = −0.5 minus p = −0.25", fontsize=8.5, color=INK, loc="left")
    ax.text(0.97, 0.97, "↑ p = −0.25 better", transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color=INK_2)  # corners where no data point is
    ax.text(0.03, 0.03, "↓ p = −0.5 better", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
            color=INK_2)

    handles, labels = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.34, -0.13), fontsize=7.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.83, -0.13), fontsize=7.5)
    return fig


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, fig in (("fig1_step_vs_time", figure_step_vs_time()), ("fig3_f1_exponent_shift", figure_exponent_shift())):
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES / f"{name}.{ext}")
        print(f"wrote report/figures/{name}.pdf/.png")


if __name__ == "__main__":
    main()

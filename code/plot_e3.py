"""Figures for E3 (late-phase spectral exponent) and E4 (noise indicator at the switch).

Usage:  python code/plot_e3.py
Reads results/e3_branches/evaluation_summary.json (from `analyze_branches.py evaluate`) and
results/e2_shared/*.json; writes report/figures/e3_*.pdf/.png and e4_*.pdf/.png.

Figure e3_relative_change: for each exponent p, the paired change in excess risk relative to
continuing Muon (p = 0), in percent, mean and 95% t-interval over the evaluation seeds. One
line per batch size (ordinal blue ramp, validated), one panel per momentum. H1 predicts the
B = 16 line falls towards p = 1 and the B = 1024 line rises.
Figure e4_delta_vs_batch: Xie's indicator delta at the switch point vs batch size.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "report" / "figures"
SPECTRAL = [("p-0.25", -0.25), ("p0", 0.0), ("p0.5", 0.5), ("p1", 1.0)]
BATCH_COLOURS = {16: "#86b6ef", 64: "#3987e5", 256: "#1c5cab", 1024: "#0d366b"}  # ordinal, light -> dark
MOMENTUM_COLOURS = {0.9: "#2a78d6", 0.95: "#eb6834"}  # categorical slots 1-2
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})


def relative_change(cond: dict, arm: str) -> tuple[float, float, float]:
    """Mean and 95% t-interval of 100 * (L_arm - L_p0) / L_p0 over seeds (paired)."""
    base = np.asarray(cond["arms"]["p0"]["values"], float)
    rel = 100 * (np.asarray(cond["arms"][arm]["values"], float) - base) / base
    if len(rel) < 2:
        return float(rel.mean()), float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(rel) - 1) * rel.std(ddof=1) / np.sqrt(len(rel))
    return float(rel.mean()), float(rel.mean() - half), float(rel.mean() + half)


def figure_relative_change(summary: dict):
    momenta = sorted({c["m"] for c in summary["conditions"]})
    fig, axes = plt.subplots(1, len(momenta), figsize=(3.4 * len(momenta), 2.9), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], momenta):
        ax.axhline(0, color=INK_2, linewidth=0.8, zorder=1)
        for cond in sorted((c for c in summary["conditions"] if c["m"] == m), key=lambda c: c["B"]):
            xs, means, lows, highs = [], [], [], []
            for arm, p in SPECTRAL:
                if arm in cond["arms"]:
                    mean, lo, hi = relative_change(cond, arm)
                    xs.append(p), means.append(mean), lows.append(lo), highs.append(hi)
            colour = BATCH_COLOURS.get(cond["B"], INK_2)
            ax.errorbar(xs, means, yerr=[np.subtract(means, lows), np.subtract(highs, means)], color=colour,
                        linewidth=1.5, marker="o", markersize=4.5, capsize=2, elinewidth=0.8,
                        label=f"B = {cond['B']}", zorder=3)
        ax.set_xticks([p for _, p in SPECTRAL], labels=["−0.25", "0\n(Muon)", "0.5", "1\n(SGD-like)"])
        ax.set_xlabel("Late-phase spectral exponent p")
        ax.set_title(f"momentum β = {m:g}", fontsize=9, color=INK, loc="left")
    axes[0][0].set_ylabel("Change in excess risk vs p = 0 (%)")
    axes[0][-1].legend(fontsize=7.5, title="batch size", title_fontsize=7.5)
    fig.suptitle("Continuing from the same Muon checkpoint with exponent p (step size tuned per p);\n"
                 "below 0 = better than continuing Muon. Mean and 95% interval over 3 evaluation seeds.",
                 fontsize=8, color=INK, x=0.0, ha="left", y=1.08)
    return fig


def figure_delta_vs_batch():
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for m, colour in MOMENTUM_COLOURS.items():
        Bs, deltas = [], []
        for B in (16, 64, 256, 1024):
            values = []
            for path in sorted((REPO / "results" / "e2_shared").glob(f"shared_B{B}_m{m:g}_s*.json")):
                diag = json.loads(path.read_text())["diagnostics"]
                values += [mat["delta"] for mat in diag[-1]["matrices"]]
            if values:
                Bs.append(B), deltas.append(np.mean(values))
        if Bs:
            ax.plot(Bs, deltas, color=colour, marker="o", markersize=5, linewidth=1.5, label=f"β = {m:g}",
                    markeredgecolor="white", markeredgewidth=0.8)
            ax.annotate(f"β = {m:g}", (Bs[-1], deltas[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK)
    ax.axhline(1, color=INK_2, linewidth=0.8, linestyle="--")
    ax.annotate("δ = 1: signal ≈ noise", (16, 1), xytext=(0, 4), textcoords="offset points", fontsize=7, color=INK_2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([16, 64, 256, 1024], labels=["16", "64", "256", "1024"])
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Xie's δ at the switch (mean over\nhidden matrices and 5 seeds)")
    ax.legend(fontsize=7.5, loc="upper left")
    return fig


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figures = [("e4_delta_vs_batch", figure_delta_vs_batch())]
    summary_path = REPO / "results" / "e3_branches" / "evaluation_summary.json"
    if summary_path.exists():
        figures.append(("e3_relative_change", figure_relative_change(json.loads(summary_path.read_text()))))
    else:
        print("no E3 evaluation summary yet; only the E4 figure is drawn")
    for name, fig in figures:
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES / f"{name}.{ext}")
        print(f"wrote report/figures/{name}.pdf/.png")


if __name__ == "__main__":
    main()

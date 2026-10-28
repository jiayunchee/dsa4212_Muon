"""Per-step vs per-second comparison of the tuned E1 baselines (the official brief asks for both).

Usage:  python code/analyze_time.py
Reads results/e1_baselines/*.json (excess-risk curves at the tuned learning rates) and
results/timing/step_time_threads1.json (median ms per step over 5 benchmark passes, E-T).

With a constant learning rate, each curve is an "anytime" curve: step k of a 6000-step run is
what a k-step run at that learning rate would give. Multiplying steps by ms/step puts every
optimiser on a wall-clock axis. The equal-time comparison uses the time SGD needs for 6000
steps at that batch size. Caveat: the learning rates were tuned for 6000 steps, and a shorter
budget might prefer a larger one, so this estimate is conservative for the slower optimisers.
Writes results/timing/per_second_summary.json and report/figures/e1_vs_time.pdf/.png.
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OPTIMISERS = [("adamw", "AdamW", "adamw", "#2a78d6"), ("sgd", "SGD + Nesterov", "sgd", "#eb6834"),
              ("muon-ns5", "Muon (NS5)", "muon-ns5", "#1baf7a"),
              ("muon-exact", "Muon (exact SVD)", "spectral p=0 (exact Muon)", "#eda100")]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "axes.edgecolor": INK_2,
                     "xtick.color": INK_2, "ytick.color": INK_2, "savefig.bbox": "tight", "savefig.dpi": 200})


def tuned_curves():
    """(label, B) -> (steps, mean excess-risk curve over seeds) at the best learning rate."""
    runs = {}
    for path in sorted((REPO / "results" / "e1_baselines").glob("*.json")):
        if path.suffix != ".json":
            continue
        r = json.loads(path.read_text())
        cfg = r["config"]
        runs.setdefault((cfg["name"].split("_")[0], cfg["batch_size"], cfg["optimizer"]["lr"]), []).append(r)
    best = {}
    for (label, B, lr), records in runs.items():
        if any(x["summary"]["diverged"] for x in records):
            continue
        score = statistics.fmean(x["summary"]["tail_mean_excess_risk"] for x in records)
        if (label, B) not in best or score < best[(label, B)][0]:
            steps = np.array([h["step"] for h in records[0]["history"]])
            curve = np.mean([[h["excess_risk"] for h in x["history"]] for x in records], axis=0)
            best[(label, B)] = (score, steps, curve)
    return {k: (v[1], v[2]) for k, v in best.items()}


def main() -> None:
    timing = json.loads((REPO / "results" / "timing" / "step_time_threads1.json").read_text())["ms_per_step"]
    curves = tuned_curves()
    batch_sizes = sorted({B for _, B in curves})
    out = {"note": "equal time = SGD's 6000 steps; learning rates tuned for 6000 steps (conservative for slower optimisers)",
           "rows": []}
    print(f"{'B':>5}  {'optimiser':17s} {'ms/step':>8}  {'excess @6000 steps':>19}  {'excess @ equal time':>20}  steps in that time")
    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(2.4 * len(batch_sizes), 2.6), sharey=False)
    for ax, B in zip(axes, batch_sizes):
        budget_ms = 6000 * timing["sgd"][str(B)]
        for label, name, timing_key, colour in OPTIMISERS:
            steps, curve = curves[(label, B)]
            ms = timing[timing_key][str(B)]
            seconds = steps * ms / 1e3
            k = min(int(budget_ms / ms), int(steps[-1]))
            at_time = float(np.interp(k, steps, curve))
            out["rows"].append({"B": B, "optimiser": name, "ms_per_step": ms, "excess_at_6000_steps": float(curve[-1]),
                                "excess_at_equal_time": at_time, "steps_in_equal_time": k})
            print(f"{B:5d}  {name:17s} {ms:8.2f}  {curve[-1]:19.5f}  {at_time:20.5f}  {k}")
            ax.plot(seconds[1:], curve[1:], color=colour, linewidth=1.3, label=name,
                    linestyle=(0, (4, 2)) if label == "muon-exact" else "-")
        ax.axvline(budget_ms / 1e3, color=INK_2, linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"B = {B}", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Optimiser compute time (s)")
        print()
    axes[0].set_ylabel("Excess risk")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(OPTIMISERS), bbox_to_anchor=(0.5, -0.22), fontsize=7.5)
    fig.suptitle("Excess risk against wall-clock compute (tuned learning rates; dotted line = SGD's 6000-step time)",
                 fontsize=8.5, color=INK, x=0.0, ha="left", y=1.06)
    (REPO / "results" / "timing" / "per_second_summary.json").write_text(json.dumps(out, indent=1))
    for ext in ("pdf", "png"):
        fig.savefig(REPO / "report" / "figures" / f"e1_vs_time.{ext}")
    print("wrote results/timing/per_second_summary.json and report/figures/e1_vs_time.pdf/.png")


if __name__ == "__main__":
    main()

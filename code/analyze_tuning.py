"""Summarise a learning-rate sweep: best learning rate per (optimiser, batch size).

Usage:  python code/analyze_tuning.py results/e1_baselines [--labels adamw,sgd]

For every optimiser label (the run-name prefix) and batch size, it averages the tail excess
risk over seeds for each learning rate, reports the best learning rate, and flags a best
value on the edge of the grid (the protocol then requires extending the grid).
Writes <results dir>/tuning_summary.csv.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(results_dir: Path, labels: set[str] | None) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        record = json.loads(path.read_text())
        cfg, summary = record["config"], record["summary"]
        label = cfg["name"].split("_")[0]
        if labels and label not in labels:
            continue
        rows.append({"label": label, "B": cfg["batch_size"], "lr": cfg["optimizer"]["lr"], "seed": cfg["seed"],
                     "tail": summary["tail_mean_excess_risk"], "diverged": summary["diverged"],
                     "ms_per_step": 1e3 * summary["seconds_per_step"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--labels", default=None, help="comma-separated run-name prefixes")
    args = parser.parse_args()
    rows = load(args.results_dir, set(args.labels.split(",")) if args.labels else None)

    cells = defaultdict(list)  # (label, B, lr) -> rows over seeds
    for r in rows:
        cells[(r["label"], r["B"], r["lr"])].append(r)
    table = []
    for (label, B, lr), runs in sorted(cells.items()):
        tails = [r["tail"] for r in runs if not r["diverged"]]
        table.append({"label": label, "B": B, "lr": lr, "seeds": len(runs),
                      "diverged": sum(r["diverged"] for r in runs),
                      "mean_tail": statistics.fmean(tails) if len(tails) == len(runs) else float("inf"),
                      "sd_tail": statistics.stdev(tails) if len(tails) > 1 else float("nan"),
                      "ms_per_step": statistics.fmean(r["ms_per_step"] for r in runs)})

    print(f"{'optimiser':12s} {'B':>5s} {'best lr':>8s} {'mean tail':>10s} {'sd':>8s} {'edge?':>6s}   all lrs (mean tail)")
    best_rows = []
    for label in sorted({t["label"] for t in table}):
        for B in sorted({t["B"] for t in table if t["label"] == label}):
            cand = sorted((t for t in table if t["label"] == label and t["B"] == B), key=lambda t: t["lr"])
            best = min(cand, key=lambda t: t["mean_tail"])
            edge = "LOW" if best is cand[0] else "HIGH" if best is cand[-1] else ""
            best_rows.append({**best, "edge": edge})
            others = "  ".join(f"{t['lr']:g}:{t['mean_tail']:.4f}" + ("(div)" if t["diverged"] else "") for t in cand)
            print(f"{label:12s} {B:5d} {best['lr']:8g} {best['mean_tail']:10.5f} {best['sd_tail']:8.5f} {edge:>6s}   {others}")

    out = args.results_dir / "tuning_summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "B", "lr", "seeds", "diverged", "mean_tail", "sd_tail", "ms_per_step"])
        writer.writeheader()
        writer.writerows(table)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

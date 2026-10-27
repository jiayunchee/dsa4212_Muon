"""Run a whole branch stage (E3, or F1 with --followup) in protocol order, so no step can be skipped:
the tuning runs and every extension until each edge is resolved (protocol Section 6), then `select`, the
evaluation runs, `evaluate` and the figures. Safe to re-run: finished runs are skipped.

Usage:  python code/run_branches.py [--followup] [--workers 4]
The same steps one at a time: code/run.py <config>, then code/analyze_branches.py extend | select | evaluate.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import analyze_branches as ab
import plot_e3


def run(config: Path, workers: int) -> None:
    print(f"\n>>> python code/run.py {config.relative_to(ab.REPO)} --workers {workers}", flush=True)
    subprocess.run([sys.executable, str(ab.REPO / "code" / "run.py"), str(config), "--workers", str(workers)],
                   check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--followup", action="store_true", help="the F1 follow-up stage (p = -0.5, -0.75)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    stage = ab.F1 if args.followup else ab.E3
    for config in [stage.tune_spec, *ab.extension_configs(stage)]:  # finishes anything interrupted earlier
        run(config, args.workers)
    previous = None
    while (config := ab.extend(stage)) is not None:
        if config == previous:  # its runs failed twice; stop rather than loop
            sys.exit(f"{config.relative_to(ab.REPO)} was run twice and some runs are still missing: see FAILED above")
        run(config, args.workers)
        previous = config
    print(f"\n>>> python code/analyze_branches.py select{' --followup' if args.followup else ''}")
    ab.select(stage)
    run(stage.eval_spec, args.workers)
    print(f"\n>>> python code/analyze_branches.py evaluate{' --followup' if args.followup else ''}")
    ab.evaluate(stage)
    print("\n>>> python code/plot_e3.py")
    plot_e3.main()


if __name__ == "__main__":
    main()

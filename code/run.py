"""Run an experiment file.

Usage:
    python code/run.py configs/e0_pilot.yaml               # run everything not yet done
    python code/run.py configs/e0_pilot.yaml --workers 4   # four runs in parallel (1 thread each)
    python code/run.py configs/e0_pilot.yaml --dry-run     # list the runs only
    python code/run.py configs/e0_pilot.yaml --only adamw  # runs whose name matches a regex
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make code/ importable (also in workers)

from muonlab.sweep import execute, expand, load_spec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_file")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", default=None, help="regular expression on run names")
    args = parser.parse_args()

    configs = expand(load_spec(args.experiment_file))
    if args.only:
        configs = [c for c in configs if re.search(args.only, c["name"])]
    if args.dry_run:
        for c in configs:
            print(c["name"])
        print(f"{len(configs)} runs")
        return
    execute(configs, workers=args.workers)


if __name__ == "__main__":
    main()

"""E-T: wall-clock time per training step for every arm, measured on a quiet machine.

Sweep runs share the CPU (several workers), so their timings include contention. Per-second
comparisons in the report use this benchmark instead (notes/protocol.md, Section 5).

Usage:  python code/benchmark_step_time.py [--threads 1] [--steps 300]
Writes results/timing/step_time_threads<k>.json and prints a table.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from muonlab.data import TeacherStudentStream  # noqa: E402
from muonlab.models import MLP  # noqa: E402
from muonlab.optim import build_optimizer  # noqa: E402
from muonlab.train import environment_info, git_commit  # noqa: E402

ARMS = {
    "adamw": {"name": "adamw", "lr": 1e-3},
    "sgd": {"name": "sgd", "lr": 0.03, "momentum": 0.9},
    "muon-ns5": {"name": "muon", "update": "ns5", "lr": 0.003, "adam_lr": 1e-3},
    "spectral p=0 (exact Muon)": {"name": "muon", "update": "spectral", "p": 0.0, "lr": 0.003, "adam_lr": 1e-3},
    "spectral p=-0.25": {"name": "muon", "update": "spectral", "p": -0.25, "lr": 0.003, "adam_lr": 1e-3},
    "plain (Xie)": {"name": "muon", "update": "plain", "lr": 0.003, "adam_lr": 1e-3},
}


def time_arm(opt_cfg: dict, batch_size: int, steps: int, warmup: int = 50) -> float:
    task = TeacherStudentStream()
    torch.manual_seed(0)
    model = MLP([128, 256, 256, 256, 16])
    opt, _ = build_optimizer(model, opt_cfg)
    gen = torch.Generator().manual_seed(0)
    batches = [task.sample(batch_size, gen) for _ in range(warmup + steps)]  # data excluded from timing
    times = []
    for i, (x, y) in enumerate(batches):
        start = time.perf_counter()
        loss = F.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i >= warmup:
            times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    batch_sizes = [16, 64, 256, 1024]
    table = {arm: {} for arm in ARMS}
    print(f"median ms per step (forward + backward + optimiser), {args.threads} thread(s)")
    print(f"{'arm':28s}" + "".join(f"{f'B={b}':>10s}" for b in batch_sizes))
    for arm, cfg in ARMS.items():
        for b in batch_sizes:
            table[arm][b] = time_arm(cfg, b, args.steps)
        print(f"{arm:28s}" + "".join(f"{table[arm][b]:10.2f}" for b in batch_sizes), flush=True)
    out = Path(__file__).resolve().parents[1] / "results" / "timing" / f"step_time_threads{args.threads}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"threads": args.threads, "steps": args.steps, "ms_per_step": table,
                               "environment": environment_info(), "git": git_commit()}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

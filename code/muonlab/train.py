"""Training loop for one run, with checkpoints and branching (notes/protocol.md, Sections 3-4).

A run config is a nested dict (built from YAML in configs/ by muonlab.sweep). Keys:
  name, experiment     run name and experiment folder for results/ and checkpoints/
  seed                 model initialisation and data stream
  threads              torch CPU threads (1 when several runs share the CPU)
  task                 TeacherStudentStream arguments
  eval_size, eval_seed fixed evaluation set (noiseless targets)
  model                {widths, activation}
  optimizer            build_optimizer config: name, lr, adam_lr, momentum, update, p, ...
  batch_size, steps    minibatch size and number of steps in this run
  eval_every           steps between evaluations
  schedule             "constant" or "linear_decay" (to zero at the last step of this run)
  checkpoint_at        steps at which to save a checkpoint (used by the shared phase, E2)
  diagnostics_at       steps at which to record noise statistics and spectra
  branch               continue from a checkpoint with a new update rule (E3):
                       {checkpoint, update, p, normalize, lr_multiplier, lr_reference, k}
                       lr_reference "checkpoint" scales the checkpoint's lr; "xie" first sets
                       each matrix's lr to Xie's matched momentum-SGD rate.
"""

from __future__ import annotations

import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from muonlab.data import TeacherStudentStream
from muonlab.diagnostics import (gaussian_nuclear_constant, gradient_noise_stats, spectrum_stats,
                                 xie_quantities)
from muonlab.models import MLP
from muonlab.optim import SVD_FALLBACKS, OptimizerBundle, build_optimizer

REPO_ROOT = Path(__file__).resolve().parents[2]


def environment_info() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "torch_threads": torch.get_num_threads()}


def git_commit() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def schedule_factor(schedule: str, i: int, total: int) -> float:
    if schedule == "constant":
        return 1.0
    if schedule == "linear_decay":
        return 1.0 - i / total
    raise ValueError(f"unknown schedule {schedule!r}")


def apply_branch(model: MLP, opt: OptimizerBundle, cfg: dict, task: TeacherStudentStream,
                 diag_gen: torch.Generator) -> dict:
    """Switch the hidden matrices to the branch's update rule and learning rate.

    The momentum buffers and the AdamW state carry over unchanged.
    """
    br = cfg["branch"]
    groups = opt.optimizers["matrix"].param_groups  # one group per hidden matrix
    multiplier = br.get("lr_multiplier", 1.0)
    info: dict = {"reference": br.get("lr_reference", "checkpoint"), "multiplier": multiplier,
                  "checkpoint_lr": [g["base_lr"] for g in groups]}
    if info["reference"] == "xie":
        stats = gradient_noise_stats(model, F.mse_loss, lambda: task.sample(cfg["batch_size"], diag_gen),
                                     model.hidden_matrices(), k=br.get("k", 32))
        info["xie"] = []
        for group, stat in zip(groups, stats):
            q = xie_quantities(stat, group["momentum"], group["nesterov"], group["base_lr"],
                               gaussian_nuclear_constant(*stat["shape"]))
            info["xie"].append({**stat, **q})
            group["base_lr"] = q["matched_lr"]
    for group in groups:
        group["base_lr"] *= multiplier
        group["lr"] = group["base_lr"]
        for key in ("update", "p", "normalize"):
            if key in br:
                group[key] = br[key]
    info["branch_lr"] = [g["base_lr"] for g in groups]
    return info


def collect_diagnostics(model: MLP, opt: OptimizerBundle, cfg: dict, task: TeacherStudentStream,
                        diag_gen: torch.Generator, step: int, k: int) -> dict:
    """Noise statistics for every hidden matrix and, for Muon runs, Xie's indicator and the
    spectrum of the momentum buffer."""
    stats = gradient_noise_stats(model, F.mse_loss, lambda: task.sample(cfg["batch_size"], diag_gen),
                                 model.hidden_matrices(), k=k)
    matrix_opt = opt.optimizers.get("matrix")
    for i, stat in enumerate(stats):
        if matrix_opt is not None:
            group = matrix_opt.param_groups[i]
            stat.update(xie_quantities(stat, group["momentum"], group["nesterov"], group["base_lr"],
                                       gaussian_nuclear_constant(*stat["shape"])))
            buffer = matrix_opt.state[group["params"][0]].get("momentum_buffer")
            if buffer is not None:
                stat["momentum_spectrum"] = spectrum_stats(buffer)
    return {"step": step, "k": k, "matrices": stats}


def run(cfg: dict) -> dict:
    """Train one configuration and return its result record."""
    torch.set_num_threads(cfg.get("threads", 1))
    task = TeacherStudentStream(**cfg["task"])
    eval_x, eval_y = task.eval_set(cfg["eval_size"], cfg["eval_seed"])
    torch.manual_seed(cfg["seed"])
    model = MLP(cfg["model"]["widths"], cfg["model"].get("activation", "tanh"))
    opt, assignment = build_optimizer(model, cfg["optimizer"])
    data_gen = torch.Generator().manual_seed(10_000 + cfg["seed"])   # training data stream
    diag_gen = torch.Generator().manual_seed(20_000 + cfg["seed"])   # diagnostics only

    step0, branch_info = 0, None
    if cfg.get("branch"):
        ckpt = torch.load(REPO_ROOT / cfg["branch"]["checkpoint"], weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        data_gen.set_state(ckpt["data_generator"])
        step0 = ckpt["step"]
        branch_info = apply_branch(model, opt, cfg, task, diag_gen)

    def evaluate(step: int, train_loss: float | None) -> dict:
        with torch.no_grad():
            excess = F.mse_loss(model(eval_x), eval_y).item()
        return {"step": step, "examples": step * cfg["batch_size"], "excess_risk": excess,
                "train_loss": train_loss, "step_seconds": step_seconds, "data_seconds": data_seconds}

    total, eval_every = cfg["steps"], cfg["eval_every"]
    fallbacks_before = SVD_FALLBACKS["count"]
    checkpoint_at = set(cfg.get("checkpoint_at", []))
    diagnostics_at = set(cfg.get("diagnostics_at", []))
    step_seconds = data_seconds = 0.0
    history = [evaluate(step0, None)]
    diagnostics = []
    loss_sum, loss_count, diverged = 0.0, 0, False
    for i in range(total):
        step = step0 + i + 1
        opt.set_lr_factor(schedule_factor(cfg.get("schedule", "constant"), i, total))
        start = time.perf_counter()
        x, y = task.sample(cfg["batch_size"], data_gen)
        data_seconds += time.perf_counter() - start
        start = time.perf_counter()
        loss = F.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step_seconds += time.perf_counter() - start
        value = loss.item()
        if not math.isfinite(value) or value > 1e6:
            diverged = True
            history.append({**evaluate(step, value), "diverged": True})
            break
        loss_sum, loss_count = loss_sum + value, loss_count + 1
        if step % eval_every == 0 or i == total - 1:
            history.append(evaluate(step, loss_sum / loss_count))
            loss_sum, loss_count = 0.0, 0
        if step in checkpoint_at:
            path = REPO_ROOT / "checkpoints" / cfg["experiment"] / f"{cfg['name']}_step{step}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "data_generator": data_gen.get_state(), "step": step, "config": cfg}, path)
        if step in diagnostics_at:
            diagnostics.append(collect_diagnostics(model, opt, cfg, task, diag_gen, step, cfg.get("diagnostics_k", 32)))

    steps_done = history[-1]["step"] - step0
    tail = [h["excess_risk"] for h in history[1:] if h["step"] > step0 + 0.9 * total]
    summary = {
        "diverged": diverged,
        "steps_completed": steps_done,
        "final_excess_risk": history[-1]["excess_risk"],
        "tail_mean_excess_risk": statistics.fmean(tail) if tail and not diverged else None,
        "step_seconds": step_seconds,
        "seconds_per_step": step_seconds / max(steps_done, 1),
        "svd_float64_fallbacks": SVD_FALLBACKS["count"] - fallbacks_before,
    }
    return {"config": cfg, "environment": environment_info(), "git": git_commit(),
            "parameter_assignment": assignment, "branch": branch_info, "history": history,
            "diagnostics": diagnostics, "summary": summary}


def run_and_save(cfg: dict, out_path: str) -> dict:
    """Worker entry point for muonlab.sweep: run, write the JSON record, return the summary."""
    record = run(cfg)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    return {"name": cfg["name"], **record["summary"]}

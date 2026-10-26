"""Expand an experiment file into run configs and execute them (in parallel, resumable).

Experiment file (YAML):
    experiment: e1_baselines
    base: {...}                         full run config shared by every run (see muonlab.train)
    grid: {dotted.key: [values, ...]}   every run is repeated for each combination
    arms:                               optional; each arm adds its own settings and sweeps
      - label: adamw
        set:   {optimizer: {name: adamw}}         merged into the config
        sweep: {optimizer.lr: [3.0e-4, 1.0e-3]}   dotted keys -> lists of values to sweep
A value may depend on another: {by: batch_size, map: {16: 1.0e-3, 1024: 3.0e-3}}.
Strings may contain {dotted.key} placeholders, filled in from the finished config.
A run whose result file already exists is skipped, so an interrupted sweep can be resumed.
"""

from __future__ import annotations

import copy
import itertools
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from muonlab.train import REPO_ROOT, run_and_save

SHORT_NAMES = {"lr": "lr", "batch_size": "B", "seed": "s", "momentum": "m", "p": "p",
               "lr_multiplier": "x", "label_noise": "sy", "adam_lr": "alr", "update": ""}


def get_key(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def set_key(cfg: dict, dotted: str, value) -> None:
    *parents, last = dotted.split(".")
    node = cfg
    for part in parents:
        node = node.setdefault(part, {})
    node[last] = value


def merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve(cfg: dict, node=None):
    """Replace {by, map} dependencies and fill {dotted.key} placeholders in strings."""
    node = cfg if node is None else node
    if isinstance(node, dict):
        if set(node) == {"by", "map"}:
            key = get_key(cfg, node["by"])
            return node["map"][key] if key in node["map"] else node["map"][str(key)]
        return {k: resolve(cfg, v) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve(cfg, v) for v in node]
    if isinstance(node, str):
        return re.sub(r"\{([A-Za-z_][\w.]*)\}", lambda m: str(get_key(cfg, m.group(1))), node)
    return node


def short_value(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def expand(spec: dict) -> list[dict]:
    """All run configs of an experiment file, each with a unique cfg["name"]."""
    grid = spec.get("grid", {})
    configs = []
    for arm in spec.get("arms") or [{}]:
        sweeps = {**arm.get("sweep", {}), **grid}
        keys = list(sweeps)
        for values in itertools.product(*(sweeps[k] for k in keys)):
            cfg = merge(spec["base"], arm.get("set", {}))
            parts = [arm["label"]] if arm.get("label") else []
            for key, value in zip(keys, values):
                set_key(cfg, key, value)
                short = SHORT_NAMES.get(key.split(".")[-1], key.split(".")[-1])
                parts.append(f"{short}{short_value(value)}")
            cfg["experiment"] = spec["experiment"]
            cfg["name"] = "_".join(p for p in parts if p)
            first_pass = resolve(cfg)  # dependencies may refer to swept values
            configs.append(resolve(first_pass))
    names = [c["name"] for c in configs]
    if len(set(names)) != len(names):
        raise ValueError("run names are not unique; give each arm a distinct label")
    return configs


def load_spec(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def result_path(cfg: dict) -> Path:
    return REPO_ROOT / "results" / cfg["experiment"] / f"{cfg['name']}.json"


def execute(configs: list[dict], workers: int = 1) -> list[dict]:
    """Run every config whose result file is missing; print one line per finished run."""
    todo = [c for c in configs if not result_path(c).exists()]
    print(f"{len(configs)} runs, {len(configs) - len(todo)} already done, {len(todo)} to run, {workers} workers")
    summaries, start = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_and_save, c, str(result_path(c))): c["name"] for c in todo}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                s = future.result()
            except Exception as exc:  # keep going; the failure is reported, not hidden
                print(f"[{i}/{len(todo)}] {futures[future]}: FAILED {exc!r}")
                continue
            summaries.append(s)
            tail = s["tail_mean_excess_risk"]
            print(f"[{i}/{len(todo)}] {s['name']}: tail excess risk "
                  f"{'DIVERGED' if s['diverged'] else f'{tail:.5f}'} | {s['seconds_per_step'] * 1e3:.2f} ms/step"
                  f" | elapsed {time.perf_counter() - start:.0f} s", flush=True)
    return summaries

"""Analyse the E3 branch experiments (notes/protocol.md, Section 5).

Usage:
    python code/analyze_branches.py extend     # after E3a: extend grids where the best is on an edge
    python code/analyze_branches.py select     # when no edges remain: pick multipliers, write E3b config
    python code/analyze_branches.py evaluate   # after E3b: the pre-registered analysis

select   For every (arm, B, momentum), average the tail excess risk over the tuning seeds for
         each learning-rate multiplier, pick the best, flag grid edges, and write
         configs/e3b_branches_eval.yaml (same arms, chosen multiplier, evaluation seeds).
evaluate On the evaluation seeds: mean excess risk per arm; paired differences to the p = 0
         branch (same checkpoint and data, so differences are paired by seed); p* per
         condition with a bootstrap over seeds; the H1 contrast
             C = [L(p=1) - L(p=-0.25)] at B=16  -  [L(p=1) - L(p=-0.25)] at B=1024
         with a Welch 95% interval; and Xie's indicator delta at the switch (E2) for H3.
         Writes results/e3_branches/evaluation_summary.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "e3_branches"
TUNE_SPEC = REPO / "configs" / "e3a_branches_tune.yaml"
EVAL_SPEC = REPO / "configs" / "e3b_branches_eval.yaml"
TUNING_SEEDS, EVAL_SEEDS = {0, 1}, {2, 3, 4}
MAX_FACTOR = 64  # stopping rule (protocol change log, 27 Oct): multipliers stay within [1/64, 64]
SPECTRAL = ["p-0.25", "p0", "p0.5", "p1"]
P_VALUE = {"p-0.25": -0.25, "p0": 0.0, "p0.5": 0.5, "p1": 1.0}
ARMS = SPECTRAL + ["ns5", "xie"]


def load(seeds: set[int]) -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        if path.name.endswith("summary.json"):
            continue
        record = json.loads(path.read_text())
        cfg, summary = record["config"], record["summary"]
        if cfg["seed"] not in seeds:
            continue
        rows.append({"arm": cfg["name"].split("_")[0], "B": cfg["batch_size"],
                     "m": cfg["optimizer"]["momentum"], "x": cfg["branch"]["lr_multiplier"],
                     "seed": cfg["seed"], "diverged": summary["diverged"],
                     "tail": summary["tail_mean_excess_risk"] if not summary["diverged"] else float("inf"),
                     "ms_per_step": 1e3 * summary["seconds_per_step"]})
    return rows


def extend(new_values: int = 2) -> None:
    """Protocol Section 6: wherever an arm's best multiplier is on the edge of its grid, add
    `new_values` more multipliers beyond that edge (halving below, doubling above) on the
    tuning seeds, and write configs/e3a_ext<k>_branches_tune.yaml. Repeat until no edges remain."""
    cells = defaultdict(list)
    for r in load(TUNING_SEEDS):
        cells[(r["arm"], r["B"], r["m"], r["x"])].append(r["tail"])
    tune = yaml.safe_load(TUNE_SPEC.read_text())
    arm_settings = {arm["label"]: arm["set"] for arm in tune["arms"]}
    arms_out, n_low, n_high, capped = [], 0, 0, []
    for arm in ARMS:
        for B in sorted({k[1] for k in cells}):
            for m in sorted({k[2] for k in cells}):
                options = sorted((x, statistics.fmean(v)) for (a, b, mm, x), v in cells.items()
                                 if a == arm and b == B and mm == m)
                if not options:
                    continue
                xs = [x for x, _ in options]
                best_x = min(options, key=lambda t: t[1])[0]
                if best_x == xs[0]:
                    new, n_low = [best_x / 2**k for k in range(1, new_values + 1)], n_low + 1
                elif best_x == xs[-1]:
                    new, n_high = [best_x * 2**k for k in range(1, new_values + 1)], n_high + 1
                else:
                    continue
                new = [x for x in new if x not in xs and 1 / MAX_FACTOR <= x <= MAX_FACTOR]
                if not new:  # the stopping rule: accept the edge value and report it
                    capped.append(f"{arm} B={B} m={m:g} (best x={best_x:g})")
                    continue
                arms_out.append({"label": arm, "set": copy.deepcopy(arm_settings[arm]),
                                 "sweep": {"branch.lr_multiplier": new,
                                           "batch_size": [B], "optimizer.momentum": [m]}})
    if capped:
        print(f"still on an edge but at the [1/{MAX_FACTOR}, {MAX_FACTOR}] limit (accepted, report it): "
              + "; ".join(capped))
    if not arms_out:
        print("no further extension needed: run `select`")
        return
    k = 1 + len(list((REPO / "configs").glob("e3a_ext*_branches_tune.yaml")))
    path = REPO / "configs" / f"e3a_ext{k}_branches_tune.yaml"
    spec = {"experiment": tune["experiment"], "base": tune["base"], "grid": {"seed": sorted(TUNING_SEEDS)},
            "arms": arms_out}
    header = (f"# E3a extension {k} (generated by `code/analyze_branches.py extend`, protocol Section 6):\n"
              f"# {n_low} cells had their best multiplier at the low edge and {n_high} at the high edge,\n"
              f"# so {new_values} more multipliers beyond each edge are tried on the tuning seeds.\n")
    path.write_text(header + yaml.safe_dump(spec, sort_keys=False))
    runs = sum(len(a["sweep"]["branch.lr_multiplier"]) for a in arms_out) * len(TUNING_SEEDS)
    print(f"{n_low} low-edge and {n_high} high-edge cells -> wrote {path.relative_to(REPO)} ({runs} runs)")


def select() -> None:
    rows = load(TUNING_SEEDS)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["arm"], r["B"], r["m"], r["x"])].append(r["tail"])
    chosen = {}
    print(f"{'arm':7s} {'B':>5s} {'m':>5s}  best x   edge   mean tail per multiplier")
    for arm in ARMS:
        for B in sorted({k[1] for k in cells}):
            for m in sorted({k[2] for k in cells}):
                options = sorted((x, statistics.fmean(v)) for (a, b, mm, x), v in cells.items()
                                 if a == arm and b == B and mm == m)
                if not options:
                    continue
                best_x, best_tail = min(options, key=lambda t: t[1])
                edge = "LOW" if best_x == options[0][0] else "HIGH" if best_x == options[-1][0] else ""
                chosen[(arm, B, m)] = best_x
                print(f"{arm:7s} {B:5d} {m:5g}  {best_x:6g}  {edge:>5s}   "
                      + "  ".join(f"{x:g}:{t:.5f}" for x, t in options))

    tune = yaml.safe_load(TUNE_SPEC.read_text())
    arm_settings = {arm["label"]: arm["set"] for arm in tune["arms"]}
    spec = {"experiment": tune["experiment"], "base": tune["base"],
            "grid": {"seed": sorted(EVAL_SEEDS)}, "arms": []}
    for (arm, B, m), x in sorted(chosen.items(), key=lambda kv: (ARMS.index(kv[0][0]), kv[0][1], kv[0][2])):
        spec["arms"].append({"label": arm, "set": copy.deepcopy(arm_settings[arm]),
                             "sweep": {"branch.lr_multiplier": [x], "batch_size": [B], "optimizer.momentum": [m]}})
    header = ("# E3b (generated by `code/analyze_branches.py select` from the E3a tuning seeds):\n"
              "# each arm uses the learning-rate multiplier chosen on seeds {0, 1}; evaluated on seeds {2, 3, 4}.\n")
    EVAL_SPEC.write_text(header + yaml.safe_dump(spec, sort_keys=False))
    print(f"\nwrote {EVAL_SPEC.relative_to(REPO)} ({len(spec['arms'])} arms x {len(EVAL_SEEDS)} seeds)")


def t_interval(values: list[float]) -> tuple[float, float, float]:
    """Mean and 95% t-interval (n >= 2)."""
    a = np.asarray(values, dtype=float)
    mean, se = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
    half = stats.t.ppf(0.975, len(a) - 1) * se
    return float(mean), float(mean - half), float(mean + half)


def welch_interval(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Difference of means mean(a) - mean(b) with a Welch 95% interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    diff, se = a.mean() - b.mean(), np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    half = stats.t.ppf(0.975, df) * se
    return float(diff), float(diff - half), float(diff + half)


def switch_delta(B: int, m: float) -> float | None:
    """Mean of Xie's delta over hidden matrices and seeds at the E2 switch point."""
    values = []
    for seed in range(5):
        path = REPO / "results" / "e2_shared" / f"shared_B{B}_m{m:g}_s{seed}.json"
        if path.exists():
            diag = json.loads(path.read_text())["diagnostics"]
            values += [mat["delta"] for mat in diag[-1]["matrices"]]
    return statistics.fmean(values) if values else None


def evaluate(bootstrap: int = 5000) -> None:
    rows = load(EVAL_SEEDS)
    table = defaultdict(dict)  # (B, m) -> arm -> {seed: tail}
    for r in rows:
        table[(r["B"], r["m"])].setdefault(r["arm"], {})[r["seed"]] = r["tail"]
    rng = np.random.default_rng(0)
    summary = {"conditions": [], "contrast": []}
    for (B, m) in sorted(table):
        arms = table[(B, m)]
        seeds = sorted(set.intersection(*(set(v) for v in arms.values())))
        base = arms["p0"]
        print(f"\n=== B = {B}, momentum = {m:g}  (seeds {seeds}; Xie delta at switch = {switch_delta(B, m)}) ===")
        print(f"{'arm':7s} {'mean excess':>12s}   {'diff vs p0 [95% CI]':>34s}")
        cond = {"B": B, "m": m, "delta_at_switch": switch_delta(B, m), "arms": {}}
        for arm in ARMS:
            if arm not in arms:
                continue
            values = [arms[arm][s] for s in seeds]
            diffs = [arms[arm][s] - base[s] for s in seeds]
            mean = statistics.fmean(values)
            d_mean, d_lo, d_hi = t_interval(diffs) if arm != "p0" else (0.0, 0.0, 0.0)
            cond["arms"][arm] = {"mean": mean, "values": values, "diff_vs_p0": [d_mean, d_lo, d_hi]}
            print(f"{arm:7s} {mean:12.5f}   {d_mean:+.5f} [{d_lo:+.5f}, {d_hi:+.5f}]")
        # p* among the spectral family, with a bootstrap over seeds (paired: same seeds for all arms)
        matrix = np.array([[arms[a][s] for s in seeds] for a in SPECTRAL])
        p_star = SPECTRAL[int(np.argmin(matrix.mean(axis=1)))]
        counts = defaultdict(int)
        for _ in range(bootstrap):
            idx = rng.integers(0, len(seeds), len(seeds))
            counts[SPECTRAL[int(np.argmin(matrix[:, idx].mean(axis=1)))]] += 1
        cond["p_star"] = P_VALUE[p_star]
        cond["p_star_bootstrap"] = {a: counts[a] / bootstrap for a in SPECTRAL}
        print(f"p* = {P_VALUE[p_star]:g}; bootstrap share as winner: "
              + ", ".join(f"{a}: {counts[a] / bootstrap:.2f}" for a in SPECTRAL))
        summary["conditions"].append(cond)

    for m in sorted({c["m"] for c in summary["conditions"]}):
        D = {}
        for B in (16, 1024):
            arms = table.get((B, m))
            if arms and "p1" in arms and "p-0.25" in arms:
                seeds = sorted(set(arms["p1"]) & set(arms["p-0.25"]))
                D[B] = [arms["p1"][s] - arms["p-0.25"][s] for s in seeds]
        if len(D) == 2 and min(len(v) for v in D.values()) >= 2:
            c, lo, hi = welch_interval(D[16], D[1024])
            verdict = "supports H1" if hi < 0 else "contradicts H1" if lo > 0 else "inconclusive"
            summary["contrast"].append({"m": m, "C": c, "ci": [lo, hi], "verdict": verdict})
            print(f"\nH1 contrast (momentum {m:g}): C = {c:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  -> {verdict}")

    controls = REPO / "results" / "e3_controls"
    if controls.exists():  # exploratory, not pre-registered: hidden matrices frozen in the branch
        frozen = defaultdict(dict)
        for path in controls.glob("frozen_*.json"):
            cfg = json.loads(path.read_text())["config"]
            if cfg["seed"] in EVAL_SEEDS:
                frozen[(cfg["batch_size"], cfg["optimizer"]["momentum"])][cfg["seed"]] = \
                    json.loads(path.read_text())["summary"]["tail_mean_excess_risk"]
        print("\nEXPLORATORY control (added after seeing tuning results): hidden matrices frozen in the branch")
        summary["exploratory_frozen_control"] = []
        for (B, m), by_seed in sorted(frozen.items()):
            arms = table.get((B, m), {})
            seeds = sorted(set(by_seed) & set(arms.get("p0", {})))
            if len(seeds) < 2:
                continue
            d_mean, d_lo, d_hi = t_interval([by_seed[s] - arms["p0"][s] for s in seeds])
            best_sgd = {a: statistics.fmean(arms[a][s] for s in seeds) for a in ("p1", "xie") if a in arms}
            summary["exploratory_frozen_control"].append({"B": B, "m": m, "mean": statistics.fmean(by_seed[s] for s in seeds),
                                                          "diff_vs_p0": [d_mean, d_lo, d_hi], "sgd_like": best_sgd})
            print(f"  B={B:5d} m={m:g}: frozen {statistics.fmean(by_seed[s] for s in seeds):.5f} "
                  f"(vs p0 {d_mean:+.5f} [{d_lo:+.5f}, {d_hi:+.5f}]); "
                  + ", ".join(f"{a} {v:.5f}" for a, v in best_sgd.items()))

    deltas = [c["delta_at_switch"] for c in summary["conditions"] if c["delta_at_switch"] is not None]
    if len(deltas) == len(summary["conditions"]) >= 3:
        rho, pval = stats.spearmanr(deltas, [c["p_star"] for c in summary["conditions"]])
        summary["h3_spearman"] = {"rho": float(rho), "p_value": float(pval), "n_conditions": len(deltas)}
        print(f"\nH3: Spearman correlation between delta at switch and p*: rho = {rho:+.2f} (p = {pval:.3f}, "
              f"n = {len(deltas)} conditions; H3 predicts rho < 0)")
    out = RESULTS / "evaluation_summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out.relative_to(REPO)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["extend", "select", "evaluate"])
    args = parser.parse_args()
    if not RESULTS.exists():
        sys.exit(f"no results in {RESULTS}")
    {"extend": extend, "select": select, "evaluate": evaluate}[args.mode]()


if __name__ == "__main__":
    main()

"""Analyse the branch experiments: E3 (pre-registered) and F1 (follow-up, not pre-registered).

Usage:
    python code/analyze_branches.py extend   [--followup]   extend grids where the best is on an edge
    python code/analyze_branches.py select   [--followup]   pick multipliers, write the evaluation config
    python code/analyze_branches.py evaluate [--followup]   the analysis on the evaluation seeds

Stages (both write to results/e3_branches/, told apart by arm label):
    E3 (default)  arms p-0.25, p0, p0.5, p1, ns5, xie; configs e3a_*.yaml / e3b_*.yaml;
                  summary results/e3_branches/evaluation_summary.json. This is the pre-registered analysis.
    F1 (--followup, added 27 Oct after E3 put p* at the edge of the p grid): new arms p-0.5 and
                  p-0.75; configs f1a_*.yaml / f1b_*.yaml. `evaluate --followup` compares all
                  eight arms and writes evaluation_summary_followup.json, leaving the E3 summary untouched.

extend   Protocol Section 6: where an arm's best multiplier is on the edge of its grid, add two more
         multipliers beyond it (halving or doubling) on the tuning seeds, within [1/64, 64]. If the
         newest extension config already lists exactly those runs, it names that config instead.
select   For every (arm, B, momentum), the multiplier with the lowest mean tail excess risk on the
         tuning seeds goes into the evaluation config (evaluation seeds 2, 3, 4). Refuses while a
         tuning run is unfinished or a best multiplier sits on an edge that can still be extended.
evaluate Paired differences to the p = 0 branch (same checkpoint and data), p* with a bootstrap
         over seeds, the H1 contrast C = [L(p=1) - L(p=-0.25)]_{B=16} - [same]_{B=1024} with a
         Welch interval, Xie's delta at the switch (E2) for H3, and the exploratory frozen control.
         Only evaluation runs at the multipliers in the evaluation configs are used. Refuses while
         tuning is unfinished or an evaluation config predates the latest tuning runs.

`python code/run_branches.py [--followup]` runs a whole stage in this order.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "e3_branches"
CONFIGS = REPO / "configs"
TUNING_SEEDS, EVAL_SEEDS = {0, 1}, {2, 3, 4}
MAX_FACTOR = 64  # stopping rule (protocol change log, 27 Oct): multipliers stay within [1/64, 64]
P_VALUE = {"p-0.75": -0.75, "p-0.5": -0.5, "p-0.25": -0.25, "p0": 0.0, "p0.5": 0.5, "p1": 1.0}
PREREG_SPECTRAL = ["p-0.25", "p0", "p0.5", "p1"]
FOLLOWUP_SPECTRAL = ["p-0.75", "p-0.5"]
OTHER_ARMS = ["ns5", "xie"]


@dataclass
class Stage:
    name: str
    tuned_arms: list[str]     # arms this stage tunes (extend / select)
    analysed_arms: list[str]  # arms compared by evaluate
    spectral: list[str]       # candidates for p*
    tune_spec: Path
    eval_spec: Path
    ext_prefix: str
    summary_name: str


E3 = Stage("E3", PREREG_SPECTRAL + OTHER_ARMS, PREREG_SPECTRAL + OTHER_ARMS, PREREG_SPECTRAL,
           CONFIGS / "e3a_branches_tune.yaml", CONFIGS / "e3b_branches_eval.yaml", "e3a_ext",
           "evaluation_summary.json")
F1 = Stage("F1", FOLLOWUP_SPECTRAL, FOLLOWUP_SPECTRAL + PREREG_SPECTRAL + OTHER_ARMS,
           FOLLOWUP_SPECTRAL + PREREG_SPECTRAL,
           CONFIGS / "f1a_branches_tune.yaml", CONFIGS / "f1b_branches_eval.yaml", "f1a_ext",
           "evaluation_summary_followup.json")


def load(seeds: set[int], arms: list[str]) -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        if path.name.startswith("evaluation_summary"):
            continue
        record = json.loads(path.read_text())
        cfg, summary = record["config"], record["summary"]
        arm = cfg["name"].split("_")[0]
        if cfg["seed"] not in seeds or arm not in arms:
            continue
        rows.append({"arm": arm, "B": cfg["batch_size"], "m": cfg["optimizer"]["momentum"],
                     "x": cfg["branch"]["lr_multiplier"], "seed": cfg["seed"], "diverged": summary["diverged"],
                     "tail": summary["tail_mean_excess_risk"] if not summary["diverged"] else float("inf")})
    return rows


def tuning_options(stage: Stage):
    """(arm, B, m) -> sorted [(multiplier, mean tail over tuning seeds, number of tuning seeds run)]."""
    cells = defaultdict(list)
    for r in load(TUNING_SEEDS, stage.tuned_arms):
        cells[(r["arm"], r["B"], r["m"], r["x"])].append(r["tail"])
    options = defaultdict(list)
    for (arm, B, m, x), tails in cells.items():
        options[(arm, B, m)].append((x, statistics.fmean(tails), len(tails)))
    return {key: sorted(v) for key, v in options.items()}


def arm_settings(stage: Stage) -> tuple[dict, dict]:
    tune = yaml.safe_load(stage.tune_spec.read_text())
    return tune, {arm["label"]: arm["set"] for arm in tune["arms"]}


def ordered(keys, arms: list[str]):
    return sorted(keys, key=lambda k: (arms.index(k[0]), k[1], k[2]))


def pending(stage: Stage, options: dict, new_values: int = 2):
    """Protocol Section 6, checked by both `extend` and `select`. Returns three lists:
    unfinished  (arm, B, m, x) not yet run on every tuning seed;
    to_extend   (arm, B, m, best x, next multipliers) where the best is on an edge of the grid tried so far;
    capped      the same, but already at the [1/64, 64] limit (the stopping rule accepts these)."""
    unfinished = [(*key, x) for key in ordered(options, stage.tuned_arms) for x, _, n in options[key]
                  if n < len(TUNING_SEEDS)]
    to_extend, capped = [], []
    for (arm, B, m) in ordered(options, stage.tuned_arms):
        opts = options[(arm, B, m)]
        xs, best_x = [x for x, *_ in opts], min(opts, key=lambda t: t[1])[0]
        if best_x not in (xs[0], xs[-1]):
            continue
        factor = 0.5 if best_x == xs[0] else 2.0
        new = [x for x in (best_x * factor**k for k in range(1, new_values + 1)) if 1 / MAX_FACTOR <= x <= MAX_FACTOR]
        (to_extend if new else capped).append((arm, B, m, best_x, new))
    return unfinished, to_extend, capped


def extension_configs(stage: Stage) -> list[Path]:
    """This stage's extension configs, oldest first."""
    paths = CONFIGS.glob(f"{stage.ext_prefix}*_branches_tune.yaml")
    return sorted(paths, key=lambda p: int(p.name[len(stage.ext_prefix):].split("_")[0]))


def extend(stage: Stage, new_values: int = 2) -> Path | None:
    """Returns the extension config to run next, or None when every edge is resolved."""
    unfinished, to_extend, capped = pending(stage, tuning_options(stage), new_values)
    if unfinished:
        arm, B, m, x = unfinished[0]
        sys.exit(f"{len(unfinished)} tuning runs are unfinished (e.g. {arm} B={B} m={m:g} x={x:g}): let the sweep "
                 "finish, or re-run its config (finished runs are skipped), then extend again")
    if capped:
        print(f"still on an edge but at the [1/{MAX_FACTOR}, {MAX_FACTOR}] limit (accepted, report it): "
              + "; ".join(f"{arm} B={B} m={m:g} (best x={best_x:g})" for arm, B, m, best_x, _ in capped))
    if not to_extend:
        print("no further extension needed: run `select`" + (" --followup" if stage is F1 else ""))
        return None
    tune, settings = arm_settings(stage)
    arms_out = [{"label": arm, "set": copy.deepcopy(settings[arm]),
                 "sweep": {"branch.lr_multiplier": new, "batch_size": [B], "optimizer.momentum": [m]}}
                for arm, B, m, _, new in to_extend]
    runs = sum(len(new) for *_, new in to_extend) * len(TUNING_SEEDS)
    existing = extension_configs(stage)
    if existing and yaml.safe_load(existing[-1].read_text())["arms"] == arms_out:  # written, not yet run
        print(f"{existing[-1].relative_to(REPO)} already lists these {runs} runs, and they have not been run. Run:\n"
              f"  python code/run.py {existing[-1].relative_to(REPO)} --workers 4\nthen extend again")
        return existing[-1]
    k = len(existing) + 1
    n_low = sum(new[0] < best_x for *_, best_x, new in to_extend)
    path = CONFIGS / f"{stage.ext_prefix}{k}_branches_tune.yaml"
    spec = {"experiment": tune["experiment"], "base": tune["base"], "grid": {"seed": sorted(TUNING_SEEDS)},
            "arms": arms_out}
    header = (f"# {stage.name} tuning extension {k} (generated by `code/analyze_branches.py extend`, "
              f"protocol Section 6):\n# {n_low} cells at the low edge and {len(to_extend) - n_low} at the high edge, "
              f"so {new_values} more multipliers beyond each edge are tried on the tuning seeds.\n")
    path.write_text(header + yaml.safe_dump(spec, sort_keys=False))
    print(f"{n_low} low-edge and {len(to_extend) - n_low} high-edge cells -> wrote {path.relative_to(REPO)} "
          f"({runs} runs)")
    return path


def select(stage: Stage) -> None:
    options = tuning_options(stage)
    unfinished, to_extend, _ = pending(stage, options)
    if unfinished or to_extend:  # protocol Section 6: choose only once every edge is resolved or at the limit
        sys.exit(f"not selecting yet: {len(unfinished)} tuning runs unfinished, {len(to_extend)} cells with the best "
                 "multiplier on an edge that can still be extended.\nNext: python code/analyze_branches.py extend"
                 + (" --followup" if stage is F1 else "") + "   (it says which config to run)")
    tune, settings = arm_settings(stage)
    spec = {"experiment": tune["experiment"], "base": tune["base"], "grid": {"seed": sorted(EVAL_SEEDS)}, "arms": []}
    print(f"{'arm':7s} {'B':>5s} {'m':>5s}  best x   edge   mean tail per multiplier")
    for (arm, B, m) in ordered(options, stage.tuned_arms):
        opts = options[(arm, B, m)]
        best_x = min(opts, key=lambda t: t[1])[0]
        edge = "LOW" if best_x == opts[0][0] else "HIGH" if best_x == opts[-1][0] else ""
        print(f"{arm:7s} {B:5d} {m:5g}  {best_x:6g}  {edge:>5s}   " + "  ".join(f"{x:g}:{t:.5f}" for x, t, _ in opts))
        spec["arms"].append({"label": arm, "set": copy.deepcopy(settings[arm]),
                             "sweep": {"branch.lr_multiplier": [best_x], "batch_size": [B], "optimizer.momentum": [m]}})
    header = (f"# {stage.name} evaluation (generated by `code/analyze_branches.py select`): each arm uses the\n"
              "# learning-rate multiplier chosen on tuning seeds {0, 1}; evaluated on seeds {2, 3, 4}.\n")
    stage.eval_spec.write_text(header + yaml.safe_dump(spec, sort_keys=False))
    print(f"\nwrote {stage.eval_spec.relative_to(REPO)} ({len(spec['arms'])} arms x {len(EVAL_SEEDS)} seeds)")


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


def selected_multipliers(stage: Stage) -> dict:
    """(arm, B, m) -> the multiplier `select` chose, read from the evaluation configs (E3's, plus F1's for
    --followup). Evaluation runs at any other multiplier, e.g. from a superseded selection, stay on disk unused."""
    chosen = {}
    for spec_path in (E3.eval_spec, F1.eval_spec) if stage is F1 else (E3.eval_spec,):
        if not spec_path.exists():
            continue
        for arm in yaml.safe_load(spec_path.read_text())["arms"]:
            sweep = arm["sweep"]
            chosen[(arm["label"], sweep["batch_size"][0], sweep["optimizer.momentum"][0])] = \
                sweep["branch.lr_multiplier"][0]
    return chosen


def stale_selections(options: dict, chosen: dict, arms: list[str]) -> list[tuple]:
    """(arm, B, m, multiplier in the evaluation config or None, best multiplier on the tuning seeds now) for
    every tuned cell where the two differ, e.g. because `select` ran before the last extension was run."""
    stale = []
    for key in ordered(options, arms):
        best_x = min(options[key], key=lambda t: t[1])[0]
        if chosen.get(key) != best_x:
            stale.append((*key, chosen.get(key), best_x))
    return stale


def evaluate(stage: Stage, bootstrap: int = 5000) -> None:
    chosen = selected_multipliers(stage)
    for tuned in (E3, F1) if stage is F1 else (E3,):  # every compared arm: tuning finished, then selected
        options, flag = tuning_options(tuned), " --followup" if tuned is F1 else ""
        unfinished, to_extend, _ = pending(tuned, options)
        if unfinished or to_extend:
            sys.exit(f"not evaluating: {tuned.name} tuning is not finished ({len(unfinished)} runs unfinished, "
                     f"{len(to_extend)} cells on an edge that can still be extended).\n"
                     f"Next: python code/analyze_branches.py extend{flag}   (it says which config to run)")
        stale = stale_selections(options, chosen, tuned.tuned_arms)
        if stale:
            arm, B, m, x, best_x = stale[0]
            have = "none" if x is None else f"{x:g}"
            sys.exit(f"not evaluating: {tuned.name}'s evaluation config predates the latest tuning runs in {len(stale)} "
                     f"cells (e.g. {arm} B={B} m={m:g}: config has x={have}, tuning best is x={best_x:g}).\n"
                     f"Next: python code/analyze_branches.py select{flag}, then "
                     f"python code/run.py {tuned.eval_spec.relative_to(REPO)} --workers 4")
    table = defaultdict(dict)  # (B, m) -> arm -> {seed: tail}
    for r in load(EVAL_SEEDS, stage.analysed_arms):
        if r["x"] == chosen.get((r["arm"], r["B"], r["m"])):
            table[(r["B"], r["m"])].setdefault(r["arm"], {})[r["seed"]] = r["tail"]
    rng = np.random.default_rng(0)
    summary = {"stage": stage.name, "conditions": [], "contrast": []}
    if stage is F1:
        print("FOLLOW-UP analysis (F1, not pre-registered): arms p-0.75 and p-0.5 added to the E3 arms")
    for (B, m) in sorted(table):
        arms = table[(B, m)]
        missing = [a for a in stage.spectral if a not in arms]
        if missing:
            print(f"\nB = {B}, momentum = {m:g}: skipped, no evaluation results yet for {missing}")
            continue
        seeds = sorted(set.intersection(*(set(v) for v in arms.values())))
        if len(seeds) < len(EVAL_SEEDS):
            print(f"\nB = {B}, momentum = {m:g}: skipped, evaluation runs unfinished (seeds done on every arm: {seeds})")
            continue
        base = arms["p0"]
        print(f"\n=== B = {B}, momentum = {m:g}  (seeds {seeds}; Xie delta at switch = {switch_delta(B, m)}) ===")
        print(f"{'arm':7s} {'mean excess':>12s}   {'diff vs p0 [95% CI]':>34s}")
        cond = {"B": B, "m": m, "delta_at_switch": switch_delta(B, m), "arms": {}}
        for arm in stage.analysed_arms:
            if arm not in arms:
                continue
            values = [arms[arm][s] for s in seeds]
            diffs = [arms[arm][s] - base[s] for s in seeds]
            d_mean, d_lo, d_hi = t_interval(diffs) if arm != "p0" else (0.0, 0.0, 0.0)
            cond["arms"][arm] = {"mean": statistics.fmean(values), "values": values, "diff_vs_p0": [d_mean, d_lo, d_hi]}
            print(f"{arm:7s} {statistics.fmean(values):12.5f}   {d_mean:+.5f} [{d_lo:+.5f}, {d_hi:+.5f}]")
        matrix = np.array([[arms[a][s] for s in seeds] for a in stage.spectral])  # paired over seeds
        p_star = stage.spectral[int(np.argmin(matrix.mean(axis=1)))]
        counts = defaultdict(int)
        for _ in range(bootstrap):
            idx = rng.integers(0, len(seeds), len(seeds))
            counts[stage.spectral[int(np.argmin(matrix[:, idx].mean(axis=1)))]] += 1
        cond["p_star"] = P_VALUE[p_star]
        cond["p_star_bootstrap"] = {a: counts[a] / bootstrap for a in stage.spectral}
        print(f"p* = {P_VALUE[p_star]:g}; bootstrap share as winner: "
              + ", ".join(f"{a}: {counts[a] / bootstrap:.2f}" for a in stage.spectral))
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
            record = json.loads(path.read_text())
            cfg = record["config"]
            if cfg["seed"] in EVAL_SEEDS:
                frozen[(cfg["batch_size"], cfg["optimizer"]["momentum"])][cfg["seed"]] = \
                    record["summary"]["tail_mean_excess_risk"]
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
    p_stars = [c["p_star"] for c in summary["conditions"]]
    if len(deltas) == len(summary["conditions"]) >= 3:
        if len(set(p_stars)) == 1:
            print(f"\nH3 not testable: p* = {p_stars[0]:g} in every condition (no variation to correlate with delta)")
            summary["h3_spearman"] = {"rho": None, "note": "p* constant across conditions"}
        else:
            rho, pval = stats.spearmanr(deltas, p_stars)
            summary["h3_spearman"] = {"rho": float(rho), "p_value": float(pval), "n_conditions": len(deltas)}
            print(f"\nH3: Spearman correlation between delta at switch and p*: rho = {rho:+.2f} (p = {pval:.3f}, "
                  f"n = {len(deltas)} conditions; H3 predicts rho < 0)")
    out = RESULTS / stage.summary_name
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out.relative_to(REPO)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["extend", "select", "evaluate"])
    parser.add_argument("--followup", action="store_true", help="the F1 follow-up stage (p = -0.5, -0.75)")
    args = parser.parse_args()
    if not RESULTS.exists():
        sys.exit(f"no results in {RESULTS}")
    stage = F1 if args.followup else E3
    {"extend": extend, "select": select, "evaluate": evaluate}[args.mode](stage)


if __name__ == "__main__":
    main()

"""Tables of the report (see report/outline.md), written as LaTeX for report/main.tex to \\input.

Usage:  python code/make_tables.py
Reads results/e3_branches/evaluation_summary*.json and the evaluation configs (configs/e3b_*, configs/f1b_*).
Writes report/tables/table2_conditions.tex and tableA1_multipliers.tex. Every number comes from these
files, so the tables follow the results without retyping.
"""

import json
from pathlib import Path

import analyze_branches as ab
from plot_e3 import relative_change

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "report" / "tables"


def multiplier(x: float) -> str:
    return f"{x:g}" if x >= 1 else f"1/{round(1 / x)}"


def table_conditions() -> str:
    e3 = {(c["B"], c["m"]): c for c in json.loads((ab.RESULTS / ab.E3.summary_name).read_text())["conditions"]}
    f1 = {(c["B"], c["m"]): c for c in json.loads((ab.RESULTS / ab.F1.summary_name).read_text())["conditions"]}
    rows = [f"{B} & {m:g} & {c['delta_at_switch']:.2f} & ${c['p_star']:g}$ & ${f1[(B, m)]['p_star']:g}$ & "
            + " & ".join(f"${relative_change(c, arm)[0]:+.1f}$" for arm in ("p-0.25", "p1", "xie")) + r" \\"
            for (B, m), c in sorted(e3.items())]
    return "\n".join([
        r"\begin{tabular}{@{}rrrrrrrr@{}}", r"\toprule",
        r" & & & \multicolumn{2}{c}{Best $p$} & \multicolumn{3}{c}{Change vs.\ Muon (\%)} \\",
        r"\cmidrule(lr){4-5}\cmidrule(l){6-8}",
        r"$B$ & $\beta$ & $\delta$ & E3 & F1 & $p=-0.25$ & $p=1$ & Xie \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", ""])


def table_multipliers() -> str:
    chosen = ab.selected_multipliers(ab.F1)  # E3's and F1's evaluation configs
    arms = ab.FOLLOWUP_SPECTRAL + ab.PREREG_SPECTRAL + ab.OTHER_ARMS
    rows = [f"{B} & {m:g} & " + " & ".join(multiplier(chosen[(arm, B, m)]) for arm in arms) + r" \\"
            for B, m in sorted({(B, m) for _, B, m in chosen})]
    return "\n".join([
        r"\begin{tabular}{@{}rrrrrrrrrr@{}}", r"\toprule",
        r" & & \multicolumn{6}{c}{Spectral exponent $p$} & & \\", r"\cmidrule(lr){3-8}",
        r"$B$ & $\beta$ & $-0.75$ & $-0.5$ & $-0.25$ & 0 & 0.5 & 1 & NS5 & Xie \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", ""])


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, text in (("table2_conditions", table_conditions()), ("tableA1_multipliers", table_multipliers())):
        (TABLES / f"{name}.tex").write_text(text)
        print(f"wrote report/tables/{name}.tex")


if __name__ == "__main__":
    main()

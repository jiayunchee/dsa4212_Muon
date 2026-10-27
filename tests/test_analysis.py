"""Protocol Section 6 tuning rules in code/analyze_branches.py: which multiplier grids get extended, when the
stopping rule accepts an edge, that unfinished tuning runs are caught (both block `select`), and that a
selection made before the last extension is caught (blocks `evaluate`)."""

import analyze_branches as ab


def cell(*tails, xs=(0.5, 1.0, 2.0), seeds=2):
    """One (arm, B, m) cell as tuning_options returns it: [(multiplier, mean tail, seeds run)]."""
    return [(x, t, seeds) for x, t in zip(xs, tails)]


def test_edge_cells_are_extended_by_halving_or_doubling_within_the_limit():
    options = {("p-0.5", 16, 0.9): cell(3.0, 2.0, 1.0),                              # best on the high edge
               ("p-0.5", 64, 0.9): cell(1.0, 2.0, 3.0, xs=(1 / 32, 1 / 16, 1 / 8)),  # low edge, one step from 1/64
               ("p-0.5", 256, 0.9): cell(2.0, 1.0, 3.0)}                             # interior: nothing to do
    unfinished, to_extend, capped = ab.pending(ab.F1, options)
    assert unfinished == [] and capped == []
    assert to_extend == [("p-0.5", 16, 0.9, 2.0, [4.0, 8.0]), ("p-0.5", 64, 0.9, 1 / 32, [1 / 64])]


def test_edge_at_the_limit_is_accepted():
    options = {("p-0.75", 16, 0.9): cell(3.0, 2.0, 1.0, xs=(16.0, 32.0, 64.0))}
    _, to_extend, capped = ab.pending(ab.F1, options)
    assert to_extend == [] and capped == [("p-0.75", 16, 0.9, 64.0, [])]


def test_unfinished_runs_are_reported():
    options = {("p-0.5", 16, 0.9): [(1.0, 2.0, 2), (2.0, 1.5, 2), (4.0, 1.0, 1)]}
    unfinished, _, _ = ab.pending(ab.F1, options)
    assert unfinished == [("p-0.5", 16, 0.9, 4.0)]


def test_selection_made_before_the_last_extension_is_stale():
    options = {("p-0.5", 16, 0.9): cell(2.0, 1.0, 3.0),                                # selection still right
               ("p-0.5", 64, 0.9): cell(3.0, 2.0, 1.5, 1.0, xs=(1.0, 2.0, 4.0, 8.0))}  # best moved to x8
    chosen = {("p-0.5", 16, 0.9): 1.0, ("p-0.5", 64, 0.9): 2.0}
    assert ab.stale_selections(options, chosen, ab.F1.tuned_arms) == [("p-0.5", 64, 0.9, 2.0, 8.0)]
    assert ab.stale_selections(options, {}, ab.F1.tuned_arms)[0] == ("p-0.5", 16, 0.9, None, 1.0)  # never selected

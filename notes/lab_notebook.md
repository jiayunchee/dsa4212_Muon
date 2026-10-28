# Lab notebook

A dated record of what was done, what was observed and what was decided. Newest entries go at the top. Measured facts and interpretations are kept apart.

## 2026-10-28: F1 finished; p* moves from −0.25 (B ≤ 64) to −0.5 (B ≥ 256)

F1 is exploratory: it was added after E3's results (protocol change log, 27 Oct). Nothing here changes the pre-registered E3 verdicts.

### Done

- Ran `code/run_branches.py --followup` in the background, 10:26–10:52. 97 runs, none failed or diverged; console log in `results/f1_console.txt`. The 26 unit tests passed before the run.
  - `f1a_ext2`: 48 runs, ×16 and ×32 for the 12 cells on the high edge (10:26–10:38).
  - `extend` wrote `f1a_ext3`: 4 runs at ×64, for p = −0.75 at B = 256 and B = 1024, β = 0.9. In both, ×64 was worse than ×32. After it no edges remained and no cell sat at the limit.
  - `select --followup` rewrote `configs/f1b_branches_eval.yaml` (10:39).
  - 45 evaluation runs (10:39–10:52). The 3 at p = −0.5, B = 16, β = 0.95 were reused, because that cell stayed at ×2.
  - Then `evaluate --followup` and the figures.
- Afterwards, neither E3 nor F1 had pending or stale cells.

### Selected multipliers (tuning seeds 0, 1)

| B | β | p = −0.75 | p = −0.5 | p = −0.25 (E3) |
|---|---|---|---|---|
| 16 | 0.9 | ×4 | ×4 | ×2 |
| 16 | 0.95 | ×4 | ×2 | ×1 |
| 64 | 0.9 | ×16 | ×8 | ×8 |
| 64 | 0.95 | ×16 | ×8 | ×4 |
| 256 | 0.9 | ×32 | ×16 | ×8 |
| 256 | 0.95 | ×16 | ×8 | ×4 |
| 1024 | 0.9 | ×32 | ×16 | ×4 |
| 1024 | 0.95 | ×16 | ×8 | ×2 |

### Results (measured; evaluation seeds 2–4)

Each cell is the change in tail excess risk against continuing Muon: the mean of the per-seed % change, as plotted in `e3_relative_change_followup`. p = 0.5, p = 1, Xie and NS5 are the E3 runs, so their numbers are unchanged.

| B | β | δ at switch | p = −0.75 | p = −0.5 | p = −0.25 | p* |
|---|---|---|---|---|---|---|
| 16 | 0.9 | 0.69 | +7.4% | +1.2% | −2.4% | −0.25 |
| 16 | 0.95 | 1.10 | +3.0% | −1.6% | −2.9% | −0.25 |
| 64 | 0.9 | 1.16 | +10.0% | +1.2% | −2.8% | −0.25 |
| 64 | 0.95 | 1.84 | +6.7% | −2.3% | −4.0% | −0.25 |
| 256 | 0.9 | 1.77 | +5.2% | −6.7% | −4.6% | −0.5 |
| 256 | 0.95 | 2.76 | −0.3% | −9.0% | −7.6% | −0.5 |
| 1024 | 0.9 | 2.62 | −4.0% | −7.7% | −6.5% | −0.5 |
| 1024 | 0.95 | 4.09 | −4.5% | −7.6% | −5.2% | −0.5 |

- **p\*** is −0.25 at B = 16 and 64 and −0.5 at B = 256 and 1024, at both momenta. Each won every bootstrap resample (share 1.00).
- **The optimum is now bracketed in every condition:** the winner beats both of its neighbours on the p grid.
- **p = −0.5 minus p = −0.25,** paired over seeds, relative to p = −0.25 (95% t-interval):
  - B = 16: +3.7% [+3.1, +4.3] at β = 0.9; +1.4% [+1.0, +1.7] at β = 0.95.
  - B = 64: +4.1% [+3.1, +5.1]; +1.8% [+1.3, +2.3].
  - B = 256: −2.2% [−3.4, −1.1]; −1.5% [−3.7, +0.6].
  - B = 1024: −1.3% [−1.8, −0.8]; −2.4% [−4.6, −0.3].
  - 7 of the 8 intervals exclude 0. Within each condition, all three seeds have the same sign.
- **H3's correlation, computed here as an exploratory result:** Spearman ρ(δ at switch, p\*) = −0.76. That is p = 0.027 asymptotically, and p = 0.029 by exact one-sided permutation (2 of 70 splits). H3 predicted ρ < 0.
- **Batch size:** Spearman ρ(log B, p\*) = −0.89 (p = 0.003), also post hoc.
- **The pre-registered H1 contrast is unchanged,** since it uses only p = 1 and p = −0.25: contradicted at both momenta.

### Interpretation (tentative; exploratory)

- **With step sizes tuned per p, the best late-phase exponent depends on batch size.** Flatten more (p = −0.5) at B ≥ 256, and slightly less (p = −0.25) at B ≤ 64.
  - That is H1's direction: p\* rises with noise.
  - But the shift is far smaller than H1 claimed. p\* stays at or below −0.25 and never moves towards the SGD-like end.
  - So H1 as stated stays contradicted. Only the direction of the dependence has exploratory support.
- **The shift tracks batch size more closely than δ.**
  - At B = 64, β = 0.95 (δ = 1.84), p\* = −0.25. At B = 256, β = 0.9 (δ = 1.77), it is −0.5.
  - At each B, raising β from 0.9 to 0.95 raises δ by 56–60% without changing p\*.
  - With p\* on a 0.25-spaced grid and only two levels observed, δ's role cannot be separated from B's.
- **The differences are small.** p = −0.5 and p = −0.25 differ by 1–4%, against 12–42% to the SGD-like arms. Practical reading: flatten at least as much as Muon, a little more at large batch.
- **The step-size tuning decided the answer.**
  - At ×2 (the provisional evaluation of 27 Oct), p = −0.5 looked up to 43% worse than Muon.
  - At its tuned step size it is between 1.2% worse and 9.0% better.
  - More negative p needed much larger multipliers: ×8–×16 for p = −0.5 and ×16–×32 for p = −0.75 at B ≥ 64.
- **Caveats:** not pre-registered; 3 evaluation seeds; a p grid with 0.25 resolution; one task, the teacher–student MLP.

### Next

- **Report:** F1 goes in the main text as an exploratory result, kept clearly apart from the pre-registered E3 tests. The current F1 figure is dominated by the p = 1 end. A zoomed panel for p ≤ 0, or p\* against B, would show the shift better.
- **Timing:** re-run the benchmark to check NS5's step time at B = 64 (27 Oct morning entry).
- **E5 (deep noise):** only if time allows; see the 27 Oct morning entry.

## 2026-10-27: follow-up F1 (p = −0.5, −0.75); selection made before tuning finished

The E1, E2, E3 and timing work is in the two entries below this one (26 Oct and the morning of 27 Oct). Both were backfilled on the evening of 27 Oct. No work was done on 21–25 Oct.

### Done

- Started F1 (protocol change log, 27 Oct). E3 put p* = −0.25 in all 8 conditions, the most negative exponent tested. F1 adds p = −0.5 and p = −0.75 from the same E2 checkpoints.
- Ran the F1 tuning: `configs/f1a_branches_tune.yaml`, 96 runs, multipliers {0.5, 1, 2}, seeds {0, 1}.
- Ran the F1 evaluation (48 runs), `evaluate --followup` and `plot_e3.py`. The evaluation turned out to be premature; see below.

### Problem found and fixed (kept on record)

**The F1 multipliers were selected before the extension runs existed.**
- 14:43:43: `extend --followup` found the best multiplier on the high edge (×2) in all 16 cells (2 arms × 8 conditions). It wrote `configs/f1a_ext1_branches_tune.yaml`: ×4 and ×8, 64 runs.
- 14:43:46: `select --followup` ran before any of those runs. It chose ×2 everywhere.
- The 48 evaluation runs and the follow-up summary all used ×2.
- The ×4/×8 runs had not been run as of 15:55.

**Why it matters.**
- At ×2 the tuning curves were still falling: −2% to −13% in tail excess risk from ×1 to ×2. The drop was steeper for p = −0.5 and at larger B.
- In E3, p = −0.25's best multiplier was above ×2 in 5 of 8 cells (×4 or ×8). There it beat ×2 by 4.5–16%.
- So the new arms were compared before their step size was found. This is the step-size/shape confound that the per-p tuning exists to remove (Shumaylov et al.; protocol Sections 2 and 6).

**Fixes in `code/analyze_branches.py`.**
- `select` now refuses while any tuning run is unfinished or any best multiplier sits on an edge that can still be extended. The check is `pending()`, shared with `extend`.
- `extend` no longer writes a duplicate config when the newest one has been written but not run. It names that config instead.
- `evaluate` used every evaluation-seed run of an arm. After a re-selection, the old ×2 runs and the new runs would have been mixed in filename order. It now uses only runs at the multipliers in the evaluation configs (`selected_multipliers()`).
- `evaluate` also skips a condition whose evaluation runs are unfinished, instead of quietly using fewer seeds.

**Checks.**
- 3 new tests in `tests/test_analysis.py`; all 25 tests pass.
- On the current data, `evaluate` and `evaluate --followup` reproduce the previous summaries byte for byte.
- E3 has no unfinished runs and no open edges, so its selection is unaffected.
- `select --followup` now exits with an error and writes nothing. `extend --followup` names `f1a_ext1`.

### Provisional numbers (superseded; the new arms at ×2)

Copied to `results/e3_branches/evaluation_summary_followup_provisional_x2.json`. The 48 evaluation runs at ×2 stay in `results/e3_branches/`.

- p* = −0.25 in all 8 conditions, with a bootstrap share of 1.00.
- p = −0.5 against continued Muon: −1.6% at B = 16, β = 0.95; +3% to +43% elsewhere.
- p = −0.75: +7% to +69% everywhere.

Interpretation: these figures likely overstate the new arms' excess risk. Whether the optimum is really bracketed at −0.25 is open until the extension runs are done. On evaluation means, p = −0.5 trails p = −0.25 by 1.4% at B = 16, β = 0.95, and by 6–50% elsewhere.

### Figures

- `e3_relative_change(_followup)`:
  - the x tick labels for p = −0.75, −0.5 and −0.25 overlapped, so they are now smaller;
  - the legend moved below the panels, clear of the data.
- `e4_delta_vs_batch`:
  - the y ticks are now plain numbers (they were in 2×10⁰ style);
  - the legend is gone, since the lines are labelled directly;
  - the δ = 1 note moved to the bottom right (it sat under the B = 16, β = 0.95 point).

### Next (superseded by the list at the end of this entry)

1. Run `configs/f1a_ext1_branches_tune.yaml` (64 runs). Repeat `extend --followup`, running each config it names, until it reports no further extension.
2. Run `select --followup`, then `configs/f1b_branches_eval.yaml` (runs at an unchanged multiplier are skipped), `evaluate --followup` and `plot_e3.py`.

### Later on 27 Oct: ×4/×8 runs done, 12 cells still on the edge, the old selection evaluated again

**What happened.**
- 17:22–17:38: `f1a_ext1` ran (64 runs).
- 17:44:01: `extend --followup` wrote `configs/f1a_ext2_branches_tune.yaml`. It tries ×16 and ×32 for the 12 cells whose best is now ×8 (48 runs). The other 4 cells now have an interior best: p = −0.75 at B = 16 (×4 at both β), and p = −0.5 at B = 16 (×4 at β = 0.9, ×2 at β = 0.95).
- 17:44:47: `evaluate --followup` ran without a new `select` (`f1b_branches_eval.yaml` unchanged since 14:43:46). It used the ×2 selection again, and its summary is byte-identical to the provisional copy (same md5).

**Fix.** `evaluate` now refuses in two cases:
- the tuning of any arm it compares is unfinished or still has an edge to extend (the same `pending()` check as `select`);
- a multiplier in an evaluation config is no longer the best on the tuning seeds (`stale_selections()`).

New: `code/run_branches.py [--followup]` runs a stage in protocol order: tuning, extensions until resolved, `select`, evaluation runs, `evaluate`, figures.

**Checks.**
- 1 new test; all 26 tests pass.
- E3: 0 of 48 cells stale, so the pre-registered evaluation is unaffected.
- F1: 15 of 16 cells stale. `evaluate --followup` and `select --followup` both refuse and write nothing.
- `run_branches.py` on E3 (0 runs left): summary byte-identical. `e3b_branches_eval.yaml` was regenerated with the current header wording (2 comment lines); its multipliers are unchanged.

**Preview on the tuning seeds only.** Each cell shows the best multiplier so far and the change in mean tail excess risk against p = 0's best. This is not the evaluation: taking the minimum over multipliers on the same seeds is optimistic, and 12 cells are still on the edge.

| B | β | p = −0.75 | p = −0.5 | p = −0.25 (E3) |
|---|---|---|---|---|
| 16 | 0.9 | ×4, +6.6% | ×4, +0.7% | ×2, −2.7% |
| 16 | 0.95 | ×4, +2.9% | ×2, −1.6% | ×1, −2.9% |
| 64 | 0.9 | ×8\*, +16.8% | ×8\*, +0.2% | ×8, −2.8% |
| 64 | 0.95 | ×8\*, +10.9% | ×8\*, −3.9% | ×4, −4.7% |
| 256 | 0.9 | ×8\*, +33.8% | ×8\*, +0.6% | ×8, −5.2% |
| 256 | 0.95 | ×8\*, +13.7% | ×8\*, −8.0% | ×4, −7.1% |
| 1024 | 0.9 | ×8\*, +26.6% | ×8\*, −3.0% | ×4, −5.0% |
| 1024 | 0.95 | ×8\*, +5.2% | ×8\*, −6.6% | ×2, −4.3% |

\* still on the high edge.

**Reading.**
- On these seeds, p = −0.5 is already ahead of p = −0.25 in 2 cells: B = 256 and B = 1024, both at β = 0.95.
- In all six B ≥ 64 cells it was still improving from ×4 to ×8, by 4.0–16.4%; p = −0.75 by 6.9–14.9%.
- At B = 16 both new arms have an interior optimum and trail p = −0.25.
- If this holds on the evaluation seeds, p* is no longer constant: it is more negative at large B and high δ. Then H3's correlation can be computed, but only as an exploratory result, since F1 was not pre-registered. The pre-registered E3 verdicts stand either way.
- The first F1 reading ("p = −0.5 is 3–43% worse than Muon") was largely a step-size artefact.
- A plausible reason, still to check: with ‖D‖_F fixed at √rank, a more negative p moves update size from the strong directions to the weak ones, so it needs a larger multiplier to make the same progress along the strong directions.

### Next (supersedes the list above)

1. `caffeinate -i python code/run_branches.py --followup`. This runs `f1a_ext2` (48 runs), any further extension (at most up to ×64), `select`, the changed `f1b` runs (45), `evaluate --followup` and the figures, in about 30–40 min. Done on 28 Oct (entry above).
2. Done the same evening: the 26 Oct and 27 Oct (morning) entries below were backfilled.

## 2026-10-27, morning (before F1): E3 tuning finished, frozen control, E3 evaluation, step timing

*Backfilled on the evening of 27 Oct from the result files, configs, commits and the session record. Times are when the result files were written.*

### Done

- **E3 tuning, extension rounds 1–3** (protocol Section 6; round 0 is in the 26 Oct entry). The number of cells whose best multiplier sat on an edge went:
  - `e3a_ext1` (148 runs, 10:55–11:23): 37 → 7 (5 low, 2 high).
  - Then the stopping rule was fixed, before round 2 ran (protocol change log, 27 Oct): multipliers stay within [1/64, 64]. Round 2's new values (16, 32, 1/16, 1/32) were all inside it.
  - `e3a_ext2` (28 runs, 11:26–11:33): 7 → 3. All three were low edges at 1/32: p = 1 at B = 16 (both β), and Xie at B = 16, β = 0.95.
  - `e3a_ext3` (6 runs, 11:37): 3 → 0, with no cell left at the limit.
  - None of the 182 extension runs diverged. There were 3 float64 SVD retries in all (2 in ext1, 1 in ext2).
- **`select`** (11:38) wrote `configs/e3b_branches_eval.yaml`. Re-checked on the evening of 27 Oct: its multipliers match the tuning best in all 48 cells.
- **Exploratory control** `configs/e3c_control_frozen.yaml` (40 runs, 11:38–11:42, 0 diverged). Every E2 checkpoint was continued with the hidden matrices frozen. It was added after seeing the tuning results (protocol change log, 27 Oct), so it is not pre-registered.
- **E3b, the pre-registered evaluation** (144 runs = 48 cells × seeds 2–4, 11:45–12:21, 0 diverged), then `evaluate` and `plot_e3.py`. Committed `a196b45` (13:58) and pushed.
- **Step timing.**
  - `benchmark_step_time.py` (14:00; 1 thread, 300 steps, clean tree at `a196b45`) wrote `results/timing/step_time_threads1.json`.
  - The new `code/analyze_time.py` (14:04) wrote `results/timing/per_second_summary.json` and the `e1_vs_time` figure.

### E3 results (measured; pre-registered analysis, evaluation seeds 2–4)

Each cell is the change in tail excess risk against continuing Muon (p = 0), as the mean over seeds of the per-seed % change. This is what `e3_relative_change` plots. Negative is better than Muon.

| B | β | δ at switch | p = −0.25 | p = 0.5 | p = 1 | Xie switch | NS5 Muon |
|---|---|---|---|---|---|---|---|
| 16 | 0.9 | 0.69 | −2.4% | +11.5% | +15.7% | +15.6% | −0.2% |
| 16 | 0.95 | 1.10 | −2.9% | +8.6% | +11.7% | +11.8% | +0.2% |
| 64 | 0.9 | 1.16 | −2.8% | +14.4% | +23.6% | +23.5% | −1.3% |
| 64 | 0.95 | 1.84 | −4.0% | +15.1% | +24.1% | +23.7% | +0.3% |
| 256 | 0.9 | 1.77 | −4.6% | +21.2% | +41.5% | +37.4% | +1.7% |
| 256 | 0.95 | 2.76 | −7.6% | +16.8% | +32.7% | +31.2% | −0.8% |
| 1024 | 0.9 | 2.62 | −6.5% | +15.5% | +35.3% | +33.2% | −0.5% |
| 1024 | 0.95 | 4.09 | −5.2% | +15.2% | +28.3% | +28.3% | −0.3% |

- **p\*:** −0.25 in all 8 conditions. It won every bootstrap resample (share 1.00).
- **H1: contradicted.** C = +0.00877 [+0.00833, +0.00922] at β = 0.9 and +0.00657 [+0.00617, +0.00698] at β = 0.95. H1 predicted C < 0.
- **H2: not supported.** Xie's switch is worse than continuing Muon in every condition, by 11.8–37.4%. That includes the noisiest condition (B = 16, β = 0.9, δ = 0.69).
- **H3: not testable.** p\* is constant, so its correlation with δ is undefined. The summary first stored NaN; since 27 Oct it records "not testable".
- **NS5 vs exact-SVD Muon:** within −1.3% to +1.7%.

### Exploratory results (not pre-registered; added after seeing results)

- **Frozen control (evaluation seeds).**
  - At B = 16, freezing the hidden matrices gives 0.0675 / 0.0610 (β = 0.9 / 0.95).
  - That is level with the SGD-like arms (p = 1: 0.0672 / 0.0606; Xie: 0.0672 / 0.0607), and 0.0094 / 0.0067 worse than continuing Muon.
  - At B = 1024 the SGD-like arms clearly beat freezing. At β = 0.9: p = 1 0.0056 and Xie 0.0055, against 0.0076 frozen.
- **Scale-free H1 contrast.** log[L(p = 1) / L(p = −0.25)] at B = 16, minus the same at B = 1024. It is −0.199 at β = 0.9 and −0.163 at β = 0.95; a negative value is H1's direction.
- **p = −0.25's gain over Muon** (100 × mean paired difference / Muon's mean) runs from 2.4% to 7.6%. It rises with δ: Spearman ρ = +0.88 (p = 0.004, n = 8).

### Timing results (measured; 1 thread, quiet machine)

Milliseconds per step:

| B | AdamW | SGD | NS5 Muon | exact Muon | Xie (plain momentum-SGD) |
|---|---|---|---|---|---|
| 16 | 0.38 | 0.28 | 2.37 | 9.70 | 0.55 |
| 64 | 0.63 | 0.55 | 4.55 | 8.90 | 0.91 |
| 256 | 1.06 | 0.95 | 2.85 | 9.46 | 1.10 |
| 1024 | 3.52 | 3.41 | 6.20 | 11.58 | 3.43 |

Excess risk from the tuned E1 runs, after 6000 steps and at equal wall-clock time. Equal time means the time SGD needs for 6000 steps; the number of steps each Muon fits into it is in brackets:

| B | SGD | NS5 Muon, 6000 steps | NS5 Muon, equal time | exact Muon, equal time |
|---|---|---|---|---|
| 16 | 0.0907 | 0.0546 | 0.1180 (697) | 0.3070 (170) |
| 64 | 0.0417 | 0.0304 | 0.0904 (729) | 0.1062 (373) |
| 256 | 0.0151 | 0.0108 | 0.0390 (1998) | 0.0790 (601) |
| 1024 | 0.0079 | 0.0039 | 0.0072 (3296) | 0.0184 (1765) |

- **Per step,** Muon is best at every batch size.
- **At equal time,** SGD beats both Muons at B = 16–256. NS5 Muon keeps a small lead at B = 1024 (0.0072 vs 0.0079).
- **Caveat (recorded in the script):** the learning rates were tuned for 6000 steps. That is conservative for the slower optimisers, which get fewer steps.

### Interpretation (tentative)

- **Flattening more than Muon wins.** p = −0.25 (DynMuon's late-training rule) beats every other arm in all 8 conditions, over a 64× range of batch size, with every arm's step size tuned.
- **The pre-registered contrast has a design flaw.**
  - It compares absolute gaps, but excess risk is about 14× larger at B = 16 than at B = 1024 (p = 0: 0.0581 vs 0.0042 at β = 0.9). So absolute gaps are larger at B = 16 almost by construction. The report must say so.
  - The scale-free version above is post hoc. In it, the ranking never flips, but flattening's relative benefit shrinks as noise rises.
- **The SGD-like arms barely update the hidden layers at B = 16.** The frozen control suggests their best there amounts to that; their best multipliers at B = 16 are 1/16–1/32. A hypothesis, not shown: the hidden-layer gradients are ill-conditioned, so any SGD-like step large enough to move the weak directions is too large for the dominant ones.
- **We never reached Xie's regime** (δ ≪ 1 near the optimum). The noisiest switch point has δ = 0.69, with the loss still falling.
- **E5 was not triggered.** On 26 Oct we decided to run E5 if H1 came out inconclusive. H1 was contradicted instead, so the trigger did not fire. E5 remains the only way to test H2 in Xie's regime.
- **The optimum is not bracketed.** p\* sits at the edge of the p grid, so we can only say p\* ≤ −0.25. This led to follow-up F1 (entry above).
- **Per second, Muon's per-step lead mostly disappears on this CPU.** Its overhead over SGD is roughly fixed per step: 1.9–4.0 ms for NS5 and 8.2–9.4 ms for exact SVD. At small batch that outweighs the per-step gain.

### Open

- NS5's step time at B = 64 (4.55 ms) is out of line with B = 16 (2.37 ms) and B = 256 (2.85 ms). Re-run the benchmark before the report relies on it.

## 2026-10-26: E1 analysed, E2 shared phase, E3 tuning round 0

*Backfilled on the evening of 27 Oct from the result files, configs, commits and the session record. Times are when the result files were written.*

### Done

- **E1** ran on 20 Oct (10:40–11:25; started in the 20 Oct entry). Task A, 6000 steps, 5 learning rates per optimiser and batch size, seeds 0 and 1.
  - E1a: AdamW and SGD, 80 runs. 2 diverged, both SGD at lr 0.3, B = 16.
  - E1a extension: SGD at B = 1024 with lr 0.6 and 1 (4 runs, all diverged). SGD's best there, 0.3, was the top of its grid.
  - E1b: NS5 and exact-SVD Muon, 80 runs, 0 diverged, 1 float64 SVD retry.
- **E1 analysis** (19:16): `analyze_tuning.py` wrote `results/e1_baselines/tuning_summary.csv`.
  - No best learning rate is on a grid edge. SGD at B = 1024 has its best at the top of the grid, but both values above it diverge.
  - Then `plot_e1.py` drew `e1_tuned_vs_batch`, `e1_curves` and `e1_per_example`.
  - Exact Muon's best rates went into `configs/e2_shared.yaml`: 0.003 at B = 16 and 0.001 at B = 64–1024.
  - Committed `989bf5a` (19:21).
- **E2, the shared phase** (40 runs = 4 B × 2 β × 5 seeds, 19:22–19:28, 0 diverged, 0 float64 SVD retries).
  - Exact-SVD Muon on the hidden matrices for 3000 steps, with AdamW at E1a's best rates for the other parameters.
  - Checkpoints are in `checkpoints/e2_shared/` (not committed).
  - Then `plot_e3.py` drew the E4 figure, δ at the switch.
- **E3a, tuning round 0** (288 runs = 6 arms × 4 B × 2 β × multipliers {0.5, 1, 2} × seeds 0 and 1; 19:31–20:17; 0 diverged).
- **`select`** (about 20:19) showed 37 of 48 cells with the best multiplier on a grid edge (16 low, 21 high).
  - Added `extend` to `code/analyze_branches.py`, plus the protocol change-log entry (26 Oct).
  - Wrote `configs/e3a_ext1_branches_tune.yaml` (148 runs).
  - The `e3b` config this `select` wrote was superseded (regenerated on 27 Oct, 11:38).
- **Machine.** Every run so far was on the laptop (MacBook Air M3); each result file records the platform. Running E3 on RONIN was considered for speed but not used.

### E1 results (measured)

Tail excess risk (the mean over the last 10% of steps), averaged over 2 seeds, at each optimiser's best learning rate:

| B | AdamW | SGD | NS5 Muon | exact Muon | best lr (AdamW / SGD / NS5 / exact) |
|---|---|---|---|---|---|
| 16 | 0.0926 | 0.0905 | 0.0564 | 0.0566 | 0.0003 / 0.03 / 0.003 / 0.003 |
| 64 | 0.0475 | 0.0418 | 0.0313 | 0.0313 | 0.001 / 0.1 / 0.003 / 0.001 |
| 256 | 0.0206 | 0.0161 | 0.0117 | 0.0114 | 0.001 / 0.1 / 0.001 / 0.001 |
| 1024 | 0.0091 | 0.0075 | 0.0041 | 0.0041 | 0.001 / 0.3 / 0.001 / 0.001 |

- **Tuned Muon has the lowest excess risk at every batch size.** Exact Muon is 37.5%, 25.2%, 29.1% and 45.1% below the better baseline at B = 16, 64, 256 and 1024. The better baseline is SGD each time.
- **NS5 and exact-SVD Muon** differ by −0.2% to +2.4%.
- **Best learning rates:** SGD's rises 10× from B = 16 to 1024 and AdamW's 3×. Muon's falls, from 0.003 to 0.001.

### E2 results (measured; at the switch, step 3000; mean over 5 seeds)

| B | β | tail excess risk | Xie's δ (range over seeds and both hidden matrices) |
|---|---|---|---|
| 16 | 0.9 | 0.0826 | 0.69 (0.57–0.81) |
| 16 | 0.95 | 0.0791 | 1.10 (0.90–1.26) |
| 64 | 0.9 | 0.0593 | 1.16 (0.98–1.30) |
| 64 | 0.95 | 0.0559 | 1.84 (1.54–2.08) |
| 256 | 0.9 | 0.0293 | 1.77 (1.60–1.97) |
| 256 | 0.95 | 0.0252 | 2.76 (2.47–3.03) |
| 1024 | 0.9 | 0.0106 | 2.62 (2.29–2.96) |
| 1024 | 0.95 | 0.0083 | 4.09 (3.56–4.67) |

- δ rises with batch size (about 4× from B = 16 to 1024) and with momentum.
- Only B = 16, β = 0.9 is clearly below 1, the noise-dominated side. B = 16, β = 0.95 sits at 1.10.

### E3a tuning, round 0 (tuning seeds only; not a result)

Where each arm's best multiplier sat on the {0.5, 1, 2} grid, out of 8 cells per arm:

| Arm | Low edge (×0.5) | High edge (×2) | Inside |
|---|---|---|---|
| p = −0.25 | 0 | 7 | 1 |
| p = 0 (Muon) | 1 | 5, all at B ≥ 64 | 2 |
| p = 0.5 | 5 | 1 | 2 |
| p = 1 | 6 | 1 | 1 |
| NS5 | 1 | 5, all at B ≥ 64 | 2 |
| Xie | 3 | 2 | 3 |

The edges are systematic. More flattening wants larger steps, and less flattening wants smaller ones. This is the step-size/shape coupling that the per-arm tuning (protocol Section 2) controls for.

### Interpretation (tentative)

- **The pilot's "Muon is worse at B = 16" (20 Oct) was a tuning artefact.** It came from Muon learning rates of 0.01–0.03, 3–10× above the tuned 0.003, and equal tuning reversed it.
- **A possible reason Muon's best learning rate does not grow with B** (not shown): under heavy noise, Muon's average step along the true gradient shrinks, so small batches need a larger nominal rate. On 26 Oct this was attributed to Xie, Prop. 3.1; check that against the paper before citing it.
- **E2 puts the batch-size extremes in different noise regimes,** as the design needs. But the noisy end is only moderately noisy, so the strongest test of Xie's theory is B = 16, β = 0.9.

### Decisions (made before any E3 result was seen)

- Run E3 exactly as planned.
- If the H1 test comes out inconclusive, run the planned robustness runs E5 (β = 0.5, label noise 1.0) for stronger noise.

## 2026-10-20: protocol, code, tests, pilot (E0), baselines started (E1a)

### Done

- Wrote `notes/protocol.md` (v1): hypotheses H1–H3, update rules, Task A design, experiment matrix, analysis plan.
- Implemented `code/muonlab/`: update rules, MLP, teacher–student stream, noise diagnostics, training loop with checkpoints and branching, and a sweep runner (`code/run.py`).
- Added a 22-test suite: `pytest`, all passing.
- Ran the E0 pilot (`configs/e0_pilot.yaml`, 16 runs, results in `results/e0_pilot/`).
- Started E1a (`configs/e1a_baselines_adamw_sgd.yaml`, 80 runs).

### Problems found and fixed (kept on record)

1. **A unit test caught the shape factor √max(1, m/n) being applied twice** to Xie's matched momentum-SGD learning rate: once in the formula, once by the optimizer. The formula now leaves it out. It made no difference for our 256×256 matrices, where the factor is 1, but it would elsewhere.
2. **Pilot run `muon-exact_lr0.03_B1024` crashed at step ≈ 5895.** `torch.linalg.svd` in float32 failed to converge. The input was finite, with Frobenius norm 3.6e-3 and condition number 4.2e4; float64 SVD of the same matrix works. Fix: retry in float64. Every fallback is counted in the run summary (`svd_float64_fallbacks`). The rerun had 1 fallback.

### Pilot observations (one seed, untuned; not results)

- **Time per step**, one thread, four parallel workers:
  - AdamW and SGD: 0.3–0.6 ms at B = 16, about 5 ms at B = 1024.
  - NS5 Muon: about 3 ms at B = 16, about 8 ms at B = 1024.
  - Exact-SVD Muon: 11–15 ms.
  - During E1a, B = 1024 steps took about 11 ms, because four workers share the CPU.
- **Xie's indicator δ at step 6000** (Muon runs): 3.4–5.6 at B = 1024 and 0.3–0.8 at B = 16. Batch size moves the hidden layers across δ ≈ 1, the boundary between signal-dominated and noise-dominated, as the design needs.
- **Muon converges fastest early but plateaus.** At step 1000 and B = 1024, its excess risk was 0.015 against AdamW's 0.040 and SGD's 0.057. Muon then flattens at about 0.010, while SGD keeps falling (0.0077 at step 6000 and still decreasing).
- **At B = 16, at the learning rates tried, Muon's plateau (0.11–0.12) sat above AdamW's (0.104) and SGD's (0.090).** This fits Xie's noise-floor prediction, but it is unconfirmed until every optimiser is tuned.
  - **Overturned by E1 (same day).** With tuned learning rates, NS5 Muon has the *lowest* excess risk at every batch size. The pilot's Muon learning rates (0.01–0.03) were simply too high. This shows why the protocol requires equal tuning before any comparison. Details are in the E1 part of the 26 Oct entry.
- **Every optimiser preferred the lower learning rate at B = 16.** The E1 grids were therefore widened downwards to 5 values spanning two decades.

## 2026-10-19: novelty check of the provisional direction

### Done

- Read the closest recent papers in full or in part. Sections read and findings are in `notes/novelty_check_2026-10-19.md`; read statuses are in `notes/reading_list.md`.
- Verified 12 more papers on arXiv and added them to the reading list.
- Timed exact spectral shaping U S^p V^T with `code/checks/spectral_shaping_cost.py`. Output: `results/smoke/spectral_shaping_cost_20261019.txt`.

### Observations (measured)

- On the M3 CPU in fp32, exact shaping via SVD takes 1.1 ms (128×128), 3.8 ms (256×256) and 17 ms (512×512). Its error against an fp64 reference is 2–7e-6.
- The Gram-matrix (eigh) route is faster, but its error reaches 2.6e-2 at 256×256 with p = −0.25. Use SVD.

### What the reading changed

- "Muon vs AdamW across batch sizes" is now covered by Dang, Wen & Malladi (Oct 2026) at language-model scale, with full retuning at each batch size. It is no longer a strong standalone question for us.
- Xie (Sep 2026) and DynMuon (May 2026) make opposite predictions about late training. Xie says flatten less (p → 1, momentum SGD). DynMuon says flatten more (p = −0.25).
- Both theories imply the best late-training p should rise with gradient noise, but neither paper varied the noise. This is the proposed new research question, still provisional.
- Shumaylov et al. (May 2026) show the step size can masquerade as a spectral-shape effect, so step sizes must be tuned for each p.

### Open decisions

- Group sign-off on the revised question.
- Canvas brief, rubric, deadline and group details: still not received.
- Whether to commit the repository; nothing is committed yet.

## 2026-10-15: environment and smoke test

### Done

- Read the official project page (alexxthiery.github.io/teaching/muon/muon.html) and the DSA4212 Ch0 slides. No project materials existed locally.
- Created the conda environment `muon`:
  - Python 3.12.15 from conda-forge;
  - from PyPI: torch 2.14.1, numpy 2.5.4, matplotlib 3.11.2, pandas 3.0.6, PyYAML 6.0.3, pytest 9.1.1;
  - full freeze in `requirements-lock.txt`.
- Created this repository with `git init`. Nothing is committed yet.
- Wrote and ran `code/smoke_test.py`. Investigated one discrepancy with `code/checks/ns_bf16_rounding.py`.

### Results files (`results/smoke/`)

| File | What it is |
|---|---|
| `smoke_20261015-170348.json` + `_console.txt` | Full run, final script |
| `smoke_20261015-165819_quick.json` | Quick run, final script |
| `smoke_20261015-165330_quick_v1.json` | First quick run, from an earlier script version that lacked the fp32 comparison and the MinimalMuon checks. Kept because it is where the bf16 problem was first seen. |
| `ns_bf16_rounding_20261015.txt` | Output of the bf16 rounding check |

### Observations (measured, full run unless stated)

1. **bfloat16 on the CPU is far too slow to use on this Mac.**
   - CPU float32 matrix multiplication reached 1,270–1,530 GFLOP/s. CPU bfloat16 reached 3.4–6.7 GFLOP/s, about 190–420× slower.
   - On the Apple GPU (MPS), bfloat16 runs at about the same speed as float32: 98–2,190 GFLOP/s depending on matrix size.
   - Consequence: `torch.optim.Muon`, which always runs Newton–Schulz in bfloat16, takes about 1.5 s per step on the CPU. AdamW takes 2–16 ms.
2. **Our Newton–Schulz is correct.**
   - In fp32 it agrees with PyTorch's operation sequence to 2–4e-6 on three test matrices.
   - Its effect on each singular value matches the scalar map φ⁵(x), with φ(x) = 3.4445x − 4.7750x³ + 2.0315x⁵, to 3 decimals.
   - Scale invariance (5.6e-6) and transpose invariance (0) hold.
3. **bfloat16 rounding moves the Newton–Schulz output by 1–11%** (relative Frobenius norm), depending on the matrix.
   - Our version, which uses Keller Jordan's separate operations, drifts 2.3–11% from fp32. PyTorch's fused `addmm` drifts 1.1–8.1%.
   - Ours and PyTorch's differ by 2.5–10% in bfloat16. The largest gap occurs on the matrix with singular values down to 1e-4.
   - MPS and CPU bfloat16 results differ by 1.2%.
   - This is rounding, not a bug (see item 2).
4. **MinimalMuon matches `torch.optim.Muon` exactly.** With PyTorch's own Newton–Schulz plugged in, the largest parameter difference after 5 steps is 0. The test used tall, wide and square matrices with weight decay 0.1. So momentum (EMA form), Nesterov, the √max(1, out/in) scaling and decoupled weight decay are implemented the same way.
5. **Five Newton–Schulz steps do not orthogonalise weak directions.**
   - Singular values below about 1e-3 of the Frobenius norm are only partly boosted: 8.5e-4 becomes 0.40, and 2.6e-5 becomes 0.013.
   - Above about 0.02 they land in [0.68, 1.13]. The overall maximum is 1.20, at x ≈ 4e-3.
6. **FLOP counts do not predict wall-clock time at this scale.**
   - For the 784-512-512-512-10 MLP, Jordan's FLOP model predicts Newton–Schulz costs 45× the forward+backward pass at batch 32, and 0.7× at batch 2048.
   - Measured step time of Muon (fp32 Newton–Schulz) over AdamW on the CPU: 5.7× at batch 32, 4.9× at 128, 3.7× at 512 and 1.9× at 2048.
   - On MPS the ratio was 1.5–3.2×. It varied between the quick and full runs because tiny GPU kernels are dominated by launch overhead.
7. **The training pipeline works.**
   - All 8 runs (4 optimisers on CPU and on MPS) reduced the loss and stayed finite.
   - The binary Fashion-MNIST task is too easy: every optimiser reaches 99.4–99.8% validation accuracy within 5 epochs.
   - The settings were untuned, so this says nothing about which optimiser is better.
8. With the same seed, AdamW and SGD results are identical on the CPU and on MPS. Muon results differ slightly between devices, and between bfloat16 and fp32 Newton–Schulz.

### Interpretation (tentative)

- Practical Muon (5 Newton–Schulz steps) leaves the weakest directions under-boosted, which acts like a soft noise floor. Exact-SVD Muon would not have that floor. This matters for the signal-vs-noise question.
- Wall-clock comparisons need their own careful protocol. FLOP estimates mispredict wall-clock time here, and MPS timings are noisy.

### Decisions needed before the protocol

- Device and Newton–Schulz precision. Option (a): CPU with fp32 Newton–Schulz. It is fast with stable timings, but deviates from the bf16 reference and the deviation must be documented. Option (b): MPS with bf16 Newton–Schulz. It matches the reference numerics, but timings are noisier. Precision could also become an experimental factor in its own right.
- Main dataset: full Fashion-MNIST or CIFAR-10. Either needs torchvision or a manual download.
- Waiting for the Canvas brief, rubric, deadline and group details.

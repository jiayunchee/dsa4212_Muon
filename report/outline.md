# Report outline (draft, 28 Oct 2026)

This is the working plan for the report of at most 10 pages. Every number below comes from `results/` and matches the lab notebook entries of 26–28 Oct. Pre-registered and exploratory results are kept apart throughout, as in the notebook.

## Constraints and assumptions

- **From the brief:**
  - at most 10 pages, including references and appendix;
  - every member's name and student number, and the GitHub URL, on the first page;
  - no code in the main text (equations only);
  - a reproducible repository.
- **From the official project page:**
  - (R1) same models and data, comparable tuning, stated budgets;
  - (R2) per-step gains reported separately from per-time gains;
  - (R3) state which parameters get Muon.
- **Assumed:** LaTeX, A4, 11 pt, single column, about 550 words per full page of text. Check this against the Canvas brief.
- **Unknown:** the deadline, the rubric, the group members and their student numbers, and any required template.

## The story in four sentences

1. **Per step vs per second.** With equal tuning, Muon beats SGD and AdamW per step at every batch size, with 25–45% lower excess risk. On a CPU its extra cost per step cancels that gain per second, except at B = 1024.
2. **The pre-registered test.** Late in training, flattening the update more than Muon (p = −0.25) beats continuing Muon in all 8 batch-size × momentum conditions, and SGD-like updates do worst (12–41% worse than Muon). Neither the pre-registered noise hypothesis H1 nor Xie's switch (H2) is supported.
3. **The exploratory follow-up.** The best exponent does move with batch size, from −0.25 at B ≤ 64 to −0.5 at B ≥ 256. That is the direction H1 predicted, but the shift is small and stays far from the SGD end.
4. **The method lesson.** Tuning each arm's step size decided every one of these answers. Without it, the pilot and the first F1 evaluation would have told the opposite story.

## Page budget (target 9.5 of 10 pages)

| Part | Pages | Figures and tables |
|---|---|---|
| Title block and abstract (about 150 words) | 0.4 | |
| 1. Introduction | 0.7 | |
| 2. Background: Muon, the U S^p Vᵀ family, two predictions | 0.8 | |
| 3. Setup and protocol | 1.5 | Table 1 |
| 4. Muon vs baselines, per step and per second | 1.2 | Figure 1 |
| 5. Pre-registered test: the late-phase exponent under noise | 1.5 | Figure 2, Table 2 |
| 6. Exploratory follow-ups | 1.1 | Figure 3 |
| 7. Discussion and limitations | 0.9 | |
| 8. Conclusion | 0.2 | |
| References (about 13) | 0.5 | |
| Appendix | 0.7 | Table A1, Figure A1 |
| **Total** | **9.5** | 3 figures and 2 tables in the main text |

The remaining 0.5 page absorbs figure placement. If the draft runs over, cut in this order:
1. Figure A1.
2. Section 2, down to 0.6 pages.
3. Section 7's limitations, into a compact list.

## Section by section

Each section starts with what the reader should come away believing.

### Title block and abstract (0.4)

- **Working titles:** "Flatten more, not less: Muon's late-training update under gradient noise", or "Should Muon flatten less when gradients are noisy? A pre-registered test".
- **First page:** names, student numbers (from `authors.md`) and the GitHub URL.
- **Abstract:** the question, the setup in one sentence, the four story sentences compressed, and one line on what is exploratory.

### 1. Introduction (0.7)

**Takeaway:** the question is open, and it can be tested here.

- Muon flattens the singular values of each hidden-layer update. The official brief asks whether boosting weak directions helps, or amplifies noise.
- Two 2026 papers disagree about late training. Xie: switch to momentum SGD when noise dominates. DynMuon: flatten even more (p = −0.25).
- Our question: does the best late-phase exponent p in U S^p Vᵀ depend on gradient noise?
- **Contributions (3 bullets):**
  - a fairly tuned per-step and per-second baseline comparison;
  - a pre-registered branch test across 8 noise conditions;
  - an exploratory follow-up that brackets p\*.

### 2. Background (0.8)

**Takeaway:** what p means, and what each theory predicts.

- **The Muon update:** EMA momentum, the Nesterov input, Newton–Schulz approximating the polar factor UVᵀ, and the shape factor √max(1, m/n). One displayed equation.
- **The family** D_p = U S^p Vᵀ, rescaled to ‖D_p‖_F = √rank: p = 0 is Muon, p = 1 is normalised momentum SGD, and p < 0 boosts weak directions further.
- **The two predictions:** Xie's indicator δ (signal over noise, per layer) with the matched switch to SGD; DynMuon's late-training p = −0.25.
- **Why step sizes must be tuned for each p** (Shumaylov et al.; read before citing, see References).

### 3. Setup and protocol (1.5)

**Takeaway:** the comparison is fair, and R1–R3 are met.

- **Task A:** teacher–student regression with a fresh minibatch every step.
  - Input covariance with eigenvalues ∝ 1/i; label noise σ_y = 0.5.
  - Excess risk is measured against the noiseless teacher on 16,384 fixed inputs, so the optimum is known.
- **Student:** a tanh MLP, 128 → 256 → 256 → 256 → 16.
  - **(R3)** Only the two 256 × 256 hidden matrices get Muon or its variants.
  - The first and last layers and the biases use AdamW at E1's best rate for that batch size.
- **Branch design:** exact Muon trains for 3000 steps (E2, 5 seeds). Every arm then continues from the same checkpoint with the same future data, so comparisons are paired.
- **(R1) Tuning:**
  - E1: 5 learning rates per optimiser and batch size.
  - E3 and F1: a multiplier per arm, from {0.5, 1, 2}. While the best value sits on an edge, the grid is extended by halving or doubling, within [1/64, 64].
  - Tuning seeds {0, 1} are kept separate from evaluation seeds {2, 3, 4}.
- **Statistics:** paired 95% t-intervals against p = 0; p\* with a bootstrap over seeds; the pre-registered contrast C for H1.
- **Pre-registration:** the protocol was fixed on 20 Oct. Its change log dates every later change: the extension rule, the stopping rule, the frozen control and F1.
- **Budget:** 1179 training runs in total, all on CPU in float32. bf16 is about 200–400× slower on this CPU.
- **Table 1:** one row per experiment (E1, E2, E3, F1, timing), with purpose, runs, seeds and whether it was pre-registered.

### 4. Muon vs baselines, per step and per second (1.2)

**Takeaway:** Muon wins per step and mostly loses per second here, and why. This is R2.

- **Per step** (E1, 6000 steps, tuned learning rates):
  - Exact Muon is 37.5 / 25.2 / 29.1 / 45.1% below the better baseline (SGD each time) at B = 16 / 64 / 256 / 1024.
  - NS5 ≈ exact Muon (−0.2% to +2.4%).
- **Best learning rates** (one sentence, or in Section 7): from B = 16 to 1024, SGD's rises 10× and AdamW's 3×, while Muon's falls from 0.003 to 0.001.
- **Per second** (1 thread; median of 5 benchmark passes):
  - Muon's overhead over SGD is 2.0–3.2 ms per step for NS5 and 7.9–9.5 ms for exact SVD.
  - In the time SGD needs for 6000 steps, SGD wins at B ≤ 256, where NS5 has 1.3–2.9× SGD's excess risk. NS5 Muon wins at B = 1024 (0.0058 vs 0.0079).
  - The winner at each batch size is the same under all three timing sets we measured (27 Oct; 28 Oct single pass; 28 Oct median of 5). Only the margins move: NS5's lead at B = 1024 is 10–27%.
- **Why equal tuning matters:** one sentence on the pilot, whose reversed ranking came from untuned Muon learning rates.
- **Figure 1** (`fig1_step_vs_time`), two panels with a shared log y-axis:
  - (a) tail excess risk against B at 6000 steps;
  - (b) excess risk against B at equal wall-clock time (SGD's 6000-step time).
- **Caution:** these timings are CPU-specific, and they vary with machine load (the sub-millisecond steps by up to 2× between passes). Report the 5-pass medians together with the robustness check above. The 27 Oct NS5 value at B = 64 was a one-off and did not recur.

### 5. Pre-registered test (1.5)

**Takeaway:** flattening more wins everywhere, and H1 and H2 fail as stated.

- **Noise at the switch:** δ ranges from 0.69 (B = 16, β = 0.9) to 4.09 (B = 1024, β = 0.95). Only one condition is clearly noise-dominated. Show δ in Table 2, with Figure A1 in the appendix.
- **Change against continuing Muon** (evaluation seeds):

  | Arm | Change |
  |---|---|
  | p = −0.25 | −2.4% to −7.6% |
  | p = 0.5 | +8.6% to +21.2% |
  | p = 1 | +11.7% to +41.5% |
  | Xie's switch | +11.8% to +37.4% |
  | NS5 | −1.3% to +1.7% |

  p\* = −0.25 in all 8 conditions.
- **Verdicts:**
  - H1 is contradicted: C = +0.00877 [+0.00833, +0.00922] at β = 0.9 and +0.00657 [+0.00617, +0.00698] at β = 0.95.
  - H2 is not supported.
  - H3 is not testable, because p\* is constant.
- **Disclose the design flaw here.** C compares absolute gaps, and excess risk is about 14× larger at B = 16 than at B = 1024.
- **Figure 2:** `e3_relative_change`.
- **Table 2:** for each condition, δ, p\* (E3 and F1), and the change for p = −0.25, p = 1 and Xie.

### 6. Exploratory follow-ups (1.1)

**Takeaway:** the optimum shifts mildly with batch size, and the SGD-like arms lose for an identifiable reason. Say once at the top, and in every caption, that this section is exploratory.

- **F1 (p = −0.5 and −0.75):**
  - p\* = −0.25 at B ≤ 64 and −0.5 at B ≥ 256, at both momenta.
  - p = −0.5 minus p = −0.25: +1.4% to +4.1% at B ≤ 64, and −1.3% to −2.4% at B ≥ 256. 7 of 8 intervals exclude 0.
  - ρ(δ, p\*) = −0.76 (exact permutation p = 0.029), and ρ(log B, p\*) = −0.89. Momentum changes δ by 56–60% without moving p\*.
  - At B ≥ 64 the new arms needed multipliers of ×8–×32. At ×2, p = −0.5 looked up to 43% worse than Muon.
- **Frozen control:**
  - At B = 16 the SGD-like arms do no better than freezing the hidden layers (p = 1: 0.0672 vs frozen 0.0675 at β = 0.9).
  - At B = 1024 they clearly beat it (p = 1: 0.0056 vs 0.0076).
- **Scale-free contrast:** −0.199 and −0.163, both in H1's direction. The gain of p = −0.25 over Muon rises with δ (ρ = +0.88).
- **Figure 3** (`fig3_f1_exponent_shift`):
  - (a, b) the change against Muon for p ≤ 0, one panel per β, with each batch size's best p enlarged;
  - (c) the paired difference p = −0.5 minus p = −0.25 against B, with 95% intervals. It shows where the sign flips (between B = 64 and B = 256), which a plot of p\* alone would not.

### 7. Discussion and limitations (0.9)

**Takeaway:** what this means for using Muon, and how far it generalises.

- **Against the two theories:**
  - DynMuon's mildly negative p holds across a 64× range of batch size.
  - Xie's regime (δ ≪ 1 near the optimum) was never reached; the smallest δ was 0.69.
- **Why SGD-like steps fail at small batch,** from the frozen control. A hypothesis, labelled as such: the hidden-layer gradients are ill-conditioned, so a step large enough for the weak directions is too large for the strong ones.
- **Step size and shape are coupled.** More negative p needs larger multipliers, and this decided both the pilot and F1.
- **Limitations:**
  - one synthetic task and a small MLP;
  - CPU float32, not bf16;
  - weight decay 0;
  - 3 evaluation seeds;
  - a p grid with steps of 0.25;
  - equal steps across batch sizes, so larger batches see more data;
  - hardware-specific timing;
  - F1 was not pre-registered.

### 8. Conclusion (0.2)

Three lines: the main answer, the exploratory refinement and the method lesson.

### Appendix (0.7)

- **Table A1:** the selected step-size multiplier for every arm and condition, plus the extension rounds. Edge cells went 37 → 7 → 3 → 0 for E3 and 16 → 12 → 2 → 0 for F1.
- **Figure A1:** `e4_delta_vs_batch`.
- **Implementation checks:** 26 unit tests. `MinimalMuon` matches `torch.optim.Muon` exactly. float64 SVD retries are counted in every run.
- **Deviations from the protocol:** the dated change-log entries, one line each.
- **Reproduction:** point to the README table; no commands in the report.

## Figures and tables

| Item | Section | Status | Source |
|---|---|---|---|
| Figure 1: per step vs per second | 4 | Made (28 Oct): `fig1_step_vs_time` (`code/plot_report.py`) | `results/e1_baselines/`, `results/timing/per_second_summary.json` |
| Figure 2: E3 change against Muon | 5 | Made (28 Oct): `fig2_e3_relative_change`, the E3 figure without its built-in title (`code/plot_report.py`) | `results/e3_branches/evaluation_summary.json` |
| Figure 3: F1 zoom and paired difference | 6 | Made (28 Oct): `fig3_f1_exponent_shift` (`code/plot_report.py`) | `results/e3_branches/evaluation_summary_followup.json` |
| Figure A1: δ at the switch | Appendix | Exists | `e4_delta_vs_batch` |
| Table 1: the experiments | 3 | Written in `main.tex` | configs, run counts |
| Table 2: per-condition summary | 5 | Generated: `report/tables/table2_conditions.tex` (`code/make_tables.py`) | both evaluation summaries |
| Table A1: selected multipliers | Appendix | Generated: `report/tables/tableA1_multipliers.tex` (`code/make_tables.py`) | `configs/e3b_branches_eval.yaml`, `configs/f1b_branches_eval.yaml` |

Not used in the report:
- `e1_tuned_vs_batch` and `e1_vs_time` (combined into Figure 1);
- `e1_curves`;
- `e1_per_example` (that result needs a careful check first, lab notebook 20 Oct);
- `e3_relative_change_followup` (replaced by Figure 3).

## Where each result comes from

| Result | Section | File | Status |
|---|---|---|---|
| Tuned learning rates, per-step excess risk | 4 | `results/e1_baselines/tuning_summary.csv` | baseline |
| Step times, equal-time excess risk | 4 | `results/timing/*.json` | baseline (benchmark) |
| δ at the switch | 5, A | `results/e2_shared/*.json` | pre-registered measurement (E4) |
| E3 changes, p\*, H1 contrast, H3 | 5 | `results/e3_branches/evaluation_summary.json` | **pre-registered** |
| Frozen control, scale-free contrast, δ–gain correlation | 6 | `evaluation_summary.json`; lab notebook, 27 Oct (morning) | exploratory |
| F1 p\*, paired differences, ρ | 6 | `evaluation_summary_followup.json`; lab notebook, 28 Oct | exploratory |
| The pilot's reversed ranking | 4 | `results/e0_pilot/`; lab notebook, 20 Oct | pilot |
| bf16 rounding, NS5 on weak directions | 3, A | `results/smoke/`; lab notebook, 15 Oct | implementation check |

## References

Cite a source only for claims in the parts we have read (the rule in `notes/reading_list.md`).

| Source | Read so far | Used for | Before citing |
|---|---|---|---|
| Jordan et al. 2024, Muon blog post | full | definition, NS5, overhead T·m/B | — |
| Xie 2026, arXiv:2609.32861 | §1–3, §7–8 | δ, the switch, H2 | check the Prop. 3.1 claim in §3 |
| Wu et al. 2026 (DynMuon), arXiv:2605.17109 | §2–4 | late-training p = −0.25 | — |
| Shumaylov et al. 2026, arXiv:2605.11181 | abstract | step size vs shape | **read the relevant sections** |
| Dang, Wen & Malladi 2026, arXiv:2610.08975 | abstract, §1, table | rankings depend on batch size | cite §1 claims only |
| Goyal & Ramkumar 2026, arXiv:2610.10965 | full | "weak directions are not noise" | — |
| Dong & Sawin 2026 (Muon^p), arXiv:2606.13867 | scanned | earlier use of U S^p Vᵀ | read the method section |
| Li et al. 2026 (Denoise First), arXiv:2606.03899 | §1, App. F.5 | momentum as a spectral filter | — |
| Jacobs & Burkholz 2026, arXiv:2610.16868 | §1–2, §6–7 | small-batch noise and Muon | — |
| Bernstein & Newhouse 2024, arXiv:2409.20325 | abstract | steepest descent under the spectral norm | read §1–2, or cite only as background |
| Liu et al. 2025, arXiv:2502.16982 | abstract | Muon at scale | read, or drop |
| Nguyen et al. 2026, arXiv:2605.13079 | abstract | stable step size (the ill-conditioning hypothesis) | read, or drop that sentence |
| Loshchilov & Hutter 2019 (AdamW) | abstract | the method only | — |

## Writing rules

- Label everything exploratory in the text and in its caption.
- Give an interval or the number of seeds with every comparison.
- Every figure is referred to in the text, and every caption can be read on its own.
- Use equations, not code. Point to the repository for commands.
- Every number is taken from `results/` or the lab notebook, never retyped from memory.

## Before writing

1. **From the group:** names and student numbers, the deadline, the rubric and any template. Then decide who writes which section.
2. ~~Make Figures 1 and 3.~~ Done on 28 Oct (`code/plot_report.py`).
3. ~~Re-run the timing benchmark for NS5 at B = 64.~~ Done on 28 Oct. The benchmark now keeps the median of 5 passes, the NS5 spike did not recur, the per-second winners are unchanged, and Figure 1 has been updated.
4. Read the sources marked above.
5. Decide on E5 (deep noise) once the deadline is known. Without E5, the verdict on H2 is "not supported in our noise regime", not "refuted".
6. ~~Set up `report/main.tex`.~~ Done on 28 Oct:
   - every section, with its page budget and plan as comments;
   - Figures 1–3 and A1 placed, with draft captions;
   - Table 1 written; Tables 2 and A1 generated by `code/make_tables.py`;
   - `references.bib`, with each entry's read status.

   It has not been compiled yet, because there is no LaTeX on this Mac.

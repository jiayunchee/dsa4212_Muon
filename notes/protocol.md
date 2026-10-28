# Experiment protocol (v1, 20 Oct 2026)

This document fixes what we run and why, *before* the main experiments. Numbers marked **(provisional)** are set by the pilot (E0) and recorded in the lab notebook. Any later change gets a dated note here.

## 0. Guidelines this protocol follows

**Official project page** (alexxthiery.github.io/teaching/muon/muon.html):

- Compare optimisers on the same models and data, with comparable tuning effort and clearly stated training budgets.
- Distinguish gains per training step from gains per unit of compute or elapsed time.
- State which parameters receive Muon updates and how the remaining parameters are optimised.
- The goal is to understand *when and why* Muon and its variants help, not just to rank them.

**Course (Ch0 slides).** Assignments are 30% of the grade. Groups have at most 4 students and receive one mark. Python only.

**Report and repository** (as stated by the user; the Canvas text has not been seen):

- The PDF is at most 10 pages, including references and appendices.
- It lists all names and student numbers and gives the GitHub URL at the top.
- No code in the main text.
- The repository holds README, `authors.md`, `report/`, `code/`, `configs/` and `results/`, and passes a reproducibility audit.

**Assumption.** The deadline is unknown. The plan is ordered so the core (Task A, E1–E4) can stand alone as a complete project. Everything after it is optional.

## 1. Question and hypotheses

**Question.** Late in training, should Muon flatten its update *more* or *less*, and does the answer depend on gradient noise?

We study the family of updates U S^p Vᵀ built from the momentum matrix (Section 2):

| p | Meaning |
|---|---|
| 0 | Muon |
| 1 | Momentum SGD (normalised) |
| < 0 | Boosts weak directions even more than Muon, as DynMuon does late in training |

**H1 (main).** Write p\*(B, β) for the best late-phase exponent, with the step size tuned separately for each p. p\* rises with gradient noise: it is ≤ 0 at large batch size B and high momentum β, and approaches 1 at small B or low β. The pre-registered test is an interaction contrast:

    C = [L(p=1) − L(p=−0.25)] at B=16  −  [L(p=1) − L(p=−0.25)] at B=1024

H1 predicts C < 0. H1 is supported if the 95% interval for C excludes 0, and contradicted if it is confidently > 0.

**H2 (Xie 2026).** Switching late from Muon to momentum SGD, with Xie's matched learning rate, lowers the final excess risk. The gain is larger at lower momentum and smaller batch.

**H3 (mechanism).** Xie's per-layer noise indicator, δ_β = ‖mean gradient‖_F / (σ_β·√(mn)), measured at the switch, predicts which p wins: small δ_β favours larger p.

Sources: Xie, arXiv:2609.32861; Wu et al. (DynMuon), arXiv:2605.17109. See `notes/novelty_check_2026-10-19.md`.

## 2. Update rules

For one hidden weight matrix W (m × n) with minibatch gradient G:

    M ← βM + (1−β)G                        momentum buffer, EMA form (as in torch.optim.Muon)
    N = (1−β)G + βM                        Nesterov input
    N = U diag(s) Vᵀ                       thin SVD, float32
    keep sᵢ > s₁·max(m,n)·ε_float32        numerical rank r; weaker directions are dropped
    D_p = U diag(sᵢᵖ) Vᵀ, rescaled to ‖D_p‖_F = √r
    W ← W − η · √max(1, m/n) · D_p

The update arms:

| Arm | Direction | Notes |
|---|---|---|
| **Exact Muon** | p = 0, D = UVᵀ | Exact polar factor via SVD |
| **Spectral p** | p ∈ {−0.25, 0.5, 1} | Same step length as exact Muon; only the spread across singular directions changes |
| **NS5 Muon** | Jordan's 5-step Newton–Schulz, fp32 | The algorithm practitioners actually run; it leaves very weak directions partly un-boosted |
| **Plain switch (Xie)** | D = N, no rescaling | Momentum SGD that keeps the buffer. Learning rate set by Xie's matched formula η_L = η_M · c_{m,n} / σ_N, then tuned over the same multipliers as the other arms. The shape factor √max(1,m/n) is applied by the optimizer to every rule, so it cancels; corrected 20 Oct after a unit test caught it being applied twice |

In the matched formula:

- c_{m,n} = E‖Z‖_*/(mn) for a Gaussian m × n matrix, computed by Monte Carlo.
- σ_N is the per-entry noise standard deviation of the Nesterov input, estimated from K = 32 minibatch gradients at the switch point.

All other parameters (first layer, output layer, biases) are trained with AdamW (β = (0.9, 0.999), weight decay 0) in every run. They are never switched.

Weight decay is 0 everywhere. The data stream is fresh, so there is no overfitting to control, and weight decay would add a confounder (Section 8).

Why rescale every arm to the same Frobenius length: so that p changes how the step is *distributed* across singular directions and not how *long* it is. Shumaylov et al. (arXiv:2605.11181) show step size can masquerade as a shape effect. The learning rate is still tuned for each arm.

## 3. Tasks

### Task A: teacher–student regression with streaming data (core)

**Inputs.** x ∈ ℝ¹²⁸ drawn from N(0, Σ). Σ is a random rotation with eigenvalues ∝ i⁻¹, scaled so the trace is 128. This makes the input ill-conditioned.

**Teacher.** A tanh MLP 128 → 64 → 64 → 16 with Gaussian weights scaled by 1/√fan_in and a fixed seed. Outputs are standardised to unit variance.

**Labels.** y = f\*(x) + σ_y·ε, with σ_y = 0.5 **(provisional)**.

**Student.** A tanh MLP 128 → 256 → 256 → 256 → 16 with PyTorch's default initialisation. The two 256 × 256 matrices are the hidden matrices: they get the Muon family. The rest get AdamW.

**Data.** A fresh minibatch every step, from a seeded generator whose state is saved in checkpoints. Every branch from one checkpoint therefore sees identical future data, so comparisons are paired.

**Metric.** Excess risk = mean (f(x) − f\*(x))² on a fixed set of 16,384 inputs, measured against the noiseless teacher. This is exactly the "excess loss" in Xie's theory, and it needs no estimate of the noise floor.

**Why this task.**

- We set the noise directly, through B (and σ_y).
- We know the optimum.
- There is no overfitting confound.
- Runs take seconds to minutes on the laptop CPU.

### Task B: CIFAR-10 MLP (replication on real data)

Planned after Task A results:

- Model: 3072 → 512 → 512 → 512 → 10, ReLU, cross-entropy loss.
- Design: the key contrast only, for B ∈ {32, 1024}, β = 0.95, p ∈ {−0.25, 0, 1}, 3 seeds.

### Task C (optional): tiny character-level transformer on Tiny Shakespeare

This is the model family Xie used for the frozen-gradient measurements.

## 4. Experiment matrix

| ID | Purpose | Arms | Factors | Seeds | Steps | Estimated runs |
|---|---|---|---|---|---|---|
| E0 | Pilot: timing, learning-rate ranges, plateau length | all | B ∈ {16, 1024} | 1 | ≤ 6000 | ~20 |
| E1 | Baselines and learning-rate tuning; context comparison | AdamW, SGD-Nesterov, NS5 Muon + AdamW, exact Muon + AdamW | B ∈ {16, 64, 256, 1024} × 5 learning rates | 2 | 6000 **(prov.)** | 160 |
| E2 | Shared phase, with checkpoints | exact Muon at its E1 learning rate | B × β ∈ {0.9, 0.95} | 5 | 3000 **(prov.)** | 40 |
| E3 | Branches (main test of H1, H2) | spectral p ∈ {−0.25, 0, 0.5, 1}; NS5 Muon; plain switch | learning-rate multiplier ∈ {0.5, 1, 2} | tune on seeds {0, 1}, evaluate on seeds {2, 3, 4} | 3000 **(prov.)** | ~460 |
| E4 | Mechanism (H3) | at each E2 checkpoint | δ_β, σ_N, numerical / effective / stable rank of N | 5 | — | 40 |
| E5 | Robustness (if time) | key arms | switch at 25% / 75%; linear learning-rate decay in the branch; σ_y ∈ {0.25, 1}; β = 0.5 | 3 | — | ~150 |
| E6 | Task B replication | p ∈ {−0.25, 0, 1} | B ∈ {32, 1024} | 3 | TBD | ~40 |

**Batch size.** Batch sizes are compared at an **equal number of steps**, so B changes only the noise in each update, which is the variable in both theories. The price is that larger batches see more examples. We also report results per example and per second.

**Learning rate.** Constant in E1–E3, because the theory concerns the stationary loss at a fixed step size. E5 adds a decaying schedule for practical relevance.

## 5. Analysis plan

1. **Outcome per run.** Mean excess risk over the last 10% of evaluations; evaluation every 100 steps **(prov.)**.
2. **Tuning.** For each arm, B and β, pick the learning-rate multiplier with the lowest mean outcome on the tuning seeds. Report the outcome only on the evaluation seeds. This avoids reusing the same seeds for both selection and reporting.
3. **Paired differences.** Compare each arm with exact Muon (p = 0) from the same checkpoint and seed. Report the mean and a 95% t-interval over the evaluation seeds.
4. **p\*.** Take p\*(B, β) as the arm with the lowest mean. Its uncertainty comes from bootstrapping over seeds.
5. **H1.** The contrast C (Section 1), with a 95% interval.
6. **H3.** Rank correlation between δ_β at the switch and the winning p, across conditions.
7. **Compute.** Report wall-clock time per step for every arm (SVD vs NS5 vs plain), and outcomes per step and per second. Per-second numbers come from the dedicated benchmark E-T (`code/benchmark_step_time.py`, run on a quiet machine). Sweep timings are not used, because they include contention between parallel workers (added 20 Oct).
8. **Reporting.** No claims beyond the stated intervals. Every run is reported, including diverged ones.

## 6. Fairness and controls

- Every arm uses the same model, initialisation, data stream and evaluation set.
- Every arm gets the same tuning budget:
  - in E1, 5 learning rates per optimiser;
  - in E3, 3 multipliers per arm.
  - If the best value lies on the edge of a grid, extend that arm's grid and record it.
- AdamW's own learning rate (for non-hidden parameters in Muon runs) is fixed at the best AdamW-baseline learning rate for that B, from E1.
- Tuning seeds and evaluation seeds are kept separate.
- Everything runs on the CPU in float32 with fixed thread counts. bf16 is 200–400× slower on this CPU (lab notebook, 15 Oct), so NS5 runs in fp32. This deviates from the bf16 reference implementations; noted as a limitation.

**Change log for E3 tuning (26 Oct).** After E3a, 37 of the 48 (arm, B, β) cells had their best multiplier on the edge of the {0.5, 1, 2} grid: 16 at the low edge and 21 at the high edge. Under the rule above, `code/analyze_branches.py extend` adds two more multipliers beyond each edge (halving, or doubling) on the tuning seeds. It writes `configs/e3a_ext<k>_branches_tune.yaml` and is repeated until no edges remain; only then does `select` write the E3b config. The edges were systematic: p = −0.25 preferred larger steps, p ∈ {0.5, 1} smaller ones, and continued Muon and NS5 larger ones at B ≥ 64. This is the step-size/shape coupling that Section 2 controls for.

**Exploratory control (27 Oct; not pre-registered).** The tuning seeds showed something odd. At B = 16 the SGD-like arms (p = 1, Xie) did best at multipliers of 1/32–1/64, with the outcome flat across a 16× range. A possible confound: their hidden layers are effectively frozen while AdamW keeps training the other layers. To test it, `configs/e3c_control_frozen.yaml` continues every checkpoint with the hidden matrices frozen (results in `results/e3_controls/`). The evaluation step reports it in a separate, labelled section.

**Stopping rule (27 Oct).** This rule was fixed after round 1, which left 7 edge cells, and before round 2 was run. Multipliers never go beyond the range 1/64 to 64. A cell whose best value is still on the edge at that limit is accepted as it is and reported. The reason: on a flat plateau, "best at the edge" can come from seed noise alone, and extension would otherwise never stop.

**Follow-up F1 (27 Oct; not pre-registered).** E3 put p* = −0.25 in all eight conditions. That is the most negative exponent tested, so the optimum was not bracketed. F1 adds p = −0.5 and p = −0.75 as new branches from the same E2 checkpoints:
- the same tuning rules (multipliers {0.5, 1, 2} on the tuning seeds, then the edge extension and the stopping rule above);
- the same evaluation seeds;
- configs `configs/f1a_*.yaml` and `configs/f1b_branches_eval.yaml`.

`analyze_branches.py evaluate --followup` compares all eight arms and writes `results/e3_branches/evaluation_summary_followup.json`. The pre-registered E3 summary and its verdicts on H1–H3 stay as they are. Anything F1 adds is reported as exploratory.

**Selection guard (27 Oct).** F1's first selection was made three seconds after `extend` had written its extension config, before those runs existed (lab notebook, 27 Oct). Now `select` refuses while any tuning run is unfinished or any best multiplier sits on an edge that can still be extended. `evaluate` uses only the evaluation runs at the selected multipliers, so runs from a superseded selection stay on disk but are not used. Later the same day, after the ×4/×8 runs, `evaluate --followup` was re-run on the old ×2 selection without a new `select`. So `evaluate` now also refuses while tuning is unfinished, or when a multiplier in an evaluation config is no longer the best on the tuning seeds. `code/run_branches.py` runs a stage's steps in this order.

**Timing benchmark (28 Oct).** A re-run of E-T did not reproduce NS5's step time at B = 64 (4.55 ms on 27 Oct, against 2.37 and 2.85 ms at its neighbours). It also showed that single passes are noisy: the sub-millisecond steps (SGD, AdamW) varied by up to 2× between passes, and the machine was not quiet (load average about 5). `code/benchmark_step_time.py` now makes 5 full passes and keeps the median per cell, and per-second results use these medians. The per-second winner at each batch size is the same under the 27 Oct pass, a single 28 Oct pass and the 5-pass medians (lab notebook, 28 Oct).

## 7. Implementation checks (must pass before E1)

Unit tests (`tests/`), via pytest:

- the exact polar factor is orthogonal and equals spectral p = 0;
- p = 1 equals the normalised N;
- the rescaling gives ‖D‖_F = √r;
- NS5 matches PyTorch's operation sequence in fp32;
- the parameter-group assignment is as intended;
- branching from a checkpoint with unchanged settings reproduces the uninterrupted run exactly;
- the noise statistics recover a known σ on synthetic gradients.

**Failure handling.**

- NaN, Inf or excess risk above 1e6: stop and mark the run diverged. It is kept in the results.
- Any failing test: no experiments.

**Recorded with every run.** Config, seed, git commit, package versions, wall-clock time and the full evaluation history (JSON, in `results/<experiment>/`). Checkpoints go in `checkpoints/`, which git ignores.

## 8. Limitations known in advance

- Small networks and a synthetic task. Results speak to mechanisms, not to LLM-scale training.
- Exact-SVD Muon is not NS5 Muon. NS5 is included as an arm to bridge the two.
- fp32 instead of bf16.
- Weight decay is 0, whereas practical Muon at scale uses weight decay.
- Equal-steps comparisons give larger batches more data.
- Few seeds (3 evaluation seeds per condition), so effects may be inconclusive. That would be reported with effect sizes.

## 9. Decisions to be able to defend in the report

1. **Why a synthetic streaming task first:** we can control the noise, we know the optimum, and nothing overfits.
2. **Why all arms are rescaled to the same step length:** to separate spectral shape from step size.
3. **Why exact SVD, plus NS5:** to isolate the effect of shape from Newton–Schulz approximation error.
4. **Why equal steps across batch sizes:** the theories concern the noise in each update.
5. **Why a constant learning rate:** the theories are about the stationary loss. A decaying schedule is checked in E5.

# Muon optimiser project (DSA4212)

**Repository:** https://github.com/jiayunchee/dsa4212_Muon

**Project status:**

- The initial experiment protocol and subsequent amendments are documented in `notes/protocol.md` and its dated change log.
- E1–E4 and the estimated compute-time comparison are complete.
- Follow-up F1 (p = −0.5 and −0.75, not pre-registered) is complete.
- The report source is in `report/main.tex`, with bibliography, figures and tables in `report/`.

Authors are listed in `authors.md`.

## Question

Late in training, should Muon flatten its update more or less, and does the answer depend on gradient noise?

- **The family.** Muon replaces each hidden-layer update matrix N = U S Vᵀ by U Vᵀ. We study U S^p Vᵀ:
  - p = 0 is exact Muon;
  - p = 1 is a normalised momentum-like update, distinct from the standalone SGD baseline;
  - p < 0 boosts weak directions even more than Muon does.
- **Why it's open.** Xie (arXiv:2609.32861) motivates response-matched momentum updates in a noise-dominated regime, while Wu et al. (arXiv:2605.17109) motivate mildly negative spectral powers late in training. These arguments concern different settings. On a synthetic teacher–student task with a known clean prediction target, we compare tuned update rules across batch sizes and momentum settings and measure a gradient-noise indicator. These comparisons can reveal associations with noise, but do not isolate noise as the cause: batch size also changes the number of examples processed, and the conditions reach different checkpoints.
- **More detail.** Background is in `notes/novelty_check_2026-10-19.md`; the full design is in `notes/protocol.md`.

## Environment

Tested on a MacBook Air (Apple M3, 16 GB, macOS 26.6) with Python 3.12.15, PyTorch 2.14.1 and SciPy 1.18.1. Everything runs on the CPU in float32.

```bash
conda create -n muon --override-channels -c conda-forge python=3.12 pip -y
conda activate muon
pip install -r requirements.txt
```

`requirements.txt` pins the packages used directly. `requirements-lock.txt` is the full `pip freeze` of the tested environment.

## How to reproduce

Run every command from the repository root.

| Step | Command | Time on the M3 (4 workers) |
|---|---|---|
| Unit tests (must pass first) | `python -m pytest -q` | ~3 s |
| Toolchain smoke test | `python code/smoke_test.py --quick` | ~2 min |
| E0 pilot | `python code/run.py configs/e0_pilot.yaml --workers 4` | ~4 min |
| E1 baselines | `python code/run.py configs/e1a_baselines_adamw_sgd.yaml --workers 4`, then `configs/e1a_ext_sgd_b1024.yaml`, then `configs/e1b_baselines_muon.yaml` | ~15 + 1 + 35 min |
| E1 analysis and figures | `python code/analyze_tuning.py results/e1_baselines` and `python code/plot_e1.py` | seconds |
| E2 shared phase | `python code/run.py configs/e2_shared.yaml --workers 4` | ~10 min |
| E3 frozen control (exploratory) | `python code/run.py configs/e3c_control_frozen.yaml --workers 4` | ~10 min |
| E3 branches: tuning, extensions, selection, evaluation, figures | `python code/run_branches.py`. It runs `e3a`, the extensions `e3a_ext1`–`e3a_ext3` (148, 28 and 6 runs), `select`, `e3b`, `evaluate` and `plot_e3.py`, in that order | ~1 h 45 min |
| F1 follow-up (not pre-registered; needs E3) | `python code/run_branches.py --followup`. The same steps with `f1a`, `f1a_ext1`–`f1a_ext3` (64, 48 and 4 runs) and `f1b` | ~1 h 10 min |
| E3, E4 and F1 figures only | `python code/plot_e3.py` | seconds |
| Timing benchmark (quiet machine) | `python code/benchmark_step_time.py` (5 passes; the median per cell is kept) | ~5 min |
| Per-second comparison | `python code/analyze_time.py` (needs E1 and the timing benchmark) | seconds |
| Report Figures 1–3 (need E1, the timing step, E3 and F1) | `python code/plot_report.py` | seconds |
| Report Tables 2 and A1 | `python code/make_tables.py` | seconds |
| Report PDF | Compile `report/main.tex` with pdflatex and bibtex, or on Overleaf | — |

**Notes on running.**

- Sweeps can be resumed. A run is skipped only when its result JSON and every checkpoint requested by its configuration exist. If a required checkpoint is missing, the run is repeated to regenerate it, and its result JSON is rewritten.
- The branch steps can also be run one at a time: `run.py <config>`, then `analyze_branches.py extend | select | evaluate [--followup]`. `select` and `evaluate` refuse while a tuning run is unfinished, a best multiplier sits on an edge that can still be extended, or the evaluation config predates the latest tuning runs (protocol Section 6). `extend` says which config to run next.
- Each run writes a JSON record to `results/<experiment>/`. It holds the config, seed, git commit, package versions, the full evaluation history and a summary.
- Checkpoints go to `checkpoints/` and are not committed. On a fresh checkout, regenerate the E2 checkpoints with `python code/run.py configs/e2_shared.yaml --workers 4`. The runner detects the missing checkpoints even when E2 result JSON files already exist. This repeats the E2 training and rewrites the corresponding E2 result files.
- Regenerating E2 checkpoints does not automatically rerun completed E3 or F1 branches. To reproduce all training from scratch, use a separate checkout, move its existing `results/` directory outside that checkout to preserve the supplied results, and follow the experiment workflow. Analyse the newly generated results separately from the supplied results.

## Layout

| Path | Contents |
|---|---|
| `code/muonlab/` | Update rules (`optim.py`), MLP, teacher–student data, noise diagnostics, training loop with branching, sweep runner |
| `code/*.py` | Command-line entry points: run, run a branch stage, analyse, plot, benchmark, smoke test |
| `code/checks/` | One-off investigations, kept so they can be re-run |
| `configs/` | One YAML file per experiment |
| `tests/` | Unit tests (pytest) |
| `results/` | Saved numerical results (committed) |
| `report/` | `main.tex` (report source), `references.bib`, and `outline.md` (plan, page budget, where each number comes from) |
| `report/figures/`, `report/tables/` | Figures and tables generated from `results/` |
| `notes/` | Protocol, lab notebook, reading list, novelty check |

## Known limitations

- **Small networks on one synthetic task:** results describe behaviour in the tested setting; they do not establish a causal noise mechanism or generalise directly to large-model training.
- **float32 throughout,** whereas the reference implementations use bfloat16. bfloat16 is 200–400× slower on this CPU; see the lab notebook (15 Oct).
- **Exact-SVD Muon is not Newton–Schulz Muon;** both are reported.
- **Few seeds:** E1 selects and reports settings on the same 2 seeds. E3/F1 tune on seeds 0 and 1 and evaluate on seeds 2, 3 and 4. F1 was designed after inspecting earlier results and remains exploratory.
- **Step timings depend on machine load:** the sub-millisecond steps varied by up to 2× between benchmark passes. Per-second results therefore use the median of 5 passes, and their conclusions are checked against single passes (lab notebook, 28 Oct).

## Report claim traceability

| Report item | Saved evidence | Generation or implementation |
|---|---|---|
| Baselines and equal-time comparison | `results/e1_baselines/tuning_summary.csv`, `results/timing/per_second_summary.json` | `code/analyze_tuning.py`, `code/analyze_time.py`, `code/plot_report.py` |
| Branch comparisons and noise indicator | `results/e3_branches/evaluation_summary.json`, `results/e2_shared/` | `code/analyze_branches.py`, `code/muonlab/diagnostics.py`, `code/plot_report.py` |
| Exploratory F1 | `results/e3_branches/evaluation_summary_followup.json` | `code/analyze_branches.py`, `code/plot_report.py` |
| Tables and multipliers | branch summaries, `configs/e3b_branches_eval.yaml`, `configs/f1b_branches_eval.yaml` | `code/make_tables.py` |


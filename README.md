# Muon optimiser project (DSA4212)

**Repository:** https://github.com/jiayunchee/dsa4212_Muon

**Status, 28 Oct 2026:**

- The experiment protocol is fixed in `notes/protocol.md`; later changes are in its change log.
- E1–E4 and the per-second timing are run.
- Follow-up F1 (p = −0.5 and −0.75, not pre-registered) is complete.
- The report has not been started.

Authors are listed in `authors.md`.

## Question

Late in training, should Muon flatten its update more or less, and does the answer depend on gradient noise?

- **The family.** Muon replaces each hidden-layer update matrix N = U S Vᵀ by U Vᵀ. We study U S^p Vᵀ:
  - p = 0 is Muon;
  - p = 1 is (normalised) momentum SGD;
  - p < 0 boosts weak directions even more than Muon does.
- **Why it's open.** Two 2026 papers make opposite predictions for late training (Xie, arXiv:2609.32861; Wu et al., arXiv:2605.17109). We test whether the best p depends on the noise level. The noise is controlled through the batch size on a synthetic teacher–student task with a known optimum.
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
| Report Figures 1–3 (need E1, the timing step, E3 and F1) | `python code/plot_report.py` | seconds |
| Report Tables 2 and A1 | `python code/make_tables.py` | seconds |
| Report PDF | Compile `report/main.tex` with pdflatex and bibtex, or on Overleaf | — |
| Timing benchmark (quiet machine) | `python code/benchmark_step_time.py` (5 passes; the median per cell is kept) | ~5 min |
| Per-second comparison | `python code/analyze_time.py` (needs E1 and the timing benchmark) | seconds |

**Notes on running.**

- Sweeps can be resumed: a run whose result file exists is skipped.
- The branch steps can also be run one at a time: `run.py <config>`, then `analyze_branches.py extend | select | evaluate [--followup]`. `select` and `evaluate` refuse while a tuning run is unfinished, a best multiplier sits on an edge that can still be extended, or the evaluation config predates the latest tuning runs (protocol Section 6). `extend` says which config to run next.
- Each run writes a JSON record to `results/<experiment>/`. It holds the config, seed, git commit, package versions, the full evaluation history and a summary.
- Checkpoints go to `checkpoints/`. They are not committed; rerun E2 to regenerate them.

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

- **Small networks on a synthetic task:** results speak to mechanisms, not to large-model training.
- **float32 throughout,** whereas the reference implementations use bfloat16. bfloat16 is 200–400× slower on this CPU; see the lab notebook (15 Oct).
- **Exact-SVD Muon is not Newton–Schulz Muon;** both are reported.
- **Few seeds:** 2 seeds for tuning, 3 for evaluation.
- **Step timings depend on machine load:** the sub-millisecond steps varied by up to 2× between benchmark passes. Per-second results therefore use the median of 5 passes, and their conclusions are checked against single passes (lab notebook, 28 Oct).

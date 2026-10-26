# Muon optimiser project (DSA4212)

**Repository:** https://github.com/jiayunchee/dsa4212_Muon

**Status, 20 Oct 2026:**

- The experiment protocol is fixed in `notes/protocol.md`.
- The baselines (E1) are run; the main experiments (E2–E4) are in progress.
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
| E3 tuning branches | `python code/run.py configs/e3a_branches_tune.yaml --workers 4`, then `python code/analyze_branches.py select` | ~50 min |
| E3 evaluation branches | `python code/run.py configs/e3b_branches_eval.yaml --workers 4`, then `python code/analyze_branches.py evaluate` and `python code/plot_e3.py` | ~25 min |
| Timing benchmark (quiet machine) | `python code/benchmark_step_time.py` | ~5 min |

**Notes on running.**

- Sweeps can be resumed: a run whose result file exists is skipped.
- Each run writes a JSON record to `results/<experiment>/`. It holds the config, seed, git commit, package versions, the full evaluation history and a summary.
- Checkpoints go to `checkpoints/`. They are not committed; rerun E2 to regenerate them.

## Layout

| Path | Contents |
|---|---|
| `code/muonlab/` | Update rules (`optim.py`), MLP, teacher–student data, noise diagnostics, training loop with branching, sweep runner |
| `code/*.py` | Command-line entry points: run, analyse, plot, benchmark, smoke test |
| `code/checks/` | One-off investigations, kept so they can be re-run |
| `configs/` | One YAML file per experiment |
| `tests/` | Unit tests (pytest) |
| `results/` | Saved numerical results (committed) |
| `report/figures/` | Figures generated from `results/` |
| `notes/` | Protocol, lab notebook, reading list, novelty check |

## Known limitations

- **Small networks on a synthetic task:** results speak to mechanisms, not to large-model training.
- **float32 throughout,** whereas the reference implementations use bfloat16. bfloat16 is 200–400× slower on this CPU; see the lab notebook (15 Oct).
- **Exact-SVD Muon is not Newton–Schulz Muon;** both are reported.
- **Few seeds:** 2 seeds for tuning, 3 for evaluation.

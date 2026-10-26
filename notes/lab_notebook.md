# Lab notebook

A dated record of what was done, what was observed and what was decided. Newest entries go at the top. Measured facts and interpretations are kept apart.

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
  - **Overturned by E1 (same day).** With tuned learning rates, NS5 Muon has the *lowest* excess risk at every batch size. The pilot's Muon learning rates (0.01–0.03) were simply too high. This shows why the protocol requires equal tuning before any comparison. Details are in the E1 entry.
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

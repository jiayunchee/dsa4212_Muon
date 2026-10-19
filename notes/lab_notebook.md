# Lab notebook

A dated record of what was done, what was observed and what was decided. Newest entries go at the top. Measured facts and interpretations are kept apart.

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

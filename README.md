# Muon optimiser project (DSA4212)

**Status, 15 Oct 2026:** the environment is set up and smoke-tested. The research question, experiment protocol, group members and repository URL are not final yet.

- GitHub URL: *to be added*
- Authors: *to be added in `authors.md`*

## Environment

Tested on a MacBook Air (Apple M3, 16 GB, macOS 26.6) with Python 3.12.15 and PyTorch 2.14.1.

```bash
conda create -n muon --override-channels -c conda-forge python=3.12 pip -y
conda activate muon
pip install -r requirements.txt
```

`requirements.txt` pins the packages used directly. `requirements-lock.txt` is the full `pip freeze` of the tested environment.

## Smoke test

```bash
python code/smoke_test.py          # about 5 minutes on an M3 MacBook Air
python code/smoke_test.py --quick  # about 2 minutes
```

This checks the toolchain. It does not produce research results: the optimisers are untuned and the task is too easy to separate them. The script:

1. records software versions and devices;
2. compares float32 and bfloat16 matrix multiplication on the CPU and the Apple GPU (MPS);
3. checks our Newton–Schulz implementation against PyTorch's, and a minimal Muon against `torch.optim.Muon`;
4. times one training step for AdamW and Muon+AdamW at batch sizes 32 to 2048;
5. trains a small MLP for a few epochs with AdamW, SGD+Nesterov and two Muon variants.

Results go to `results/smoke/`. By default the script uses the binary Fashion-MNIST subset from the DSA4212 exercises (set the folder with `--data-dir`). If that folder is missing, it falls back to synthetic data.

`code/checks/ns_bf16_rounding.py` reproduces the investigation of a bf16 discrepancy found during the smoke test (see `notes/lab_notebook.md`).

## Known limitation on Apple-silicon Macs

bfloat16 matrix multiplication on the CPU is roughly 200–400× slower than float32 with this PyTorch build. `torch.optim.Muon` always runs Newton–Schulz in bfloat16, so on the CPU use an fp32 Newton–Schulz, or run on the Apple GPU (MPS), where bfloat16 is fast.

## Layout (provisional; will follow the official brief)

| Path | Contents |
|---|---|
| `code/` | training, optimiser and analysis code |
| `code/checks/` | one-off investigations, kept so they can be re-run |
| `results/` | saved numerical results (committed) |
| `notes/` | lab notebook and reading list |

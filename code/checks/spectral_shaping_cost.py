"""Cost and accuracy of exact spectral shaping U S^p V^T on this machine.

Checked on 19 Oct 2026 for the provisional design, which needs exact U S^p V^T
(any exponent p, including negative ones) every step for small hidden matrices.
Two routes:
  - SVD:  M = U S V^T, then U S^p V^T;
  - eigh: (M M^T)^((p-1)/2) M, from the Gram matrix. Faster, but squaring M also
    squares its condition number, so it loses accuracy for negative p in fp32.

Usage:  python code/checks/spectral_shaping_cost.py
"""

import statistics
import time

import torch


def median_ms(fn, repeats: int = 30) -> float:
    for _ in range(3):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


def shaped_svd(M: torch.Tensor, p: float) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return (U * S.clamp_min(1e-12).pow(p)) @ Vh


def shaped_eigh(M: torch.Tensor, p: float) -> torch.Tensor:
    evals, Q = torch.linalg.eigh(M @ M.T)
    return (Q * evals.clamp_min(1e-24).pow((p - 1) / 2)) @ Q.T @ M


def main() -> None:
    torch.manual_seed(0)
    p = -0.25
    print(f"torch {torch.__version__}, CPU threads {torch.get_num_threads()}, float32, p = {p}")
    for n in (128, 256, 512):
        M = torch.randn(n, n)
        svd_ms = median_ms(lambda: shaped_svd(M, p))
        eigh_ms = median_ms(lambda: shaped_eigh(M, p))
        ref = shaped_svd(M.double(), p)
        err_svd = ((shaped_svd(M, p).double() - ref).norm() / ref.norm()).item()
        err_eigh = ((shaped_eigh(M, p).double() - ref).norm() / ref.norm()).item()
        print(f"n={n:4d}  SVD {svd_ms:6.2f} ms (err vs fp64 {err_svd:.1e}) | "
              f"eigh {eigh_ms:6.2f} ms (err vs fp64 {err_eigh:.1e})")


if __name__ == "__main__":
    main()

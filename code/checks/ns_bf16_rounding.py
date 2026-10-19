"""Why does our bf16 Newton-Schulz differ from PyTorch's bf16 Newton-Schulz?

Found during the smoke test on 15 Oct 2026: the two bf16 outputs differed by ~10%
(relative Frobenius norm). This script checks whether that is a bug or rounding:
  - replica check: our copy of PyTorch's operation order reproduces PyTorch exactly;
  - in fp32, our implementation and PyTorch's operation order must agree (~1e-6);
  - in bf16, both drift from the fp32 answer; we measure by how much.

Usage:  python code/checks/ns_bf16_rounding.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make code/ importable
from smoke_test import NS_COEFFS, NS_STEPS, newton_schulz5, torch_style_newton_schulz5  # noqa: E402
from torch.optim._muon import _zeropower_via_newtonschulz  # noqa: E402


def rel(P: torch.Tensor, Q: torch.Tensor) -> float:
    return ((P.double() - Q.double()).norm() / Q.double().norm()).item()


def matrix_with_spectrum(gen, rows, cols, log10_smallest):
    U, _ = torch.linalg.qr(torch.randn(rows, cols, generator=gen, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(cols, cols, generator=gen, dtype=torch.float64))
    s = torch.logspace(0, log10_smallest, cols, dtype=torch.float64)
    return ((U * s) @ V.T).float()


def main() -> None:
    gen = torch.Generator().manual_seed(0)
    cases = {
        "512x256, singular values 1 ... 1e-4 (smoke-test matrix)": matrix_with_spectrum(gen, 512, 256, -4),
        "512x512, singular values 1 ... 1e-2": matrix_with_spectrum(gen, 512, 512, -2),
        "512x512, Gaussian entries": torch.randn(512, 512, generator=gen),
    }
    torch_bf16 = lambda G: _zeropower_via_newtonschulz(G, NS_COEFFS, NS_STEPS, 1e-7)  # noqa: E731
    for name, G in cases.items():
        fp32 = newton_schulz5(G, dtype=torch.float32)
        print(name)
        print(f"  replica of PyTorch order (bf16) vs PyTorch   {rel(torch_style_newton_schulz5(G, torch.bfloat16), torch_bf16(G)):.1e}")
        print(f"  ours vs PyTorch order, both fp32              {rel(fp32, torch_style_newton_schulz5(G, torch.float32)):.1e}")
        print(f"  ours bf16 vs ours fp32                        {rel(newton_schulz5(G, dtype=torch.bfloat16), fp32):.3f}")
        print(f"  PyTorch bf16 vs ours fp32                     {rel(torch_bf16(G), fp32):.3f}")
        print(f"  ours bf16 vs PyTorch bf16                     {rel(newton_schulz5(G, dtype=torch.bfloat16), torch_bf16(G)):.3f}")


if __name__ == "__main__":
    main()

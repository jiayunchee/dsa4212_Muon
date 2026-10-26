"""Data for the Muon project."""

from __future__ import annotations

import math

import torch


class TeacherStudentStream:
    """Streaming regression data with a known optimum (Task A in notes/protocol.md).

    Inputs x ~ N(0, Sigma) with Sigma = Q diag(lam) Q^T, lam_i proportional to i^(-power)
    and trace(Sigma) = d_in, so the input is ill-conditioned. Targets are
    y = f*(x) + label_noise * eps, where f* is a fixed random tanh MLP whose outputs are
    standardised to unit variance. The excess risk of a model f is E (f(x) - f*(x))^2, which
    `eval_set` lets us measure exactly, without estimating the noise floor.
    """

    def __init__(self, d_in: int = 128, teacher_hidden: tuple[int, ...] = (64, 64), d_out: int = 16,
                 power: float = 1.0, label_noise: float = 0.5, teacher_seed: int = 0):
        gen = torch.Generator().manual_seed(teacher_seed)
        Q, _ = torch.linalg.qr(torch.randn(d_in, d_in, generator=gen))
        lam = torch.arange(1, d_in + 1, dtype=torch.float32).pow(-power)
        lam = lam * d_in / lam.sum()
        self.input_map = Q * lam.sqrt()  # x = input_map @ z with z ~ N(0, I)
        widths = [d_in, *teacher_hidden, d_out]
        self.teacher = [torch.randn(b, a, generator=gen) / math.sqrt(a)
                        for a, b in zip(widths[:-1], widths[1:])]
        z = torch.randn(20_000, d_in, generator=gen)
        raw = self._teacher_raw(z @ self.input_map.T)
        self.out_mean, self.out_std = raw.mean(dim=0), raw.std(dim=0)
        self.d_in, self.d_out, self.label_noise = d_in, d_out, label_noise

    def _teacher_raw(self, x: torch.Tensor) -> torch.Tensor:
        for W in self.teacher[:-1]:
            x = torch.tanh(x @ W.T)
        return x @ self.teacher[-1].T

    def target(self, x: torch.Tensor) -> torch.Tensor:
        """Noiseless teacher output f*(x), standardised to zero mean and unit variance."""
        return (self._teacher_raw(x) - self.out_mean) / self.out_std

    def sample(self, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """A fresh minibatch (x, noisy y); `generator` fixes the data stream."""
        x = torch.randn(n, self.d_in, generator=generator) @ self.input_map.T
        y = self.target(x) + self.label_noise * torch.randn(n, self.d_out, generator=generator)
        return x, y

    def eval_set(self, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed evaluation inputs with *noiseless* targets, so the MSE on it is the excess risk."""
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn(n, self.d_in, generator=gen) @ self.input_map.T
        return x, self.target(x)

"""Noise and spectrum diagnostics (hypothesis H3 and Xie's matched learning rate).

Xie (arXiv:2609.32861) models the minibatch gradient of a matrix as G = Y + sigma_g * Z, with Y
the mean gradient and Z standard Gaussian. Muon's polar map sees the Nesterov input N, whose
noise has standard deviation sigma_N per entry. Two quantities follow:
    delta  = ||Y||_F / (sigma_N * sqrt(m n))    signal-to-noise indicator (Xie, eq. 32)
    lr_sgd = lr_muon * c_{m,n} / sigma_N         momentum SGD with the same mean response as Muon
                                                 (Xie, Prop. 3.1)
where c_{m,n} = E||Z||_* / (m n) for an m x n standard Gaussian matrix Z. MatrixMomentum applies
the shape factor sqrt(max(1, m/n)) to every update rule, so it cancels and is not in lr_sgd.
"""

from __future__ import annotations

import math
from typing import Callable

import torch


def gaussian_nuclear_constant(m: int, n: int, samples: int = 32, seed: int = 0) -> float:
    """Monte Carlo estimate of c_{m,n} = E||Z||_* / (m n); c_{N,N} * sqrt(N) -> 8 / (3 pi)."""
    gen = torch.Generator().manual_seed(seed)
    total = sum(torch.linalg.svdvals(torch.randn(m, n, generator=gen, dtype=torch.float64)).sum().item()
                for _ in range(samples))
    return total / samples / (m * n)


def momentum_noise_factor(beta: float, nesterov: bool = True) -> float:
    """sigma_N / sigma_g in stationarity, for independent gradient noise.

    EMA buffer M = beta*M + (1-beta)*G has noise variance sigma_g^2 (1-beta)/(1+beta).
    The Nesterov input N = (1-beta)(1+beta) G_t + beta^2 M_{t-1} adds the two independent parts.
    """
    ema = (1 - beta) / (1 + beta)
    if not nesterov:
        return math.sqrt(ema)
    return math.sqrt(((1 - beta) * (1 + beta)) ** 2 + beta**4 * ema)


def moments_from_gradients(grads: torch.Tensor) -> tuple[float, float]:
    """From k minibatch gradients of one matrix (shape k x m x n), estimate
    (||Y||_F, sigma_g): the mean-gradient norm, bias-corrected for the noise left in the
    average, and the per-entry noise standard deviation."""
    k = grads.shape[0]
    G = grads.double()
    mean = G.mean(dim=0)
    sigma_g = G.var(dim=0, unbiased=True).mean().sqrt().item()
    signal_sq = mean.square().sum().item() - mean.numel() * sigma_g**2 / k
    return math.sqrt(max(signal_sq, 0.0)), sigma_g


def gradient_noise_stats(model: torch.nn.Module, loss_fn: Callable, sample_batch: Callable,
                         matrices: list[torch.Tensor], k: int = 32) -> list[dict]:
    """Mean-gradient norm and noise level for each matrix from k fresh minibatches.

    The weights are not changed. Use a `sample_batch` with its own random generator so the
    diagnostic does not alter the training data stream.
    """
    collected = [[] for _ in matrices]
    for _ in range(k):
        x, y = sample_batch()
        model.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        for store, W in zip(collected, matrices):
            store.append(W.grad.detach().clone())
    model.zero_grad(set_to_none=True)
    stats = []
    for store, W in zip(collected, matrices):
        signal, sigma_g = moments_from_gradients(torch.stack(store))
        stats.append({"shape": list(W.shape), "mean_grad_norm": signal, "sigma_g": sigma_g})
    return stats


def xie_quantities(stat: dict, beta: float, nesterov: bool, lr_muon: float, c_mn: float) -> dict:
    """Signal-to-noise indicator delta and the matched momentum-SGD learning rate for one matrix."""
    m, n = stat["shape"]
    sigma_N = stat["sigma_g"] * momentum_noise_factor(beta, nesterov)
    return {
        "sigma_N": sigma_N,
        "delta": stat["mean_grad_norm"] / (sigma_N * math.sqrt(m * n)),
        "matched_lr": lr_muon * c_mn / sigma_N,  # the optimizer adds the shape factor itself
    }


def spectrum_stats(N: torch.Tensor) -> dict:
    """Numerical, effective (entropy) and stable rank of a matrix, plus its extreme singular values."""
    s = torch.linalg.svdvals(N.double())
    tol = s[0] * max(N.shape) * torch.finfo(torch.float32).eps
    q = s[s > 0] / s.sum()
    return {
        "numerical_rank": int((s > tol).sum()),
        "effective_rank": math.exp(-(q * q.log()).sum().item()),
        "stable_rank": (s.square().sum() / s[0] ** 2).item(),
        "top_singular_value": s[0].item(),
        "smallest_singular_value": s[-1].item(),
    }

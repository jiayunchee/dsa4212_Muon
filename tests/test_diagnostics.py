"""Tests for the noise diagnostics used for hypothesis H3 and Xie's matched learning rate."""

import math

import pytest
import torch

from muonlab.diagnostics import (gaussian_nuclear_constant, moments_from_gradients,
                                 momentum_noise_factor, spectrum_stats, xie_quantities)


def test_moments_recover_known_signal_and_noise():
    gen = torch.Generator().manual_seed(0)
    Y = 0.05 * torch.randn(32, 16, generator=gen)
    sigma = 0.3
    grads = Y + sigma * torch.randn(400, 32, 16, generator=gen)
    signal, sigma_hat = moments_from_gradients(grads)
    assert sigma_hat == pytest.approx(sigma, rel=0.02)
    assert signal == pytest.approx(Y.norm().item(), rel=0.1)


@pytest.mark.parametrize("nesterov", [False, True])
def test_momentum_noise_factor_matches_simulation(nesterov):
    beta, chains, steps = 0.9, 20_000, 300
    gen = torch.Generator().manual_seed(1)
    M = torch.zeros(chains, dtype=torch.float64)
    for _ in range(steps):
        g = torch.randn(chains, generator=gen, dtype=torch.float64)
        M = beta * M + (1 - beta) * g
        N = (1 - beta) * g + beta * M if nesterov else M
    assert N.std().item() == pytest.approx(momentum_noise_factor(beta, nesterov), rel=0.03)


def test_gaussian_nuclear_constant_approaches_its_limit():
    c = gaussian_nuclear_constant(128, 128, samples=8)
    assert c * math.sqrt(128) == pytest.approx(8 / (3 * math.pi), rel=0.03)


def test_xie_quantities_formula():
    stat = {"shape": [256, 256], "mean_grad_norm": 2.0, "sigma_g": 0.01}
    q = xie_quantities(stat, beta=0.9, nesterov=False, lr_muon=0.02, c_mn=0.05)
    sigma_N = 0.01 * math.sqrt(0.1 / 1.9)
    assert q["sigma_N"] == pytest.approx(sigma_N)
    assert q["delta"] == pytest.approx(2.0 / (sigma_N * 256))
    assert q["matched_lr"] == pytest.approx(0.02 * 0.05 / sigma_N)


def test_spectrum_stats_on_a_known_matrix():
    D = torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.0]))
    s = spectrum_stats(D)
    assert s["numerical_rank"] == 3
    assert s["stable_rank"] == pytest.approx((16 + 4 + 1) / 16)

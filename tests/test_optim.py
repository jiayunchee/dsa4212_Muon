"""Tests for the update rules and the parameter split (notes/protocol.md, Section 7)."""

import math

import pytest
import torch
import torch.nn as nn

import muonlab.optim as mo
from muonlab.models import MLP


def randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def test_spectral_p0_is_the_polar_factor():
    N = randn(64, 32)
    D = mo.spectral_direction(N, 0.0)
    U, _, Vh = torch.linalg.svd(N, full_matrices=False)
    assert torch.allclose(D, U @ Vh, atol=1e-5)
    assert torch.allclose(D.T @ D, torch.eye(32), atol=1e-5)


def test_spectral_p1_is_the_normalised_input():
    N = randn(48, 64, seed=1)
    D = mo.spectral_direction(N, 1.0)
    assert torch.allclose(D, N * math.sqrt(48) / N.norm(), atol=1e-5)


@pytest.mark.parametrize("p", [-0.25, 0.5, 2.0])
def test_rescaling_gives_frobenius_norm_sqrt_rank(p):
    N = randn(40, 56, seed=2)
    assert mo.spectral_direction(N, p).norm().item() == pytest.approx(math.sqrt(40), rel=1e-4)


@pytest.mark.parametrize("p", [0.0, -0.25])
def test_rank_deficient_input_does_not_amplify_rounding_error(p):
    N = randn(64, 5, seed=3) @ randn(5, 64, seed=4)  # exact rank 5
    D = mo.spectral_direction(N, p)
    assert torch.linalg.matrix_rank(D).item() == 5
    assert D.norm().item() ** 2 == pytest.approx(5.0, rel=1e-3)


def test_spectral_direction_falls_back_to_float64_when_svd_fails(monkeypatch):
    real_svd = torch.linalg.svd

    def flaky_svd(A, full_matrices=True):
        if A.dtype == torch.float32:
            raise torch.linalg.LinAlgError("simulated non-convergence")
        return real_svd(A, full_matrices=full_matrices)

    monkeypatch.setattr(torch.linalg, "svd", flaky_svd)
    before = mo.SVD_FALLBACKS["count"]
    N = randn(32, 16, seed=6)
    D = mo.spectral_direction(N, 0.0)
    U, _, Vh = real_svd(N.double(), full_matrices=False)
    assert mo.SVD_FALLBACKS["count"] == before + 1
    assert D.dtype == torch.float32
    assert torch.allclose(D.double(), U @ Vh, atol=1e-5)


def test_ns5_matches_pytorch_operation_order_in_fp32():
    a, b, c = mo.NS_COEFFS
    G = randn(96, 48, seed=5)
    X = G.T / G.norm().clamp(min=1e-7)  # PyTorch 2.14 sequence (fused addmm), here in fp32
    for _ in range(5):
        A = X @ X.T
        X = torch.addmm(X, torch.addmm(A, A, A, beta=b, alpha=c), X, beta=a)
    reference = X.T
    ours = mo.newton_schulz5(G)
    assert ((ours - reference).norm() / reference.norm()).item() < 1e-5


def test_ns5_rule_matches_torch_muon_step_by_step(monkeypatch):
    """With PyTorch's own Newton-Schulz plugged in, every step must equal torch.optim.Muon:
    this checks the momentum, Nesterov, shape scaling and weight-decay logic."""
    from torch.optim._muon import _zeropower_via_newtonschulz
    monkeypatch.setattr(mo, "newton_schulz5",
                        lambda G: _zeropower_via_newtonschulz(G, mo.NS_COEFFS, 5, 1e-7).to(G.dtype))
    shapes = [(64, 32), (32, 64), (48, 48)]  # tall, wide, square
    init = [randn(*s, seed=i) for i, s in enumerate(shapes)]
    p_torch = [nn.Parameter(w.clone()) for w in init]
    p_ours = [nn.Parameter(w.clone()) for w in init]
    reference = torch.optim.Muon(p_torch, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.1,
                                 adjust_lr_fn="original")
    ours = mo.MatrixMomentum(p_ours, lr=0.02, momentum=0.95, nesterov=True, update="ns5", weight_decay=0.1)
    for step in range(5):
        for i, (pt, po) in enumerate(zip(p_torch, p_ours)):
            g = randn(*pt.shape, seed=100 + 10 * step + i)
            pt.grad, po.grad = g.clone(), g.clone()
        reference.step()
        ours.step()
    assert max((pt - po).abs().max().item() for pt, po in zip(p_torch, p_ours)) == 0.0


def test_plain_rule_is_momentum_sgd_on_the_ema_buffer():
    """Plain rule = momentum SGD on Muon's EMA buffer, times the shape factor sqrt(max(1, m/n))
    that MatrixMomentum applies to every rule (here 4 x 3, so sqrt(4/3))."""
    W = nn.Parameter(torch.zeros(4, 3))
    opt = mo.MatrixMomentum([W], lr=0.1, momentum=0.9, nesterov=False, update="plain")
    M, expected = torch.zeros(4, 3), torch.zeros(4, 3)
    for step in range(3):
        G = randn(4, 3, seed=step)
        W.grad = G.clone()
        opt.step()
        M = 0.9 * M + 0.1 * G
        expected -= 0.1 * math.sqrt(4 / 3) * M
    assert torch.allclose(W.detach(), expected, atol=1e-7)


def test_parameter_split_for_muon():
    model = MLP([8, 16, 16, 16, 4])
    bundle, assignment = mo.build_optimizer(model, {"name": "muon", "lr": 0.02, "adam_lr": 1e-3})
    matrix = [q for g in bundle.optimizers["matrix"].param_groups for q in g["params"]]
    adam = [q for g in bundle.optimizers["adamw"].param_groups for q in g["params"]]
    assert [tuple(q.shape) for q in matrix] == [(16, 16), (16, 16)]
    assert len(matrix) + len(adam) == len(list(model.parameters()))  # nothing missed or doubled
    by_name = {a["name"]: a["optimizer"] for a in assignment}
    assert by_name["layers.1.weight"] == by_name["layers.2.weight"] == "MatrixMomentum"
    assert by_name["layers.0.weight"] == by_name["layers.3.weight"] == "AdamW"
    assert all(v == "AdamW" for k, v in by_name.items() if k.endswith("bias"))


def test_matrix_momentum_rejects_non_matrices():
    with pytest.raises(ValueError):
        mo.MatrixMomentum([nn.Parameter(torch.zeros(5))], lr=0.1)

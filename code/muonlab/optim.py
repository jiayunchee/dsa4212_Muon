"""Optimisers for the Muon project.

Matrix optimisers act on the hidden 2-D weight matrices only. Everything else (first layer,
output layer, biases) is trained with AdamW; `build_optimizer` makes that split.

For one hidden matrix W (m x n) with minibatch gradient G (notes/protocol.md, Section 2):
    M <- beta*M + (1-beta)*G              momentum buffer, EMA form (as in torch.optim.Muon)
    N  = (1-beta)*G + beta*M              Nesterov input (N = M without Nesterov)
    W <- W - lr * sqrt(max(1, m/n)) * D(N)
where the direction D(N) depends on the update rule:
    "ns5"       Keller Jordan's quintic Newton-Schulz, 5 steps (practical Muon)
    "spectral"  U diag(s^p) V^T from the exact SVD of N, rescaled to Frobenius norm sqrt(rank):
                p = 0 is exact Muon (the polar factor), p = 1 is normalised momentum SGD,
                p < 0 boosts weak directions beyond Muon (DynMuon's late-training choice)
    "plain"     N itself, unscaled: momentum SGD that keeps Muon's buffer (Xie's comparator)
"""

from __future__ import annotations

import torch
import torch.nn as nn

NS_COEFFS = (3.4445, -4.7750, 2.0315)
UPDATE_RULES = ("ns5", "spectral", "plain")
SVD_FALLBACKS = {"count": 0}  # float32 SVDs that failed and were redone in float64 (recorded per run)


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Approximate polar factor U V^T of G (Jordan, 2024).

    Runs in float32 by default: bfloat16 is 200-400x slower on this project's CPU
    (lab notebook, 15 Oct 2026). Each iteration maps every singular value s to
    a*s + b*s^3 + c*s^5 after Frobenius normalisation, so values end up near 1.
    """
    a, b, c = NS_COEFFS
    X = G.to(dtype)
    transposed = X.size(0) > X.size(1)
    if transposed:  # work with the wide orientation so X X^T is the small Gram matrix
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    X = X.T if transposed else X
    return X.to(G.dtype)


def spectral_direction(N: torch.Tensor, p: float, normalize: bool = True) -> torch.Tensor:
    """U diag(s^p) V^T for the thin SVD N = U diag(s) V^T.

    Singular values below the numerical-rank threshold s_1 * max(m, n) * eps are treated as
    zero; otherwise p <= 0 would blow rounding error up to full size. With normalize=True the
    result is rescaled to Frobenius norm sqrt(r), the norm of the polar factor of a rank-r
    matrix, so that p changes how the step is spread across directions but not its length.
    """
    try:
        U, s, Vh = torch.linalg.svd(N, full_matrices=False)
    except torch.linalg.LinAlgError:
        # LAPACK's float32 divide-and-conquer SVD occasionally fails to converge on finite,
        # moderately conditioned matrices (E0 pilot, 20 Oct: condition number 4e4); float64 works.
        SVD_FALLBACKS["count"] += 1
        U, s, Vh = (t.to(N.dtype) for t in torch.linalg.svd(N.double(), full_matrices=False))
    if s.numel() == 0 or s[0] <= 0:
        return torch.zeros_like(N)
    keep = s > s[0] * max(N.shape) * torch.finfo(s.dtype).eps
    weights = torch.zeros_like(s)
    weights[keep] = s[keep].pow(p)
    if normalize:
        weights = weights * torch.sqrt(keep.sum() / weights.square().sum())
    return (U * weights) @ Vh


class MatrixMomentum(torch.optim.Optimizer):
    """Muon-style optimiser for 2-D parameters with a selectable update rule.

    The rule, the exponent p and the learning rate live in each param group, so a run can
    switch rule mid-training (a "branch") while keeping its momentum buffer.
    """

    def __init__(self, params, lr: float, momentum: float = 0.95, nesterov: bool = True,
                 update: str = "ns5", p: float = 0.0, normalize: bool = True,
                 weight_decay: float = 0.0):
        if update not in UPDATE_RULES:
            raise ValueError(f"update must be one of {UPDATE_RULES}, got {update!r}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, update=update, p=p,
                        normalize=normalize, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for W in group["params"]:
                if W.ndim != 2:
                    raise ValueError(f"MatrixMomentum takes 2-D parameters only, got {tuple(W.shape)}")

    @staticmethod
    def direction(N: torch.Tensor, group: dict) -> torch.Tensor:
        rule = group["update"]
        if rule == "ns5":
            return newton_schulz5(N)
        if rule == "spectral":
            return spectral_direction(N, group["p"], group["normalize"])
        if rule == "plain":
            return N
        raise ValueError(f"unknown update rule {rule!r}")

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            beta = group["momentum"]
            for W in group["params"]:
                if W.grad is None:
                    continue
                state = self.state[W]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(W)
                M = state["momentum_buffer"]
                M.lerp_(W.grad, 1 - beta)
                N = W.grad.lerp(M, beta) if group["nesterov"] else M
                D = self.direction(N, group)
                if group["weight_decay"]:
                    W.mul_(1 - group["lr"] * group["weight_decay"])
                W.add_(D, alpha=-group["lr"] * max(1.0, W.size(0) / W.size(1)) ** 0.5)


class OptimizerBundle:
    """Several optimisers stepped as one, e.g. MatrixMomentum for hidden matrices + AdamW."""

    def __init__(self, optimizers: dict[str, torch.optim.Optimizer]):
        self.optimizers = optimizers
        for opt in optimizers.values():
            for group in opt.param_groups:
                group.setdefault("base_lr", group["lr"])

    def zero_grad(self) -> None:
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)

    def step(self) -> None:
        for opt in self.optimizers.values():
            opt.step()

    def set_lr_factor(self, factor: float) -> None:
        """Learning-rate schedule hook: every group's lr becomes base_lr * factor."""
        for opt in self.optimizers.values():
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * factor

    def state_dict(self) -> dict:
        return {name: opt.state_dict() for name, opt in self.optimizers.items()}

    def load_state_dict(self, state: dict) -> None:
        for name, opt in self.optimizers.items():
            opt.load_state_dict(state[name])


def build_optimizer(model: nn.Module, cfg: dict) -> tuple[OptimizerBundle, list[dict]]:
    """Create the optimiser named in cfg["name"] and report which parameters it trains.

    adamw : AdamW on every parameter.
    sgd   : SGD with Nesterov momentum on every parameter (PyTorch's sum-form momentum).
    muon  : MatrixMomentum on model.hidden_matrices(), AdamW (lr = cfg["adam_lr"]) on the rest.
    """
    name = cfg["name"]
    if name == "adamw":
        bundle = OptimizerBundle({"adamw": torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=cfg.get("weight_decay", 0.0))})
        hidden_ids: set[int] = set()
    elif name == "sgd":
        bundle = OptimizerBundle({"sgd": torch.optim.SGD(
            model.parameters(), lr=cfg["lr"], momentum=cfg.get("momentum", 0.9), nesterov=True,
            weight_decay=cfg.get("weight_decay", 0.0))})
        hidden_ids = set()
    elif name == "muon":
        hidden = model.hidden_matrices()
        hidden_ids = {id(W) for W in hidden}
        rest = [q for q in model.parameters() if id(q) not in hidden_ids]
        bundle = OptimizerBundle({
            # one param group per matrix, so a branch can give each matrix its own lr
            "matrix": MatrixMomentum([{"params": [W]} for W in hidden], lr=cfg["lr"],
                                     momentum=cfg.get("momentum", 0.95),
                                     nesterov=cfg.get("nesterov", True), update=cfg.get("update", "ns5"),
                                     p=cfg.get("p", 0.0), normalize=cfg.get("normalize", True),
                                     weight_decay=cfg.get("weight_decay", 0.0)),
            "adamw": torch.optim.AdamW(rest, lr=cfg["adam_lr"], betas=(0.9, 0.999), weight_decay=0.0),
        })
    else:
        raise ValueError(f"unknown optimiser {name!r}")
    trained_by = {"adamw": "AdamW", "sgd": "SGD"}.get(name, "AdamW")
    assignment = [{"name": n, "shape": list(q.shape),
                   "optimizer": "MatrixMomentum" if id(q) in hidden_ids else trained_by}
                  for n, q in model.named_parameters()]
    return bundle, assignment

"""Smoke test for the Muon project environment (DSA4212).

This is a toolchain check, not an experiment. Nothing here is tuned, so no
number this script produces should be quoted as a research result.

It checks five things:
  1. Versions and devices: Python / PyTorch versions, CPU threads, Apple GPU (MPS).
  2. bfloat16 (bf16) matrix multiplication: does it run on each device, how
     accurate is it, and how fast is it compared with float32?
  3. Newton-Schulz (NS) sanity: our minimal NS5 matches PyTorch's internal one,
     and it maps singular values to roughly [0.7, 1.1] instead of exactly 1
     (by design, see Jordan 2024).
  4. Cost: one training step (forward + backward + optimiser step) with AdamW
     versus Muon+AdamW for several batch sizes, next to Jordan's FLOP estimate.
  5. A short MLP training run with AdamW, SGD+Nesterov and Muon+AdamW: the loss
     must decrease and stay finite.

Usage:
    python code/smoke_test.py            # full run (a few minutes on an M3 laptop)
    python code/smoke_test.py --quick    # fewer repeats and epochs

Results are written to results/smoke/smoke_<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path.home() / "Desktop" / "DSA4212" / "Exercises" / "logistic_regression"

# Keller Jordan's quintic Newton-Schulz coefficients (also PyTorch's defaults).
NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5

# Untuned settings for the smoke run only. Real experiments will tune every
# optimiser with the same budget (see the protocol, once agreed).
SMOKE_OPTIMIZERS = {
    "adamw": "AdamW on all parameters: lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01",
    "sgd_nesterov": "SGD on all parameters: lr=0.05, momentum=0.9, nesterov=True",
    "muon_adamw": (
        "torch.optim.Muon (Newton-Schulz in bf16) on hidden weight matrices: lr=0.02, momentum=0.95, "
        "nesterov=True, weight_decay=0, adjust_lr_fn='original'; AdamW (lr=1e-3, weight_decay=0.01) on the rest"
    ),
    "muon_fp32_adamw": (
        "MinimalMuon (this file: torch.optim.Muon's update with Newton-Schulz in fp32) on hidden weight "
        "matrices, same hyperparameters; AdamW (lr=1e-3, weight_decay=0.01) on the rest"
    ),
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def tidy_path(path: Path | str) -> str:
    """Replace the home directory with '~' so saved results carry no user name."""
    return str(path).replace(str(Path.home()), "~")


def available_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work, so wall-clock timings measure the actual compute."""
    if device.type == "mps":
        torch.mps.synchronize()


def median_seconds(fn, device: torch.device, warmup: int, repeats: int) -> float:
    """Median wall-clock time of fn(), synchronising the device after every call."""
    for _ in range(warmup):
        fn()
    synchronize(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def error_record(exc: Exception) -> dict:
    return {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(limit=3)}


# --------------------------------------------------------------------------- #
# Newton-Schulz orthogonalisation (our minimal version)
# --------------------------------------------------------------------------- #
def newton_schulz5(G: torch.Tensor, steps: int = NS_STEPS, eps: float = 1e-7,
                   dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Approximate the orthogonal factor U V^T of G = U S V^T (Jordan, 2024).

    Each iteration X <- aX + b(XX^T)X + c(XX^T)^2 X keeps U and V and maps every
    singular value s to a*s + b*s^3 + c*s^5. Dividing by the Frobenius norm first
    puts all singular values in [0, 1], where repeating the map pushes them
    towards ~1.
    """
    assert G.ndim == 2, "Newton-Schulz needs a matrix"
    a, b, c = NS_COEFFS
    X = G.to(dtype)
    transposed = X.size(0) > X.size(1)
    if transposed:  # use the wide orientation so the Gram matrix XX^T is the small one
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


def ns_scalar_map(x: torch.Tensor, steps: int = NS_STEPS) -> torch.Tensor:
    """What `steps` NS iterations do to a single normalised singular value x."""
    a, b, c = NS_COEFFS
    for _ in range(steps):
        x = a * x + b * x**3 + c * x**5
    return x


def torch_style_newton_schulz5(G: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """PyTorch 2.14's operation sequence (fused addmm) in a chosen dtype.

    PyTorch's own function always runs in bf16, where rounding differs with the
    order of operations. This replica lets us compare implementations in fp32,
    where a real bug cannot hide behind rounding.
    """
    a, b, c = NS_COEFFS
    X = G.to(dtype)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / X.norm().clamp(min=1e-7)
    for _ in range(NS_STEPS):
        A = X @ X.T
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a)
    return X.T if transposed else X


# --------------------------------------------------------------------------- #
# Model and optimisers
# --------------------------------------------------------------------------- #
def make_mlp(widths: list[int]) -> nn.Sequential:
    """Fully connected ReLU network, e.g. widths=[784, 256, 256, 256, 2]."""
    layers: list[nn.Module] = []
    for i, (d_in, d_out) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(d_in, d_out))
        if i < len(widths) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def split_params_for_muon(model: nn.Sequential):
    """Muon gets the hidden weight matrices only. AdamW gets the input layer, the
    output head and every bias (Jordan 2024, 'Empirical considerations')."""
    linears = [m for m in model if isinstance(m, nn.Linear)]
    muon_params = [layer.weight for layer in linears[1:-1]]
    muon_ids = {id(p) for p in muon_params}
    adamw_params = [p for p in model.parameters() if id(p) not in muon_ids]
    assignment = [
        {"name": name, "shape": list(p.shape), "optimizer": "Muon" if id(p) in muon_ids else "AdamW"}
        for name, p in model.named_parameters()
    ]
    return muon_params, adamw_params, assignment


class OptimizerBundle:
    """Steps several optimisers together (e.g. Muon for hidden matrices + AdamW for the rest)."""

    def __init__(self, *optimizers: torch.optim.Optimizer):
        self.optimizers = optimizers

    def zero_grad(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()


class MinimalMuon(torch.optim.Optimizer):
    """Smoke-test helper: torch.optim.Muon's update with a pluggable Newton-Schulz.

    It exists so we can run Muon with fp32 Newton-Schulz on the CPU, where bf16 is
    very slow on this Mac. The project's own implementation will be built and
    unit-tested properly later. For each 2-D parameter W with gradient G:
        M <- beta*M + (1-beta)*G                      momentum (EMA form)
        N <- (1-beta)*G + beta*M if nesterov else M   Nesterov look-ahead
        O <- NS5(N)                                   approximately U V^T
        W <- (1 - lr*wd)*W - lr*sqrt(max(1, out/in))*O
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0, ns_fn=None):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))
        self.ns_fn = ns_fn or (lambda M: newton_schulz5(M, dtype=torch.float32))

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            beta = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - beta)
                update = p.grad.lerp(buf, beta) if group["nesterov"] else buf
                O = self.ns_fn(update).to(p.dtype)
                shape_scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(O, alpha=-group["lr"] * shape_scale)


def build_optimizer(name: str, model: nn.Sequential) -> OptimizerBundle:
    if name == "adamw":
        return OptimizerBundle(torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01))
    if name == "sgd_nesterov":
        return OptimizerBundle(torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True))
    if name == "muon_adamw":
        muon_params, adamw_params, _ = split_params_for_muon(model)
        return OptimizerBundle(
            torch.optim.Muon(muon_params, lr=0.02, momentum=0.95, nesterov=True,
                             weight_decay=0.0, adjust_lr_fn="original"),
            torch.optim.AdamW(adamw_params, lr=1e-3, weight_decay=0.01),
        )
    if name == "muon_fp32_adamw":
        muon_params, adamw_params, _ = split_params_for_muon(model)
        return OptimizerBundle(
            MinimalMuon(muon_params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0),
            torch.optim.AdamW(adamw_params, lr=1e-3, weight_decay=0.01),
        )
    raise ValueError(f"unknown optimiser {name!r}")


# --------------------------------------------------------------------------- #
# Check 1: environment
# --------------------------------------------------------------------------- #
def environment_info() -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "python_executable": tidy_path(sys.executable),
    }


# --------------------------------------------------------------------------- #
# Check 2: bf16 matmul accuracy and speed
# --------------------------------------------------------------------------- #
def check_bf16_matmul(devices: list[torch.device], sizes: tuple[int, ...], repeats: int) -> list[dict]:
    rows = []
    for device in devices:
        for n in sizes:
            gen = torch.Generator().manual_seed(n)
            A = torch.randn(n, n, generator=gen)
            B = torch.randn(n, n, generator=gen)
            reference = A.double() @ B.double()  # float64 on CPU (MPS has no float64)
            row = {"device": device.type, "n": n}
            for dtype in (torch.float32, torch.bfloat16):
                key = "fp32" if dtype == torch.float32 else "bf16"
                try:
                    Ad, Bd = A.to(device=device, dtype=dtype), B.to(device=device, dtype=dtype)
                    C = Ad @ Bd
                    synchronize(device)
                    rel_err = ((C.float().cpu().double() - reference).norm() / reference.norm()).item()
                    seconds = median_seconds(lambda: Ad @ Bd, device, warmup=3, repeats=repeats)
                    row[key] = {"ok": True, "rel_err": rel_err, "ms": seconds * 1e3,
                                "gflops": 2 * n**3 / seconds / 1e9}
                except Exception as exc:  # record and continue: a smoke test should report, not crash
                    row[key] = error_record(exc)
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Check 3: Newton-Schulz sanity
# --------------------------------------------------------------------------- #
def check_newton_schulz(devices: list[torch.device], seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    n_rows, n_cols = 512, 256
    U, _ = torch.linalg.qr(torch.randn(n_rows, n_cols, generator=gen, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(n_cols, n_cols, generator=gen, dtype=torch.float64))
    s = torch.logspace(0, -4, n_cols, dtype=torch.float64)  # singular values from 1 down to 1e-4
    G = (U * s) @ V.T                                        # G = U diag(s) V^T, known SVD
    exact = U @ V.T
    x = s / s.norm()                                         # what NS sees after Frobenius scaling

    out: dict = {"matrix_shape": [n_rows, n_cols], "singular_values": "logspace(1, 1e-4, 256)"}
    O32 = newton_schulz5(G.float(), dtype=torch.float32).double()
    O16 = newton_schulz5(G.float(), dtype=torch.bfloat16).double()

    # diag(U^T O V) recovers what NS did to each singular value of G
    gain32 = torch.diagonal(U.T @ O32 @ V)
    gain16 = torch.diagonal(U.T @ O16 @ V)
    predicted = ns_scalar_map(x)
    picks = torch.linspace(0, n_cols - 1, 9).round().long()
    out["transfer_table"] = [
        {"x_normalised": x[i].item(), "predicted_phi5": predicted[i].item(),
         "measured_fp32": gain32[i].item(), "measured_bf16": gain16[i].item()}
        for i in picks
    ]
    out["max_abs_error_fp32_vs_scalar_prediction"] = (gain32 - predicted).abs().max().item()

    sv32 = torch.linalg.svdvals(O32)
    out["output_singular_values_fp32"] = {
        "min": sv32.min().item(), "max": sv32.max().item(),
        "fraction_in_[0.6,1.2]": ((sv32 > 0.6) & (sv32 < 1.2)).double().mean().item(),
    }
    out["rel_dist_to_exact_UVt"] = {
        "fp32": ((O32 - exact).norm() / exact.norm()).item(),
        "bf16": ((O16 - exact).norm() / exact.norm()).item(),
    }

    # Invariances that must hold: Ortho(cG) = Ortho(G) and Ortho(G^T) = Ortho(G)^T
    O32_scaled = newton_schulz5(1000.0 * G.float(), dtype=torch.float32).double()
    O32_T = newton_schulz5(G.float().T, dtype=torch.float32).double()
    out["scale_invariance_rel_diff"] = ((O32_scaled - O32).norm() / O32.norm()).item()
    out["transpose_rel_diff"] = ((O32_T.T - O32).norm() / O32.norm()).item()

    # Our implementation against PyTorch's. In fp32 they must agree (same maths);
    # in bf16 they differ through rounding, because PyTorch fuses operations (addmm).
    O32_torch_style = torch_style_newton_schulz5(G.float(), dtype=torch.float32).double()
    out["ours_vs_torch_style_fp32_rel_diff"] = ((O32 - O32_torch_style).norm() / O32.norm()).item()
    try:
        from torch.optim._muon import _zeropower_via_newtonschulz
        O_torch = _zeropower_via_newtonschulz(G.float(), NS_COEFFS, NS_STEPS, 1e-7).double()
        out["bf16_rel_diff"] = {
            "ours_bf16_vs_ours_fp32": ((O16 - O32).norm() / O32.norm()).item(),
            "torch_bf16_vs_ours_fp32": ((O_torch - O32).norm() / O32.norm()).item(),
            "ours_bf16_vs_torch_bf16": ((O16 - O_torch).norm() / O_torch.norm()).item(),
        }
        out["minimal_muon_vs_torch_muon_max_abs_diff"] = minimal_muon_matches_torch(
            lambda M: _zeropower_via_newtonschulz(M, NS_COEFFS, NS_STEPS, 1e-7), seed=seed)
    except Exception as exc:
        out["bf16_rel_diff"] = error_record(exc)

    # Same computation on the Apple GPU
    for device in devices:
        if device.type == "cpu":
            continue
        try:
            O_dev = newton_schulz5(G.float().to(device), dtype=torch.bfloat16).float().cpu().double()
            out[f"{device.type}_vs_cpu_bf16_rel_diff"] = ((O_dev - O16).norm() / O16.norm()).item()
        except Exception as exc:
            out[f"{device.type}_vs_cpu_bf16_rel_diff"] = error_record(exc)
    return out


def minimal_muon_matches_torch(ns_fn, seed: int = 0, steps: int = 5) -> float:
    """Run MinimalMuon (given PyTorch's own Newton-Schulz) and torch.optim.Muon side by
    side on identical parameters and gradients. Returns the largest parameter difference;
    0 means the momentum, Nesterov, shape-scaling and weight-decay logic all match."""
    gen = torch.Generator().manual_seed(seed)
    shapes = [(256, 128), (128, 256), (64, 64)]  # tall, wide and square: exercises the shape scaling
    init = [torch.randn(s, generator=gen) for s in shapes]
    grads = [[torch.randn(s, generator=gen) for s in shapes] for _ in range(steps)]
    p_torch = [nn.Parameter(w.clone()) for w in init]
    p_ours = [nn.Parameter(w.clone()) for w in init]
    reference = torch.optim.Muon(p_torch, lr=0.02, momentum=0.95, nesterov=True,
                                 weight_decay=0.1, adjust_lr_fn="original")
    ours = MinimalMuon(p_ours, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.1, ns_fn=ns_fn)
    for step_grads in grads:
        for pt, po, g in zip(p_torch, p_ours, step_grads):
            pt.grad, po.grad = g.clone(), g.clone()
        reference.step()
        ours.step()
    return max((pt - po).abs().max().item() for pt, po in zip(p_torch, p_ours))


# --------------------------------------------------------------------------- #
# Check 4: cost of a training step, AdamW vs Muon+AdamW
# --------------------------------------------------------------------------- #
def check_step_cost(devices: list[torch.device], width: int, batch_sizes: tuple[int, ...],
                    warmup: int, repeats: int, seed: int = 0) -> list[dict]:
    widths = [784, width, width, width, 10]  # two hidden width x width matrices get Muon
    rows = []
    for device in devices:
        for batch in batch_sizes:
            torch.manual_seed(seed)
            x = torch.randn(batch, widths[0]).to(device)
            y = torch.randint(0, widths[-1], (batch,)).to(device)
            row: dict = {"device": device.type, "batch_size": batch, "width": width}
            try:
                for name in ("adamw", "muon_adamw", "muon_fp32_adamw"):
                    torch.manual_seed(seed)
                    model = make_mlp(widths).to(device)
                    opt = build_optimizer(name, model)

                    def train_step(model=model, opt=opt):
                        opt.zero_grad()
                        F.cross_entropy(model(x), y).backward()
                        opt.step()

                    row[f"{name}_step_ms"] = median_seconds(train_step, device, warmup, repeats) * 1e3

                torch.manual_seed(seed)
                model = make_mlp(widths).to(device)

                def forward_backward():
                    model.zero_grad(set_to_none=True)
                    F.cross_entropy(model(x), y).backward()

                row["fwd_bwd_ms"] = median_seconds(forward_backward, device, warmup, repeats) * 1e3
                G = torch.randn(width, width).to(device)
                for dtype, key in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
                    row[f"ns5_{key}_ms_per_matrix"] = median_seconds(
                        lambda: newton_schulz5(G, dtype=dtype), device, warmup, repeats) * 1e3

                # FLOP model (Jordan 2024): forward+backward ~ 6 * batch * (weights used in matmuls);
                # each NS step on an m x n matrix (m <= n) ~ 2(2nm^2 + m^3) = 6m^3 when square.
                matmul_weights = sum(a * b for a, b in zip(widths[:-1], widths[1:]))
                ns_flops = 2 * NS_STEPS * 6 * width**3  # two hidden square matrices
                row["predicted_ns_flops_over_fwd_bwd_flops"] = ns_flops / (6 * batch * matmul_weights)
                row["measured_muon_over_adamw_step_time"] = row["muon_adamw_step_ms"] / row["adamw_step_ms"]
                row["measured_muon_fp32_over_adamw_step_time"] = row["muon_fp32_adamw_step_ms"] / row["adamw_step_ms"]
                row["ok"] = True
            except Exception as exc:
                row.update(error_record(exc))
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Check 5: short training run
# --------------------------------------------------------------------------- #
def load_data(data_dir: Path, seed: int):
    names = ("img_training", "label_training", "img_val", "label_val")
    files = {k: data_dir / f"fashion_{k}.npy" for k in names}
    if all(f.exists() for f in files.values()):
        X_tr = np.load(files["img_training"]).reshape(-1, 784).astype(np.float32)
        y_tr = ((np.load(files["label_training"]) + 1) / 2).astype(np.int64)  # labels -1/+1 -> 0/1
        X_va = np.load(files["img_val"]).reshape(-1, 784).astype(np.float32)
        y_va = ((np.load(files["label_val"]) + 1) / 2).astype(np.int64)
        source = f"DSA4212 binary Fashion-MNIST subset ({tidy_path(data_dir)})"
    else:  # fallback so the script runs anywhere: two Gaussian clouds in 784 dimensions
        rng = np.random.default_rng(seed)
        centre = rng.normal(size=784).astype(np.float32) * 0.1

        def blobs(n):
            y = rng.integers(0, 2, size=n)
            X = rng.normal(size=(n, 784)).astype(np.float32) + np.where(y[:, None] == 1, centre, -centre)
            return X, y.astype(np.int64)

        (X_tr, y_tr), (X_va, y_va) = blobs(10_000), blobs(2_000)
        source = "synthetic Gaussian blobs (course data not found)"
    mean = X_tr.mean(axis=0, keepdims=True)  # centre with training statistics only
    tensors = [torch.from_numpy(a) for a in (X_tr - mean, y_tr, X_va - mean, y_va)]
    return (*tensors, source)


def train_short(name: str, data, device: torch.device, epochs: int, batch_size: int, seed: int) -> dict:
    X_tr, y_tr, X_va, y_va = (t.to(device) for t in data)
    torch.manual_seed(seed)
    model = make_mlp([784, 256, 256, 256, 2]).to(device)  # initialised on CPU: identical on every device
    opt = build_optimizer(name, model)
    order = torch.Generator().manual_seed(seed)           # same data order for every optimiser
    n = X_tr.shape[0]
    history, first_batch_loss = [], None
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(n, generator=order)
        start = time.perf_counter()
        batch_losses = []
        for i in range(0, n, batch_size):
            idx = permutation[i:i + batch_size].to(device)
            loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(loss.detach())
        synchronize(device)
        seconds = time.perf_counter() - start
        if first_batch_loss is None:
            first_batch_loss = batch_losses[0].item()
        model.eval()
        with torch.no_grad():
            logits = model(X_va)
            val_loss = F.cross_entropy(logits, y_va).item()
            val_acc = (logits.argmax(dim=1) == y_va).float().mean().item()
        history.append({"epoch": epoch, "train_loss": torch.stack(batch_losses).mean().item(),
                        "val_loss": val_loss, "val_acc": val_acc, "epoch_seconds": seconds})
    finite = bool(np.all(np.isfinite([[h["train_loss"], h["val_loss"]] for h in history])))
    return {
        "optimizer": name, "device": device.type, "ok": True, "first_batch_loss": first_batch_loss,
        "history": history, "all_finite": finite,
        "loss_decreased": finite and history[-1]["train_loss"] < first_batch_loss,
    }


def check_training(devices, data_dir: Path, epochs: int, batch_size: int, seed: int) -> dict:
    *data, source = load_data(data_dir, seed)
    _, _, assignment = split_params_for_muon(make_mlp([784, 256, 256, 256, 2]))
    runs = []
    for device in devices:
        for name in SMOKE_OPTIMIZERS:
            try:
                runs.append(train_short(name, data, device, epochs, batch_size, seed))
            except Exception as exc:
                runs.append({"optimizer": name, "device": device.type, **error_record(exc)})
    return {"data_source": source, "n_train": int(data[0].shape[0]), "n_val": int(data[2].shape[0]),
            "model": "MLP 784-256-256-256-2, ReLU, PyTorch default init", "batch_size": batch_size,
            "epochs": epochs, "seed": seed, "optimizer_settings": SMOKE_OPTIMIZERS,
            "parameter_assignment": assignment, "runs": runs}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_summary(r: dict) -> None:
    env = r["environment"]
    print(f"\n=== 1. Environment ===\nPython {env['python']} | torch {env['torch']} | numpy {env['numpy']} | "
          f"{env['platform']} | threads {env['torch_num_threads']} | MPS available: {env['mps_available']}")

    print("\n=== 2. Matrix multiply: fp32 vs bf16 (median ms; relative error vs float64) ===")
    for row in r["bf16_matmul"]:
        cells = []
        for key in ("fp32", "bf16"):
            c = row[key]
            cells.append(f"{key}: {c['ms']:8.3f} ms ({c['gflops']:6.1f} GFLOP/s, err {c['rel_err']:.1e})"
                         if c.get("ok") else f"{key}: FAILED {c['error']}")
        print(f"{row['device']:>3} n={row['n']:<5} " + " | ".join(cells))

    ns = r["newton_schulz"]
    print("\n=== 3. Newton-Schulz sanity (512x256 matrix, singular values 1 ... 1e-4) ===")
    print(f"{'x = s/||G||_F':>14} {'phi^5(x)':>9} {'fp32':>8} {'bf16':>8}")
    for t in ns["transfer_table"]:
        print(f"{t['x_normalised']:14.2e} {t['predicted_phi5']:9.3f} {t['measured_fp32']:8.3f} {t['measured_bf16']:8.3f}")
    sv = ns["output_singular_values_fp32"]
    print(f"output singular values (fp32): min {sv['min']:.3f}, max {sv['max']:.3f}, "
          f"fraction in [0.6, 1.2]: {sv['fraction_in_[0.6,1.2]']:.2f}")
    print(f"distance to exact UV^T: fp32 {ns['rel_dist_to_exact_UVt']['fp32']:.3f}, "
          f"bf16 {ns['rel_dist_to_exact_UVt']['bf16']:.3f}")
    print(f"invariances: scale {ns['scale_invariance_rel_diff']:.1e}, transpose {ns['transpose_rel_diff']:.1e}")
    print(f"ours vs PyTorch's op sequence, both fp32: {ns['ours_vs_torch_style_fp32_rel_diff']:.1e}  (must be ~1e-6)")
    bf = ns["bf16_rel_diff"]
    if bf.get("ok", True):
        print(f"bf16 rounding: ours vs fp32 {bf['ours_bf16_vs_ours_fp32']:.3f} | torch vs fp32 "
              f"{bf['torch_bf16_vs_ours_fp32']:.3f} | ours vs torch {bf['ours_bf16_vs_torch_bf16']:.3f}")
        print(f"MinimalMuon vs torch.optim.Muon over 5 steps, max |diff|: "
              f"{ns['minimal_muon_vs_torch_muon_max_abs_diff']:.1e}  (must be ~0)")
    else:
        print(f"PyTorch comparison FAILED {bf['error']}")
    for k, v in ns.items():
        if k.endswith("_vs_cpu_bf16_rel_diff"):
            print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: FAILED {v['error']}")

    print("\n=== 4. Training-step cost, MLP 784-512-512-512-10 (median ms per step; ratio = vs AdamW) ===")
    print(f"{'dev':>3} {'batch':>5} {'AdamW':>7} {'Muon bf16':>10} {'ratio':>6} {'Muon fp32':>10} {'ratio':>6} "
          f"{'fwd+bwd':>8} {'NS5 bf16':>9} {'NS5 fp32':>9} {'FLOP-pred NS/fb':>16}")
    for row in r["step_cost"]:
        if not row.get("ok"):
            print(f"{row['device']:>3} {row['batch_size']:>5} FAILED {row.get('error')}")
            continue
        print(f"{row['device']:>3} {row['batch_size']:>5} {row['adamw_step_ms']:7.2f} {row['muon_adamw_step_ms']:10.2f} "
              f"{row['measured_muon_over_adamw_step_time']:6.2f} {row['muon_fp32_adamw_step_ms']:10.2f} "
              f"{row['measured_muon_fp32_over_adamw_step_time']:6.2f} {row['fwd_bwd_ms']:8.2f} "
              f"{row['ns5_bf16_ms_per_matrix']:9.2f} {row['ns5_fp32_ms_per_matrix']:9.2f} "
              f"{row['predicted_ns_flops_over_fwd_bwd_flops']:16.2f}")

    tr = r["training"]
    print(f"\n=== 5. Short training run ({tr['data_source']}; {tr['n_train']} train / {tr['n_val']} val; "
          f"batch {tr['batch_size']}, {tr['epochs']} epochs, UNTUNED) ===")
    for run in tr["runs"]:
        if not run.get("ok"):
            print(f"{run['device']:>3} {run['optimizer']:<13} FAILED {run.get('error')}")
            continue
        last = run["history"][-1]
        secs = sum(h["epoch_seconds"] for h in run["history"])
        print(f"{run['device']:>3} {run['optimizer']:<13} first-batch loss {run['first_batch_loss']:.3f} -> "
              f"last-epoch train {last['train_loss']:.4f}, val loss {last['val_loss']:.4f}, "
              f"val acc {last['val_acc']:.4f} | {secs:5.1f} s | finite {run['all_finite']} | "
              f"decreased {run['loss_decreased']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="fewer repeats and epochs")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="folder with the DSA4212 fashion_*.npy files (synthetic data if missing)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "smoke")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repeats, epochs = (5, 2) if args.quick else (20, 5)
    devices = available_devices()
    t0 = time.perf_counter()
    results = {"environment": environment_info(), "args": {"quick": args.quick, "seed": args.seed}}
    results["bf16_matmul"] = check_bf16_matmul(devices, sizes=(256, 512, 1024), repeats=repeats)
    results["newton_schulz"] = check_newton_schulz(devices, seed=args.seed)
    results["step_cost"] = check_step_cost(devices, width=512, batch_sizes=(32, 128, 512, 2048),
                                           warmup=3, repeats=repeats, seed=args.seed)
    results["training"] = check_training(devices, args.data_dir, epochs=epochs, batch_size=128, seed=args.seed)
    results["total_seconds"] = time.perf_counter() - t0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / f"smoke_{datetime.now():%Y%m%d-%H%M%S}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print_summary(results)
    print(f"\nTotal {results['total_seconds']:.0f} s. Saved {tidy_path(out_file)}")


if __name__ == "__main__":
    main()

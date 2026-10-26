"""Tests for the training loop, branching and sweep expansion."""

import shutil

import pytest

from muonlab.sweep import expand
from muonlab.train import REPO_ROOT, run

EXPERIMENT = "pytest_tmp"


def tiny_config(**overrides):
    cfg = {
        "experiment": EXPERIMENT, "name": "shared", "seed": 3, "threads": 1,
        "task": {"d_in": 8, "teacher_hidden": [8], "d_out": 2, "power": 1.0, "label_noise": 0.5, "teacher_seed": 0},
        "eval_size": 256, "eval_seed": 7,
        "model": {"widths": [8, 16, 16, 16, 2], "activation": "tanh"},
        "optimizer": {"name": "muon", "lr": 0.02, "adam_lr": 1e-3, "momentum": 0.9, "update": "spectral", "p": 0.0},
        "batch_size": 8, "steps": 20, "eval_every": 10, "schedule": "constant",
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture(autouse=True)
def clean_checkpoints():
    yield
    shutil.rmtree(REPO_ROOT / "checkpoints" / EXPERIMENT, ignore_errors=True)


def test_branch_with_unchanged_settings_reproduces_the_uninterrupted_run():
    uninterrupted = run(tiny_config(name="full"))
    run(tiny_config(steps=10, checkpoint_at=[10]))
    branch = run(tiny_config(name="branch", steps=10, branch={
        "checkpoint": f"checkpoints/{EXPERIMENT}/shared_step10.pt", "lr_multiplier": 1.0}))
    assert branch["history"][-1]["step"] == uninterrupted["history"][-1]["step"] == 20
    assert branch["history"][-1]["excess_risk"] == uninterrupted["history"][-1]["excess_risk"]


def test_xie_branch_records_matched_learning_rates():
    run(tiny_config(steps=10, checkpoint_at=[10]))
    branch = run(tiny_config(name="xie", steps=10, branch={
        "checkpoint": f"checkpoints/{EXPERIMENT}/shared_step10.pt", "update": "plain",
        "lr_reference": "xie", "lr_multiplier": 2.0, "k": 8}))
    info = branch["branch"]
    assert len(info["xie"]) == 2 and all(q["matched_lr"] > 0 and q["delta"] >= 0 for q in info["xie"])
    assert info["branch_lr"] == pytest.approx([2.0 * q["matched_lr"] for q in info["xie"]])
    assert not branch["summary"]["diverged"]


def test_sweep_expansion_resolves_dependencies_and_placeholders():
    spec = {
        "experiment": "demo",
        "base": {"batch_size": 16, "seed": 0, "optimizer": {"name": "muon", "lr": 0.02},
                 "branch": {"checkpoint": "checkpoints/e2/B{batch_size}_s{seed}.pt"}},
        "grid": {"batch_size": [16, 1024], "seed": [0, 1]},
        "arms": [{"label": "muon", "set": {"optimizer": {"adam_lr": {"by": "batch_size", "map": {16: 1e-3, 1024: 3e-3}}}},
                  "sweep": {"optimizer.lr": [0.01, 0.02]}}],
    }
    configs = expand(spec)
    assert len(configs) == 8 and len({c["name"] for c in configs}) == 8
    big = next(c for c in configs if c["batch_size"] == 1024 and c["seed"] == 1)
    assert big["optimizer"]["adam_lr"] == 3e-3
    assert big["branch"]["checkpoint"] == "checkpoints/e2/B1024_s1.pt"
    assert big["name"].startswith("muon_lr")

"""scripts/ramen_l_preflight.py: positive control plus one forced-fail control per refusal class (CPU, tiny fake weights file)."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ramen_l_preflight as P  # noqa: E402


@pytest.fixture()
def fake(tmp_path):
    w = tmp_path / "w.pth"
    w.write_bytes(b"not a real checkpoint" * 10)
    return w, hashlib.sha256(w.read_bytes()).hexdigest(), w.stat().st_size


def run(cfg, w, sha, size, *extra):
    return P.main(["--config", str(ROOT / "configs/experiments" / cfg), "--no-git", "--no-paths", "--weights", str(w), "--weights-sha", sha, "--weights-size", str(size), *extra])


def test_wrong_weights_sha_refuses(fake, capsys):
    w, sha, size = fake
    assert run("ramen_l_c3_n46.yaml", w, P.WEIGHTS_SHA, size) == 2          # wrong weights file (sha) refuses
    assert "REFUSED: weights sha256" in capsys.readouterr().out


def test_positive_with_matching_pin(fake, monkeypatch, capsys):
    w, sha, size = fake
    monkeypatch.setattr(P, "WEIGHTS_SHA", sha)                              # a config pinning the same sha as the file under test
    cfgdir = ROOT / "configs/experiments"
    tmp = cfgdir / "_tmp_ramen_l_ctrl.yaml"
    tmp.write_text(f"extends: ramen_l_c3_n46.yaml\nmodel:\n  vda_checkpoint_sha256: {sha}\n")
    try:
        assert run(tmp.name, w, sha, size) == 0
        assert "PREFLIGHT OK" in capsys.readouterr().out
    finally:
        tmp.unlink()


@pytest.mark.parametrize("override,needle", [
    ("model:\n  vda_encoder: vits\n", "vda_encoder"),                                   # ViT-S judged as ViT-L recipe
    ("data:\n  seed: 47\n", "outside the allow-list"),                                  # recipe drift: seed
    ("loss:\n  tgm_weight: 5.0\n", "outside the allow-list"),                           # recipe drift: loss weight
    ("training:\n  max_steps: 6000\n  lr_decay_steps: 6000\n", "outside the allow-list"),  # a full run with fewer steps
    ("model:\n  st_stage2_width: wide8x\n", "st_stage2_width"),                         # unknown restorer width
    ("model:\n  vda_encoder_grad_ckpt: false\n", "vda_encoder_grad_ckpt"),
])
def test_forced_fails(fake, override, needle, capsys):
    w, sha, size = fake
    tmp = ROOT / "configs/experiments" / "_tmp_ramen_l_neg.yaml"
    tmp.write_text("extends: ramen_l_c3_n46.yaml\n" + override)
    try:
        assert run(tmp.name, w, P.WEIGHTS_SHA, size) == 2
        assert needle in capsys.readouterr().out
    finally:
        tmp.unlink()


def test_missing_weights_and_bad_pin(tmp_path, capsys):
    assert P.main(["--config", str(ROOT / "configs/experiments/ramen_l_c3_n46.yaml"), "--no-git", "--no-paths", "--weights", str(tmp_path / "none.pth")]) == 2
    assert "weights file missing" in capsys.readouterr().out
    assert P.main(["--config", str(ROOT / "configs/experiments/ramen_l_c3_n46.yaml"), "--no-paths", "--weights", str(tmp_path / "none.pth"), "--pin", "UNSET"]) == 2
    assert "not a 40-hex sha" in capsys.readouterr().out


def test_baseline_drift_refuses(fake, capsys):
    w, sha, size = fake
    assert run("ramen_l_c3_n46.yaml", w, P.WEIGHTS_SHA, size, "--base-resolved-sha", "0" * 64) == 2
    assert "drifted" in capsys.readouterr().out

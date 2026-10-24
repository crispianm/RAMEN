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


# ---- --profile ft (depth-only VDA-L fine-tune) ----
def run_ft(cfg, w, sha, size, *extra):
    return P.main(["--profile", "ft", "--config", str(ROOT / "configs/experiments" / cfg), "--no-git", "--no-paths", "--weights", str(w), "--weights-sha", sha, "--weights-size", str(size), *extra])


def test_ft_profile_positive(fake, monkeypatch, capsys):
    w, sha, size = fake
    tmp = ROOT / "configs/experiments" / "_tmp_ft_ctrl.yaml"
    tmp.write_text(f"extends: vdal_ft_n46.yaml\nmodel:\n  vda_checkpoint_sha256: {sha}\n")
    try:
        assert run_ft(tmp.name, w, sha, size) == 0
        assert "PREFLIGHT OK" in capsys.readouterr().out
    finally:
        tmp.unlink()


@pytest.mark.parametrize("override,needle", [
    ("data:\n  seed: 47\n", "outside the allow-list"),          # seed drift
    ("model:\n  vda_encoder: vits\n", "vda_encoder"),             # wrong encoder
])
def test_ft_profile_forced_fails(fake, override, needle, capsys):
    w, sha, size = fake
    tmp = ROOT / "configs/experiments" / "_tmp_ft_neg.yaml"
    tmp.write_text("extends: vdal_ft_n46.yaml\n" + override)
    try:
        assert run_ft(tmp.name, w, P.WEIGHTS_SHA, size) == 2
        assert needle in capsys.readouterr().out
    finally:
        tmp.unlink()


def test_ft_profile_rejects_ramen_l_config(fake, capsys):
    w, sha, size = fake
    assert run_ft("ramen_l_c3_n46.yaml", w, P.WEIGHTS_SHA, size) == 2   # the Tables 1/2-based config is not an FT config


# ---- R3: the ft profile requires a metric SiLog loss ----
import copy  # noqa: E402

import yaml  # noqa: E402

FIXTURE = ROOT / "tests/fixtures/teach03_vda_l_metric_released_fixed1_cfg.yaml"


def _ft_cfg():
    sys.path.insert(0, str(ROOT))
    import train
    return train.load_config(ROOT / "configs/experiments/vdal_ft_n46.yaml")


def test_metric_assert_passes_on_vdal_ft():
    assert P.metric_loss_problems(_ft_cfg()) == []


def test_metric_assert_refuses_september_cfg():
    cfg = yaml.safe_load(FIXTURE.read_text())
    assert "metric_silog_weight" not in cfg["loss"]
    probs = P.metric_loss_problems(cfg)
    assert len(probs) == 2 and "loss.metric_silog_weight = None" in probs[0] and "metric_supervision" in probs[1]


def test_metric_assert_refuses_zero_weight():
    cfg = copy.deepcopy(_ft_cfg()); cfg["loss"]["metric_silog_weight"] = 0.0
    probs = P.metric_loss_problems(cfg)
    assert len(probs) == 1 and "loss.metric_silog_weight = 0.0" in probs[0]


def test_metric_assert_refuses_no_supervised_source():
    cfg = copy.deepcopy(_ft_cfg())
    for s in cfg["data"]["sources"]:
        s["metric_supervision"] = False
    probs = P.metric_loss_problems(cfg)
    assert len(probs) == 1 and "metric_supervision: true" in probs[0]


def test_ft_preflight_end_to_end_zero_metric_weight_exits_2(fake, capsys):
    w, sha, size = fake
    tmp = ROOT / "configs/experiments" / "_tmp_ft_nometric.yaml"
    tmp.write_text("extends: vdal_ft_n46.yaml\nloss:\n  metric_silog_weight: 0.0\n")
    try:
        assert run_ft(tmp.name, w, P.WEIGHTS_SHA, size) == 2
        out = capsys.readouterr().out
        assert "metric_silog_weight > 0" in out
    finally:
        tmp.unlink()


def test_ft_preflight_end_to_end_no_supervised_source_exits_2(fake, capsys):
    w, sha, size = fake
    tmp = ROOT / "configs/experiments" / "_tmp_ft_nosup.yaml"
    c = _ft_cfg()
    srcs = copy.deepcopy(c["data"]["sources"])
    for s in srcs:
        s["metric_supervision"] = False
    tmp.write_text("extends: vdal_ft_n46.yaml\n" + yaml.safe_dump({"data": {"sources": srcs}}))
    try:
        assert run_ft(tmp.name, w, P.WEIGHTS_SHA, size) == 2
        assert "metric_supervision" in capsys.readouterr().out
    finally:
        tmp.unlink()

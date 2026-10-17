"""make_lr_scale: default schedule byte-identical to the pre-refactor closure; opt-in lr_restart_* values (screen continuation from step 12000)."""
import math
from pathlib import Path
import pytest, yaml
import train

C3 = Path(__file__).resolve().parents[1] / "outdoor_debug/parts/C2_config_train/effective_c3_seed45.yaml"


def _legacy(t, warmup, total_steps):   # verbatim copy of the closure removed from train.py (reference for the identity test only)
    lr_decay_steps = int(t.get("lr_decay_steps", 0)) or total_steps
    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, lr_decay_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return lr_scale


def test_default_identical_on_c3_config():
    t = yaml.safe_load(open(C3))["training"]
    new = train.make_lr_scale(t, int(t["warmup_steps"]), int(t["max_steps"]))
    old = _legacy(t, int(t["warmup_steps"]), int(t["max_steps"]))
    for s in range(0, 14001):   # exact float equality, incl. the 12000+ tail (0.0 on the C3 config)
        assert new(s) == old(s)
    assert new(12000) == 0.0 and new(13999) == 0.0


def test_default_other_decay_identical():
    for t in ({"lr_decay_steps": 14000}, {"lr_decay_steps": 10**9}, {}):
        a, b = train.make_lr_scale(t, 500, 14000), _legacy(t, 500, 14000)
        assert all(a(s) == b(s) for s in range(0, 14001))


def test_restart_values():
    t = {"lr_decay_steps": 14000, "lr_restart_step": 12000, "lr_restart_warmup": 100, "lr_restart_peak": 0.5}
    f = train.make_lr_scale(t, 500, 14000)
    base = _legacy(t, 500, 14000)
    assert all(f(s) == base(s) for s in range(0, 12000))     # before the restart: unchanged
    assert f(12000) == pytest.approx(0.5 * 1 / 100)
    assert f(12001) == pytest.approx(0.5 * 2 / 100)
    assert f(12099) == pytest.approx(0.5)
    assert f(12100) == pytest.approx(0.5)                    # cosine starts at peak
    assert f(13000) == pytest.approx(0.5 * 0.5 * (1 + math.cos(math.pi * 900 / 1900)))
    assert f(13999) == pytest.approx(0.5 * 0.5 * (1 + math.cos(math.pi * 1899 / 1900)))
    assert 0 < f(13999) < 1e-5 and f(14000) == pytest.approx(0.0, abs=1e-12)


def test_restart_requires_all_keys_and_valid():
    with pytest.raises(ValueError):
        train.make_lr_scale({"lr_restart_step": 12000}, 500, 14000)
    with pytest.raises(ValueError):
        train.make_lr_scale({"lr_decay_steps": 12050, "lr_restart_step": 12000, "lr_restart_warmup": 100, "lr_restart_peak": 0.5}, 500, 14000)


def test_restart_peak_1_full():
    """Addendum 45: S3 with peak 1.0 (warmup convention (k+1)/W as in the base schedule: step R -> 0.01, not 0)."""
    t = {"lr_decay_steps": 14000, "lr_restart_step": 12000, "lr_restart_warmup": 100, "lr_restart_peak": 1.0}
    f = train.make_lr_scale(t, 500, 14000)
    assert f(12000) == pytest.approx(0.01)
    assert f(12050) == pytest.approx(0.51)
    assert f(12099) == pytest.approx(1.0)
    assert f(12100) == pytest.approx(1.0)
    assert f(13050) == pytest.approx(0.5, abs=1e-12)
    assert f(13999) == pytest.approx(0.5 * (1 + math.cos(math.pi * 1899 / 1900)))
    assert f(13999) < 1e-6


def test_arm_yaml_keys_real_lr_scale():
    """Auditor O17 L2: the REAL lr_scale built from a committed arm yaml's training keys. Convention: scale(step) = (step - R + 1)/W inside the restart warmup, i.e. the FIRST optimizer update
    after the resume (step 12000, lr_scale(gstep) called BEFORE gstep += 1) uses 0.01, not 0; step 12099 -> 1.0."""
    arm = Path(__file__).resolve().parents[1] / "outdoor_debug/parts/arm_checks/configs/arm_A0.yaml"
    t = yaml.safe_load(open(arm))["training"]
    assert (t["lr_restart_step"], t["lr_restart_warmup"], t["lr_restart_peak"], t["lr_decay_steps"], t["max_steps"]) == (12000, 100, 1.0, 14000, 14000)
    f = train.make_lr_scale(t, int(t["warmup_steps"]), int(t["max_steps"]))
    assert f(12000) == pytest.approx(0.01)
    assert f(12049) == pytest.approx(0.5)
    assert f(12099) == pytest.approx(1.0)
    assert f(12100) == pytest.approx(1.0)
    assert f(13050) == pytest.approx(0.5, abs=1e-12)
    assert f(13999) == pytest.approx(0.0, abs=1e-6) and f(13999) > 0

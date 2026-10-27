"""Pins the released `ramen` / `ramen_l` entries of configs/paper_results.yaml (sha256, spec, path)."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "ramen": ("334623ec4fe0415896efa66a48278d4e6051dddaf571620c1d83bec11a313f47", "checkpoints_release/ramen.pt"),
    "ramen_l": ("cad96e2c1c5219ac6a9fccb30ba6b327f95656b00a8ad804265ec795e3f2e8bc", "checkpoints_release/ramen_l.pt"),
}


def _check(models):
    for name, (sha, path) in PINS.items():
        m = models[name]
        assert m["spec"] == "l15d:{ckpt}", name
        assert m["ckpt"]["sha256"] == sha, name
        assert m["ckpt"]["path"] == path, name


def _models():
    return yaml.safe_load((ROOT / "configs" / "paper_results.yaml").read_text())["models"]


def test_released_entries_pinned():
    _check(_models())


@pytest.mark.parametrize("field", ["sha", "spec", "path"])
def test_tampering_is_detected(field):
    m = copy.deepcopy(_models())
    if field == "sha":
        m["ramen"]["ckpt"]["sha256"] = "0" * 64
    elif field == "spec":
        m["ramen_l"]["spec"] = "ours:{ckpt}"
    else:
        m["ramen"]["ckpt"]["path"] = "elsewhere/ramen.pt"
    with pytest.raises(AssertionError):
        _check(m)

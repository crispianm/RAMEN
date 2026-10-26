"""Pins the released `ramen` / `ramen_l` entries of configs/paper_results.yaml (sha256, spec, path)."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "ramen": ("898c50da69841b5c6710a026ffd42e570fba08a349f7485ee7e35747a22f46dd", "checkpoints_release/ramen.pt"),
    "ramen_l": ("9484ef5bf41a3ba6a5ad58dc56320c8c94baaf54c9bf0bbbb69a65a564c039b2", "checkpoints_release/ramen_l.pt"),
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

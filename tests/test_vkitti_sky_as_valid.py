"""Add.44 arm TS2+VK655: with ``sky_as_valid_depth`` the VKITTI 65535 sentinel is a VALID 655.35 m target (mask=1, sky=0); without it the released behaviour holds (mask=0, sky=1, depth clipped to 655)."""
from __future__ import annotations

import numpy as np
import pytest

from data.vkitti_dataset import VKITTIDataset
from tests.test_vda_semantics_dataset import COMMON, _write_16bit_depth, _write_rgb
from train import effective_depth_bound


@pytest.fixture()
def root(tmp_path):
    r = tmp_path / "Scene02" / "clone" / "frames" / "rgb" / "Camera_0"
    dp = tmp_path / "Scene02" / "clone" / "frames" / "depth" / "Camera_0"
    r.mkdir(parents=True)
    dp.mkdir(parents=True)
    for i in range(3):
        _write_rgb(r / f"rgb_{i:05d}.jpg")
        _write_16bit_depth(dp / f"depth_{i:05d}.png", value=65535)   # all sentinel
    return tmp_path


def _ds(root, **kw):
    return VKITTIDataset(root=str(root), split="train", val_fraction=0.0, no_holdout=True, **{k: v for k, v in COMMON.items()}, **kw)


def test_released_behaviour_without_flag(root):
    s = _ds(root)[0]
    assert float(s["mask"].max()) == 0.0 and float(s["sky"].min()) == 1.0
    assert float(s["depth"].max()) <= 655.0 + 1e-3


def test_sentinel_is_valid_metric_depth_with_flag(root):
    ds = _ds(root)
    ds.sky_as_valid_depth = True
    s = ds[0]
    assert float(s["mask"].min()) == 1.0 and float(s["sky"].max()) == 0.0
    assert np.allclose(s["depth"].numpy(), 655.35, atol=1e-3)


def test_effective_bound_includes_sentinel_only_with_flag_and_raised_cap(root):
    ds = _ds(root)
    ds.vkitti_max_depth = 656.0
    assert effective_depth_bound(ds) == 655.0       # released: cap 656 still clipped to the strict 655 sentinel bound
    ds.sky_as_valid_depth = True
    assert effective_depth_bound(ds) == 656.0

"""DIST-01 arm (e): real bright/dark twin source and its per-sample teacher routing."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from data.vda_semantics_dataset import VDASemanticsPairedVideoDataset
from train import _assert_no_video_only_val_source, online_teacher_input


def _write_twins(root, n_seq=3, n_frames=4, wh=(96, 64)):
    rng = np.random.RandomState(0)
    for s in range(n_seq):
        for side in ("normal", "low"):
            (root / side / f"S{s:02d}__10").mkdir(parents=True)
        for f in range(n_frames):
            bright = rng.randint(0, 256, (wh[1], wh[0], 3), dtype=np.uint8)
            Image.fromarray(bright).save(root / "normal" / f"S{s:02d}__10" / f"{f + 1:05d}.png")
            Image.fromarray(bright // 4).save(root / "low" / f"S{s:02d}__10" / f"{f + 1:05d}.png")


def test_dark_twin_shares_the_bright_clips_random_crop(tmp_path):
    _write_twins(tmp_path)
    ds = VDASemanticsPairedVideoDataset(
        rgb_root=str(tmp_path / "normal"), dark_root=str(tmp_path / "low"), num_frames=2, clip_stride=2,
        image_size=(28, 28), split="train", val_fraction=0.34)
    assert len(ds) > 0
    for i in range(len(ds)):
        item = ds[i]
        assert item["frames"].shape == item["degraded_frames"].shape == (3, 2, 28, 28)
        # dark = bright // 4 pixelwise before resize, so a shared crop keeps dark ~ bright / 4;
        # a mismatched crop on random-noise frames would not.
        assert torch.allclose(item["degraded_frames"], item["frames"] / 4, atol=0.02)
        assert not item["mask"].any() and not item["depth"].any()
        assert (item["degrade_params"] == -1).all() and item["degrade_params"].shape == (5,)


def test_teacher_sees_the_bright_twin_only_for_paired_clips():
    rgb, clean = torch.zeros(2, 3, 2, 4, 4), torch.ones(2, 3, 2, 4, 4)
    batch = {"frames": clean, "degrade_params": torch.tensor([[1.0, 2.8, 0.4, 0.02, 2.5], [-1.0] * 5])}
    out = online_teacher_input("degraded_or_bright_twin", rgb, batch, "cpu")
    assert not out[0].any() and out[1].all()          # synthetic -> student input; paired -> bright twin
    assert online_teacher_input("degraded", rgb, batch, "cpu") is rgb
    assert online_teacher_input("clean", rgb, batch, "cpu").all()


def test_paired_video_is_rejected_as_a_val_source():
    with pytest.raises(ValueError):
        _assert_no_video_only_val_source({"data": {"val_sources": [{"type": "paired_video"}]}})

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from data.vda_semantics_dataset import VDASemanticsDegradedTartanAirDataset, VDASemanticsTartanAirDataset
from train import _make_source

COMMON = dict(
    num_frames=2, image_size=(518, 518), clip_stride=1, frame_stride=1,
    max_depth=80.0, seed=42, degrade=False, degrade_mode="full",
    degrade_severity=None, max_clips_per_video=None, resize_mode="vda",
)


def _tartanair_root(tmp_path, num_images: int = 1):
    img_dir = tmp_path / "Env" / "Data_easy" / "P000" / "image_lcam_front"
    dep_dir = tmp_path / "Env" / "Data_easy" / "P000" / "depth_lcam_front"
    img_dir.mkdir(parents=True)
    dep_dir.mkdir(parents=True)
    import cv2
    for i in range(num_images):
        img = Image.new("RGB", (640, 640), color=(i * 10 % 256, 0, 0))
        img.save(img_dir / f"{i:06d}_lcam_front.png")
        depth = np.full((640, 640), 5.0, dtype=np.float32)
        bgra = depth.reshape(640, 640, 1).view(np.uint8).reshape(640, 640, 4)
        cv2.imwrite(str(dep_dir / f"{i:06d}_lcam_front.png"), bgra)
    return tmp_path


def test_resize_mode_vda_dispatches_to_the_vda_semantics_tartanair_class(tmp_path):
    root = _tartanair_root(tmp_path)
    spec = {"type": "tartanair", "root": str(root), "val_fraction": 1.0}
    src = _make_source(spec, "val", COMMON)
    assert isinstance(src, VDASemanticsTartanAirDataset)


def test_resize_mode_vda_is_not_set_by_default(tmp_path):
    root = _tartanair_root(tmp_path)
    spec = {"type": "tartanair", "root": str(root), "val_fraction": 1.0}
    common = {**COMMON, "resize_mode": None}
    src = _make_source(spec, "val", common)
    assert not isinstance(src, VDASemanticsTartanAirDataset)


def test_resize_mode_vda_plus_degrade_dispatches_to_the_degraded_class(tmp_path):
    """Matrix item 3 (Opus/Fable, 2026-08-28): resize_mode='vda' + degrade=True is no
    longer rejected -- it dispatches to the further VDASemanticsDegraded* variant, which
    is a VDASemanticsTartanAirDataset subclass (so the clean isinstance check alone can't
    tell them apart) but adds degraded_frames to every real item."""
    root = _tartanair_root(tmp_path, num_images=2)
    spec = {"type": "tartanair", "root": str(root), "val_fraction": 1.0}
    common = {**COMMON, "degrade": True}
    src = _make_source(spec, "val", common)
    assert isinstance(src, VDASemanticsDegradedTartanAirDataset)
    item = src[0]
    assert "degraded_frames" in item
    assert item["degraded_frames"].shape == item["frames"].shape
    assert not torch.equal(item["degraded_frames"], item["frames"])


def test_resize_mode_vda_without_degrade_still_uses_the_clean_class(tmp_path):
    root = _tartanair_root(tmp_path, num_images=2)
    spec = {"type": "tartanair", "root": str(root), "val_fraction": 1.0}
    src = _make_source(spec, "val", COMMON)   # COMMON has degrade=False
    assert not isinstance(src, VDASemanticsDegradedTartanAirDataset)
    assert "degraded_frames" not in src[0]


def test_resize_mode_vda_dispatches_pseudo_gt_to_the_vda_loader(tmp_path):
    """K80 option B implemented resize_mode='vda' for pseudo_gt (data/vda_semantics_dataset.py); the old 'not implemented' guard is gone, so an empty dir now fails INSIDE the VDA loader."""
    spec = {"type": "pseudo_gt", "rgb_root": str(tmp_path), "depth_root": str(tmp_path)}
    with pytest.raises(FileNotFoundError):
        _make_source(spec, "val", COMMON)


def test_resize_mode_vda_per_source_override_beats_common_default(tmp_path):
    """A source-level resize_mode wins over the common default -- lets one config mix a
    VDA-semantics source with a legacy-stretch source (e.g. auditing one at a time)."""
    root = _tartanair_root(tmp_path)
    spec = {"type": "tartanair", "root": str(root), "val_fraction": 1.0, "resize_mode": None}
    common = {**COMMON, "resize_mode": "vda"}
    src = _make_source(spec, "val", common)
    assert not isinstance(src, VDASemanticsTartanAirDataset)

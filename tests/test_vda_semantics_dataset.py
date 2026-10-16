from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from data.vda_semantics_dataset import (
    VDASemanticsIRSDataset,
    VDASemanticsPointOdysseyDataset,
    VDASemanticsTartanAirDataset,
    VDASemanticsVKITTIDataset,
    _vda_frame_and_crop,
)
from data.vda_resize import vda_resize_size

# Non-square, matching the real dataset aspect ratios found in VAL-01's pipeline audit
# (provenance/inventory/val01_pipeline_audit/), so a stretch bug would be visible if
# reintroduced -- not an arbitrary test shape.
RAW_W, RAW_H = 960, 540
COMMON = dict(num_frames=2, image_size=(518, 518), clip_stride=1, frame_stride=1, max_depth=80.0, seed=42)


def _write_rgb(path, w=RAW_W, h=RAW_H):
    img = Image.new("RGB", (w, h))
    arr = np.asarray(img).copy()
    # A simple ramp so the frame isn't flat -- lets a resize/crop bug show up as a shape or
    # value-range problem rather than being invisible on a uniform colour.
    arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    Image.fromarray(arr).save(path)


def _write_tartanair_depth(path, w=RAW_W, h=RAW_H, value=5.0):
    depth = np.full((h, w), value, dtype=np.float32)
    bgra = depth.reshape(h, w, 1).view(np.uint8).reshape(h, w, 4)
    cv2.imwrite(str(path), bgra)


def _write_16bit_depth(path, w=RAW_W, h=RAW_H, value=5000):
    cv2.imwrite(str(path), np.full((h, w), value, dtype=np.uint16))


def _write_exr_depth(path, w=RAW_W, h=RAW_H, disparity=2.0):
    cv2.imwrite(str(path), np.full((h, w), disparity, dtype=np.float32))


def _assert_common_contract(sample, num_frames=2, size=518):
    assert sample["frames"].shape == (3, num_frames, size, size)
    assert sample["depth"].shape == (1, num_frames, size, size)
    assert sample["mask"].shape == (1, num_frames, size, size)
    assert sample["sky"].shape == (1, num_frames, size, size)
    # cv2.INTER_CUBIC (matching VDA's own transform) can slightly overshoot/undershoot near
    # sharp edges -- real behaviour, not a bug here, so this tolerance is wide on purpose
    # rather than clipping (clipping would diverge from what VDA's own transform actually does)
    assert -0.05 <= sample["frames"].min() and sample["frames"].max() <= 1.05


def test_tartanair_vda_semantics_shape_and_range(tmp_path):
    img_dir = tmp_path / "AbandonedFactory" / "Data_easy" / "P000" / "image_lcam_front"
    dep_dir = tmp_path / "AbandonedFactory" / "Data_easy" / "P000" / "depth_lcam_front"
    img_dir.mkdir(parents=True)
    dep_dir.mkdir(parents=True)
    for i in range(3):
        _write_rgb(img_dir / f"{i:06d}_lcam_front.png")
        _write_tartanair_depth(dep_dir / f"{i:06d}_lcam_front.png")

    ds = VDASemanticsTartanAirDataset(root=str(tmp_path), split="val", val_fraction=1.0, **COMMON)
    assert len(ds) > 0
    sample = ds[0]
    _assert_common_contract(sample)
    assert sample["depth"].max() > 0  # real depth value survived resize+crop, not all zero


def test_vkitti_vda_semantics_shape_and_range(tmp_path):
    rgb_dir = tmp_path / "Scene01" / "clone" / "frames" / "rgb" / "Camera_0"
    dep_dir = tmp_path / "Scene01" / "clone" / "frames" / "depth" / "Camera_0"
    rgb_dir.mkdir(parents=True)
    dep_dir.mkdir(parents=True)
    for i in range(3):
        _write_rgb(rgb_dir / f"rgb_{i:05d}.jpg")
        _write_16bit_depth(dep_dir / f"depth_{i:05d}.png")

    ds = VDASemanticsVKITTIDataset(root=str(tmp_path), split="val", val_fraction=1.0, **COMMON)
    assert len(ds) > 0
    sample = ds[0]
    _assert_common_contract(sample)


def test_irs_vda_semantics_shape_and_range(tmp_path):
    scene_dir = tmp_path / "Home" / "ArchvizKitchen"
    scene_dir.mkdir(parents=True)
    for i in range(3):
        _write_rgb(scene_dir / f"l_{i:05d}.png")
        _write_exr_depth(scene_dir / f"d_{i:05d}.exr")

    ds = VDASemanticsIRSDataset(root=str(tmp_path), split="test", val_fraction=0.0, test_fraction=1.0, **COMMON)
    assert len(ds) > 0
    sample = ds[0]
    _assert_common_contract(sample)


def test_pointodyssey_vda_semantics_shape_and_range(tmp_path):
    scene_dir = tmp_path / "ani1"
    (scene_dir / "rgbs").mkdir(parents=True)
    (scene_dir / "depths").mkdir(parents=True)
    for i in range(3):
        _write_rgb(scene_dir / "rgbs" / f"rgb_{i:05d}.jpg")
        _write_16bit_depth(scene_dir / "depths" / f"depth_{i:05d}.png")

    ds = VDASemanticsPointOdysseyDataset(root=str(tmp_path), split="val", val_fraction=0.0, **COMMON)
    assert len(ds) > 0
    sample = ds[0]
    _assert_common_contract(sample)


def test_val_split_uses_the_same_crop_position_deterministically(tmp_path):
    """Fable's decision, 2026-08-27: centre crop for val -- must be reproducible, not
    randomly sampled, so a val-set number is comparable run to run."""
    img_dir = tmp_path / "Env" / "Data_easy" / "P000" / "image_lcam_front"
    dep_dir = tmp_path / "Env" / "Data_easy" / "P000" / "depth_lcam_front"
    img_dir.mkdir(parents=True)
    dep_dir.mkdir(parents=True)
    for i in range(3):
        _write_rgb(img_dir / f"{i:06d}_lcam_front.png")
        _write_tartanair_depth(dep_dir / f"{i:06d}_lcam_front.png")

    ds1 = VDASemanticsTartanAirDataset(root=str(tmp_path), split="val", val_fraction=1.0, **COMMON)
    ds2 = VDASemanticsTartanAirDataset(root=str(tmp_path), split="val", val_fraction=1.0, **COMMON)
    assert torch.equal(ds1[0]["frames"], ds2[0]["frames"])


def test_train_split_frames_within_a_clip_share_one_crop_position(tmp_path):
    """A clip must stay spatially consistent across time -- verified indirectly: since
    every synthetic frame in this fixture is identical, if per-frame crop position varied
    within one clip, that alone wouldn't show up here (frames are the same image) -- so
    this test checks the OTHER real invariant: depth values placed at a known offset
    survive at the same relative position across every frame of the clip, which only
    holds if every frame used the same crop box."""
    img_dir = tmp_path / "Env" / "Data_easy" / "P000" / "image_lcam_front"
    dep_dir = tmp_path / "Env" / "Data_easy" / "P000" / "depth_lcam_front"
    img_dir.mkdir(parents=True)
    dep_dir.mkdir(parents=True)
    for i in range(4):
        _write_rgb(img_dir / f"{i:06d}_lcam_front.png")
        # distinct depth per frame so a crop-position mismatch across frames would show up
        # as different valid-region shapes/values between frames
        _write_tartanair_depth(dep_dir / f"{i:06d}_lcam_front.png", value=float(i + 1))
    # a second trajectory: VideoDepthClipDataset reserves max(1, ...) trajectories for val
    # even at val_fraction=0.0, so a single-trajectory fixture leaves nothing for train
    img_dir2 = tmp_path / "Env2" / "Data_easy" / "P001" / "image_lcam_front"
    dep_dir2 = tmp_path / "Env2" / "Data_easy" / "P001" / "depth_lcam_front"
    img_dir2.mkdir(parents=True)
    dep_dir2.mkdir(parents=True)
    for i in range(4):
        _write_rgb(img_dir2 / f"{i:06d}_lcam_front.png")
        _write_tartanair_depth(dep_dir2 / f"{i:06d}_lcam_front.png", value=float(i + 1))

    ds = VDASemanticsTartanAirDataset(root=str(tmp_path), split="train", val_fraction=0.0, **COMMON)
    sample = ds[0]
    # every frame's depth is uniform (constant per synthetic frame) and masked-valid
    # (0 < value < max_depth) after resize+crop -- if crop boxes differed per frame in a
    # way that clipped differently, mask coverage would differ frame to frame
    mask_sum_per_frame = sample["mask"].sum(dim=(0, 2, 3))
    assert mask_sum_per_frame[0] == mask_sum_per_frame[1]


# Real raw resolutions from VAL-01's pipeline audit (provenance/inventory/val01_pipeline_audit/),
# one per dataset -- not the uniform (960, 540) used by the shape/range tests above.
REAL_RAW_WH_PER_DATASET = [
    ("tartanair", 640, 640),
    ("vkitti", 1242, 375),
    ("irs", 960, 540),
    ("pointodyssey", 960, 540),
]


def test_mismatched_raw_dims_within_a_clip_raise_instead_of_silently_mis_cropping(tmp_path):
    """Opus's audit ask: the crop box is chosen from the first frame and reused for the rest
    of the clip -- if a later frame's raw resolution differs, reusing that box would silently
    mis-crop it. Assert loudly instead."""
    img_dir = tmp_path / "Env" / "Data_easy" / "P000" / "image_lcam_front"
    dep_dir = tmp_path / "Env" / "Data_easy" / "P000" / "depth_lcam_front"
    img_dir.mkdir(parents=True)
    dep_dir.mkdir(parents=True)
    _write_rgb(img_dir / "000000_lcam_front.png", w=640, h=640)
    _write_tartanair_depth(dep_dir / "000000_lcam_front.png", w=640, h=640)
    _write_rgb(img_dir / "000001_lcam_front.png", w=800, h=600)  # different raw resolution
    _write_tartanair_depth(dep_dir / "000001_lcam_front.png", w=800, h=600)

    ds = VDASemanticsTartanAirDataset(root=str(tmp_path), split="val", val_fraction=1.0, **COMMON)
    with pytest.raises(AssertionError, match="same raw resolution"):
        ds[0]


@pytest.mark.parametrize("name,w,h", REAL_RAW_WH_PER_DATASET)
def test_pre_crop_resized_frame_aspect_matches_raw_within_multiple_of_14_rounding(name, w, h):
    """Fable's step-3 spec: the PRE-CROP resized tensor (before the square crop is applied)
    must preserve the raw frame's aspect ratio, up to the multiple-of-14 rounding that
    vda_resize_size introduces -- this is the invariant that distinguishes VDA-semantics
    mode from the old anisotropic-stretch bug (data/video_clip_dataset.py:247), so it is
    checked per dataset here rather than only once against vda_resize_size in isolation."""
    raw_img = Image.new("RGB", (w, h), color=(100, 150, 200))
    resized, _crop_xy, (rw, rh) = _vda_frame_and_crop(raw_img, target=518, split="val")
    assert resized.shape == (rh, rw, 3)
    expected_w, expected_h = vda_resize_size(w, h, target=518)
    assert (rw, rh) == (expected_w, expected_h)
    raw_ar = w / h
    resized_ar = rw / rh
    assert abs(resized_ar - raw_ar) / raw_ar < 0.02, (
        f"{name}: raw {w}x{h} (ar={raw_ar:.4f}) -> pre-crop resized {rw}x{rh} (ar={resized_ar:.4f})"
    )

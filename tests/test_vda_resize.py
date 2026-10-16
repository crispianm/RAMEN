from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from data.vda_resize import (
    vda_crop_box,
    vda_resize_depth_like,
    vda_resize_frame,
    vda_resize_size,
)

# Ground truth from VDA's own real Resize class (video_depth_anything/util/transform.py),
# run empirically on real dataset frames -- VAL-01's pipeline audit, 2026-08-27.
REAL_VDA_CASES = [
    ((1242, 375), (1722, 518)),   # vkitti
    ((960, 540), (924, 518)),     # irs / pointodyssey
    ((640, 640), (518, 518)),     # tartanair (square sample)
]


@pytest.mark.parametrize("wh,expected", REAL_VDA_CASES)
def test_vda_resize_size_matches_the_real_vda_transform(wh, expected):
    assert vda_resize_size(*wh) == expected


def test_vda_resize_size_output_is_a_multiple_of_14():
    for w, h in [(1242, 375), (960, 540), (640, 640), (1920, 1080), (333, 217)]:
        new_w, new_h = vda_resize_size(w, h)
        assert new_w % 14 == 0
        assert new_h % 14 == 0


def test_vda_resize_size_output_is_at_least_the_target():
    for w, h in [(1242, 375), (100, 50), (10, 10)]:
        new_w, new_h = vda_resize_size(w, h, target=518)
        assert new_w >= 518
        assert new_h >= 518


def test_vda_resize_size_preserves_aspect_ratio_within_rounding():
    """The whole point of this mode -- resize must not distort the frame, only the final
    ensure_multiple_of=14 rounding may perturb the aspect ratio slightly."""
    for w, h in [(1242, 375), (960, 540), (1920, 1080), (333, 217)]:
        new_w, new_h = vda_resize_size(w, h)
        raw_ar = w / h
        new_ar = new_w / new_h
        assert abs(new_ar - raw_ar) / raw_ar < 0.02, f"{w}x{h} -> {new_w}x{new_h} distorts aspect ratio"


def test_vda_resize_frame_output_shape_matches_vda_resize_size():
    img = Image.new("RGB", (1242, 375), color=(100, 150, 200))
    arr = vda_resize_frame(img)
    expected_w, expected_h = vda_resize_size(1242, 375)
    assert arr.shape == (expected_h, expected_w, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_vda_resize_depth_like_uses_nearest_not_interpolated():
    """Depth/mask must never be interpolated -- a nearest-neighbour resize keeps every
    output pixel's value identical to some input pixel's value (no new values created)."""
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    resized = vda_resize_depth_like(depth, new_w=4, new_h=4)
    assert set(np.unique(resized).tolist()) <= {1.0, 2.0, 3.0, 4.0}


def test_vda_crop_box_val_split_is_centred_and_deterministic():
    for _ in range(5):
        assert vda_crop_box(1722, 518, target=518, split="val") == (602, 0)


def test_vda_crop_box_train_split_is_random_and_stays_in_bounds():
    random.seed(0)
    positions = {vda_crop_box(1722, 518, target=518, split="train") for _ in range(50)}
    assert len(positions) > 1, "train split should vary crop position across calls"
    for x, y in positions:
        assert 0 <= x <= 1722 - 518
        assert 0 <= y <= 518 - 518 == 0


def test_vda_crop_box_no_slack_when_already_square():
    assert vda_crop_box(518, 518, target=518, split="train") == (0, 0)
    assert vda_crop_box(518, 518, target=518, split="val") == (0, 0)

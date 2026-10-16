"""VAL-01: VDA's exact resize semantics, as a drop-in opt-in mode for the existing dataset
classes -- short-edge resize to a target size (aspect preserved, ensure_multiple_of=14,
"lower_bound"), then a square crop (random for train, centre for val). Mirrors
Video-Depth-Anything's own ``video_depth_anything/util/transform.py`` Resize class exactly
(same get_size/constrain_to_multiple_of algorithm), reimplemented here rather than imported
so this repo has no runtime dependency on the vendored VDA checkout.

Existing datasets default to their current (`"stretch"`/bespoke) behaviour unchanged --
this is opt-in via ``resize_mode="vda"``, so nothing about the frozen, paused pipeline
changes for any existing config (data/video_clip_dataset.py:247's stretch and
vkitti_dataset.py's own crop-then-resize both stay exactly as they were).
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np
from PIL import Image


def _constrain_to_multiple_of(x: float, multiple_of: int, min_val: int = 0) -> int:
    """Matches VDA's own constrain_to_multiple_of exactly (round, then ceil-adjust up if
    under min_val) -- caught in review: a floor-division-based ceiling
    ``(x // multiple_of + 1) * multiple_of`` diverges from ``math.ceil`` on exact
    multiples (e.g. x=504, multiple_of=14 gives 518 via floor-div+1, but VDA's own
    ceil(504/14)*14 = 504 -- must use real ceil, not the floor-division shortcut).

    VDA's real method also takes a ``max_val`` that floor-adjusts down when over it, but
    that branch is only ever reached from VDA's own ``resize_method="upper_bound"`` path
    (video_depth_anything/util/transform.py's ``get_size``) -- the "lower_bound" method this
    module implements never passes ``max_val``, so it's dead code here and correctly
    omitted rather than reimplemented unused."""
    y = int(round(x / multiple_of) * multiple_of)
    if y < min_val:
        y = int(math.ceil(x / multiple_of) * multiple_of)
    return y


def vda_resize_size(width: int, height: int, target: int = 518, multiple_of: int = 14) -> tuple[int, int]:
    """Short-edge resize target (new_width, new_height), aspect-preserved, "lower_bound"
    method -- matches VDA's Resize(keep_aspect_ratio=True, resize_method="lower_bound")
    exactly. Both outputs are >= target and a multiple of `multiple_of`."""
    scale_h = target / height
    scale_w = target / width
    if scale_w > scale_h:
        scale_h = scale_w
    else:
        scale_w = scale_h
    new_h = _constrain_to_multiple_of(scale_h * height, multiple_of, min_val=target)
    new_w = _constrain_to_multiple_of(scale_w * width, multiple_of, min_val=target)
    return new_w, new_h


def vda_resize_frame(img: Image.Image, target: int = 518, multiple_of: int = 14) -> np.ndarray:
    """Short-edge resize an RGB PIL image via VDA's own cubic interpolation. Returns a
    float32 [0, 1] HWC array (pre-crop) -- the caller crops to a square afterward."""
    w, h = img.size
    new_w, new_h = vda_resize_size(w, h, target, multiple_of)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def vda_resize_depth_like(arr: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Nearest-neighbour resize for depth/mask/sky -- matches VDA's Resize class (which
    resizes depth/mask with cv2.INTER_NEAREST, never interpolating GT)."""
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def vda_crop_box(resized_w: int, resized_h: int, target: int, split: str) -> tuple[int, int]:
    """(x0, y0) top-left corner of a `target`x`target` square crop -- random position for
    train (matching VDA's own random_crop augmentation), centred for val (deterministic,
    matching VDA's own eval path, which never randomly augments)."""
    max_x = max(0, resized_w - target)
    max_y = max(0, resized_h - target)
    if split == "train":
        return random.randint(0, max_x), random.randint(0, max_y)
    return max_x // 2, max_y // 2

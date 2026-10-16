"""Protocol v3 real-data clip sources (KITTI raw, NYUv2 train, ScanNet++ train, Spring): one VDA-semantics clip base shared by the new loaders.

A subclass builds ``self.clips = [(img_files, depth_files, video_id, start), ...]`` and implements ``read_depth(path) -> (H,W) float32 metres`` (<= 0 / non-finite = invalid). ``__getitem__`` is the same
short-edge resize + one square crop per clip as ``data/vda_semantics_dataset.py`` (RGB: VDA's cubic resize; depth/mask: nearest, never interpolated, so SPARSE LiDAR stays sparse), with the same output dict
(``frames`` / ``depth`` / ``mask`` / ``sky`` zeros / ``video_id`` / ``start_frame``). ``with_degradation(cls)`` returns the sibling that adds ``degraded_frames`` AFTER resize+crop through the shared
``_VDADegradedMixin`` (same degradation machinery as every other degraded source).

``sparse_gt`` (the source spec's ``sparse_gt``; None = nothing emitted): True / False emits ``sparse`` (1.0 / 0.0) on EVERY clip of the source, and train.py's VDA loss then uses a sparse-safe masked SSI
(per-frame median / scale over VALID pixels only) for the clips with 1.0 -- VDA's own normalisation takes the median over the whole frame including the masked-out zeros, which is wrong for LiDAR density.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .vda_resize import vda_resize_depth_like
from .vda_semantics_dataset import _VDADegradedMixin, _crop, _vda_frame_and_crop


class V3ClipDataset(Dataset):
    sparse_gt: Optional[bool] = None
    metric_supervision: Optional[bool] = None   # same per-source keys as the VDA-semantics classes (train.py _make_source sets them; None = nothing emitted)
    silog_max_depth: Optional[float] = None

    def __init__(self, num_frames: int = 32, image_size: Tuple[int, int] = (518, 518), split: str = "train", clip_stride: int = 32,
                 frame_stride: int = 1, max_depth: float = 80.0, seed: int = 42, degrade: bool = True, degrade_mode="full",
                 degrade_severity: Optional[float] = None, max_clips_per_video: Optional[int] = None, **_ignored) -> None:
        super().__init__()
        assert image_size[0] == image_size[1], "VDA-semantics mode requires a square target (e.g. 518x518)"
        self.num_frames, self.image_size, self.split = num_frames, tuple(image_size), split
        self.clip_stride, self.frame_stride, self.max_depth = clip_stride, frame_stride, float(max_depth)
        self.seed, self.degrade, self.degrade_mode, self.degrade_severity = int(seed), degrade, degrade_mode, degrade_severity
        self.severity_max = torch.ones(1).share_memory_()
        self.max_clips_per_video = max_clips_per_video
        self.clips: List[Tuple[list, list, str, int]] = []

    # -- subclass hooks ---------------------------------------------------------------------------------------------
    def read_depth(self, path) -> np.ndarray:
        raise NotImplementedError

    def read_rgb(self, path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def load_clip(self, idx: int, indices: List[int]):
        """``(list of PIL RGB frames, list of (H,W) float32 metre depth maps)`` for clip ``idx``. Default: one file per frame (``read_rgb`` / ``read_depth``); a source with a video stream overrides it."""
        img_files, dep_files, _vid, _start = self.clips[idx]
        return [self.read_rgb(img_files[i]) for i in indices], [self.read_depth(dep_files[i]) for i in indices]

    # -- shared ------------------------------------------------------------------------------------------------------
    def add_video(self, video_id: str, imgs: list, deps: list) -> None:
        """Index the clips of one video (aligned ``imgs`` / ``deps``): starts every ``clip_stride`` frames, optionally thinned to ``max_clips_per_video``."""
        m = min(len(imgs), len(deps))
        span = (self.num_frames - 1) * self.frame_stride + 1
        if m < span:
            return
        starts = list(range(0, m - span + 1, self.clip_stride))
        if self.max_clips_per_video and len(starts) > self.max_clips_per_video:
            step = len(starts) / self.max_clips_per_video
            starts = [starts[int(i * step)] for i in range(self.max_clips_per_video)]
        imgs_m, deps_m = imgs[:m], deps[:m]      # protocol v3 WP27 F3: slice ONCE per video; every clip of the video shares these two lists (a per-clip slice copy made memory = clips x frames)
        for s in starts:
            self.clips.append((imgs_m, deps_m, video_id, s))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        _imgs, _deps, video_id, start = self.clips[idx]
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]
        H = self.image_size[0]
        frames, depths, masks = [], [], []
        crop_xy = rwh = raw_wh = None
        pil_frames, depth_frames = self.load_clip(idx, indices)
        for i, raw_img, d_raw in zip(indices, pil_frames, depth_frames):
            if raw_wh is None:
                raw_wh = raw_img.size
            else:
                assert raw_img.size == raw_wh, f"every frame in a clip must share one raw resolution: {video_id} frame {i} is {raw_img.size}, clip started at {raw_wh}"
            resized, crop_i, rwh_i = _vda_frame_and_crop(raw_img, H, self.split)
            if crop_xy is None:
                crop_xy, rwh = crop_i, rwh_i
            x0, y0 = crop_xy
            frames.append(torch.from_numpy(_crop(resized, x0, y0, H).copy()).permute(2, 0, 1))
            assert d_raw.shape == (raw_wh[1], raw_wh[0]), f"{video_id} frame {i}: depth {d_raw.shape} != RGB {raw_wh[::-1]}"
            valid = np.isfinite(d_raw) & (d_raw > 0) & (d_raw < self.max_depth)
            d_resized = vda_resize_depth_like(np.where(valid, d_raw, 0.0).astype(np.float32), *rwh)
            valid_r = vda_resize_depth_like(valid.astype(np.float32), *rwh)
            depths.append(torch.from_numpy(_crop(d_resized, x0, y0, H).copy()).unsqueeze(0).float())
            masks.append(torch.from_numpy(_crop(valid_r, x0, y0, H).copy()).unsqueeze(0))
        mask = torch.stack(masks, 1)
        out = {"frames": torch.stack(frames, 1).contiguous(), "depth": torch.stack(depths, 1), "mask": mask, "sky": torch.zeros_like(mask),
               "video_id": video_id, "start_frame": start}
        if self.sparse_gt is not None:
            out["sparse"] = torch.tensor(1.0 if self.sparse_gt else 0.0)
        if self.metric_supervision is not None:
            out["metric_sup"] = torch.tensor(1.0 if self.metric_supervision else 0.0)
        if self.silog_max_depth is not None:
            out["silog_ceil"] = torch.tensor(float(self.silog_max_depth))
        return out


def with_degradation(cls):
    """``cls`` + the post-resize degradation (the same mixin every ``VDASemanticsDegraded*`` class uses)."""
    return type(f"Degraded{cls.__name__}", (_VDADegradedMixin, cls), {})

"""Unlabeled real-video source -- RGB only, NO ground-truth depth of any kind.

DIST-01 arm (c) (docs/RAMEN_V3_PROGRAMME.md D1, 2026-09-18): the parity-B GT recipe on the
four synthetic sources does not transfer to real degraded video (LOOP_JOURNAL 14:05), so the
student also trains on the real unlabeled video already on Isambard (``youtube_vos``,
``bvi_aom``) where the ONLY supervision is the ONLINE TEACHER's disparity on the same
degraded frames (``train.online_distillation_term``, ``loss.distill_weight``).

This is deliberately NOT ``pseudo_gt`` with the depth tree removed: no depth file is opened,
listed or required. Every batch carries ``depth`` and ``mask`` as all-zero tensors, so every
existing ground-truth loss sees an empty mask and contributes exactly zero (train.py's
``compute_loss`` / ``_compute_loss_vda`` return a finite zero for an empty mask rather than
NaN, so the trainer's ``torch.isfinite(loss)`` guard does not silently drop these clips).

Batch-key contract -- byte-identical key SET to every other training source, which matters
because sources are mixed in one ``ConcatDataset`` and collated together (a source emitting an
extra or missing key would break ``default_collate`` on a mixed batch):

    frames          (3, T, H, W)  clean RGB in [0, 1]
    depth           (1, T, H, W)  ALL ZEROS      (no GT)
    mask            (1, T, H, W)  ALL ZEROS      (no GT)
    sky             (1, T, H, W)  ALL ZEROS      (no sky label)
    video_id        str           sequence directory name
    start_frame     int
    degraded_frames (3, T, H, W)  only when degrade=True
    degrade_params  (5,)          only when degrade=True

Frame enumeration is shared with the pseudo-GT source
(``pseudo_gt_dataset.enumerate_video_clips``), so ``num_frames`` / ``clip_stride`` /
``frame_stride`` / ``max_clips_per_video`` / the deterministic train-val sequence split all
behave exactly as they do there.

``VideoOnlyDataset`` below is the legacy-loader variant (direct resize to ``image_size``,
matching ``PseudoGTVideoDataset``). The ``resize_mode: vda`` variants -- short-edge 518 +
square crop, and the degraded sibling -- live in ``data/vda_semantics_dataset.py`` alongside
the other VDA-semantics classes so they reuse the same crop helpers and ``_VDADegradedMixin``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .pseudo_gt_dataset import enumerate_video_clips
from .video_clip_dataset import _degrade_clip


class VideoOnlyDataset(Dataset):
    """Real video clips with no depth supervision at all (teacher-only source)."""

    def __init__(
        self,
        rgb_root: str,
        num_frames: int = 8,
        image_size: Tuple[int, int] = (518, 518),
        split: str = "train",
        clip_stride: int = 8,
        frame_stride: int = 1,
        max_depth: float = 80.0,
        val_fraction: float = 0.1,
        seed: int = 42,
        degrade: bool = True,
        degrade_mode="full",
        degrade_severity: Optional[float] = None,
        max_trajectories: Optional[int] = None,
        max_clips_per_video: Optional[int] = None,
        teacher_cache_dir: Optional[str] = None,  # accepted+ignored (shared `common` kwargs)
        teacher_name: str = "depth_pro",
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.image_size = tuple(image_size)
        self.split = split
        self.clip_stride = clip_stride
        self.frame_stride = frame_stride
        self.max_depth = max_depth
        self.degrade = degrade
        self.degrade_mode = degrade_mode
        self.degrade_severity = degrade_severity
        # curriculum cap, shared memory so forked workers see main-process ramps (as VKITTI/TA)
        self.severity_max = torch.ones(1).share_memory_()
        self.seed = int(seed)

        self.clips: List[Tuple[List[Path], None, str, int]] = enumerate_video_clips(
            rgb_root, depth_root=None, num_frames=num_frames, clip_stride=clip_stride,
            frame_stride=frame_stride, split=split, val_fraction=val_fraction, seed=seed,
            max_trajectories=max_trajectories, max_clips_per_video=max_clips_per_video,
        )

    def __len__(self) -> int:
        return len(self.clips)

    def _empty_gt(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """All-zero depth/mask/sky matching ``frames`` (3, T, H, W) -> (1, T, H, W)."""
        z = torch.zeros((1,) + frames.shape[1:], dtype=torch.float32)
        return {"depth": z, "mask": z.clone(), "sky": z.clone()}

    def _apply_degradation(self, result: Dict[str, torch.Tensor], idx: int) -> Dict[str, torch.Tensor]:
        """Same rng / severity / mode selection as every other degraded dataset class."""
        if not self.degrade:
            return result
        rng = np.random.RandomState(self.seed + idx) if self.split != "train" else np.random.RandomState()
        sev = self.degrade_severity
        mode = rng.choice(self.degrade_mode) if isinstance(self.degrade_mode, list) else self.degrade_mode
        if sev is None and self.split == "train" and mode != "elvis":
            sev = float(rng.uniform(0.0, float(self.severity_max[0])))
        degraded, params = _degrade_clip(result["frames"].permute(1, 0, 2, 3), rng, mode, severity=sev)
        result["degraded_frames"] = degraded.permute(1, 0, 2, 3).contiguous()
        result["degrade_params"] = torch.from_numpy(params)
        return result

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_files, _, video_id, start = self.clips[idx]
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]
        H, W = self.image_size

        frames = []
        for i in indices:
            img = Image.open(img_files[i]).convert("RGB").resize((W, H), Image.BILINEAR)
            frames.append(torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1))
        frames = torch.stack(frames, 1).contiguous()          # (3, T, H, W)

        result: Dict[str, torch.Tensor] = {"frames": frames, **self._empty_gt(frames),
                                           "video_id": video_id, "start_frame": start}
        return self._apply_degradation(result, idx)

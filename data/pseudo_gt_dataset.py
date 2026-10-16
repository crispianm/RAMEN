"""Pseudo-GT video clip dataset — real video + VDA-L-metric teacher depth.

Adds diverse real-world video to the degraded-metric-depth training mix (the VDA
data recipe): the VDA-L-metric teacher depth is the GT target, and the synthetic
low-light/blur/noise degradation hits the *student* input only (teacher depth /
mask stay clean). RGB frames live in a source tree; per-frame metric depth lives
in a parallel pseudo-GT tree written by ``scripts/teacher_pseudo_gt.py``.

Layout::

    rgb_root/<seq>/<frame>.{jpg,png}                       (sorted, 1:1 with depth)
    depth_root/<seq>/depth/depth_%04d.npz   {"depth": float16 (H,W) metres}

Same per-sample contract and degradation path as ``VideoDepthClipDataset`` so it
drops into ``ConcatDataset`` alongside VKITTI / TartanAir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .video_clip_dataset import _degrade_clip  # reuse the shared ELVIS-style degradation

_IMG_EXTS = ("*.jpg", "*.png")


def enumerate_video_clips(
    rgb_root: str,
    *,
    num_frames: int,
    clip_stride: int,
    frame_stride: int,
    split: str,
    val_fraction: float,
    seed: int,
    max_trajectories: Optional[int] = None,
    max_clips_per_video: Optional[int] = None,
    depth_root: Optional[str] = None,
) -> List[Tuple[List[Path], Optional[List[Path]], str, int]]:
    """Enumerate ``<root>/<seq>/<frame>.{jpg,png}`` video directories into clip indices.

    Extracted verbatim from ``PseudoGTVideoDataset.__init__`` (2026-09-18, DIST-01 arm (c))
    so the unlabeled-real-video source (``data/video_only_dataset.py``) shares ONE frame
    enumeration with the pseudo-GT source rather than a near-copy that can drift: same
    deterministic sequence shuffle + train/val split convention, same ``span``/``clip_stride``
    windowing, same ``max_clips_per_video`` even-thinning.

    ``depth_root=None`` is the video-only mode: sequences are not filtered on a parallel
    ``<seq>/depth`` directory and no depth files are listed or read at all. Each clip is
    ``(image_paths, depth_paths_or_None, seq_name, start_index)``.
    """
    rgb_p = Path(rgb_root)
    seqs = sorted(p.name for p in rgb_p.iterdir() if p.is_dir() and not p.name.startswith("."))
    dep_p = Path(depth_root) if depth_root is not None else None
    if dep_p is not None:
        # keep only sequences that actually have a pseudo-GT depth dir
        seqs = [s for s in seqs if (dep_p / s / "depth").is_dir()]
        if not seqs:
            raise FileNotFoundError(f"No <seq>/depth dirs under {depth_root} matching {rgb_root}.")
    elif not seqs:
        raise FileNotFoundError(f"No video directories under {rgb_root}.")

    # deterministic seq-level train/val split (same convention as VideoDepthClipDataset)
    import random
    rng = random.Random(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(len(seqs) * val_fraction))
    seqs = seqs[:n_val] if split == "val" else seqs[n_val:]
    if max_trajectories is not None:
        seqs = seqs[:max_trajectories]

    clips: List[Tuple[List[Path], Optional[List[Path]], str, int]] = []
    span = (num_frames - 1) * frame_stride + 1
    for s in seqs:
        imgs = sorted(sum((list((rgb_p / s).glob(e)) for e in _IMG_EXTS), []))
        deps = sorted((dep_p / s / "depth").glob("depth_*.npz")) if dep_p is not None else None
        n = min(len(imgs), len(deps)) if deps is not None else len(imgs)
        if n < span:
            continue
        starts = list(range(0, n - span + 1, clip_stride))
        if max_clips_per_video and len(starts) > max_clips_per_video:
            step = len(starts) / max_clips_per_video
            starts = [starts[int(i * step)] for i in range(max_clips_per_video)]
        for start in starts:
            clips.append((imgs[:n], deps[:n] if deps is not None else None, s, start))
    return clips


class PseudoGTVideoDataset(Dataset):
    """Real-video clips with VDA-L-metric pseudo-GT depth as the training target."""

    def __init__(
        self,
        rgb_root: str,
        depth_root: str,
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
        split_seed: Optional[int] = None,   # E4: seeds ONLY the held-out trajectory shuffle (None = `seed`: byte-identical); val degradation keeps `seed`
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

        self.clips: List[Tuple[List[Path], List[Path], str, int]] = enumerate_video_clips(
            rgb_root, depth_root=depth_root, num_frames=num_frames, clip_stride=clip_stride,
            frame_stride=frame_stride, split=split, val_fraction=val_fraction, seed=seed if split_seed is None else split_seed,
            max_trajectories=max_trajectories, max_clips_per_video=max_clips_per_video,
        )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_files, dep_files, video_id, start = self.clips[idx]
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]
        H, W = self.image_size

        frames, depths, masks = [], [], []
        for i in indices:
            img = Image.open(img_files[i]).convert("RGB").resize((W, H), Image.BILINEAR)
            frames.append(torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1))
            d = np.load(dep_files[i])["depth"].astype(np.float32)          # (H0, W0) metres
            valid = np.isfinite(d) & (d > 0) & (d < self.max_depth)
            depths.append(torch.from_numpy(np.clip(d, 0.0, self.max_depth)).unsqueeze(0))
            masks.append(torch.from_numpy(valid.astype(np.float32)).unsqueeze(0))

        frames = torch.stack(frames, 0)                       # (T, 3, H, W)
        depth = torch.stack(depths, 1)                        # (1, T, H0, W0)
        mask = torch.stack(masks, 1)
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(depth.permute(1, 0, 2, 3), (H, W), mode="nearest").permute(1, 0, 2, 3)
            mask = F.interpolate(mask.permute(1, 0, 2, 3), (H, W), mode="nearest").permute(1, 0, 2, 3)

        result: Dict[str, torch.Tensor] = {
            "frames": frames.permute(1, 0, 2, 3).contiguous(),  # (C, T, H, W)
            "depth": depth,
            "mask": mask,
            "sky": torch.zeros_like(mask),   # no reliable sky label from pseudo-GT
            "video_id": video_id,
            "start_frame": start,
        }
        if self.degrade:
            rng = np.random.RandomState(self.seed + idx) if self.split != "train" else np.random.RandomState()
            sev = self.degrade_severity
            mode = rng.choice(self.degrade_mode) if isinstance(self.degrade_mode, list) else self.degrade_mode
            if sev is None and self.split == "train" and mode != "elvis":
                sev = float(rng.uniform(0.0, float(self.severity_max[0])))
            degraded, params = _degrade_clip(frames, rng, mode, severity=sev)
            result["degraded_frames"] = degraded.permute(1, 0, 2, 3).contiguous()
            result["degrade_params"] = torch.from_numpy(params)
        return result

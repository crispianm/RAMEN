"""Dark Zurich test loader — real night-driving RGB clips (QUALITATIVE; no depth GT).

Dark Zurich ships no metric depth, so this feeds the paper's qualitative real-low-light panels:
the real night footage IS the degradation (no synthetic degradation applied). Returns the same
clip contract as the other loaders minus real depth (zeros stand in), with the night frames placed
in BOTH `frames` and `degraded_frames` so the eval/viz path is unchanged.

Layout::  <root>/rgb_anon/<split>/night/<seq>/<frame>.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class DarkZurichDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,                    # e.g. data/dark_zurich/Dark_Zurich_val_anon
        num_frames: int = 8,
        image_size: Tuple[int, int] = (518, 518),
        split: str = "val",
        clip_stride: int = 8,
        frame_stride: int = 1,
        max_clips_per_video: Optional[int] = None,
        **_ignore,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.image_size = tuple(image_size)
        self.frame_stride = frame_stride
        self.degrade = False          # real night footage — no synthetic degradation

        night = Path(root) / "rgb_anon" / split / "night"
        seqs = [p for p in sorted(night.iterdir()) if p.is_dir()] if night.is_dir() else []
        if not seqs:
            raise FileNotFoundError(f"No Dark Zurich night sequences under {night}.")
        self.clips: List[Tuple[List[Path], str, int]] = []
        span = (num_frames - 1) * frame_stride + 1
        for sc in seqs:
            imgs = sorted(sc.glob("*.png"))
            if len(imgs) < span:
                continue
            starts = list(range(0, len(imgs) - span + 1, clip_stride))
            if max_clips_per_video and len(starts) > max_clips_per_video:
                step = len(starts) / max_clips_per_video
                starts = [starts[int(i * step)] for i in range(max_clips_per_video)]
            for start in starts:
                self.clips.append((imgs, sc.name, start))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_files, video_id, start = self.clips[idx]
        H, W = self.image_size
        frames = []
        for t in range(self.num_frames):
            im = Image.open(img_files[start + t * self.frame_stride]).convert("RGB").resize((W, H), Image.BILINEAR)
            frames.append(torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1))
        clip = torch.stack(frames, 0).permute(1, 0, 2, 3).contiguous()   # (3,T,H,W)
        zeros = torch.zeros(1, self.num_frames, H, W)
        return {
            "frames": clip,
            "degraded_frames": clip,          # night footage is already the degraded input
            "depth": zeros, "mask": zeros, "sky": zeros,   # no GT (qualitative only)
            "video_id": video_id, "start_frame": start,
        }

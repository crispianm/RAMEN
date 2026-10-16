"""NYUv2 test loader — official 654-image indoor depth benchmark (zero-shot reporting).

Samples are single RGB-D `.h5` files (`rgb` (3,H,W) uint8, `depth` (H,W) float32 metres). The
video model consumes clips, so each image is replicated to a `num_frames` static clip; when
`degrade` is on, each frame gets INDEPENDENT degradation (so temporal aggregation is exercised on
a static scene — the degraded-benchmark condition). Reported through the same harness as every
other test set.

Layout::  <root>/*/{train,val}/official/*.h5   (val-* shards hold the official test split)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .video_clip_dataset import _degrade_clip   # h5py imported lazily in __getitem__ (eval-only dep)


class NYUDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        num_frames: int = 8,
        image_size: Tuple[int, int] = (518, 518),
        split: str = "val",           # NYU official test lives in the val-* shards
        max_depth: float = 80.0,
        seed: int = 42,
        degrade: bool = True,
        degrade_mode="full",
        degrade_severity: Optional[float] = None,
        max_samples: Optional[int] = None,
        **_ignore,                     # swallow shared clip kwargs (clip_stride, frame_stride, ...)
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.image_size = tuple(image_size)
        self.split = split
        self.max_depth = max_depth
        self.degrade = degrade
        self.degrade_mode = degrade_mode
        self.degrade_severity = degrade_severity
        self.severity_max = torch.ones(1).share_memory_()
        self.seed = int(seed)

        root_p = Path(root)
        self.files: List[Path] = sorted(
            p for p in root_p.rglob("*.h5") if f"/{split}/" in str(p) or f"{split}-" in p.parts[-3]
        )
        if not self.files:  # fall back: any .h5 under a matching-split shard dir
            self.files = sorted(p for p in root_p.rglob("*.h5") if split in str(p))
        if not self.files:
            raise FileNotFoundError(f"No NYU .h5 files under {root} for split={split}.")
        if max_samples:
            self.files = self.files[:max_samples]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import h5py
        H, W = self.image_size
        with h5py.File(self.files[idx], "r") as h:
            rgb = np.asarray(h["rgb"], np.float32) / 255.0        # (3, H0, W0)
            d = np.asarray(h["depth"], np.float32)                # (H0, W0) metres
        rgb = torch.from_numpy(rgb)
        rgb = F.interpolate(rgb.unsqueeze(0), (H, W), mode="bilinear", align_corners=False)[0]
        depth = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)     # (1,1,H0,W0)
        depth = F.interpolate(depth, (H, W), mode="nearest")[0]   # (1,H,W)
        valid = (depth > 0) & (depth < self.max_depth) & torch.isfinite(depth)

        T = self.num_frames
        frames = rgb.unsqueeze(1).repeat(1, T, 1, 1)              # (3,T,H,W) static clip
        result: Dict[str, torch.Tensor] = {
            "frames": frames.contiguous(),
            "depth": depth.unsqueeze(1).repeat(1, T, 1, 1),
            "mask": valid.float().unsqueeze(1).repeat(1, T, 1, 1),
            "sky": torch.zeros(1, T, H, W),
            "video_id": self.files[idx].stem,
            "start_frame": 0,
        }
        if self.degrade:
            rng = np.random.RandomState(self.seed + idx)          # deterministic per sample
            sev = self.degrade_severity
            mode = rng.choice(self.degrade_mode) if isinstance(self.degrade_mode, list) else self.degrade_mode
            # frames is (3,T,H,W) identical; _degrade_clip wants (T,3,H,W) and degrades each frame
            degraded, params = _degrade_clip(frames.permute(1, 0, 2, 3), rng, mode, severity=sev)
            result["degraded_frames"] = degraded.permute(1, 0, 2, 3).contiguous()
            result["degrade_params"] = torch.from_numpy(params)
        return result

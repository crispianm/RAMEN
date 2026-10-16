"""PointOdyssey clip dataset — synthetic RGB + metric depth video.

One of the four synthetic training corpora. Official train/val/test splits are separate
directories (point at the split via ``root``). Depth is a 16-bit PNG; the decode was verified
against the dataset's own 3D point tracks (anno.npz): metres = png / 65535 * 1000.

Layout::

    root/<scene>/{rgbs/rgb_%05d.jpg, depths/depth_%05d.png (uint16), masks/, anno.npz}
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from torch.utils.data import Dataset

from .video_clip_dataset import _degrade_clip

_PO_DEPTH_SCALE = 1000.0 / 65535.0   # uint16 png -> metres (verified vs anno.npz 3D tracks)


def filter_scenes(scenes: List[Path], keep_list: str, keep_sha256: Optional[str] = None) -> List[Path]:
    """K80 option B: restrict ``scenes`` to the committed kept list (``kept`` names in the JSON written by scripts/dn01_po_filter.py). The list's own hash is re-computed (sha256 of the sorted names joined by
    newlines) and must equal its stored ``kept_sha256`` and, when given, ``keep_sha256``; a kept scene missing under ``root`` fails loudly (a host with a different copy)."""
    import hashlib
    import json
    d = json.loads(Path(keep_list).read_text())
    kept = sorted(d["kept"])
    h = hashlib.sha256("\n".join(kept).encode()).hexdigest()
    if h != d.get("kept_sha256") or (keep_sha256 is not None and h != keep_sha256):
        raise ValueError(f"PointOdyssey keep list {keep_list}: sha256 {h} != stored {d.get('kept_sha256')} / pinned {keep_sha256}")
    have = {p.name: p for p in scenes}
    missing = [n for n in kept if n not in have]
    if missing:
        raise FileNotFoundError(f"PointOdyssey keep list names {len(missing)} scene(s) absent under the root, e.g. {missing[:3]}")
    return [have[n] for n in kept]


class PointOdysseyDataset(Dataset):
    def __init__(
        self,
        root: str,
        num_frames: int = 8,
        image_size: Tuple[int, int] = (518, 518),
        split: str = "train",
        clip_stride: int = 8,
        frame_stride: int = 1,
        max_depth: float = 80.0,
        val_fraction: float = 0.0,   # PointOdyssey ships official splits; default: use all of `root`
        seed: int = 42,
        degrade: bool = True,
        degrade_mode="full",
        degrade_severity: Optional[float] = None,
        max_trajectories: Optional[int] = None,
        max_clips_per_video: Optional[int] = None,
        teacher_cache_dir: Optional[str] = None,  # accepted+ignored (shared `common` kwargs)
        teacher_name: str = "depth_pro",
        split_seed: Optional[int] = None,   # E4: seeds ONLY the held-out scene shuffle (None = `seed`: byte-identical); val degradation keeps `seed`
        keep_list: Optional[str] = None,    # K80 option B: the committed background-depth filter list (scripts/dn01_po_filter.py); None = every scene (byte-identical)
        keep_sha256: Optional[str] = None,  # pin of that list's kept_sha256: a different list is refused
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
        self.severity_max = torch.ones(1).share_memory_()
        self.seed = int(seed)

        root_p = Path(root)
        scenes = [p for p in sorted(root_p.iterdir())
                  if p.is_dir() and (p / "rgbs").is_dir() and (p / "depths").is_dir()]
        if not scenes:
            raise FileNotFoundError(f"No PointOdyssey scenes under {root} with rgbs/ + depths/.")
        if keep_list is not None:
            scenes = filter_scenes(scenes, keep_list, keep_sha256)
        if val_fraction > 0:   # optional internal held-out split when official splits aren't used
            rng = random.Random(seed if split_seed is None else split_seed)
            rng.shuffle(scenes)
            n_val = max(1, int(len(scenes) * val_fraction))
            scenes = scenes[:n_val] if split == "val" else scenes[n_val:]
            scenes = sorted(scenes)
        if max_trajectories is not None:
            scenes = scenes[:max_trajectories]

        self.clips: List[Tuple[List[Path], List[Path], str, int]] = []
        span = (num_frames - 1) * frame_stride + 1
        for sc in scenes:
            imgs = sorted((sc / "rgbs").glob("rgb_*.jpg"))
            deps = sorted((sc / "depths").glob("depth_*.png"))
            m = min(len(imgs), len(deps))
            if m < span:
                continue
            starts = list(range(0, m - span + 1, clip_stride))
            if max_clips_per_video and len(starts) > max_clips_per_video:
                step = len(starts) / max_clips_per_video
                starts = [starts[int(i * step)] for i in range(max_clips_per_video)]
            for start in starts:
                self.clips.append((imgs[:m], deps[:m], sc.name, start))

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
            raw = cv2.imread(str(dep_files[i]), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(f"Could not read PO depth: {dep_files[i]}")
            d = raw.astype(np.float32) * _PO_DEPTH_SCALE
            valid = np.isfinite(d) & (d > 0) & (d < self.max_depth)
            depths.append(torch.from_numpy(np.clip(d, 0.0, self.max_depth)).unsqueeze(0))
            masks.append(torch.from_numpy(valid.astype(np.float32)).unsqueeze(0))

        frames = torch.stack(frames, 0)
        depth = torch.stack(depths, 1)
        mask = torch.stack(masks, 1)
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(depth.permute(1, 0, 2, 3), (H, W), mode="nearest").permute(1, 0, 2, 3)
            mask = F.interpolate(mask.permute(1, 0, 2, 3), (H, W), mode="nearest").permute(1, 0, 2, 3)

        result: Dict[str, torch.Tensor] = {
            "frames": frames.permute(1, 0, 2, 3).contiguous(),
            "depth": depth,
            "mask": mask,
            "sky": torch.zeros_like(mask),
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

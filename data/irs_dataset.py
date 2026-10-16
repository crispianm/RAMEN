"""IRS (Indoor Robotics Stereo) clip dataset — synthetic indoor RGB + stereo-disparity GT.

One of the four synthetic training corpora. Per-scene rendered camera sequences; we use the
left RGB and the d_*.exr GT (right image + normals ignored). The EXRs store STEREO DISPARITY
in PIXELS in a single channel (the others are all-zero) — NOT metres. Depth = f·b/disparity
with f=480 px (90° HFOV at 960 px width) and baseline 0.1 m. Verified 2026-07-16: read as
metres the "depth" medians are an impossible 16–50 m indoors and the flagship scores 0.2–2.5
AbsRel on clean frames; read as disparity it scores 0.08–0.17 (Home/Office/Restaurant).

Layout::

    root/<family>/<scene>/{l_%05d.png (left RGB, RGBA), d_%05d.exr (depth, metres)}

Deterministic scene-level 3-way split (train/val/test) so a held-out subset can serve as
in-loop validation while the test split feeds the paper benchmark.
"""

from __future__ import annotations

import logging  # protocol v3 WP25 irsfix
import os
import random
import re  # protocol v3 WP25 irsfix
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from .video_clip_dataset import _degrade_clip  # shared ELVIS-style degradation


logger = logging.getLogger(__name__)   # protocol v3 WP25 irsfix
_FRAME_RE = re.compile(r"^(?P<p>[a-z])_(?P<i>\d+)\.(?P<e>png|exr)$")


def _index_files(sc: Path, prefix: str, ext: str) -> Dict[int, Path]:
    """frame number -> path for ``<prefix>NNNNN<ext>`` in a scene dir. Strict full-name match: dot-files / partial downloads (``.l_00002.png.abc``) are ignored."""
    out: Dict[int, Path] = {}
    for p in sc.iterdir():
        m = _FRAME_RE.match(p.name)
        if m and p.name.startswith(prefix) and p.name.endswith(ext):
            out[int(m.group("i"))] = p
    return out


def _consecutive_runs(idx: List[int]) -> List[List[int]]:
    """Split a sorted index list into maximal runs of consecutive integers."""
    runs: List[List[int]] = []
    for k in idx:
        if runs and k == runs[-1][-1] + 1:
            runs[-1].append(k)
        else:
            runs.append([k])
    return runs


_IRS_FB = 48.0  # focal(480 px) × baseline(0.1 m): depth[m] = 48 / disparity[px]


def _read_exr_depth(path: Path) -> np.ndarray:
    """IRS d_*.exr → (H, W) float32 metres, converted from the stored stereo DISPARITY
    (pixels; single non-zero channel). Invalid disparity (≤0) maps to a huge depth that
    the caller's ``d < max_depth`` validity mask rejects."""
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(f"Could not read EXR depth: {path} (OPENCV_IO_ENABLE_OPENEXR set?)")
    if a.ndim == 3:
        a = a.max(axis=2)          # disparity is in the single non-zero channel
    return _IRS_FB / np.clip(a.astype(np.float32), 1e-3, None)


class IRSDataset(Dataset):
    def __init__(
        self,
        root: str,
        num_frames: int = 8,
        image_size: Tuple[int, int] = (518, 518),
        split: str = "train",
        clip_stride: int = 8,
        frame_stride: int = 1,
        max_depth: float = 80.0,
        val_fraction: float = 0.1,
        test_fraction: float = 0.0,
        seed: int = 42,
        degrade: bool = True,
        degrade_mode="full",
        degrade_severity: Optional[float] = None,
        families: Optional[List[str]] = None,
        max_trajectories: Optional[int] = None,
        max_clips_per_video: Optional[int] = None,
        teacher_cache_dir: Optional[str] = None,  # accepted+ignored (shared `common` kwargs)
        teacher_name: str = "depth_pro",
        split_seed: Optional[int] = None,   # E4: seeds ONLY the held-out scene shuffle (None = `seed`: byte-identical); val degradation keeps `seed`
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
        fams = sorted(p.name for p in root_p.iterdir() if p.is_dir() and not p.name.startswith("."))
        if families is not None:
            fams = [f for f in fams if f in families]
        scenes: List[Path] = []
        for fam in fams:
            for sc in sorted((root_p / fam).iterdir()):
                if sc.is_dir() and _index_files(sc, "l_", ".png") and _index_files(sc, "d_", ".exr"):   # protocol v3 WP25 irsfix
                    scenes.append(sc)
        if not scenes:
            raise FileNotFoundError(f"No IRS scenes under {root} with l_*.png + d_*.exr.")

        # deterministic 3-way scene split: [test | val | train]
        rng = random.Random(seed if split_seed is None else split_seed)
        rng.shuffle(scenes)
        n = len(scenes)
        n_test = int(n * test_fraction)
        n_val = 0 if val_fraction <= 0 else max(1, int(n * val_fraction))   # protocol v3 WP11: val_fraction 0 = a TRUE zero hold-out (every scene trains); > 0 keeps the old at-least-one-scene rule
        if split == "val" and n_val == 0:
            raise ValueError("IRS: split='val' with val_fraction 0 (nothing is held out)")
        if split == "test":
            scenes = scenes[:n_test]
        elif split == "val":
            scenes = scenes[n_test:n_test + n_val]
        else:
            scenes = scenes[n_test + n_val:]
        if max_trajectories is not None:
            scenes = scenes[:max_trajectories]

        # protocol v3 WP25 irsfix: pair l_/d_ BY FRAME NUMBER (never by list index) and build clips only inside maximal runs of CONSECUTIVE frame numbers.
        self.clips: List[Tuple[List[Path], List[Path], str, int]] = []
        self.pairing_stats: Dict[str, Dict[str, int]] = {}   # protocol v3 WP25 irsfix
        span = (num_frames - 1) * frame_stride + 1
        for sc in scenes:
            video_id = f"{sc.parent.name}/{sc.name}"
            imap, dmap = _index_files(sc, "l_", ".png"), _index_files(sc, "d_", ".exr")
            paired = sorted(set(imap) & set(dmap))
            runs = _consecutive_runs(paired)
            n_unpaired = len(set(imap) ^ set(dmap))
            n_clips = n_short = 0
            for run in runs:
                if len(run) < span:
                    n_short += len(run)
                    continue
                rid = video_id if len(runs) == 1 else f"{video_id}@{run[0]}"   # one run (a gap-free scene) keeps the old id; split scenes get a per-run id
                imgs, deps = [imap[k] for k in run], [dmap[k] for k in run]
                starts = list(range(0, len(run) - span + 1, clip_stride))
                if max_clips_per_video and len(starts) > max_clips_per_video:
                    step = len(starts) / max_clips_per_video
                    starts = [starts[int(i * step)] for i in range(max_clips_per_video)]
                for start in starts:
                    self.clips.append((imgs, deps, rid, start))
                n_clips += len(starts)
            self.pairing_stats[video_id] = {"paired": len(paired), "dropped_unpaired": n_unpaired, "dropped_short_run": n_short, "runs": len(runs), "clips": n_clips}   # protocol v3 WP25 irsfix
            if n_unpaired or n_short or len(runs) > 1:
                logger.warning("IRS %s: %d paired frames in %d consecutive run(s); dropped %d frames present in only one of l_/d_, %d frames in runs shorter than the clip span %d; %d clips",
                               video_id, len(paired), len(runs), n_unpaired, n_short, span, n_clips)

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
            d = _read_exr_depth(dep_files[i])
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
            "sky": torch.zeros_like(mask),   # indoor: no sky
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

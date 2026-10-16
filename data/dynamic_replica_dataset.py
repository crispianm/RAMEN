"""Dynamic Replica (DR) clip dataset -- synthetic indoor stereo video, LEFT camera RGB + metric depth.  # protocol v3 WP5 dr-loader

Facebook's Dynamic Replica train split (facebookresearch/dynamic_stereo). Layout after unzipping every ``train_NNN.zip`` into one directory (the official ``<root>/train``)::

    root/<seq>_source_left/images/<seq>_source_left-NNNN.png          RGBA 1280x720 (dash before the frame number)
    root/<seq>_source_left/depths/<seq>_source_left_NNNN.geometric.png  uint16 PNG holding a float16 bit pattern = depth in METRES
    root/<seq>_source_left/masks/<seq>_source_left_NNNN.png           0/255 DYNAMIC-OBJECT mask (NOT a validity mask; ignored here)
    (+ the right camera, flow, instance ids, trajectories: unused; ``frame_annotations_train.jgz`` is not read)

Depth decode = the official ``_load_16big_png_depth``: reinterpret the uint16 as float16, cast to float32. The official code applies NO unit scale (annotation ``scale_adjustment`` = 1.0) and
the scene is Replica-scale (a sample frame: median 1.4 m, max 5.5 m), so the value is taken as metres. Depth <= ``_DEPTH_EPS`` (1e-5, the official ``depth_eps``; windows / outside render 0) is invalid.
Pairing is by FRAME NUMBER (never list index) and clips are built only inside maximal runs of CONSECUTIVE frame numbers (the irsfix convention). Dot-files / partial downloads are ignored (strict name match).
Scene-level deterministic val split (``val_fraction``; the v3 SYN config uses 0.0 = every sequence trains).
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .irs_dataset import _consecutive_runs
from .video_clip_dataset import _degrade_clip

logger = logging.getLogger(__name__)
_IMG_RE = re.compile(r"^(?P<s>[^.].*)-(?P<i>\d+)\.png$")
_DEP_RE = re.compile(r"^(?P<s>[^.].*)_(?P<i>\d+)\.geometric\.png$")
_DEPTH_EPS = 1e-5


def _index(d: Path, rx: re.Pattern) -> Dict[int, Path]:
    """frame number -> path for strictly-matching names in ``d`` (dot-files, ``done.ok``, partial downloads ignored)."""
    out: Dict[int, Path] = {}
    if not d.is_dir():
        return out
    for p in d.iterdir():
        m = rx.match(p.name)
        if m:
            out[int(m.group("i"))] = p
    return out


def _read_dr_depth(path) -> np.ndarray:
    """DR ``*.geometric.png`` -> (H, W) float32 metres (uint16 bits reinterpreted as float16; no scale)."""
    with Image.open(path) as im:
        raw = np.array(im, dtype=np.uint16)
    return np.frombuffer(raw.tobytes(), dtype=np.float16).astype(np.float32).reshape(raw.shape)


class DynamicReplicaDataset(Dataset):
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
        max_trajectories: Optional[int] = None,
        max_clips_per_video: Optional[int] = None,
        teacher_cache_dir: Optional[str] = None,  # accepted+ignored (shared `common` kwargs)
        teacher_name: str = "depth_pro",
        split_seed: Optional[int] = None,
        families=None,                            # accepted+ignored (irs-style spec key)
    ) -> None:
        super().__init__()
        self.num_frames, self.image_size, self.split = num_frames, tuple(image_size), split
        self.clip_stride, self.frame_stride, self.max_depth = clip_stride, frame_stride, max_depth
        self.degrade, self.degrade_mode, self.degrade_severity = degrade, degrade_mode, degrade_severity
        self.severity_max = torch.ones(1).share_memory_()
        self.seed = int(seed)

        root_p = Path(root)
        seqs: List[Path] = []
        for sq in sorted(root_p.iterdir()):
            if sq.is_dir() and not sq.name.startswith(".") and sq.name.endswith("_left") and _index(sq / "images", _IMG_RE) and _index(sq / "depths", _DEP_RE):
                seqs.append(sq)
        if not seqs:
            raise FileNotFoundError(f"No Dynamic Replica *_left sequences under {root} with images/ + depths/.")
        rng = random.Random(seed if split_seed is None else split_seed)
        rng.shuffle(seqs)
        n = len(seqs)
        n_test = int(n * test_fraction)
        n_val = 0 if val_fraction <= 0 else max(1, int(n * val_fraction))
        if split == "val" and n_val == 0:
            raise ValueError("DynamicReplica: split='val' with val_fraction 0 (nothing is held out)")
        seqs = seqs[:n_test] if split == "test" else seqs[n_test:n_test + n_val] if split == "val" else seqs[n_test + n_val:]
        if max_trajectories is not None:
            seqs = seqs[:max_trajectories]

        self.clips: List[Tuple[List[Path], List[Path], str, int]] = []
        self.pairing_stats: Dict[str, Dict[str, int]] = {}      # per-sequence stats
        span = (num_frames - 1) * frame_stride + 1
        for sq in seqs:
            imap, dmap = _index(sq / "images", _IMG_RE), _index(sq / "depths", _DEP_RE)
            paired = sorted(set(imap) & set(dmap))
            runs = _consecutive_runs(paired)
            n_unpaired = len(set(imap) ^ set(dmap))
            n_clips = n_short = 0
            for run in runs:
                if len(run) < span:
                    n_short += len(run)
                    continue
                rid = sq.name if len(runs) == 1 else f"{sq.name}@{run[0]}"
                imgs, deps = [imap[k] for k in run], [dmap[k] for k in run]
                starts = list(range(0, len(run) - span + 1, clip_stride))
                if max_clips_per_video and len(starts) > max_clips_per_video:
                    step = len(starts) / max_clips_per_video
                    starts = [starts[int(i * step)] for i in range(max_clips_per_video)]
                for start in starts:
                    self.clips.append((imgs, deps, rid, start))
                n_clips += len(starts)
            self.pairing_stats[sq.name] = {"paired": len(paired), "dropped_unpaired": n_unpaired, "dropped_short_run": n_short, "runs": len(runs), "clips": n_clips}
            if n_unpaired or n_short or len(runs) > 1:
                logger.warning("DR %s: %d paired frames in %d consecutive run(s); dropped %d unpaired, %d in runs shorter than the span %d; %d clips", sq.name, len(paired), len(runs), n_unpaired, n_short, span, n_clips)

    def __len__(self) -> int:
        return len(self.clips)

    def _gt(self, path) -> Tuple[np.ndarray, np.ndarray]:
        d = _read_dr_depth(path)
        valid = np.isfinite(d) & (d > _DEPTH_EPS) & (d < self.max_depth)
        return np.where(valid, d, 0.0).astype(np.float32), valid

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_files, dep_files, video_id, start = self.clips[idx]
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]
        H, W = self.image_size
        frames, depths, masks = [], [], []
        for i in indices:
            img = Image.open(img_files[i]).convert("RGB").resize((W, H), Image.BILINEAR)
            frames.append(torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1))
            d, valid = self._gt(dep_files[i])
            depths.append(torch.from_numpy(d).unsqueeze(0))
            masks.append(torch.from_numpy(valid.astype(np.float32)).unsqueeze(0))
        frames, depth, mask = torch.stack(frames, 0), torch.stack(depths, 1), torch.stack(masks, 1)
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(depth.permute(1, 0, 2, 3), (H, W), mode="nearest").permute(1, 0, 2, 3)
            mask = F.interpolate(mask.permute(1, 0, 2, 3), (H, W), mode="nearest").permute(1, 0, 2, 3)
        result: Dict[str, torch.Tensor] = {"frames": frames.permute(1, 0, 2, 3).contiguous(), "depth": depth, "mask": mask,
                                           "sky": torch.zeros_like(mask), "video_id": video_id, "start_frame": start}
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


def _vda_classes():
    """The VDA-semantics (short-edge 518 resize + crop, post-resize degradation) variants, built lazily so importing this module does not import vda_semantics_dataset."""
    from .vda_semantics_dataset import _VDADegradedMixin, _crop, _vda_frame_and_crop, vda_resize_depth_like

    class VDASemanticsDynamicReplicaDataset(DynamicReplicaDataset):
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            img_files, dep_files, video_id, start = self.clips[idx]
            indices = [start + t * self.frame_stride for t in range(self.num_frames)]
            H, W = self.image_size
            assert H == W, "VDA-semantics mode requires a square target (e.g. 518x518)"
            frames, depths, masks = [], [], []
            crop_xy = rwh = raw_wh = None
            for i in indices:
                raw_img = Image.open(img_files[i]).convert("RGB")
                if raw_wh is None:
                    raw_wh = raw_img.size
                else:
                    assert raw_img.size == raw_wh, f"every frame in a clip must share the raw resolution: {video_id} frame {i} is {raw_img.size}, clip started at {raw_wh}"
                resized, crop_xy_i, rwh_i = _vda_frame_and_crop(raw_img, H, self.split)
                if crop_xy is None:
                    crop_xy, rwh = crop_xy_i, rwh_i
                x0, y0 = crop_xy
                frames.append(torch.from_numpy(_crop(resized, x0, y0, H).copy()).permute(2, 0, 1))
                d, valid = self._gt(dep_files[i])
                d_r = vda_resize_depth_like(d, *rwh)
                v_r = vda_resize_depth_like(valid.astype(np.float32), *rwh)
                depths.append(torch.from_numpy(_crop(d_r, x0, y0, H).copy()).unsqueeze(0).float())
                masks.append(torch.from_numpy(_crop(v_r, x0, y0, H).copy()).unsqueeze(0))
            mask = torch.stack(masks, 1)
            return {"frames": torch.stack(frames, 1).contiguous(), "depth": torch.stack(depths, 1), "mask": mask,
                    "sky": torch.zeros_like(mask), "video_id": video_id, "start_frame": start}

    class VDASemanticsDegradedDynamicReplicaDataset(_VDADegradedMixin, VDASemanticsDynamicReplicaDataset):
        pass

    return VDASemanticsDynamicReplicaDataset, VDASemanticsDegradedDynamicReplicaDataset

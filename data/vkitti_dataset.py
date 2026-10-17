"""
Virtual KITTI 2 clip dataset (standalone)
=========================================

KITTI-domain driving clips (dense GT depth) to complement TartanAir and close the
gap to the real-KITTI / VKITTI-Scene18 evaluation.  Same per-sample contract as
:class:`data.video_clip_dataset.VideoDepthClipDataset` so the two concatenate.

Layout: ``<root>/Scene*/<variant>/frames/{rgb,depth}/Camera_0/{rgb_*.jpg,depth_*.png}``.
Depth: 16-bit PNG, centimetres → metres = value / 100.  Sky / no-return is the uint16 max
(65535 cm ≈ 655.35 m) and is the ONLY thing masked — real geometry past 80 m (far trees,
hillsides) is kept with true depth.  Wide frames (1242×375) are square-cropped to full
height (random train / centre val) then resized to ``image_size`` (squashing hurts DA3).
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .video_clip_dataset import _degrade_clip

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("opencv-python is required to decode VKITTI depth PNGs.") from e


def _decode_vkitti_depth(path: str) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Could not read VKITTI depth: {path}")
    return arr.astype(np.float32) / 100.0  # cm -> m


_SKY_VALID_DEPTH_M = 655.35  # Add.44 arm VK655: the sentinel value (65535 cm) used as a VALID metric target when ``sky_as_valid_depth`` is set
_SKY_DEPTH_M = 655.0  # VKITTI sky / no-return sentinel (65535 cm ≈ 655.35 m); real geometry sits below it


# Ultrareview: VKITTI2 Scene01 / Scene20 are synthetic re-renders of real KITTI-raw drives (the public KITTI tracking OXTS of seq 0001 / 0020 equal these raw drives' OXTS exactly), and those drives are
# in the KITTI DEV clips (2011_10_03_drive_0047) / the Eigen KITTI-C list (both). Training on them puts a synthetic twin of an evaluation drive in-distribution for the fine-tuned models only.
# Default behaviour is unchanged (every shipped V2/F1/FV2 config trains on them); `exclude_eval_overlap=True` (config: the vkitti source's `exclude_eval_overlap`) drops them, and a train split
# that includes them warns. Train and val must be built with the SAME flag (the scene shuffle runs over the remaining sequences).
KITTI_EVAL_OVERLAP_SCENES = {"Scene01": "2011_09_26_drive_0009_sync", "Scene20": "2011_10_03_drive_0047_sync"}


class VKITTIDataset(Dataset):
    """RGB + dense-depth clips from Virtual KITTI 2 (square-crop + resize)."""

    def __init__(
        self,
        root: str,
        num_frames: int = 4,
        image_size: Tuple[int, int] = (518, 518),
        split: str = "train",
        clip_stride: int = 8,
        frame_stride: int = 1,
        max_depth: float = 80.0,
        val_fraction: float = 0.1,
        seed: int = 42,
        scenes: Optional[List[str]] = None,
        variants: Optional[List[str]] = None,
        cameras: Optional[List[str]] = None,
        degrade: bool = True,
        degrade_mode = "full",   # str or list[str] — sampled per clip during training
        degrade_severity: Optional[float] = None,
        exclude_scenes: Optional[List[str]] = None,
        teacher_cache_dir: Optional[str] = None,
        teacher_name: str = "depth_pro",
        max_clips_per_video: Optional[int] = None,
        split_seed: Optional[int] = None,   # E4: seeds ONLY the held-out scene shuffle (None = `seed`: byte-identical); val degradation keeps `seed`
        exclude_eval_overlap: bool = False,   # drop KITTI_EVAL_OVERLAP_SCENES (False = unchanged)
        no_holdout: bool = False,           # K80 option B amendment 3: the TRAIN split is EVERY enumerated sequence (no validation hold-out; the val split is then undefined and refused)
        scene_holdout: bool = False,        # K80 option B (all 10 variants x 2 cameras): hold out WHOLE SCENES, the ones the clone / Camera_0 split would hold out (so every variant and camera of a held-out scene is out of train)
    ) -> None:
        super().__init__()
        # VKITTI uses a per-frame square crop, so a per-frame teacher cache won't match the
        # student view — v1 distills on TartanAir only. Kept for batch-key consistency.
        # teacher_name / max_clips_per_video accepted for shared-kwargs compatibility with the
        # TartanAir loader; VKITTI scenes are short so the cap never binds here.
        self.teacher_cache_dir = teacher_cache_dir
        self.teacher_name = teacher_name
        self.max_clips_per_video = max_clips_per_video
        self.num_frames = num_frames
        self.image_size = tuple(image_size)
        self.split = split
        self.frame_stride = frame_stride
        self.max_depth = max_depth
        self.degrade = degrade
        self.degrade_mode = degrade_mode
        self.degrade_severity = degrade_severity  # fixed severity (eval sweeps); None = sample
        self.severity_max = torch.ones(1).share_memory_()  # train-time curriculum cap
        self.seed = int(seed)
        variants = variants or ["clone"]
        cameras = cameras or ["Camera_0"]

        root_p = Path(root)
        all_scenes = sorted(p.name for p in root_p.iterdir() if p.is_dir() and p.name.startswith("Scene"))
        if scenes is not None:
            all_scenes = [s for s in all_scenes if s in scenes]
        if exclude_scenes:
            all_scenes = [s for s in all_scenes if s not in exclude_scenes]
        if exclude_eval_overlap:
            all_scenes = [s for s in all_scenes if s not in KITTI_EVAL_OVERLAP_SCENES]

        sequences: List[Tuple[List[Path], List[Path], str]] = []
        span = (num_frames - 1) * frame_stride + 1
        for scene in all_scenes:
            for variant in variants:
                for cam in cameras:
                    rgb_dir = root_p / scene / variant / "frames" / "rgb" / cam
                    dep_dir = root_p / scene / variant / "frames" / "depth" / cam
                    if not (rgb_dir.is_dir() and dep_dir.is_dir()):
                        continue
                    rgb = sorted(rgb_dir.glob("rgb_*.jpg"))
                    dep = sorted(dep_dir.glob("depth_*.png"))
                    n = min(len(rgb), len(dep))
                    if n >= span:
                        sequences.append((rgb[:n], dep[:n], f"{scene}/{variant}/{cam}"))
        if not sequences:
            raise FileNotFoundError(f"No VKITTI sequences under {root} (scenes={scenes}).")

        rng = random.Random(seed if split_seed is None else split_seed)
        if no_holdout:
            if split != "train":
                raise ValueError("VKITTIDataset(no_holdout=True) has no validation split: build no VKITTI val source (amendment 3)")
            chosen = list(sequences)
        elif scene_holdout:
            # The clone / Camera_0 sequence list, built and shuffled EXACTLY as the default split does, names the held-out scenes (the same scenes the F1 / V2 recipe holds out for the same seed);
            # every other variant and camera of those scenes goes with them (a held-out sequence is never a relighted twin of a trained one).
            ref = []
            for sc in all_scenes:
                rgb_dir, dep_dir = (root_p / sc / "clone" / "frames" / k / "Camera_0" for k in ("rgb", "depth"))
                if rgb_dir.is_dir() and dep_dir.is_dir() and min(len(list(rgb_dir.glob("rgb_*.jpg"))), len(list(dep_dir.glob("depth_*.png")))) >= span:
                    ref.append(f"{sc}/clone/Camera_0")
            rng.shuffle(ref)
            held = {sid.split("/")[0] for sid in ref[:max(1, int(len(ref) * val_fraction))]}
            chosen = [q for q in sequences if (q[2].split("/")[0] in held) == (split == "val")]
        else:
            rng.shuffle(sequences)
            n_val = max(1, int(len(sequences) * val_fraction))
            chosen = sequences[:n_val] if split == "val" else sequences[n_val:]

        self.eval_overlap_scenes = sorted({sid.split("/")[0] for _, _, sid in chosen} & set(KITTI_EVAL_OVERLAP_SCENES))
        if split == "train" and self.eval_overlap_scenes:
            warnings.warn(f"VKITTIDataset: training split includes {self.eval_overlap_scenes}, synthetic re-renders of KITTI-raw drives "
                          f"{[KITTI_EVAL_OVERLAP_SCENES[s] for s in self.eval_overlap_scenes]} that are in the KITTI evaluation sets (pass exclude_eval_overlap=True to drop them)", stacklevel=2)

        self.clips: List[Tuple[List[Path], List[Path], str, int]] = []
        for rgb, dep, sid in chosen:
            for start in range(0, len(rgb) - span + 1, clip_stride):
                self.clips.append((rgb, dep, sid, start))

    def __len__(self) -> int:
        return len(self.clips)

    def _crop_offset(self, h: int, w: int) -> int:
        max_off = max(0, w - h)
        if max_off == 0:
            return 0
        return random.randint(0, max_off) if self.split == "train" else max_off // 2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_files, dep_files, sid, start = self.clips[idx]
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]
        H, W = self.image_size

        imgs, depths, masks, skies, x_off = [], [], [], [], None
        for i in indices:
            img = Image.open(rgb_files[i]).convert("RGB")
            w, h = img.size
            if x_off is None:
                x_off = self._crop_offset(h, w)
            t = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)  # (3,h,w)
            d_raw = _decode_vkitti_depth(str(dep_files[i]))
            # Mask ONLY sky / no-return (the 65535 sentinel) and ≤0 — keep far trees/hillsides
            # past max_depth with their true depth (the 80 m cap is applied later, at eval).
            valid = np.isfinite(d_raw) & (d_raw > 0) & (d_raw < _SKY_DEPTH_M)
            if getattr(self, "sky_as_valid_depth", False):   # Add.44 VK655 (default off = byte-identical): the 65535 sentinel is a VALID 655.35 m target, sky=0
                valid = np.isfinite(d_raw) & (d_raw > 0)
            # Sky sentinel: target disparity 0 for the VDA-style sky loss; NOT in `mask`.
            sky = d_raw >= _SKY_DEPTH_M
            d = np.clip(d_raw, 0.0, _SKY_DEPTH_M)
            if getattr(self, "sky_as_valid_depth", False):
                sky = np.zeros_like(sky)
                d = np.clip(d_raw, 0.0, _SKY_VALID_DEPTH_M)
            dt = torch.from_numpy(d).unsqueeze(0).float()
            mt = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)
            st = torch.from_numpy(sky.astype(np.float32)).unsqueeze(0)
            imgs.append(t[:, :, x_off:x_off + h])
            depths.append(dt[:, :, x_off:x_off + h])
            masks.append(mt[:, :, x_off:x_off + h])
            skies.append(st[:, :, x_off:x_off + h])

        frames = torch.stack(imgs, 1)   # (3, T, h, h)  centre/rand square crop
        depth = torch.stack(depths, 1)  # (1, T, h, h)
        mask = torch.stack(masks, 1)
        sky = torch.stack(skies, 1)
        if frames.shape[-2:] != (H, W):
            frames = F.interpolate(frames.permute(1, 0, 2, 3), (H, W), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
            depth = F.interpolate(depth.permute(1, 0, 2, 3), (H, W), mode="nearest").permute(1, 0, 2, 3)
            mask = F.interpolate(mask.permute(1, 0, 2, 3), (H, W), mode="nearest").permute(1, 0, 2, 3)
            sky = F.interpolate(sky.permute(1, 0, 2, 3), (H, W), mode="nearest").permute(1, 0, 2, 3)

        result: Dict[str, torch.Tensor] = {
            "frames": frames.contiguous(), "depth": depth, "mask": mask, "sky": sky,
            "video_id": f"vkitti/{sid}", "start_frame": start,
        }
        if self.teacher_cache_dir is not None:  # no VKITTI teacher in v1 (crop mismatch); keep keys
            result["teacher_depth"] = torch.zeros(1, self.num_frames, H, W)
            result["teacher_valid"] = torch.zeros(self.num_frames)
        if self.degrade:
            rng = np.random.RandomState(self.seed + idx) if self.split != "train" else np.random.RandomState()
            sev = self.degrade_severity
            mode = rng.choice(self.degrade_mode) if isinstance(self.degrade_mode, list) \
                else self.degrade_mode
            if sev is None and self.split == "train" and mode != "elvis":
                sev = float(rng.uniform(0.0, float(self.severity_max[0])))  # curriculum cap
            degraded, params = _degrade_clip(frames.permute(1, 0, 2, 3), rng, mode, severity=sev)
            result["degraded_frames"] = degraded.permute(1, 0, 2, 3).contiguous()
            result["degrade_params"] = torch.from_numpy(params)
        return result

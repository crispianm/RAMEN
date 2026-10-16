"""KITTI raw TRAIN source (protocol v3 WP13): CAM_02 RGB at 10 Hz + the sparse LiDAR GT of ``data_depth_annotated`` (the depth-completion benchmark's own annotated depth: 16-bit PNG, metres * 256, 0 = no return).

Drive list = ALL 138 depth-annotated train drives (``provenance/dn01_v3/kitti_raw_train_drives.json``), INCLUDING the 21 that also sit in data/TEST/kitti_raw_val_drives (Manager decision, runs/WP6/RUNLOG.md; the
older KITTI-C / Eigen DEV reservation is not a K96 headline set). VDA's KITTI test = the 13 depth-completion val drives: pinned below and ASSERTED disjoint from the train list, whatever the manifest says.

Layout::  <root>/<date>/<drive>/image_02/data/%010d.png   and   <gt_root>/train/<drive>/proj_depth/groundtruth/image_02/%010d.png   (GT exists for the frames 5 .. N-6 of a drive; clips never cross a gap).
Depth is SPARSE (~5% of pixels): set ``sparse_gt: true`` on the source so train.py uses the sparse-safe masked SSI for it (``data/v3_clip.py``). The 80 m cap is ``max_depth`` (per-source override allowed).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .v3_clip import V3ClipDataset

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "provenance" / "dn01_v3" / "kitti_raw_train_drives.json"
VDA_VAL_DRIVES = (   # the 13 val drives of data_depth_annotated = VDA's KITTI test set (listing of data_depth_annotated.zip val/)
    "2011_09_26_drive_0002_sync", "2011_09_26_drive_0005_sync", "2011_09_26_drive_0013_sync", "2011_09_26_drive_0020_sync", "2011_09_26_drive_0023_sync", "2011_09_26_drive_0036_sync",
    "2011_09_26_drive_0079_sync", "2011_09_26_drive_0095_sync", "2011_09_26_drive_0113_sync", "2011_09_28_drive_0037_sync", "2011_09_29_drive_0026_sync", "2011_09_30_drive_0016_sync",
    "2011_10_03_drive_0047_sync")
N_TRAIN_DRIVES = 138


def read_kitti_gt(path) -> np.ndarray:
    """16-bit depth PNG -> metres (0 = invalid)."""
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None or a.dtype != np.uint16:
        raise FileNotFoundError(f"KITTI depth PNG unreadable or not 16-bit: {path}")
    return a.astype(np.float32) / 256.0


def load_drives(manifest: Path | str = DEFAULT_MANIFEST) -> List[str]:
    m = json.loads(Path(manifest).read_text())
    train, val = list(m["train_drives"]), list(m["vda_val_drives"])
    assert tuple(val) == VDA_VAL_DRIVES, "manifest vda_val_drives differs from the pinned 13 VDA val drives"
    assert len(train) == N_TRAIN_DRIVES == len(set(train)), f"expected {N_TRAIN_DRIVES} distinct train drives, got {len(train)}"
    assert not set(train) & set(VDA_VAL_DRIVES), f"train drives overlap VDA's val drives: {sorted(set(train) & set(VDA_VAL_DRIVES))}"
    return sorted(train)


def _runs(ids: List[int]) -> List[List[int]]:
    runs: List[List[int]] = []
    for i in ids:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


class KITTIRawTrainDataset(V3ClipDataset):
    def __init__(self, root: str, gt_root: str, manifest: str | None = None, val_fraction: float = 0.0, split_seed: int | None = None,
                 allow_missing: bool = False, drives: List[str] | None = None, **kw) -> None:
        super().__init__(**kw)
        root_p, gt_p = Path(root), Path(gt_root)
        ds = load_drives(manifest or DEFAULT_MANIFEST) if drives is None else sorted(drives)
        assert not set(ds) & set(VDA_VAL_DRIVES), "a VDA val drive was requested for training"
        if val_fraction > 0:                                             # drive-level hold-out (default 0: train on all 138)
            rng = random.Random(self.seed if split_seed is None else split_seed)
            shuf = sorted(ds)
            rng.shuffle(shuf)
            n_val = max(1, int(len(shuf) * val_fraction))
            ds = sorted(shuf[:n_val] if self.split == "val" else shuf[n_val:])
        elif self.split == "val":
            raise ValueError("kitti_raw: split='val' needs val_fraction > 0 (the default trains on all drives)")
        self.drives, self.missing = [], []
        for d in ds:
            rgb_dir = root_p / d[:10] / d / "image_02" / "data"
            gt_dir = gt_p / "train" / d / "proj_depth" / "groundtruth" / "image_02"
            if not rgb_dir.is_dir() or not gt_dir.is_dir():
                self.missing.append(d)
                continue
            gt_ids = sorted(int(p.stem) for p in gt_dir.glob("*.png"))
            rgb_ids = {int(p.stem) for p in rgb_dir.glob("*.png")}
            ids = [i for i in gt_ids if i in rgb_ids]
            for k, run in enumerate(_runs(ids)):
                self.add_video(f"{d}#{k}" if k else d, [rgb_dir / f"{i:010d}.png" for i in run], [gt_dir / f"{i:010d}.png" for i in run])
            self.drives.append(d)
        if self.missing and not allow_missing:
            raise FileNotFoundError(f"kitti_raw: {len(self.missing)} drive(s) without RGB or GT under {root} / {gt_root}: {self.missing[:5]} ... (allow_missing: true for a partial tree)")
        if not self.clips:
            raise FileNotFoundError(f"kitti_raw: no clips under {root} / {gt_root}")

    def read_depth(self, path) -> np.ndarray:
        return read_kitti_gt(path)

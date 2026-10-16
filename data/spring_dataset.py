"""Spring (Mehl et al., CVPR 2023) TRAIN source (protocol v3 WP12): rendered 1920x1080 stereo video with dense disparity GT; we use the left camera. Disparity -> metric depth.

Layout (zips of the official release extracted under ``<root>/extracted/`` or, on Isambard, directly under ``<root>/``)::  spring/train/<seq>/frame_left/frame_left_%04d.png, spring/train/<seq>/disp1_left/disp1_left_%04d.dsp5, spring/train/<seq>/cam_data/intrinsics.txt
Frames are 1-based. ``.dsp5`` is HDF5 with one float16 dataset ``disparity`` at DOUBLE resolution (2160x3840); like the official toolkit we take ``disp[::2, ::2]`` to the frame grid, where the values are in pixels of the
1920-wide frame. Depth Z = fx * baseline / disparity with the PER-FRAME ``fx`` of ``cam_data/intrinsics.txt`` (fx fy cx cy per line; fx varies over a sequence) and the stereo baseline ``BASELINE_M`` = 0.065 m (Spring
paper). Invalid / non-finite / <= 0 disparity -> 0. 37 train sequences (test GT is hidden), ~24 fps; clips are consecutive frames. Dense GT (``sparse_gt: false``); per-source ``max_depth`` (default 80 m) masks the far field.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from .v3_clip import V3ClipDataset

BASELINE_M = 0.065


def resolve_spring_train(root: str) -> Path:
    """Spring train dir. Two layouts are accepted, tried in this order (the one used is logged): ``<root>/extracted/spring/train`` (the local layout) then ``<root>/spring/train`` (Isambard: the zips extracted in place)."""
    cands = [Path(root) / "extracted" / "spring" / "train", Path(root) / "spring" / "train"]
    for c in cands:
        if c.is_dir():
            print(f"[spring] train dir = {c}", flush=True)
            return c
    raise FileNotFoundError(f"spring: train dir not found, tried {[str(c) for c in cands]} (extract train_frame_left / train_disp1_left / train_cam_data under {root}/extracted or {root}/spring)")


class SpringDataset(V3ClipDataset):
    def __init__(self, root: str, sequences: List[str] | None = None, val_fraction: float = 0.0, split_seed: int | None = None, **kw) -> None:
        super().__init__(**kw)
        base = resolve_spring_train(root)
        seqs = sorted(p.name for p in base.iterdir() if p.is_dir())
        if sequences is not None:
            seqs = [s for s in seqs if s in set(sequences)]
        if val_fraction > 0:
            import random
            rng = random.Random(self.seed if split_seed is None else split_seed)
            shuf = list(seqs)
            rng.shuffle(shuf)
            n_val = max(1, int(len(shuf) * val_fraction))
            seqs = sorted(shuf[:n_val] if self.split == "val" else shuf[n_val:])
        elif self.split == "val":
            raise ValueError("spring: split='val' needs val_fraction > 0 (the default trains on all sequences)")
        self.sequences, self._fx = [], {}
        for s in seqs:
            fr = sorted((base / s / "frame_left").glob("frame_left_*.png"))
            dp = sorted((base / s / "disp1_left").glob("disp1_left_*.dsp5"))
            intr = base / s / "cam_data" / "intrinsics.txt"
            if not fr or len(fr) != len(dp) or not intr.is_file():
                raise FileNotFoundError(f"spring: sequence {s} incomplete ({len(fr)} frames, {len(dp)} disparities, intrinsics {intr.is_file()})")
            self._fx[s] = np.loadtxt(intr, ndmin=2)[:, 0]
            assert len(self._fx[s]) == len(fr), f"spring {s}: {len(self._fx[s])} intrinsics lines for {len(fr)} frames"
            self.add_video(s, fr, dp)
            self.sequences.append(s)
        if not self.clips:
            raise FileNotFoundError(f"spring: no sequence under {base} has >= {self.num_frames} frames")

    def read_depth(self, path) -> np.ndarray:
        import h5py
        m = re.search(r"/train/(\d+)/disp1_left/disp1_left_(\d+)\.dsp5$", str(path))
        seq, frame = m.group(1), int(m.group(2))
        with h5py.File(path, "r") as h:
            disp = np.asarray(h["disparity"])[::2, ::2].astype(np.float32)
        ok = np.isfinite(disp) & (disp > 1e-3)
        return np.where(ok, self._fx[seq][frame - 1] * BASELINE_M / np.where(ok, disp, 1.0), 0.0).astype(np.float32)

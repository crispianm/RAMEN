"""
Offline Depth Pro teacher cache reader (FocusMamba clip-format).

The frozen Depth Pro teacher depth is precomputed per clean clip and cached as
``<cache_dir>/<video_id>/<start>/<teacher>.npy`` = float16 ``(1, clip_t, h, w)``.
Training loads it per clip frame and distils the degraded-input student toward the
clean-input teacher.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_teacher_clip_cache(cache_dir: str | Path, video_id: str, indices, H: int, W: int,
                            teacher: str = "depth_pro", clip_t: int = 8):
    """Read the **clip-format** teacher cache (FocusMamba layout):
    ``<cache_dir>/<video_id>/<start>/<teacher>.npy`` = ``(1, clip_t, h, w)`` float16,
    non-overlapping T=``clip_t`` tiling (every frame covered once).

    Returns ``(teacher_depth (1,T,H,W), valid (T,))`` for the requested absolute frame
    ``indices``; frames whose clip is missing get 0 + valid=0 (masked in the distill loss).
    Each needed clip npy is loaded at most once.
    """
    root = Path(cache_dir)
    loaded: dict = {}
    out, valid = [], []
    for i in indices:
        start = (int(i) // clip_t) * clip_t
        if start not in loaded:
            f = root / video_id / str(start) / f"{teacher}.npy"
            loaded[start] = np.load(f).astype(np.float32)[0] if f.exists() else None  # (clip_t,h,w)
        arr, pos = loaded[start], int(i) % clip_t
        if arr is None or pos >= arr.shape[0]:
            out.append(torch.zeros(1, H, W)); valid.append(0.0)
            continue
        d = torch.from_numpy(arr[pos])[None, None]
        if d.shape[-2:] != (H, W):
            d = F.interpolate(d, (H, W), mode="bilinear", align_corners=False)
        out.append(d[0]); valid.append(1.0)
    return torch.stack(out, 1), torch.tensor(valid)

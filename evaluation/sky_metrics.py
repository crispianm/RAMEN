"""DN-01 sky-fix guard metrics (audit/SKY_screen_predeclared_20261001.md section 2; ADDITIVE scorer keys, DESCRIPTIVE or decisive only as that file says).
`sky_mask_kitti` is the investigator's rule VERBATIM (session_briefs/sky_hunt_20261001/skyprobe.py, sha256 be0fe64f871ef952...; tests/test_dn01_sky.py compares the function source to that file when it is present): on the CLEAN frame at GT size,
bright (luma > 0.55), smooth (7x7 local std of luma < 0.035), no GT, in the top 55 % of rows, in a connected component touching the top 3 rows, eroded 5x5. It was fixed before any depth was seen.
`sky_stats` (the same `stats` fields): over the sky pixels of a clip's UNCLAMPED metric depth at GT size: sky_frac, sky_med (median depth, m), sky_ge80 (share >= 80 m), sky_nearer_than_scene_p95 (mean over frames of the share of sky pixels nearer than the
95th percentile of the model's own GT-valid scene depth in that frame; valid = 0.1 < gt < 80 and not sky), far40_80_ratio_med (median pred/GT on 40 < gt < 80, not sky)."""
from __future__ import annotations

import cv2
import numpy as np

SKYPROBE_SHA256_PREFIX = "be0fe64f871ef952"


def sky_mask_kitti(clean_raw: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """clean_raw (H,W,3) [0,1] at GT size; gt (H,W) (-1 invalid). Rule (fixed before looking at any depth):
    bright (luma > 0.55), smooth (7x7 local std of luma < 0.035), no GT, in the top 55% of rows, and in a
    connected component that touches the top 3 rows."""
    H, W = gt.shape
    L = clean_raw @ np.array([0.299, 0.587, 0.114], np.float32)
    mu = cv2.blur(L, (7, 7)); sd = np.sqrt(np.clip(cv2.blur(L * L, (7, 7)) - mu * mu, 0, None))
    cand = (L > 0.55) & (sd < 0.035) & (gt <= 0)
    cand[int(0.55 * H):] = False
    n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=4)
    top = set(np.unique(lab[:3])) - {0}
    m = np.isin(lab, list(top))
    return cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def sky_stats(depth: np.ndarray, sky: np.ndarray, gt: np.ndarray) -> dict:
    """depth (N,H,W) metres, UNCLAMPED, at GT size; sky (N,H,W) bool; gt (N,H,W) metres (<= 0 invalid). NaN fields when the clip has no sky pixel."""
    sky = np.broadcast_to(sky, depth.shape)
    ds = depth[sky]
    out = {"sky_frac": float(sky.mean()), "sky_med": float(np.median(ds)) if ds.size else float("nan"), "sky_ge80": float((ds >= 80).mean()) if ds.size else float("nan"),
           "sky_nearer_than_scene_p95": float("nan"), "far40_80_ratio_med": float("nan")}
    fv = (gt > 40) & (gt < 80) & ~sky                                   # far GT-valid geometry: median pred / GT (the hinge's spill onto far geometry; PRINTED, not decisive)
    if fv.any():
        out["far40_80_ratio_med"] = float(np.median(depth[fv] / gt[fv]))
    if ds.size:
        valid = (gt > 0.1) & (gt < 80) & ~sky
        v = []
        for t in range(depth.shape[0]):
            s = depth[t][sky[t]]
            if s.size and valid[t].any():
                v.append((s < np.percentile(depth[t][valid[t]], 95)).mean())
        if v:
            out["sky_nearer_than_scene_p95"] = float(np.mean(v))
    return out


def kitti_swap_testonly() -> bool:
    """DN-01 KITTI B/R fix (audit/SWAP_impact_predeclared_20261001.md, G-FIX-3): ``BENCH_KITTI_SWAP_TESTONLY=1`` re-introduces the OLD channel swap on the (fixed) KITTI frames, to reproduce the old swapped
    rows and for the calibration's swapped arm. Never set by any result stage except the swap-repro / calibration jobs, and recorded in every result json it touches."""
    import os
    return os.environ.get("BENCH_KITTI_SWAP_TESTONLY") == "1"


def kitti_sky_masks(root, frames: list, crop: tuple, gt: np.ndarray) -> np.ndarray:
    """(N,H,W) bool sky masks of a KITTI clip from the RAW clean frames (cropped to the GT crop, resized to GT size, as the probe), one per frame."""
    from pathlib import Path
    a, b, c, d = crop
    out = []
    for f, g in zip(frames, gt):
        im = cv2.cvtColor(cv2.imread(str(Path(root) / f["image"])), cv2.COLOR_BGR2RGB)
        if kitti_swap_testonly():
            im = np.ascontiguousarray(im[..., ::-1])                  # the same TEST-ONLY re-introduction of the old B/R swap as bench_joint.load_frames (the sky mask reads the SAME frames as the model)
        im = cv2.resize(im[a:b, c:d].astype(np.float32) / 255.0, (gt.shape[2], gt.shape[1]))
        out.append(sky_mask_kitti(im, g))
    return np.stack(out)

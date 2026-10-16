"""
Temporal-stability metrics for D3 screening (pose-free, cv2/numpy only)
======================================================================

Two complementary temporal metrics, computed post-hoc on a clip's predicted
depth (scale-shift aligned) — no camera poses required, so they run on any
TartanAir/VKITTI clip the trial harness produces:

  * ``tof``  — temporal Optical-Flow consistency (FMA-Net, Youk et al. CVPR 2024;
    ``utils.get_tOF``/``crop_8x8`` ported verbatim). Compares the dense optical
    flow of the GT depth sequence with that of the predicted sequence; low = the
    prediction's motion matches the ground truth's.
  * ``opw``  — Optical-flow Warping self-consistency (the OPW criterion from
    Video-Depth-Anything, Chen et al. CVPR 2025). Estimates flow t→t+1 on the
    clean RGB, warps predicted depth[t] forward, and measures the relative error
    against depth[t+1] on co-visible valid pixels; low = temporally stable depth.
    Pose-free proxy for VDA's pose-based TAE (which D3's loader has no poses for).

Both are flicker/consistency scores in [0, ∞); lower is better. Pure cv2/numpy.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

_FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def _np(x) -> np.ndarray:
    """Detach a torch tensor (or pass an array) to a float32 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _to_grey_u8(frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalise a 2D depth/disparity map to a uint8 grey image with a shared range."""
    g = (frame - lo) / max(hi - lo, 1e-6)
    return np.clip(g * 255.0, 0, 255).astype(np.uint8)


def crop_8x8(img: np.ndarray):
    """FMA-Net ``crop_8x8`` (verbatim): centre-crop to a 32-aligned size."""
    ori_h, ori_w = img.shape[0], img.shape[1]
    h = (ori_h // 32) * 32
    w = (ori_w // 32) * 32
    while h > ori_h - 16:
        h -= 32
    while w > ori_w - 16:
        w -= 32
    y = (ori_h - h) // 2
    x = (ori_w - w) // 2
    return img[y : y + h, x : x + w], y, x


def get_tOF(pre_gt_grey: np.ndarray, gt_grey: np.ndarray, pre_out_grey: np.ndarray, out_grey: np.ndarray) -> float:
    """FMA-Net ``get_tOF`` (verbatim): mean L2 difference of GT vs output optical flow."""
    target_OF = cv2.calcOpticalFlowFarneback(pre_gt_grey, gt_grey, None, **_FARNEBACK)
    output_OF = cv2.calcOpticalFlowFarneback(pre_out_grey, out_grey, None, **_FARNEBACK)
    target_OF, _, _ = crop_8x8(target_OF)
    output_OF, _, _ = crop_8x8(output_OF)
    OF_diff = np.absolute(target_OF - output_OF)
    OF_diff = np.sqrt(np.sum(OF_diff * OF_diff, axis=-1))  # per-pixel L2 of the flow-vector difference
    return float(OF_diff.mean())


def tof_clip(gt_depth, pred_depth) -> float:
    """Mean tOF over a clip. ``gt_depth``/``pred_depth``: ``(T, H, W)`` depth maps."""
    gt = _np(gt_depth)
    pr = _np(pred_depth)
    if gt.ndim != 3 or gt.shape[0] < 2:
        return float("nan")
    g_lo, g_hi = float(np.nanmin(gt)), float(np.nanmax(gt))
    p_lo, p_hi = float(np.nanmin(pr)), float(np.nanmax(pr))
    g = [_to_grey_u8(gt[t], g_lo, g_hi) for t in range(gt.shape[0])]
    p = [_to_grey_u8(pr[t], p_lo, p_hi) for t in range(pr.shape[0])]
    vals = [get_tOF(g[t - 1], g[t], p[t - 1], p[t]) for t in range(1, len(g))]
    return float(np.mean(vals)) if vals else float("nan")


def opw_clip(rgb, pred_depth, mask=None) -> float:
    """Mean OPW warping error over a clip.

    ``rgb``: ``(T, 3, H, W)`` clean frames in [0, 1] (flow source); ``pred_depth``:
    ``(T, H, W)`` aligned depth; ``mask``: optional ``(T, H, W)`` validity.
    """
    rgb = _np(rgb)
    pr = _np(pred_depth)
    T, _, H, W = rgb.shape
    if T < 2:
        return float("nan")
    m = _np(mask) if mask is not None else np.ones((T, H, W), np.float32)
    grey = [cv2.cvtColor((np.clip(rgb[t].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for t in range(T)]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    errs = []
    for t in range(1, T):
        # Backward flow t -> t-1: defined on frame-t pixels, so (gx+flow, gy+flow) is the
        # correct backward sampling map that pulls depth[t-1] onto frame t's grid. Using the
        # forward flow t-1 -> t here (as before) mis-warps and inflates the error ~7x.
        flow = cv2.calcOpticalFlowFarneback(grey[t], grey[t - 1], None, **_FARNEBACK)
        map_x = gx + flow[..., 0]
        map_y = gy + flow[..., 1]
        warped = cv2.remap(pr[t - 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        warped_m = cv2.remap(m[t - 1], map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cov = (m[t] > 0.5) & (warped_m > 0.5) & (pr[t] > 1e-3)
        if cov.sum() == 0:
            continue
        rel = np.abs(warped[cov] - pr[t][cov]) / np.clip(pr[t][cov], 1e-3, None)
        errs.append(float(rel.mean()))
    return float(np.mean(errs)) if errs else float("nan")


def clip_temporal_metrics(gt_depth, pred_depth, rgb, mask=None) -> Dict[str, float]:
    """Convenience: both temporal metrics for one clip (``T`` ≥ 2)."""
    return {"tof": tof_clip(gt_depth, pred_depth), "opw": opw_clip(rgb, pred_depth, mask)}

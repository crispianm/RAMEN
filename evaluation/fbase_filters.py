"""Protocol v3 WP29 / FBASE (D7): causal post-hoc temporal depth filters applied to MVDA-S, as bench_joint depth post tokens (``metric_vda_s+kal0.05``).

All three act on the stitched depth (N,H,W) at GT size before scoring, per pixel, and are CAUSAL: output[t] depends only on depth[:t+1] (and, for flowema, on the input frames t-1, t).
No parameter is tuned on TEST; the defaults below are the standard published ones (or the documented adaptation) and are recorded in every result by the spec string itself.

  kal<q>      per-pixel scalar Kalman filter, random-walk state x_t = x_{t-1} + w (var q*r), measurement z_t = x_t + v (var r); only the ratio q/r matters, token value = q/r
              (default token ``kal0.05``: steady-state gain 0.2). x_0 = z_0, P_0 = r.
  oneeuro[<mincutoff>b<beta>]   the 1-euro filter (Casiez, Roussel, Vogel, CHI 2012), per pixel, 30 fps: low-pass with cutoff = mincutoff + beta * |speed|, speed = low-passed derivative (dcutoff 1 Hz).
              ADAPTATION: depth units differ (metres vs disparity), so the speed is RELATIVE (derivative / |previous filtered value|, 1/s). Defaults mincutoff 1.0 Hz, beta 1.0
              (a 10 %/frame change = 3/s adds 3 Hz of cutoff); bare ``oneeuro`` = ``oneeuro1.0b1.0``.
  flowema<a>  flow-aligned causal EMA: y_t = a*d_t + (1-a)*warp(y_{t-1}), warp = backward warp of the previous OUTPUT by the RAFT flow t -> t-1 of the guide (input) frames, the same flow
              network the TOF metric uses; border pixels fall back to the current depth. a=1 is the identity. No occlusion handling (a plain baseline).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

KAL_DEFAULT_QR = 0.05
EURO_DEFAULT = (1.0, 1.0)        # mincutoff (Hz), beta (relative-speed units)
EURO_FPS, EURO_DCUTOFF = 30.0, 1.0


def depth_kalman(depth: np.ndarray, qr: float = KAL_DEFAULT_QR) -> np.ndarray:
    """Per-pixel random-walk Kalman filter with process/measurement variance ratio ``qr`` (measurement variance 1)."""
    if qr <= 0:
        raise ValueError(f"kalman q/r must be > 0: {qr}")
    d = depth.astype(np.float64)
    out = np.empty_like(d)
    x, p = d[0].copy(), 1.0
    out[0] = x
    for t in range(1, d.shape[0]):
        pp = p + qr
        k = pp / (pp + 1.0)
        x = x + k * (d[t] - x)
        p = (1.0 - k) * pp
        out[t] = x
    return out.astype(depth.dtype)


def _lp_alpha(cutoff, fps: float):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau * fps)


def depth_oneeuro(depth: np.ndarray, mincutoff: float = EURO_DEFAULT[0], beta: float = EURO_DEFAULT[1], fps: float = EURO_FPS, dcutoff: float = EURO_DCUTOFF) -> np.ndarray:
    """Per-pixel 1-euro filter on RELATIVE speed (see the module docstring)."""
    d = depth.astype(np.float64)
    out = np.empty_like(d)
    xh, dxh, raw_prev = d[0].copy(), np.zeros_like(d[0]), d[0].copy()
    out[0] = xh
    ad = _lp_alpha(dcutoff, fps)
    for t in range(1, d.shape[0]):
        dx = (d[t] - raw_prev) * fps / (np.abs(xh) + 1e-6)
        dxh = dxh + ad * (dx - dxh)
        a = _lp_alpha(mincutoff + beta * np.abs(dxh), fps)
        xh = xh + a * (d[t] - xh)
        raw_prev = d[t]
        out[t] = xh
    return out.astype(depth.dtype)


_RAFT = {}


def raft_back_flows(frames: torch.Tensor) -> torch.Tensor:
    """(N,3,h,w) in [0,1] -> (N-1,2,h,w): element j = flow of the pair (frame j+1 -> frame j) (backward flow at t=j+1), by the TOF metric's RAFT."""
    from evaluation.temporal_metrics import TOF
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev not in _RAFT:
        _RAFT[dev] = TOF(dev)
    with torch.no_grad():
        return _RAFT[dev].flows(frames.flip(0).float().to(dev)).flip(0).cpu()


FLOW_FN = raft_back_flows          # replaceable (tests)


def warp_backward(prev: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """prev (1,1,H,W), flow (1,2,H,W) in pixels (x, y) at the same size: out(p) = prev(p + flow(p)) (bilinear); also the in-bounds mask (1,1,H,W)."""
    _, _, h, w = prev.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    gx, gy = xs + flow[0, 0], ys + flow[0, 1]
    ok = ((gx >= 0) & (gx <= w - 1) & (gy >= 0) & (gy <= h - 1))[None, None].to(prev.dtype)
    grid = torch.stack([2 * gx / max(w - 1, 1) - 1, 2 * gy / max(h - 1, 1) - 1], -1)[None]
    return F.grid_sample(prev, grid.to(prev.dtype), mode="bilinear", padding_mode="border", align_corners=True), ok


def depth_flowema(depth: np.ndarray, frames: torch.Tensor | None, alpha: float) -> np.ndarray:
    """Flow-aligned causal EMA (module docstring). ``frames``: the guide (input) frames (N,3,h,w); required unless alpha == 1."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"flowema alpha must be in (0, 1]: {alpha}")
    if alpha == 1.0:
        return depth.copy()
    if frames is None:
        raise ValueError("flowema needs the input frames as guides")
    n, h, w = depth.shape
    flows = FLOW_FN(frames.detach().float().cpu())                       # (N-1,2,fh,fw)
    fh, fw = flows.shape[-2:]
    if (fh, fw) != (h, w):
        flows = F.interpolate(flows, size=(h, w), mode="bilinear", align_corners=False) * torch.tensor([w / fw, h / fh]).view(1, 2, 1, 1)
    d = torch.from_numpy(depth.astype(np.float32))
    out = torch.empty_like(d)
    out[0] = d[0]
    for t in range(1, n):
        prev, ok = warp_backward(out[t - 1][None, None], flows[t - 1][None])
        y = alpha * d[t][None, None] + (1 - alpha) * prev
        out[t] = (ok * y + (1 - ok) * d[t][None, None])[0, 0]
    return out.numpy().astype(depth.dtype)

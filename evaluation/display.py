"""DN-01 display path (K44; user-adopted after the ELVIS research brief): restored frames at NATIVE exposure -> a blind,
CAUSAL per-frame global gain -> a fixed global monotone tone curve. The same function for EVERY pipeline (its inputs are
the pipeline's restored frames and the shared degraded input); it is never trained and never sees GT. Its parameters
come ONLY from the frozen fit file (scripts/dn01_display_fit.py -> provenance/dn01_display/display_fit.json), fitted
once on the BVI-RLV FIT scenes. A SECONDARY (display) metric path; the native-exposure metrics stay primary.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

LUMA = (0.299, 0.587, 0.114)
Q, DOWN, GAIN_MAX, EMA_ALPHA = 0.995, 4, 16.0, 0.2    # EMA_ALPHA declared a priori (K44(c)), not fitted
FIT = Path(os.environ.get("DN01_DISPLAY_FIT",   # the unattended chain writes the frozen fit outside its clean clone
                          Path(__file__).resolve().parents[1] / "provenance/dn01_display/display_fit.json"))


def load_fit(path: Path = FIT) -> dict:
    return json.loads(Path(path).read_text())


def frame_stat(x: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) -> (N,): each frame's 99.5th percentile of its 4x average-pooled luma."""
    lum = (x.float() * torch.tensor(LUMA, device=x.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
    return torch.quantile(F.avg_pool2d(lum, DOWN).flatten(1), Q, dim=1)


def causal_gain(x: torch.Tensor, gain_ref: float, alpha: float = EMA_ALPHA) -> torch.Tensor:
    """(N,) per-frame gains: log g_t = clip(log(gain_ref / stat_t), 0, log GAIN_MAX), smoothed by a CAUSAL EMA
    s_t = alpha log g_t + (1 - alpha) s_{t-1}, s_0 = log g_0 (frame t uses frames <= t only)."""
    lg = torch.log(gain_ref / frame_stat(x).clamp_min(1e-4)).clamp(0.0, math.log(GAIN_MAX))
    s, out = lg[0], []
    for t in range(lg.shape[0]):
        s = lg[t] if t == 0 else alpha * lg[t] + (1.0 - alpha) * s
        out.append(s)
    return torch.stack(out).exp()


def apply_lut(z: torch.Tensor, lut: list[float]) -> torch.Tensor:
    """A 256-entry global tone curve on [0,1], linear interpolation between entries (all channels alike)."""
    t = torch.tensor(lut, device=z.device, dtype=z.dtype)
    p = z.clamp(0.0, 1.0) * 255.0
    i = p.floor().clamp(max=254).long()
    return t[i] + (p - i) * (t[i + 1] - t[i])


def display(restored: torch.Tensor, degraded: torch.Tensor, fit: dict, curve: bool = True) -> torch.Tensor:
    """The display output of a pipeline: clip(g_t x restored_t) then (curve) the tone LUT. Images (N,3,H,W)."""
    z = (restored * causal_gain(degraded, fit["gain_ref"], fit["ema_alpha"]).view(-1, 1, 1, 1)).clamp(0.0, 1.0)
    return apply_lut(z, fit["lut"]) if curve else z

"""ELVIS-DERIVED degradation Deg(·), fork-safe numpy/cv2 port (not a faithful replica).

Port of `VDP-Net/src/dataset/noise.py:generate_noise` from ELVIS (Lin et al., CVPR 2026,
arXiv:2512.01495). Applies, per frame, a 9-param profile φ (normalized [0,1], same order as
VDP-Net): exposure (2^EV in XYZ) → anisotropic Gaussian blur → Poisson shot → Gaussian read →
uniform quantization → directional banding → clip.

Known differences from upstream (the ELVIS research brief, 2026-09-26; the paper says "ELVIS-derived"):
  1. blur border: cv2 BORDER_REFLECT here, zero padding (F.conv2d padding=10) upstream;
  2. no final 8-bit re-quantisation: upstream's GenVIS mapper floors the degraded clip to 8 bits
     ((noisy*255).int()); here the output stays float (the only 8-bit step is the truncation
     before the Poisson draw, as upstream);
  3. profile sampling (sample_profile): one 5-frame WINDOW uniformly from the bank (source mix
     DID 28%, LMOT 16%, BVI-RLV 32%, SDSD 25%), stretched over T by nearest index; upstream picks
     a video key uniformly (DID ~66%, LMOT ~3%) and concatenates that video's windows.
The DN-01 `lowlight` mode (lowlight_profile / degrade_clip_lowlight) departs further, by design:
its own EV draw (independent of φ), isotropic cv2 blur, every noise amplitude x kappa (shot gain
K*kappa^2), and one profile per clip averaged over its 5 frames: "ELVIS noise terms", not
"the ELVIS degradation". Its input is truncated to 8 bits after exposure + blur and before the
noise, then stays float (no final re-quantisation); lowlight_reference (the exposed target)
is on the 8-bit grid.

Numpy/cv2 (no torch): `_degrade_clip` runs inside forked DataLoader workers where torch
convs deadlock — same constraint the rest of the degradation path already obeys.

φ index → param (ELVIS noise.py `actual_labels`):
  0 exposure[0,-3.5]  1 shot_log[ln.1,ln30]  2 read[0,.1]  3 quant[0,.1]  4 band[0,.03]
  5 band_angle(round→0/1)  6 blur_σ1[.5,10]  7 blur_σ2[.5,10]  8 blur_angle[0,.25]·π
"""
from __future__ import annotations

import cv2
import numpy as np

BLUR_KSIZE = 21
_LN01, _LN30 = np.log(0.1), np.log(30.0)

# kornia sRGB<->XYZ matrices (linear, no gamma — matches kornia.color.rgb_to_xyz).
_RGB2XYZ = np.array([[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]], dtype=np.float32)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ).astype(np.float32)


def _scale(phi: np.ndarray) -> dict:
    """Normalized [0,1] φ (9,) → physical params (ELVIS (vmax-vmin)·x+vmin scaling)."""
    return dict(
        ev=-3.5 * phi[0],                              # negative EV (darken)
        k=float(np.exp((_LN30 - _LN01) * phi[1] + _LN01)),  # shot gain
        read=0.1 * phi[2],
        quant=0.1 * phi[3],
        band=0.03 * phi[4],
        band_h=int(round(float(phi[5]))) == 0,        # 0 → horizontal, 1 → vertical
        s1=9.5 * phi[6] + 0.5,
        s2=9.5 * phi[7] + 0.5,
        ang=(0.25 * phi[8]) * np.pi,                  # radians
    )


def _exposure(img: np.ndarray, ev: float) -> np.ndarray:
    """img (H,W,3) → XYZ, ×2^ev, → RGB. ev<0 darkens."""
    xyz = img @ _RGB2XYZ.T
    xyz *= (2.0 ** ev)
    return xyz @ _XYZ2RGB.T


def _aniso_kernel(s1: float, s2: float, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    D = np.diag([s1 ** 2, s2 ** 2]).astype(np.float32)
    cov_inv = np.linalg.inv(R @ D @ R.T)
    ax = np.arange(BLUR_KSIZE) - BLUR_KSIZE // 2
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    coords = np.stack([xx, yy], -1).astype(np.float32)            # (k,k,2)
    expo = -0.5 * np.einsum("...i,ij,...j->...", coords, cov_inv, coords)
    ker = np.exp(expo)
    return (ker / ker.sum()).astype(np.float32)


def degrade_clip_elvis(clip: np.ndarray, phi: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """clip (T,3,H,W) float[0,1], phi (T,9) or (9,) normalized → degraded (T,3,H,W) float[0,1]."""
    T = clip.shape[0]
    if phi.ndim == 1:
        phi = np.repeat(phi[None], T, 0)
    out = np.empty_like(clip)
    for t in range(T):
        p = _scale(phi[t])
        img = np.ascontiguousarray(clip[t].transpose(1, 2, 0))    # (H,W,3)
        img = _exposure(img, p["ev"])
        ker = _aniso_kernel(p["s1"], p["s2"], p["ang"])
        img = cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_REFLECT)
        img = np.clip(img, 0.0, None)
        # Poisson shot in 8-bit photon space (matches noise.py shot_noise for x<=1).
        q = (img * 255.0).astype(np.int32)
        img = rng.poisson(np.maximum(q, 0) / p["k"]).astype(np.float32) * p["k"] / 255.0
        img = img + rng.standard_normal(img.shape).astype(np.float32) * p["read"]   # read
        img = img + rng.uniform(0.0, p["quant"], img.shape).astype(np.float32)       # quant (additive)
        H, W, _ = img.shape                                                          # banding stripes
        if p["band_h"]:
            band = rng.standard_normal((H, 1, 3)).astype(np.float32) * p["band"]
        else:
            band = rng.standard_normal((1, W, 3)).astype(np.float32) * p["band"]
        img = img + band
        out[t] = np.clip(img, 0.0, 1.0).transpose(2, 0, 1)
    return out


# Φ bank (author-provided VDP-Net profiles) — lazy module-level cache.
_PHI = None
_PHI_HARDRANK = None   # indices sorted easy→hard by noise magnitude (for severe-val selection)


def _load_phi(path: str = "data/elvis_phi.npy"):
    global _PHI, _PHI_HARDRANK
    if _PHI is None:
        bank = np.load(path).astype(np.float32)          # (M,5,9)
        _PHI = bank
        mag = bank[:, :, [1, 2, 3, 4]].mean((1, 2))      # shot+read+quant+band → severity proxy
        _PHI_HARDRANK = np.argsort(mag)
    return _PHI


def sample_profile(rng: np.random.RandomState, severity=None, path: str = "data/elvis_phi.npy") -> np.ndarray:
    """Return one clip profile (5,9). severity=None → uniform (train); severity∈[0,1] → that
    quantile of the easy→hard ranking (deterministic severe val pin)."""
    bank = _load_phi(path)
    if severity is None:
        idx = rng.randint(bank.shape[0])
    else:
        rank = _PHI_HARDRANK
        lo = int(np.clip(severity, 0, 1) * (len(rank) - 1))
        idx = rank[rng.randint(lo, len(rank))] if lo < len(rank) - 1 else rank[-1]
    return bank[idx]


# --- DN-01 final-v2 `lowlight` mode (user 2026-09-26): exposure darkening + ELVIS's physics noise terms (the model
# PocketDVDNet's current trainer uses; temporally independent per frame) with a real-footage profile bank, an amplitude
# scale kappa calibrated on paired BVI-RLV (scripts/dn01_lowlight_calibrate.py), and bench_joint's ISOTROPIC Gaussian
# blur in place of ELVIS's anisotropic one. The restoration target stays the normally exposed clean frame.
ELVIS_PHI_PATHS = ("/mnt/DATA/DegradedDepth/RAMEN_aaai/data/elvis_phi.npy",       # crispwork (gitignored data)
                   "/projects/b5dh/benchmark/weights/elvis_phi.npy",               # Isambard
                   "data/elvis_phi.npy")


def load_bank() -> np.ndarray:
    """The ELVIS real-footage profile bank (M,5,9), from the first existing ELVIS_PHI_PATHS entry."""
    import os
    path = next((p for p in ELVIS_PHI_PATHS if os.path.isfile(p)), None)
    if path is None:
        raise FileNotFoundError(f"ELVIS profile bank not found in {ELVIS_PHI_PATHS}")
    return _load_phi(path)


def lowlight_profile(rng: np.random.RandomState) -> np.ndarray:
    """One clip's noise profile: a random bank entry, averaged over its 5 frames (9,) (one profile per clip, as
    ELVIS/Pocket sample; the noise REALISATION is still drawn independently per frame)."""
    bank = load_bank()
    return bank[rng.randint(bank.shape[0])].mean(0)


def lowlight_reference(clip: np.ndarray, ev: float) -> np.ndarray:
    """The exposure-equivariant target of a lowlight clip (user 2026-09-26, after the shadow-lifting defect,
    provenance/dn01_lowlight_defect/): clean (T,3,H,W) at the SAME exposure 2^ev, with degrade_clip_lowlight's own
    clipping and 8-bit truncation but no blur and no noise. It is both the training target (data.lowlight.target
    'exposed') and bench_joint's native-exposure GT of the LL cells."""
    clip = np.clip(clip, 0.0, 1.0)                     # as _degrade_clip clamps the clean clip before degrading
    out = np.empty_like(clip)
    for t in range(clip.shape[0]):
        img = np.clip(_exposure(np.ascontiguousarray(clip[t].transpose(1, 2, 0)), ev), 0.0, None)
        out[t] = np.clip((img * 255.0).astype(np.int32) / 255.0, 0.0, 1.0).transpose(2, 0, 1)
    return out


def degrade_clip_lowlight(clip: np.ndarray, ev: float, phi: np.ndarray, kappa: float, blur_sigma: float,
                          rng: np.random.RandomState) -> np.ndarray:
    """clip (T,3,H,W) float[0,1] -> exposure 2^ev (XYZ, as ELVIS) -> isotropic Gaussian blur sigma px (cv2, as
    _blur_frames) -> ELVIS shot/read/quant/band noise from ``phi`` with every amplitude scaled by ``kappa`` (shot gain
    K*kappa^2, so its std scales by kappa too) -> clip [0,1]. kappa = 1 is ELVIS's noise exactly."""
    p = _scale(phi)
    k = p["k"] * kappa ** 2
    out = np.empty_like(clip)
    ksz = int(2 * round(3 * blur_sigma) + 1)
    for t in range(clip.shape[0]):
        img = _exposure(np.ascontiguousarray(clip[t].transpose(1, 2, 0)), ev)
        if blur_sigma > 0:
            img = cv2.GaussianBlur(img, (ksz, ksz), blur_sigma)
        img = np.clip(img, 0.0, None)
        q = (img * 255.0).astype(np.int32)
        img = rng.poisson(np.maximum(q, 0) / k).astype(np.float32) * k / 255.0
        img = img + rng.standard_normal(img.shape).astype(np.float32) * (p["read"] * kappa)
        img = img + rng.uniform(0.0, p["quant"] * kappa, img.shape).astype(np.float32)
        H, W, _ = img.shape
        shape = (H, 1, 3) if p["band_h"] else (1, W, 3)
        img = img + rng.standard_normal(shape).astype(np.float32) * (p["band"] * kappa)
        out[t] = np.clip(img, 0.0, 1.0).transpose(2, 0, 1)
    return out


if __name__ == "__main__":  # self-check: degradation is present, bounded, banding visible
    rng = np.random.RandomState(0)
    base = np.ones((4, 3, 64, 64), np.float32) * 0.5
    phi_hard = np.array([0.3, 0.9, 0.6, 0.5, 0.9, 0.0, 0.05, 0.05, 0.3], np.float32)  # heavy noise, horiz band
    deg = degrade_clip_elvis(base, phi_hard, rng)
    assert deg.min() >= 0 and deg.max() <= 1, "out of [0,1]"
    noise_std = deg[0].std()
    assert noise_std > 0.02, f"no noise added (std={noise_std:.4f})"
    # horizontal banding → row means vary more than a clean flat image
    rowvar = deg[0, 0].mean(1).std()
    assert rowvar > 0.005, f"no banding (rowvar={rowvar:.4f})"
    # sample real profiles
    p = sample_profile(rng)
    assert p.shape == (5, 9)
    ps = sample_profile(rng, severity=0.9)
    print(f"[selfcheck] OK  std={noise_std:.3f} rowvar={rowvar:.3f}  "
          f"profile{p.shape} severe-mean={ps.mean(0).round(2)}")

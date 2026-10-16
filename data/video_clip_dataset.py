"""
Video depth clip dataset — standalone TartanAir-V2 loader
=========================================================

Self-contained clip loader for D3.  Loads
short RGB + metric-depth clips from a TartanAir-V2 tree and, when enabled,
applies a compact synthetic degradation (low-light + noise + defocus blur) to
the *student* input while keeping GT depth/mask clean.

Per-sample contract (collated by ``DataLoader`` into a leading batch dim ``B``):

    frames          (C, T, H, W)  float32 [0, 1]   — clean RGB clip
    degraded_frames (C, T, H, W)  float32 [0, 1]   — degraded student input (if enabled)
    depth           (1, T, H, W)  float32 metres   — metric GT depth (clean)
    mask            (1, T, H, W)  float32 {0, 1}    — valid-pixel mask (sky excluded)
    video_id        str                             — e.g. "AbandonedFactory/P000"
    start_frame     int                             — first frame index

H and W must be divisible by 14 (DINOv2 patch size); 518 = 37×14 is canonical.

Directory layout (TartanAir-V2)::

    root/<Env>/Data_easy/P00X/{image_lcam_front/*.png, depth_lcam_front/*.png}

Depth PNGs pack a little-endian float32 metre value into 4 uint8 channels; they
must be decoded with OpenCV's unchanged-read path (PIL RGBA is not equivalent).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("opencv-python is required to decode TartanAir depth PNGs.") from e


def _decode_depth(path: str) -> np.ndarray:
    """Decode a TartanAir 4-channel depth PNG to a float32 metre map ``(H, W)``."""
    bgra = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgra is None:
        raise FileNotFoundError(f"Could not read depth PNG: {path}")
    if bgra.ndim != 3 or bgra.shape[2] != 4 or bgra.dtype != np.uint8:
        raise ValueError(f"Expected 4-channel uint8 depth PNG, got {bgra.shape}/{bgra.dtype}")
    return np.squeeze(np.ascontiguousarray(bgra).view("<f4"), axis=-1)


AWGN_BLUR_MAX_SIGMA_PX = 3.0   # DN-01 awgn_blur: defocus sigma at s_blur=1, px at model resolution


def _llvar_opts(lowlight: dict, rng: np.random.RandomState, phi: np.ndarray, blur_sigma: float):
    """DN-01 protocol v3 WP20 llvar-train (opt-in, data.lowlight keys; all absent = byte-identical, NO extra rng draw): returns
    ``(phi, kappa, blur_sigma)`` for one lowlight clip.  ``kappa_mult_choices`` [..]: kappa x one choice drawn uniformly per clip
    (llvar: [0.5, 1, 2]); ``blur_px`` b: the Gaussian blur is pinned to b px (llvar: 3.0) instead of the s_b x 3 px draw;
    ``profile: median``: the ELVIS bank's median profile instead of a random bank entry (llvar's noise profile)."""
    kappa = float(lowlight["kappa"])
    if lowlight.get("profile") == "median":
        from .elvis_noise import load_bank
        phi = np.median(load_bank().reshape(-1, 9), 0)
    elif lowlight.get("profile") is not None:
        raise ValueError(f"data.lowlight.profile must be absent or 'median', got {lowlight['profile']!r}")
    mults = lowlight.get("kappa_mult_choices")
    if mults:
        kappa *= float(mults[int(rng.randint(len(mults)))])      # drawn AFTER the default draws: off => same stream
    if lowlight.get("blur_px") is not None:
        blur_sigma = float(lowlight["blur_px"])
    return phi, kappa, blur_sigma



def _blur_frames(out: np.ndarray, blur_sigma: float) -> np.ndarray:
    """cv2 Gaussian defocus blur of every ``(3, H, W)`` frame of ``out`` (``(T, 3, H, W)``),
    in place; shared by ``full`` and ``awgn_blur`` (and DN-01's eval scripts)."""
    k = int(2 * round(3 * blur_sigma) + 1)
    for t in range(out.shape[0]):
        img = np.ascontiguousarray(out[t].transpose(1, 2, 0))   # (H, W, 3)
        img = cv2.GaussianBlur(img, (k, k), blur_sigma)
        out[t] = img.transpose(2, 0, 1)
    return out


def _disk_blur_frames(out: np.ndarray, blur_sigma: float) -> np.ndarray:
    """DN-01 proxy screen (data.blur_disk_prob): pillbox (disk) defocus of radius R = 2*blur_sigma px, in place --
    bench_joint's ``d<R>`` cells (same kernel as bench_joint.disk_psf, reflect-101 border), matched to the
    Gaussian by per-axis second moment (disk variance R^2/4 = sigma^2). R < 1 px is a 1-pixel identity kernel."""
    r = 2.0 * blur_sigma
    n = int(np.ceil(r))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    ker = (xx ** 2 + yy ** 2 <= r ** 2).astype(np.float32)
    ker /= ker.sum()
    for t in range(out.shape[0]):
        img = np.ascontiguousarray(out[t].transpose(1, 2, 0))
        out[t] = cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_REFLECT_101).transpose(2, 0, 1)
    return out


def _degrade_clip(
    clip: torch.Tensor,
    rng: np.random.RandomState,
    mode: str = "full",
    severity: Optional[float] = None,
    force_blur: Optional[bool] = None,
    blur_severity: Optional[float] = None,
    blur_disk: bool = False,
    lowlight: Optional[dict] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """ELVIS-style low-light + noise + defocus degradation on a ``(T, 3, H, W)`` clip.

    Follows the analytic low-light synthesis of ELVIS (Lin et al., arXiv:2512.01495):
    illumination reduction (gain/gamma) + Gaussian defocus blur + Poisson shot noise +
    Gaussian read noise, applied to the student input while GT depth/mask stay clean.

    A single severity scalar ``s ∈ [0, 1]`` drives all factors jointly (sampled from
    ``rng`` when ``severity`` is None). This removes the unlearnable corners of the old
    independent sampling (e.g. gamma 3 × gain 0.2 × σ 0.06 ≈ black noise, SNR < 2) and
    gives a controllable severity axis for curriculum training and robustness sweeps.

    ``mode``:
      * ``"full"``     — low-light (gamma/gain) + defocus blur + shot + read noise.
      * ``"gaussian"`` — additive Gaussian read noise ONLY (no darkening, no blur). An
        ablation to disentangle whether degraded-training erosion is driven by low-light
        domain shift or just sensor noise.
      * ``"awgn"``     — additive white Gaussian noise, sigma = s*50/255 (DN-01 denoising).
      * ``"awgn_blur"`` — Gaussian defocus blur, sigma_b = s_b*3 px (``AWGN_BLUR_MAX_SIGMA_PX``,
        at the clip's own resolution), THEN ``awgn``'s noise at severity s. ``s_b`` is
        ``blur_severity`` (an independent draw, see ``_VDADegradedMixin``); None -> shares s.

    ``force_blur``, when not ``None``, overrides the ``mode="full"`` blur Bernoulli's
    outcome (used by stratified blur assignment, docs/DEGRADED_EVAL_PROTOCOL.md §1)
    without changing how much of the RNG stream this call consumes, so severity/noise
    stay identical to an unforced call with the same seed.

    Returns ``(degraded_clip, params)`` with ``params`` =
    ``float32 [s, gamma, gain, sigma_read, blur_sigma]`` — the true synthesis profile,
    free supervision for a degradation-profile estimator.

    Pure cv2/numpy (NO torch ops) — torch's OpenMP conv2d deadlocks inside forked
    DataLoader workers; cv2/numpy are fork-safe (the decode path already uses them).
    """
    arr = np.clip(clip.numpy(), 0.0, 1.0)                # (T, 3, H, W) in [0, 1]
    if mode == "elvis":
        # ELVIS degradation (Lin et al., arXiv:2512.01495): sample a real VDP-Net profile φ
        # from the author-provided bank and apply the full Deg(·) pipeline. severity=None →
        # uniform over Φ (train, the ELVIS recipe); a fixed s → hard-tail quantile (severe val).
        from .elvis_noise import sample_profile, degrade_clip_elvis
        phi = sample_profile(rng, severity)                          # (5, 9) normalized
        T = arr.shape[0]
        phi_t = phi[np.clip(np.round(np.linspace(0, phi.shape[0] - 1, T)).astype(int),
                            0, phi.shape[0] - 1)]                    # (T, 9)
        out = degrade_clip_elvis(arr, phi_t, rng)
        return torch.from_numpy(out), phi.mean(0).astype(np.float32)  # params = clip-mean φ (9,)
    s = float(rng.uniform(0.0, 1.0)) if severity is None else float(severity)
    if mode == "lowlight":
        # DN-01 final-v2 (user 2026-09-26): exposure darkening + ELVIS physics noise (calibrated kappa, real-footage
        # profile bank) + isotropic Gaussian blur sigma_b = s_b * 3 px (awgn_blur's draw). s == 0 -> clean (clean_prob).
        # ``lowlight`` = {"ev": [lo, hi], "kappa": k} (data.lowlight). This function returns only the degraded input; the
        # restoration target is set by the caller: with data.lowlight.target 'exposed' (the v2 recipe) it is the clean
        # frame at the input's exposure (elvis_noise.lowlight_reference(clean, EV), ~ clean x 2^EV; _VDADegradedMixin),
        # otherwise the normally exposed clean frame.
        from .elvis_noise import degrade_clip_lowlight, lowlight_profile
        if lowlight is None:
            raise ValueError("degrade_mode 'lowlight' needs data.lowlight {ev: [lo, hi], kappa}")
        blur_sigma = float(blur_severity or 0.0) * AWGN_BLUR_MAX_SIGMA_PX
        if s == 0:
            return torch.from_numpy(arr.copy()), np.array([0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        ev = float(rng.uniform(*lowlight["ev"]))
        phi = lowlight_profile(rng)
        phi, kappa, blur_sigma = _llvar_opts(lowlight, rng, phi, blur_sigma)   # protocol v3 WP20 llvar-train (keys absent: no-op)
        out = degrade_clip_lowlight(arr, ev, phi, kappa, blur_sigma, rng)
        if lowlight.get("quantize", False):     # K44 proxy arm (b): upstream GenVIS's final 8-bit floor
            out = np.floor(out * 255.0) / 255.0
        params = np.array([s, 1.0, 2.0 ** ev, 0.1 * phi[2] * kappa, blur_sigma], dtype=np.float32)   # protocol v3 WP20 llvar-train
        return torch.from_numpy(out.astype(np.float32)), params
    if mode == "gaussian":
        gamma, gain, blur_sigma = 1.0, 1.0, 0.0
        sigma_r = 0.01 + 0.05 * s                        # same marginal as the old U(0.01, 0.06)
        out = arr.copy()
    elif mode == "awgn":
        # DN-01: plain additive white Gaussian noise at the standard denoising scale,
        # sigma = s * 50/255 (s=1 -> the sigma=50 benchmark level; s=0 -> exactly clean).
        gamma, gain, blur_sigma = 1.0, 1.0, 0.0
        sigma_r = s * 50.0 / 255.0
        out = arr.copy()
    elif mode == "awgn_blur":
        # DN-01 A1-blur: defocus first, then awgn's noise (added after the branches). s = s_b = 0 -> clean.
        gamma, gain = 1.0, 1.0
        if blur_severity is None:
            blur_severity = s
        blur_sigma = float(blur_severity) * AWGN_BLUR_MAX_SIGMA_PX
        sigma_r = s * 50.0 / 255.0
        out = arr.copy()
        if blur_sigma > 0:                               # cv2 derives sigma from ksize at 0: skip, stay exact
            (_disk_blur_frames if blur_disk else _blur_frames)(out, blur_sigma)
    elif mode == "full":
        gamma = 1.0 + 1.8 * s                            # darken midtones
        # ponytail: 0.15 gain floor — only binds at s>1.42, so s<=1.0 (all prior training + eval)
        # is byte-identical. Without it, severity>1.67 gives gain<=0 → black frames (unlearnable).
        gain = max(0.15, 1.0 - 0.6 * s)                  # reduce exposure
        out = (arr ** gamma) * gain

        blur_sigma = 0.0
        blur_coin = rng.uniform()                        # always drawn, even if force_blur overrides
        # the outcome below, so the RNG stream position (and everything drawn after
        # it) stays identical to an unforced call -- force_blur only overrides which
        # branch runs, not how much of the stream this call consumes.
        do_blur = (blur_coin < 0.7) if force_blur is None else force_blur
        if do_blur:                                       # defocus blur via cv2
            blur_sigma = 0.3 + 2.2 * s
            _blur_frames(out, blur_sigma)

        # Poisson shot noise: photon count per unit intensity falls with severity
        # (signal-dependent noise — the dominant term in real low light).
        n_ph = float(4000.0 * (600.0 / 4000.0) ** s)
        out = rng.poisson(out * n_ph).astype(np.float32) / n_ph
        sigma_r = 0.005 + 0.015 * s                      # read noise
    else:
        raise ValueError(f"_degrade_clip: unknown mode {mode!r} (use 'full' | 'gaussian' | 'awgn' | 'awgn_blur' | 'elvis').")

    out = out + rng.standard_normal(out.shape).astype(np.float32) * sigma_r
    params = np.array([s, gamma, gain, sigma_r, blur_sigma], dtype=np.float32)
    return torch.from_numpy(np.clip(out, 0.0, 1.0).astype(np.float32)), params


STATIC_MODES = ("awgn", "awgn_blur", "lowlight")     # DN-01 WS2 arm S: the modes with an independent per-frame draw
STATIC_NOISE_MULT = (0.5, 1.5)                        # per-frame noise-level multiplier U[lo, hi] (kappa / sigma)
STATIC_EV_JITTER = 0.15                               # lowlight: per-frame exposure jitter U[-j, +j] stops


def _degrade_static_clip(clip: torch.Tensor, rng: np.random.RandomState, mode: str, severity: float,
                         blur_severity: Optional[float] = None, blur_disk: bool = False,
                         lowlight: Optional[dict] = None, noise_mult=STATIC_NOISE_MULT,
                         ev_jitter: float = STATIC_EV_JITTER):
    """DN-01 WS2 arm S (K49): degrade a STATIC clip ``(T, 3, H, W)`` (one still frame repeated T times) with an
    INDEPENDENT draw per frame, so any change between the degraded frames is noise. The clip keeps its ONE severity /
    EV / noise profile; per frame the noise level is scaled by m_t ~ U[noise_mult] (lowlight: kappa x m_t, plus an
    exposure jitter j_t ~ U[-ev_jitter, +ev_jitter] stops, the auto-gain case; awgn / awgn_blur: sigma x m_t).
    ``severity`` == 0 is a clean clip (returned as is). Returns ``(degraded (T,3,H,W), params (5,), evs)``: ``params``
    as :func:`_degrade_clip` (the clip-level EV for lowlight; the mean over frames for awgn*), ``evs`` (T,) = each
    frame's exposure in stops (lowlight, s > 0) else None -- the exposed target follows it (mixin)."""
    if mode not in STATIC_MODES:
        raise ValueError(f"_degrade_static_clip: mode {mode!r} not in {STATIC_MODES}")
    T = clip.shape[0]
    s = float(severity)
    if mode == "lowlight":
        from .elvis_noise import degrade_clip_lowlight, lowlight_profile
        if lowlight is None:
            raise ValueError("degrade_mode 'lowlight' needs data.lowlight {ev: [lo, hi], kappa}")
        if s == 0:
            return clip.clone(), np.array([0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32), None
        arr = np.clip(clip.numpy(), 0.0, 1.0)
        blur_sigma = float(blur_severity or 0.0) * AWGN_BLUR_MAX_SIGMA_PX
        ev = float(rng.uniform(*lowlight["ev"]))
        phi = lowlight_profile(rng)
        phi, kappa, blur_sigma = _llvar_opts(lowlight, rng, phi, blur_sigma)   # protocol v3 WP20 llvar-train (keys absent: no-op)
        out, evs = np.empty_like(arr), np.empty(T)
        for t in range(T):
            m, j = float(rng.uniform(*noise_mult)), float(rng.uniform(-ev_jitter, ev_jitter))
            evs[t] = ev + j
            out[t:t + 1] = degrade_clip_lowlight(arr[t:t + 1], evs[t], phi, kappa * m, blur_sigma, rng)
        if lowlight.get("quantize", False):
            out = np.floor(out * 255.0) / 255.0
        params = np.array([s, 1.0, 2.0 ** ev, 0.1 * phi[2] * kappa, blur_sigma], dtype=np.float32)
        return torch.from_numpy(out.astype(np.float32)), params, evs
    if mode == "awgn_blur" and blur_severity is None:
        blur_severity = s                     # _degrade_clip would tie the blur to the per-frame severity s x m_t
    frames, params = [], []
    for t in range(T):
        m = float(rng.uniform(*noise_mult))
        f, p = _degrade_clip(clip[t:t + 1], rng, mode, severity=s * m, blur_severity=blur_severity,
                             blur_disk=blur_disk)
        frames.append(f)
        params.append(p)
    return torch.cat(frames), np.mean(params, 0).astype(np.float32), None


def endpoint_degradation_seed(benchmark: str, video_id: str, mode: str, severity: float) -> int:
    """The one-profile-per-video seed derivation pre-registered in
    docs/DEGRADED_EVAL_PROTOCOL.md §4: ``sha256(f"{benchmark}|{video_id}|{mode}|{severity}")``,
    first 8 hex digits as an int. A single shared helper so every endpoint manifest
    builder (Bonn, VKITTI, KITTI) derives this identically instead of each hand-copying
    the formula -- the same drift risk :func:`evaluation.alignment_free.update_pixel_digest`
    exists to close for the digest side of this mechanism.
    """
    import hashlib
    key = f"{benchmark}|{video_id}|{mode}|{severity}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def stratified_blur_assignment(video_ids: list[str]) -> dict[str, bool]:
    """docs/DEGRADED_EVAL_PROTOCOL.md §1: within a benchmark's PRIMARY set, force the
    blurred fraction to the nearest achievable value to 0.7, assigned as the first k
    videos by sorted video id (lexicographic), k = nearest-achievable count to 0.7*n,
    round up on ties. Literal, checked counts: 4/5 (VKITTI-5scene), 18/25 (Bonn
    primary), 9/13 (KITTI 13-drive). A single shared helper, alongside
    :func:`endpoint_degradation_seed` and :func:`degrade_frame_range`, so every
    endpoint manifest builder (Bonn, VKITTI, KITTI) computes this identically
    instead of each hand-copying the formula.
    """
    n = len(video_ids)
    # nearest achievable count to 0.7*n, tie -> round up (§1's explicit tie-break)
    exact = 0.7 * n
    lower, upper = int(exact), int(exact) + (0 if exact == int(exact) else 1)
    if exact - lower == 0.5:
        k = upper
    else:
        k = lower if (exact - lower) < 0.5 else upper
    ordered = sorted(video_ids)
    blurred_set = set(ordered[:k])
    return {vid: (vid in blurred_set) for vid in video_ids}


def degrade_frame_range(
    frames: torch.Tensor,
    rng: np.random.RandomState,
    severity: float,
    blurred: bool,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Apply ``_degrade_clip``'s ``mode="full"`` formulas to a range of frames from a
    longer video, drawing Poisson shot noise and Gaussian read noise ONE FRAME AT A
    TIME so the result does not depend on how the caller partitions the video into
    ranges -- required by ``evaluation.alignment_free.RecipeManifest``'s contract,
    where a whole video's pixels must never be held in memory at once (see
    provenance/plan_state.json's MATRIX-03 note for the OOM this avoids: Bonn's
    ``rgbd_bonn_static`` alone is ~44GB of raw frames).

    ``frames`` is ``(n, C, H, W)``, already resized, in ``[0, 1]``. ``rng`` must be
    freshly seeded per video (:func:`endpoint_degradation_seed`) and reused, unmodified,
    across every call for that same video's ranges, called with strictly increasing,
    non-overlapping, contiguous ranges starting at local frame 0 -- anything else
    silently desyncs the stream and produces the wrong (not obviously wrong) pixels.
    ``severity``/``blurred`` are the video's already-decided one-profile-per-video
    values (§1) -- fixed inputs here, not drawn.

    This is a DELIBERATELY DIFFERENT random-number decomposition from
    ``_degrade_clip``'s own whole-clip call, which draws Poisson/Gaussian noise in two
    single bulk calls over the FULL clip -- verified empirically that bulk-over-a-
    sub-range and bulk-over-the-whole-clip do NOT produce the same bytes when
    concatenated (numpy's Poisson/Gaussian generators are each individually
    chunk-size-independent, but interleaving Poisson-then-Gaussian per chunk is a
    different draw order than Poisson-over-everything-then-Gaussian-over-everything).
    Reproducing ``_degrade_clip`` bit-for-bit under chunking would require holding the
    whole video's frames in memory to draw noise in the same two whole-clip calls --
    exactly what this function exists to avoid. ``docs/DEGRADED_EVAL_PROTOCOL.md`` §4
    anticipates and licenses this: "If a future implementation needs to degrade a
    video in per-frame pieces for memory reasons, the SAME video_id-derived seed must
    still drive one continuous RNG stream across those pieces, not a fresh seed per
    piece." What matters (§0's non-negotiable) is every arm seeing byte-identical
    pixels for the same video under the same manifest, which holds by construction
    here since the decomposition is fixed (always one frame at a time, in order) and
    therefore independent of ``RecipeManifest``'s read-side ``chunk_size`` -- not that
    this reproduces ``_degrade_clip``'s own whole-clip draw order.

    Formulas (gamma, gain, blur sigma, photon count, read sigma) are ported verbatim
    from ``_degrade_clip``'s ``mode="full"`` branch; only ``mode="full"`` is
    implemented (``"gaussian"``/``"elvis"`` are not used by the endpoint protocol,
    docs/DEGRADED_EVAL_PROTOCOL.md §1).

    Returns ``(degraded_frames, params)`` with ``params`` = ``float32
    [s, gamma, gain, sigma_read, blur_sigma]`` -- the same field order as
    ``_degrade_clip``'s own return, so a caller processing a video in multiple
    ranges can assert the profile is identical across calls (the one-profile-
    per-video invariant, §1's own requirement, now enforced by a different
    function than the one §1 originally described -- see
    docs/DEGRADED_EVAL_PROTOCOL.md §4's per-frame-derivation amendment).
    """
    s = float(severity)
    gamma = 1.0 + 1.8 * s
    gain = max(0.15, 1.0 - 0.6 * s)
    blur_sigma = 0.3 + 2.2 * s if blurred else 0.0
    n_ph = float(4000.0 * (600.0 / 4000.0) ** s)
    sigma_r = 0.005 + 0.015 * s

    arr = np.clip(frames.numpy(), 0.0, 1.0)              # (n, C, H, W)
    out = (arr ** gamma) * gain
    if blurred:
        k = int(2 * round(3 * blur_sigma) + 1)
        for t in range(out.shape[0]):
            img = np.ascontiguousarray(out[t].transpose(1, 2, 0))   # (H, W, C)
            img = cv2.GaussianBlur(img, (k, k), blur_sigma)
            out[t] = img.transpose(2, 0, 1)

    degraded = np.empty_like(out)
    for t in range(out.shape[0]):
        frame = rng.poisson(out[t] * n_ph).astype(np.float32) / n_ph
        frame = frame + rng.standard_normal(frame.shape).astype(np.float32) * sigma_r
        degraded[t] = frame
    params = np.array([s, gamma, gain, sigma_r, blur_sigma], dtype=np.float32)
    return torch.from_numpy(np.clip(degraded, 0.0, 1.0).astype(np.float32)), params


class VideoDepthClipDataset(Dataset):
    """Standalone TartanAir-V2 RGB + metric-depth clip dataset.

    Args:
        root: TartanAir-V2 root directory (contains environment folders).
        num_frames: Clip length ``T``.
        image_size: ``(H, W)``; both divisible by 14.
        split: ``"train"`` or ``"val"`` (deterministic trajectory split).
        clip_stride: Stride between clip start frames within a trajectory.
        frame_stride: Step between frames inside a clip.
        max_depth: Depths ``≥`` this (and non-finite sky) are masked out, then clipped.
        val_fraction: Fraction of trajectories held out for validation.
        seed: Split / degradation RNG seed.
        difficulty / camera: TartanAir sub-folder selectors.
        degrade: If True, also return ``degraded_frames``.
        envs: Optional whitelist of environment names.
        max_trajectories: Optional cap on number of trajectories.
    """

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
        seed: int = 42,
        difficulty: str = "Data_easy",
        camera: str = "lcam_front",
        degrade: bool = True,
        degrade_mode = "full",   # str or list[str] — sampled per clip during training
        degrade_severity: Optional[float] = None,
        envs: Optional[List[str]] = None,
        max_trajectories: Optional[int] = None,
        teacher_cache_dir: Optional[str] = None,
        teacher_name: str = "depth_pro",
        max_clips_per_video: Optional[int] = None,
        split_seed: Optional[int] = None,   # E4: seeds ONLY the held-out trajectory shuffle (None = `seed`: byte-identical); val degradation keeps `seed`
    ) -> None:
        super().__init__()
        self.teacher_cache_dir = teacher_cache_dir
        self.teacher_name = teacher_name
        self.max_clips_per_video = max_clips_per_video
        self.num_frames = num_frames
        self.image_size = tuple(image_size)
        self.split = split
        self.clip_stride = clip_stride
        self.frame_stride = frame_stride
        self.max_depth = max_depth
        self.degrade = degrade
        self.degrade_mode = degrade_mode
        self.degrade_severity = degrade_severity  # fixed severity (eval sweeps); None = sample
        # Curriculum cap on train-time severity; shared memory so the main process can
        # ramp it while forked persistent DataLoader workers see the update.
        self.severity_max = torch.ones(1).share_memory_()
        self.seed = int(seed)

        root_p = Path(root)
        trajectories: List[Tuple[Path, Path]] = []
        env_dirs = sorted(p for p in root_p.iterdir() if p.is_dir() and not p.name.startswith("."))
        if envs is not None:
            env_dirs = [d for d in env_dirs if d.name in envs]
        for env in env_dirs:
            diff = env / difficulty
            if not diff.is_dir():
                continue
            for traj in sorted(diff.iterdir()):
                img_dir, dep_dir = traj / f"image_{camera}", traj / f"depth_{camera}"
                if img_dir.is_dir() and dep_dir.is_dir():
                    trajectories.append((img_dir, dep_dir))
        if not trajectories:
            raise FileNotFoundError(
                f"No trajectories under {root} with image_{camera}/ + depth_{camera}/."
            )

        # Deterministic train/val trajectory split.
        rng = random.Random(seed if split_seed is None else split_seed)
        rng.shuffle(trajectories)
        n_val = max(1, int(len(trajectories) * val_fraction))
        trajectories = trajectories[:n_val] if split == "val" else trajectories[n_val:]
        if max_trajectories is not None:
            trajectories = trajectories[:max_trajectories]

        # Build clip index: (image_files, depth_files, video_id, start).
        self.clips: List[Tuple[List[Path], List[Path], str, int]] = []
        span = (num_frames - 1) * frame_stride + 1
        for img_dir, dep_dir in trajectories:
            imgs = sorted(img_dir.glob("*.png"))
            deps = sorted(dep_dir.glob("*.png"))
            n = min(len(imgs), len(deps))
            if n < span:
                continue
            video_id = f"{img_dir.parent.parent.parent.name}/{img_dir.parent.name}"
            starts = list(range(0, n - span + 1, clip_stride))
            if max_clips_per_video and len(starts) > max_clips_per_video:
                # evenly thin the huge trajectories (113k-frame GreatMarsh vs 2k median)
                # so per-env clip counts stay balanced; keeps temporal spread, not just the head
                step = len(starts) / max_clips_per_video
                starts = [starts[int(i * step)] for i in range(max_clips_per_video)]
            for start in starts:
                self.clips.append((imgs[:n], deps[:n], video_id, start))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_files, dep_files, video_id, start = self.clips[idx]
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]
        H, W = self.image_size

        frames, depths, masks, skies = [], [], [], []
        for i in indices:
            img = Image.open(img_files[i]).convert("RGB").resize((W, H), Image.BILINEAR)
            frames.append(torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1))

            d_raw = _decode_depth(str(dep_files[i]))  # (H0, W0) metres; inf for sky
            valid = np.isfinite(d_raw) & (d_raw > 0) & (d_raw < self.max_depth)
            # Sky / no-return (non-finite or beyond 10 km): target disparity 0 for the
            # VDA-style sky loss; kept OUT of `mask` so metrics are unchanged.
            sky = ~np.isfinite(d_raw) | (d_raw > 1.0e4)
            d = np.clip(d_raw, 0.0, self.max_depth)
            depths.append(torch.from_numpy(d).unsqueeze(0).float())            # (1, H0, W0)
            masks.append(torch.from_numpy(valid.astype(np.float32)).unsqueeze(0))
            skies.append(torch.from_numpy(sky.astype(np.float32)).unsqueeze(0))

        frames = torch.stack(frames, 0)                       # (T, 3, H, W)
        depth = torch.stack(depths, 1)                        # (1, T, H0, W0)
        mask = torch.stack(masks, 1)                          # (1, T, H0, W0)
        sky = torch.stack(skies, 1)                           # (1, T, H0, W0)
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(depth.permute(1, 0, 2, 3), (H, W), mode="nearest").permute(1, 0, 2, 3)
            mask = F.interpolate(mask.permute(1, 0, 2, 3), (H, W), mode="nearest").permute(1, 0, 2, 3)
            sky = F.interpolate(sky.permute(1, 0, 2, 3), (H, W), mode="nearest").permute(1, 0, 2, 3)

        result: Dict[str, torch.Tensor] = {
            "frames": frames.permute(1, 0, 2, 3).contiguous(),  # (C, T, H, W)
            "depth": depth,
            "mask": mask,
            "sky": sky,
            "video_id": video_id,
            "start_frame": start,
        }
        if self.teacher_cache_dir is not None:
            # Pre-cached Depth Pro teacher depth (FocusMamba clip-format cache, keyed by
            # video_id + absolute frame index; non-overlapping T=8 tiling covers every frame).
            # Frames whose clip is missing → 0 + valid=0 (masked in the distill loss).
            from .teacher_cache import load_teacher_clip_cache
            result["teacher_depth"], result["teacher_valid"] = load_teacher_clip_cache(
                self.teacher_cache_dir, video_id, indices, H, W, teacher=self.teacher_name)
        if self.degrade:
            # Deterministic degradation for val so metrics are comparable across runs.
            rng = np.random.RandomState(self.seed + idx) if self.split != "train" else np.random.RandomState()
            sev = self.degrade_severity
            # degrade_mode may be a list — sample per clip during training
            mode = rng.choice(self.degrade_mode) if isinstance(self.degrade_mode, list) \
                else self.degrade_mode
            if sev is None and self.split == "train" and mode != "elvis":
                sev = float(rng.uniform(0.0, float(self.severity_max[0])))  # curriculum cap
                # elvis: train samples the full Φ uniformly (severity stays None).
            degraded, params = _degrade_clip(frames, rng, mode, severity=sev)
            result["degraded_frames"] = degraded.permute(1, 0, 2, 3).contiguous()
            result["degrade_params"] = torch.from_numpy(params)
        return result

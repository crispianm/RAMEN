"""Per-frame depth-change statistics for a STATIC clip (one still repeated N times, fresh noise per frame), numpy only.

The scorer's ``bench_joint.flicker_static`` (eval clone @368b5c53) is, for t0 = 8 warm-up frames:
    flicker = mean_valid_px [ mean_{t=t0..N-1} |D_t - D_{t-1}| / Dbar ],   Dbar = mean_{t=t0-1..N-1} D_t   (per pixel)
            = mean_{t=t0..N-1} f_t   with   f_t = mean_valid_px |D_t - D_{t-1}| / Dbar        (linear, so it is exactly recoverable from the series f_t)
This module computes, for every pair t = 1..N-1, a family of per-frame statistics on two depth versions:
  'un'  = metric depth as scored by flicker_static (resized to the GT size, clamped to [min_depth, max_depth])
  'al'  = the same depth after the scorer's per-clip least-squares (scale, shift) on DISPARITY against the (constant) GT, exactly as score_clip
and four reductions: |dD| (metres), |dD|/D_{t-1}, |dD|/D_t, |dD|/Dbar (scorer's normaliser; 'un' only), each over (a) the valid-GT pixels ('v') and (b) all pixels ('a').
HEADLINE DEFINITION (FIXED by the user, relayed by the Manager 2026-10-12; the plot and the README headline use ONLY this one): per clip and per frame t >= 1,
    mean over pixels valid in BOTH frames t and t-1 of  |d_t - d_{t-1}| / d_t * 100     (relative change, % of depth)
on the model's UNALIGNED metric depth in metres (for the post-filtered curve: the post-filtered output), exactly as the scorer sees it: resized to the GT size and clamped to
[min_depth, max_depth] (bench_joint.flicker_static). Valid = the benchmark's GT-valid mask min_depth < gt < max_depth, required in both frames (on a static clip the GT is one
repeated still, so "valid in both" equals the single GT mask; the pair mask is nevertheless computed explicitly and asserted equal in the tests). Then averaged over the clips at each frame index.
All other variants stay in the dump for recomputation (they are pure functions of the dumped series) but are not used for the headline.
"""
from __future__ import annotations

import numpy as np

T0 = 8                                       # bench_joint.STATIC_T0 (warm-up frames of the scorer)

# ----------------------------------------------------------------------------- definitions (swappable)
# name -> (statistic key in the dump, one-line meaning). "al" = scale-aligned (scorer's per-clip LS fit), "un" = unaligned metric depth as scored.
DEFINITIONS = {
    "rel_cur_pct_unaligned_pairmasked": ("relcurpct_un_pv", "HEADLINE (user-fixed): mean over px valid in both frames of |D_t - D_{t-1}| / D_t * 100, unaligned metric depth as scored (clamped)"),
    "rel_cur_pct_unclamped_pairmasked": ("relcurpct_raw_pv", "diagnostic only: as the headline but on the resized, UNCLAMPED model output (floored at 1e-3 m)"),
    "abs_aligned_masked":   ("abs_al_v",     "mean over valid-GT px of |D_t - D_{t-1}|, scale-aligned depth (metres)   [alternative; not the headline]"),
    "abs_unaligned_masked": ("abs_un_v",     "mean over valid-GT px of |D_t - D_{t-1}|, unaligned metric depth (metres)"),
    "rel_prev_aligned_masked":   ("relprev_al_v", "mean over valid-GT px of |D_t - D_{t-1}| / D_{t-1}, scale-aligned depth"),
    "rel_prev_unaligned_masked": ("relprev_un_v", "mean over valid-GT px of |D_t - D_{t-1}| / D_{t-1}, unaligned metric depth"),
    "rel_cur_unaligned_masked":  ("relcur_un_v",  "mean over valid-GT px of |D_t - D_{t-1}| / D_t (the change-map formula), unaligned"),
    "rel_scorer_unaligned_masked": ("reldbar_un_v", "mean over valid-GT px of |D_t - D_{t-1}| / Dbar (the scorer's own per-pixel normaliser; its mean over t >= t0 IS the scored 'flicker')"),
    "abs_aligned_unmasked": ("abs_al_a",     "as the default but over ALL pixels (no GT mask)"),
    "abs_unaligned_unmasked": ("abs_un_a",   "unaligned, all pixels"),
    "rel_prev_aligned_unmasked": ("relprev_al_a", "|dD|/D_{t-1}, aligned, all pixels"),
    "rel_prev_unaligned_unmasked": ("relprev_un_a", "|dD|/D_{t-1}, unaligned, all pixels"),
}
DEFAULT_DEFINITION = "rel_cur_pct_unaligned_pairmasked"      # the user-fixed headline definition
HEADLINE_DEFINITION = DEFAULT_DEFINITION


def prep_depth(depth: np.ndarray, gt_np: np.ndarray, max_depth: float, min_depth: float):
    """As flicker_static / score_clip: resize to the GT size, float64. Returns (d_clamped_for_flicker, d_raw_resized, g0, valid_flicker, valid_fit)."""
    import cv2
    H, W = gt_np.shape[-2:]
    if depth.shape[-2:] != (H, W):
        depth = np.stack([cv2.resize(d, (W, H)) for d in depth])
    d_raw = depth.astype(np.float64)
    d = np.clip(d_raw, min_depth, max_depth)
    g0 = gt_np[0].astype(np.float64)
    return d, d_raw, g0, (g0 > min_depth) & (g0 < max_depth), (gt_np > 1e-3) & (gt_np < max_depth)


def fit_aligned(d_raw: np.ndarray, gt_np: np.ndarray, valid_fit: np.ndarray, max_depth: float):
    """score_clip's alignment (metric model): one least-squares (scale, shift) on disparity per clip (normal equations, fp64), over all frames' valid pixels.
    Returns (aligned depth clamped to [1e-3, max_depth], scale, shift)."""
    disp = (1.0 / np.maximum(d_raw, 1e-3))
    disp = np.maximum(disp, 1e-3)
    gd = np.where(valid_fit, 1.0 / (gt_np.astype(np.float64) + 1e-8), 0.0)
    p, t = disp[valid_fit], gd[valid_fit]
    n = float(p.size)
    A = np.array([[(p * p).sum(), p.sum()], [p.sum(), n]])
    try:
        s, b = np.linalg.solve(A, np.array([(p * t).sum(), t.sum()]))
    except np.linalg.LinAlgError:                      # degenerate (constant) prediction: the scorer's torch.linalg.solve would raise too; keep the dump alive with a least-squares fit
        s, b = np.linalg.lstsq(A, np.array([(p * t).sum(), t.sum()]), rcond=None)[0]
    al = 1.0 / np.maximum(s * disp + b, 1e-3)
    return np.clip(al, 1e-3, max_depth), float(s), float(b)


def _series(dd: np.ndarray, valid: np.ndarray, dbar: np.ndarray | None, suffix: str, out: dict):
    N = dd.shape[0]
    keys = ["abs", "relprev", "relcur"] + (["reldbar"] if dbar is not None else [])
    for k in keys:
        for m in ("v", "a"):
            out.setdefault(f"{k}_{suffix}_{m}", np.full(N - 1, np.nan))
    for t in range(1, N):
        diff = np.abs(dd[t] - dd[t - 1])
        parts = {"abs": diff, "relprev": diff / dd[t - 1], "relcur": diff / dd[t]}
        if dbar is not None:
            parts["reldbar"] = diff / dbar
        for k, a in parts.items():
            out[f"{k}_{suffix}_v"][t - 1] = a[valid].mean()
            out[f"{k}_{suffix}_a"][t - 1] = a.mean()


def clip_series(depth: np.ndarray, gt_np: np.ndarray, max_depth: float, min_depth: float, t0: int = T0) -> dict:
    """Everything for ONE static clip. ``depth`` (N,H0,W0) metric metres, ``gt_np`` (N,H,W). Series have length N-1 (index i <-> frame t = i+1)."""
    d, d_raw, g0, valid, valid_fit = prep_depth(depth, gt_np, max_depth, min_depth)
    out: dict = {}
    dbar = d[t0 - 1:].mean(0)
    _series(d, valid, dbar, "un", out)
    al, s, b = fit_aligned(d_raw, gt_np, valid_fit, max_depth)
    _series(al, valid, None, "al", out)
    # headline (user-fixed) series: pixels valid in BOTH frames t and t-1 (per-frame GT validity), relative change in % of D_t
    N = d.shape[0]
    vt = (gt_np > min_depth) & (gt_np < max_depth)
    draw_f = np.maximum(d_raw, 1e-3)
    out["relcurpct_un_pv"] = np.full(N - 1, np.nan); out["relcurpct_raw_pv"] = np.full(N - 1, np.nan); out["n_pair_valid_px"] = np.zeros(N - 1)
    for t in range(1, N):
        vp = vt[t] & vt[t - 1]
        out["n_pair_valid_px"][t - 1] = vp.sum()
        out["relcurpct_un_pv"][t - 1] = 100.0 * (np.abs(d[t] - d[t - 1]) / d[t])[vp].mean()
        out["relcurpct_raw_pv"][t - 1] = 100.0 * (np.abs(draw_f[t] - draw_f[t - 1]) / draw_f[t])[vp].mean()
    out["fit_scale"], out["fit_shift"] = np.float64(s), np.float64(b)
    out["n_valid_px"] = np.float64(valid.sum()); out["n_px"] = np.float64(valid.size)
    return out


def flicker_from_series(series: dict, t0: int = T0) -> float:
    """The scorer's ``flicker`` recovered from the per-frame series: mean over pairs t = t0..N-1 of the Dbar-normalised masked series (index t-1)."""
    return float(np.mean(series["reldbar_un_v"][t0 - 1:]))


def curve(per_clip: list[dict], definition: str = DEFAULT_DEFINITION):
    """Per-frame-index mean over clips of the chosen definition: (frame_index t = 1..N-1, mean, sem, n_clips)."""
    key = DEFINITIONS[definition][0]
    M = np.stack([c[key] for c in per_clip])
    t = np.arange(1, M.shape[1] + 1)
    return t, M.mean(0), (M.std(0, ddof=1) / np.sqrt(M.shape[0]) if M.shape[0] > 1 else np.zeros(M.shape[1])), M.shape[0]

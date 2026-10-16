"""DN-01 probe package (scoring only, DEV only): causal post-hoc depth filters beside ``ema<a>``, the flicker diagnostic, and the
grid machinery that scores many post-filter variants from ONE model pass. Nothing here trains or changes a model.

PRE-DECLARED (2026-09-30, before any result exists; Auditor conditions 1-9 of the same day):
  * Every filter is CAUSAL (frame t sees frames <= t only), acts on the depth model's stitched output in the SAME domain as ``ema`` (metres for a
    metric model, disparity for a relative one), and uses DEPLOYABLE single-stream signals only. Clean frames and variance over noise draws are never inputs.
  * ``agf<amin>t<tau>``  adaptive-gain EMA. u_t = (d_t - y_{t-1}) / |y_{t-1}| (relative local temporal difference); s_t = B s_{t-1} + (1-B) u_t^2 (B = AGF_BETA,
    a short running innovation variance); per-pixel gain a_t = amin + (1-amin)(1 - exp(-s_t / tau^2)); y_t = a_t d_t + (1-a_t) y_{t-1}. Small persistent
    differences (noise) -> gain ~ amin (heavy smoothing); large ones (real change) -> gain -> 1 (follow). ``agf1.0t<tau>`` (amin = 1) is the exact identity.
  * ``sgf<r>e<eps>[l<lam>]<g>``  RGB-guided causal spatio-temporal guided filter (He et al.'s guided filter with the moments (I, p, Ip, II) averaged by a
    causal EMA over frames with weight ``lam`` on the new frame, default SGF_LAMBDA). Radius r (box, border-clipped), regulariser eps in the units of the
    NORMALISED guide, guide g = ``i`` the depth model's input frames (the noisy frames) or ``r`` the model's restored frames. For a model with no restorer
    ``r`` IS ``i`` (restored := input), so the two guides are the same run and are labelled so. The guide is the luma, resized to the depth size and
    normalised per clip: (Y - median) / (P95 - P5), never the clean frame. ``sgf0e<eps>l1.0<g>`` (r = 0, no temporal averaging) is the exact identity.
  * FRONTIERS (grids fixed here, <= 12 settings per filter): EMA alpha 0.3..0.9 step 0.1; AGF amin {0.2, 0.3, 0.5} x tau {0.03, 0.06, 0.12, 0.25};
    SGF r {2, 4, 8} x eps {0.001, 0.01, 0.1, 1} (guide ``r``), the same 12 on top of ema0.7 (``sgf..+ema0.7``), and guide ``i`` at r = 4 (4 settings, joint models).
  * HEADROOM RULE (for the Manager / Auditor to confirm; written before results): a filter shows headroom over EMA only if, at MATCHED night aligned AbsRel
    (within 0.5% of an EMA setting on the frontier) and matched BVI level and motion-cell lag_k, its F_noise or static flicker is below that EMA setting's on
    the stock_ctrl seed pair AND on r7 x3, DEV only. A single best setting is never reported without its frontier.
"""
from __future__ import annotations

import os
import re

import numpy as np
import torch
import torch.nn.functional as F

AGF_BETA = 0.5           # running innovation-variance memory of agf
SGF_LAMBDA = 0.5         # weight of the NEW frame in sgf's causal moment EMA (ema-equivalent alpha for the statistics)
LUMA = (0.299, 0.587, 0.114)
_NUM = r"\d+(?:\.\d+)?"
# E2 (2026-09-30): ``rgf<amin>t<tau>k<kappa><g>`` RGB-gated agf; a trailing ``c`` on an sgf token = the CAUSAL running guide normalisation (the plain sgf tokens keep the per-clip one)
RGF_RADIUS = 4           # box radius (px, at the depth size) of the local RGB innovation
RGF_NOISE_LAM = 0.1      # weight of the new frame in the running RGB-innovation noise floor
CAUSAL_LAM = 0.1         # weight of the new frame in the running guide statistics (median / P5 / P95) of the causal normalisation
PROBE_RE = re.compile(rf"agf(?P<amin>{_NUM})t(?P<tau>{_NUM})|sgf(?P<r>\d+)e(?P<eps>{_NUM})(?:l(?P<lam>{_NUM}))?(?P<g>[ir])(?P<c>c)?"
                      rf"|rgf(?P<ra>{_NUM})t(?P<rt>{_NUM})k(?P<rk>{_NUM})(?P<rg>[ir])")

EMA_ALPHAS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
AGF_GRID = tuple(f"agf{a}t{t}" for a in (0.2, 0.3, 0.5) for t in (0.03, 0.06, 0.12, 0.25))
SGF_GRID = tuple(f"sgf{r}e{e}r" for r in (2, 4, 8) for e in (0.001, 0.01, 0.1, 1))
GROUPS = {
    "none": ("none",),
    "ema": tuple(f"ema{a}" for a in EMA_ALPHAS),
    "agf": AGF_GRID,
    "sgf": SGF_GRID,
    "sgfe": tuple(f"{v}+ema0.7" for v in SGF_GRID),
    "sgfi": tuple(f"sgf4e{e}i" for e in (0.001, 0.01, 0.1, 1)),
    "soup": ("none", "ema0.7"),
    "repro": ("none", "ema0.7"),
    # E4 (audit/E4_predeclared_20261001.md): the declared variants, agf0.2t0.06 primary / agf0.2t0.03 secondary, none and ema0.7 descriptive; e4ema = the rest of the EMA frontier (optional, E4_EMA=1)
    "e4": ("none", "ema0.7", "agf0.2t0.06", "agf0.2t0.03"),
    "e4ema": tuple(f"ema{a}" for a in EMA_ALPHAS if a != 0.7),
    # E2: RGB-gated agf (amin 0.2; depth-innovation scale tau {0.06, 0.12, 0.25} x RGB-excess scale kappa {0.5, 1, 2, 4}; guide = restored frames) = 12; rgfi = the input-frame guide, tau 0.12 (joint models)
    "rgf": tuple(f"rgf0.2t{t}k{k}r" for t in (0.06, 0.12, 0.25) for k in (0.5, 1, 2, 4)),
    "rgfi": tuple(f"rgf0.2t0.12k{k}i" for k in (0.5, 1, 2, 4)),
    "sgfc": tuple(f"sgf{r}e{e}rc" for r in (2, 4, 8) for e in (0.001, 0.01, 0.1, 1)),       # the causal-normalisation twin of the sgf grid (only if an sgf setting ever clears the rule)
    # F3 (audit/F3_predeclared_20261001.md Q2): the blind temporal-consistency post-process (Deep Video Prior, bench_joint token `dvp`, NON-causal; evaluation/f3_baselines.py) alone and
    # with the stage after it (descriptive). Not a probe filter: `dvp` is routed by bench_joint.DEPTH_POST_RE group 6.
    "f3blind": ("dvp", "dvp+agf0.2t0.06", "dvp+agf0.2t0.03"),
}


# protocol v3 WP24: ``RAMEN_AGF_BACKEND`` = ``cpu`` (default: the numpy REFERENCE below, every existing result) or ``gpu`` (``depth_agf_torch``: the same recursion in float64 on the input's device, no host copy for a CUDA tensor).
AGF_BACKEND = os.environ.get("RAMEN_AGF_BACKEND", "cpu")


def is_probe_token(tok: str) -> bool:
    return PROBE_RE.fullmatch(tok) is not None


def depth_agf(depth: np.ndarray, amin: float, tau: float, beta: float = AGF_BETA) -> np.ndarray:
    """Causal adaptive-gain EMA (see the module docstring). ``amin`` = 1 returns ``depth`` unchanged, bit for bit."""
    out = np.empty_like(depth)
    acc = depth[0].astype(np.float64)
    out[0] = acc
    s = np.zeros_like(acc)
    for t in range(1, depth.shape[0]):
        d = depth[t].astype(np.float64)
        u = (d - acc) / np.maximum(np.abs(acc), 1e-6)
        s = beta * s + (1.0 - beta) * u * u
        a = amin + (1.0 - amin) * (1.0 - np.exp(-s / (tau * tau)))
        acc = a * d + (1.0 - a) * acc
        out[t] = acc
    return out


@torch.no_grad()
def depth_agf_torch(depth, amin: float, tau: float, beta: float = AGF_BETA):
    """``depth_agf`` on a torch device (protocol v3 WP24): the SAME recursion, in float64 like the reference, over a ``(N,H,W)`` tensor on its own device (a CUDA input stays on the GPU: no host copy) or a numpy array
    (copied to CUDA and back). Output dtype = input dtype. Differs from ``depth_agf`` only by the last bits of exp (libm vs CUDA) and the final cast to the depth dtype: |diff| << 1e-4 m (tested on random and real
    clips). ``amin`` = 1 returns ``depth`` unchanged, bit for bit, as the reference does."""
    is_np = isinstance(depth, np.ndarray)
    x = torch.from_numpy(np.ascontiguousarray(depth)).to("cuda" if torch.cuda.is_available() else "cpu") if is_np else depth
    out = torch.empty_like(x)
    acc = x[0].double()
    out[0] = acc
    s = torch.zeros_like(acc)
    for t in range(1, x.shape[0]):
        d = x[t].double()
        u = (d - acc) / acc.abs().clamp_min(1e-6)
        s = beta * s + (1.0 - beta) * u * u
        a = amin + (1.0 - amin) * (1.0 - torch.exp(-s / (tau * tau)))
        acc = a * d + (1.0 - a) * acc
        out[t] = acc
    return out.cpu().numpy() if is_np else out


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean over the (2r+1)^2 window clipped at the borders, (N,H,W) float64, by two separable cumulative sums (O(1) per pixel). r = 0 -> x."""
    if r == 0:
        return x

    def one(x: torch.Tensor, dim: int) -> torch.Tensor:
        n = x.shape[dim]
        c = torch.cumsum(x, dim)
        z = torch.zeros_like(c.narrow(dim, 0, 1))
        c = torch.cat([z, c], dim)
        i = torch.arange(n, device=x.device)
        hi, lo = (i + r + 1).clamp(max=n), (i - r).clamp(min=0)
        shape = [1, 1, 1]
        shape[dim] = n
        return (c.index_select(dim, hi) - c.index_select(dim, lo)) / (hi - lo).to(x.dtype).view(shape)
    return one(one(x, 1), 2)


def guide_luma(frames: torch.Tensor, hw: tuple[int, int]) -> np.ndarray:
    """(N,3,h,w) frames in [0,1] -> (N,H,W) float64 luma resized to ``hw`` and normalised per clip: (Y - median) / (P95 - P5) on a strided sample."""
    y = (frames.detach().double().cpu() * torch.tensor(LUMA, dtype=torch.float64).view(1, 3, 1, 1)).sum(1, keepdim=True)
    if tuple(y.shape[-2:]) != tuple(hw):
        y = F.interpolate(y, size=hw, mode="bilinear", align_corners=False)
    y = y[:, 0].numpy()
    sub = y[:, ::4, ::4]
    lo, med, hi = np.percentile(sub, [5, 50, 95])
    return (y - med) / (hi - lo + 1e-6)


def guide_luma_causal(frames: torch.Tensor, hw: tuple[int, int], lam: float = CAUSAL_LAM) -> np.ndarray:
    """Like guide_luma but the normalisation is CAUSAL: frame t is normalised by running statistics (median, P5, P95 of each frame's strided luma, each smoothed by an EMA with
    weight ``lam`` on the new frame, first frame = its own), so frame t never sees a later frame (the per-clip version does, which the Auditor disclosed)."""
    y = (frames.detach().double().cpu() * torch.tensor(LUMA, dtype=torch.float64).view(1, 3, 1, 1)).sum(1, keepdim=True)
    if tuple(y.shape[-2:]) != tuple(hw):
        y = F.interpolate(y, size=hw, mode="bilinear", align_corners=False)
    y = y[:, 0].numpy()
    lo, med, hi = np.percentile(y[:, ::4, ::4].reshape(y.shape[0], -1), [5, 50, 95], axis=1)
    for v in (lo, med, hi):
        for t in range(1, len(v)):
            v[t] = lam * v[t] + (1.0 - lam) * v[t - 1]
    return (y - med[:, None, None]) / (hi - lo + 1e-6)[:, None, None]


def depth_rgf(depth: np.ndarray, guide: np.ndarray, amin: float, tau: float, kappa: float, beta: float = AGF_BETA) -> np.ndarray:
    """RGB-gated adaptive-gain EMA. As depth_agf, but the gain also follows the temporal innovation of the GUIDE frames: w_t = box_RGF_RADIUS |G_t - G_{t-1}| (local), its running
    noise floor m_t = EMA of the frame's spatial median of w (first = w_1's), excess ex_t = max(w_t / m_t - 1, 0) (in units of the typical frame-to-frame change, ~0 where only noise
    moves), and a_t = amin + (1 - amin)(1 - exp(-(s_t / tau^2 + ex_t^2 / kappa^2))). Real scene change moves RGB; depth noise does not. ``guide`` (N,H,W) must be causally normalised
    (guide_luma_causal). Causal in depth AND guide. ``amin`` = 1 returns ``depth`` unchanged, bit for bit."""
    out = np.empty_like(depth)
    acc = depth[0].astype(np.float64)
    out[0] = acc
    s = np.zeros_like(acc)
    m = None
    g = guide.astype(np.float64)
    for t in range(1, depth.shape[0]):
        d = depth[t].astype(np.float64)
        u = (d - acc) / np.maximum(np.abs(acc), 1e-6)
        s = beta * s + (1.0 - beta) * u * u
        w = _box(torch.from_numpy(np.abs(g[t] - g[t - 1]))[None], RGF_RADIUS)[0].numpy()
        med = float(np.median(w[::4, ::4]))
        m = med if m is None else RGF_NOISE_LAM * med + (1.0 - RGF_NOISE_LAM) * m
        ex = np.maximum(w / (m + 1e-6) - 1.0, 0.0)
        a = amin + (1.0 - amin) * (1.0 - np.exp(-(s / (tau * tau) + ex * ex / (kappa * kappa))))
        acc = a * d + (1.0 - a) * acc
        out[t] = acc
    return out


def depth_sgf(depth: np.ndarray, guide: np.ndarray, r: int, eps: float, lam: float = SGF_LAMBDA) -> np.ndarray:
    """Causal spatio-temporal guided filter; ``guide`` (N,H,W) already normalised (guide_luma). r = 0 and lam = 1 return ``depth`` unchanged, bit for bit."""
    p = torch.from_numpy(np.ascontiguousarray(depth)).double()
    g = torch.from_numpy(np.ascontiguousarray(guide)).double()
    m = [_box(g, r), _box(p, r), _box(g * p, r), _box(g * g, r)]          # I, p, Ip, II
    if lam < 1.0:                                                            # causal EMA of the moments over frames
        for k in range(4):
            acc = m[k][0].clone()
            for t in range(1, m[k].shape[0]):
                acc = lam * m[k][t] + (1.0 - lam) * acc
                m[k][t] = acc
    mI, mp, mIp, mII = m
    a = (mIp - mI * mp) / (mII - mI * mI + eps)
    b = mp - a * mI
    q = _box(a, r) * g + _box(b, r)
    return q.numpy().astype(depth.dtype, copy=False)


def make_post(token: str):
    """A probe token -> fn(depth (N,H,W) numpy, guides {'i': frames, 'r': frames}) -> (N,H,W) numpy; ``fn.guide`` names the guide ('' for depth-only)."""
    m = PROBE_RE.fullmatch(token)
    if m is None:
        raise ValueError(f"not a probe token: {token}")
    if token.startswith("agf"):
        amin, tau = float(m["amin"]), float(m["tau"])
        if not 0.0 < amin <= 1.0 or tau <= 0.0:
            raise ValueError(f"agf needs 0 < amin <= 1 and tau > 0: {token}")

        def fn(depth, guides=None):
            if AGF_BACKEND == "gpu" or not isinstance(depth, np.ndarray):          # protocol v3 WP24: the device implementation (a tensor input always takes it); default = the numpy reference
                return depth_agf_torch(depth, amin, tau)
            return depth_agf(depth, amin, tau)
        fn.guide = ""
        return fn
    if token.startswith("rgf"):
        amin, tau, kappa, which = float(m["ra"]), float(m["rt"]), float(m["rk"]), m["rg"]
        if not 0.0 < amin <= 1.0 or tau <= 0.0 or kappa <= 0.0:
            raise ValueError(f"rgf needs 0 < amin <= 1, tau > 0, kappa > 0: {token}")

        def fn(depth, guides=None):
            if guides is None or which not in guides:
                raise ValueError(f"{token} needs the '{which}' guide frames (run_pipeline provides them)")
            return depth_rgf(depth, guide_luma_causal(guides[which], depth.shape[-2:]), amin, tau, kappa)
        fn.guide = which
        return fn
    r, eps, lam = int(m["r"]), float(m["eps"]), float(m["lam"]) if m["lam"] else SGF_LAMBDA
    if eps <= 0.0 or not 0.0 < lam <= 1.0:
        raise ValueError(f"sgf needs eps > 0 and 0 < lam <= 1: {token}")
    which, norm = m["g"], (guide_luma_causal if m["c"] else guide_luma)

    def fn(depth, guides=None):
        if guides is None or which not in guides:
            raise ValueError(f"{token} needs the '{which}' guide frames (run_pipeline provides them)")
        return depth_sgf(depth, norm(guides[which], depth.shape[-2:]), r, eps, lam)
    fn.guide = which
    return fn


# ----------------------------------------------------------------------------- flicker diagnostic (item 1)

SEAM_STRIDE = 22          # VDA window stride: INFER_LEN 32 - OVERLAP 10
SPEC_BANDS = ((0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.5001))     # cycles / frame, DC excluded


def flicker_diag(depth: np.ndarray, t0: int = 8, px: int = 4) -> dict:
    """Decomposes the temporal residual of a STATIC-scene depth video (all change is flicker) in log depth: R_t = L_t - mean_t L (frames >= t0, every
    ``px``-th pixel); the per-frame scale mode c_t = median_xy R_t; structure S_t = R_t - c_t. Returns variance shares, mean |frame step| of each part,
    the step profile by position in the VDA window stride (seams), the structure variance by far (top-quartile temporal-mean depth) vs near pixels and by
    image thirds, and the temporal power spectrum of S and of c in SPEC_BANDS (shares of non-DC power)."""
    L = np.log(np.clip(depth[t0:, ::px, ::px].astype(np.float64), 1e-3, None))
    n = L.shape[0]
    mean = L.mean(0)
    R = L - mean
    c = np.median(R.reshape(n, -1), axis=1)
    S = R - c[:, None, None]
    var = lambda z: float((z * z).mean())
    out = {"var_total": var(R), "var_scale": var(c), "var_struct": var(S), "frames": int(n)}
    out["scale_share"] = out["var_scale"] / max(out["var_total"], 1e-30)
    step = lambda z: float(np.abs(np.diff(z, axis=0)).mean())
    out.update(step_total=step(R), step_scale=step(c), step_struct=step(S))
    pos = (np.arange(t0, t0 + n - 1) % SEAM_STRIDE)
    stepR = np.abs(np.diff(R, axis=0)).mean((1, 2))
    prof = [float(stepR[pos == k].mean()) if (pos == k).any() else None for k in range(SEAM_STRIDE)]
    out["seam_profile"] = prof
    vals = [v for v in prof if v is not None]
    out["seam_peak_over_median"] = float(max(vals) / max(float(np.median(vals)), 1e-30))
    far = mean >= np.quantile(mean, 0.75)
    out["struct_var_far"] = float((S ** 2)[:, far].mean()) if far.any() else None
    out["struct_var_near"] = float((S ** 2)[:, ~far].mean()) if (~far).any() else None
    h = S.shape[1]
    out["struct_var_thirds"] = [float((S[:, i * h // 3:(i + 1) * h // 3] ** 2).mean()) for i in range(3)]
    for name, z in (("struct", S.reshape(n, -1)[:, ::7]), ("scale", c[:, None])):
        P = (np.abs(np.fft.rfft(z, axis=0)) ** 2).mean(1)
        fr = np.fft.rfftfreq(n)
        tot = max(float(P[1:].sum()), 1e-30)
        out[f"spectrum_{name}"] = [float(P[(fr >= lo) & (fr < hi) & (fr > 0)].sum() / tot) for lo, hi in SPEC_BANDS]
    return out


# ----------------------------------------------------------------------------- variants (one model pass, many post-filters)

def expand_grid(items) -> list[str]:
    """Group names (GROUPS) and literal variants -> the ordered, de-duplicated variant list ('none' = the unsmoothed model)."""
    out: list[str] = []
    for it in items:
        out += list(GROUPS[it]) if it in GROUPS else [it]
    return list(dict.fromkeys(out))


def parse_variant(name: str):
    """'none' -> []; 'ema0.7', 'agf0.3t0.06', 'sgf4e0.1r+ema0.7' -> the list of post tokens (order kept)."""
    return [] if name == "none" else name.split("+")


def variant_spec(model: str, name: str) -> str:
    """The model string a SINGLE-token run of this variant would carry (its result file name and 'model' field)."""
    return model if name == "none" else f"{model}+{name}"


def dump_look(path, depths: dict, guide_frames: torch.Tensor, gt: np.ndarray, n: int = 16) -> None:
    """Save the first ``n`` frames of each variant's depth (half size, float16), the guide luma (uint8) and GT (float16) for the ghosting look."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    hw = next(iter(depths.values())).shape[-2:]
    g = np.clip(guide_luma(guide_frames[:n], hw) * 0.5 + 0.5, 0, 1)
    np.savez_compressed(path, guide=(g * 255).astype(np.uint8), gt=gt[:n].astype(np.float16),
                        **{f"depth::{v}": d[:n, ::2, ::2].astype(np.float16) for v, d in depths.items()})


class MemoTOF:
    """Wraps evaluation.temporal_metrics.TOF for grid scoring: the flow of (restored|degraded, clean) does not depend on the depth post-filter, so it is
    computed once per clip and reused by every variant. Exact: a hit needs the same first tensor and an EXACTLY equal second tensor. Call ``new_clip()`` per clip."""

    def __init__(self, tof):
        self.tof, self.cache = tof, []

    def new_clip(self) -> None:
        self.cache = []

    def __call__(self, a: torch.Tensor, b: torch.Tensor):
        for pa, sb, res in self.cache:
            if pa == a.data_ptr() and sb.shape == b.shape and torch.equal(sb, b):
                return res
        res = self.tof(a, b)
        self.cache.append((a.data_ptr(), b.detach().clone(), res))
        return res

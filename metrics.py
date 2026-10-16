"""Depth metrics with explicit alignment and invalid-prediction semantics.

The standard scale-shift-aligned video-depth evaluation protocol (per-video
affine alignment), so D3 numbers are directly comparable across models:

  1. ``align_disparity_to_depth``: the model output is affine-invariant disparity;
     solve ``s·pred + t ≈ 1/GT`` (per-sample, shared over the clip), invert to depth.
  2. ``compute_depth_metrics``: AbsRel / δ1 / RMSE / SI-log on the aligned depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


METRIC_KEYS = (
    "abs_rel",
    "sq_rel",
    "rmse",
    "rmse_log",
    "si_log",
    "delta1",
    "delta2",
    "delta3",
    "si_log_pct",
)


@dataclass(frozen=True)
class AffineFit:
    """Least-squares fit ``scale * prediction + shift ~= target``."""

    scale: float
    shift: float
    valid_count: int
    solved: bool


@dataclass(frozen=True)
class ScaleFit:
    """Median-ratio scale-only fit in depth space: ``scale = median(target)/median(pred)``,
    no shift term. The classic Eigen/Zhou "median scaling" protocol, as distinct from
    :class:`AffineFit`'s scale+shift solve in disparity space."""

    scale: float
    valid_count: int
    solved: bool


def fit_median_scale(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> ScaleFit:
    """Fit one depth-space scale ratio over all selected elements without mutating inputs."""
    valid = torch.isfinite(pred) & torch.isfinite(target) & (pred > 0)
    if mask is not None:
        valid &= mask.bool()
    p, g = pred.float()[valid], target.float()[valid]
    n = p.numel()
    if n < 1:
        return ScaleFit(1.0, n, False)
    scale = float(g.median() / p.median())
    if not torch.isfinite(torch.tensor(scale)) or scale <= 0:
        return ScaleFit(1.0, n, False)
    return ScaleFit(scale, n, True)


def fit_scale_shift(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-9,
) -> AffineFit:
    """Fit one affine transform over all selected elements without mutating inputs."""
    valid = torch.isfinite(pred) & torch.isfinite(target)
    if mask is not None:
        valid &= mask.bool()
    p, g = pred.float()[valid], target.float()[valid]
    n = p.numel()
    if n < 2:
        return AffineFit(1.0, 0.0, n, False)
    a00, a01, a11 = (p * p).sum(), p.sum(), p.new_tensor(float(n))
    b0, b1 = (p * g).sum(), g.sum()
    det = a00 * a11 - a01.square()
    if not torch.isfinite(det) or det.abs() <= eps:
        return AffineFit(1.0, 0.0, n, False)
    scale = (a11 * b0 - a01 * b1) / det
    shift = (a00 * b1 - a01 * b0) / det
    return AffineFit(float(scale), float(shift), n, True)


def align_scale_shift(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-sample affine align ``s·pred + t`` to ``target`` (shared over the clip)."""
    if mask is None:
        valid = torch.isfinite(target) & torch.isfinite(pred)
    else:
        valid = mask.bool() & torch.isfinite(target) & torch.isfinite(pred)

    aligned = pred.clone()
    for b in range(pred.shape[0]):
        vb = valid[b]
        if vb.sum() < 2:
            continue
        fit = fit_scale_shift(pred[b], target[b], vb, eps)
        if fit.solved:
            aligned[b] = fit.scale * pred[b] + fit.shift
    return aligned


def garg_crop_mask(height: int, width: int) -> torch.Tensor:
    """Boolean valid-region mask from Garg et al. (2016), ``Unsupervised CNN for
    Single View Depth Estimation``. Excludes the untrustworthy image borders
    (sky, car hood, unannotated edges) before scoring on KITTI. Bounds match the
    ``monodepth2``/``BTS``/``AdaBins``-lineage evaluation scripts nearly every
    subsequent KITTI depth paper reuses verbatim."""
    mask = torch.zeros((height, width), dtype=torch.bool)
    y0, y1 = int(0.40810811 * height), int(0.99189189 * height)
    x0, x1 = int(0.03594771 * width), int(0.96405229 * width)
    mask[y0:y1, x0:x1] = True
    return mask


def eigen_crop_mask(height: int, width: int, dataset: str = "kitti") -> torch.Tensor:
    """Boolean valid-region mask from Eigen et al. (2014), ``Depth Map Prediction
    from a Single Image using a Multi-Scale Deep Network``.

    This is **not** the same region as :func:`garg_crop_mask` -- the two crops
    coexist as distinct branches in every upstream script this repo compares
    against, e.g. ``competition/DepthAnythingAC/evaluate_depth.py`` lines 162-172
    and ``competition/unidepth_repo/unidepth/datasets/kitti.py`` lines 154-165,
    which select between them with a ``garg_crop``/``eigen_crop`` flag. The Eigen
    crop keeps a taller-but-higher band of the image (top 33.2% to 91.4%) where
    Garg keeps 40.8% to 99.2%.

    For non-KITTI datasets the convention is a fixed pixel box defined on
    640x480 NYU frames, so ``dataset`` selects between the two forms.
    """
    mask = torch.zeros((height, width), dtype=torch.bool)
    if dataset == "kitti":
        y0, y1 = int(0.3324324 * height), int(0.91351351 * height)
        x0, x1 = int(0.0359477 * width), int(0.96405229 * width)
        mask[y0:y1, x0:x1] = True
    else:
        if (height, width) != (480, 640):
            raise ValueError(
                f"the non-KITTI Eigen crop is defined only for 480x640 frames, got {height}x{width}"
            )
        mask[45:471, 41:601] = True
    return mask


def align_disparity_to_depth(
    pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    min_depth: float = 1e-3,
) -> torch.Tensor:
    """Align affine-invariant disparity to metric GT: ``s·pred+t ≈ 1/GT`` → depth."""
    disp_gt = 1.0 / depth_gt.clamp(min=min_depth)
    aligned_disp = align_scale_shift(pred, disp_gt, mask=mask, eps=eps)
    return 1.0 / aligned_disp.clamp(min=min_depth)


@torch.no_grad()
def compute_depth_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    invalid_pred: str = "exclude",
) -> Dict[str, float]:
    """Compute standard metrics.

    ``invalid_pred='exclude'`` preserves the historical API.  For alignment-free
    reporting use ``'penalize'``: validity is determined only by GT/mask and every
    non-finite or non-positive prediction is replaced by ``min_depth``.  This
    prevents invalid outputs from improving delta accuracy through silent removal.
    """
    pred = pred.float().flatten()
    gt = gt.float().flatten()
    mask = mask.flatten().bool() if mask is not None else torch.ones_like(gt, dtype=torch.bool)

    valid = mask & torch.isfinite(gt) & (gt > min_depth) & (gt < max_depth)
    if invalid_pred == "exclude":
        valid &= (pred > min_depth) & torch.isfinite(pred)
    elif invalid_pred == "penalize":
        pred = torch.where(torch.isfinite(pred) & (pred > min_depth), pred, pred.new_tensor(min_depth))
    else:
        raise ValueError("invalid_pred must be 'exclude' or 'penalize'")
    pred, gt = pred[valid], gt[valid]
    if pred.numel() == 0:
        return {k: float("nan") for k in METRIC_KEYS}

    diff = pred - gt
    log_diff = torch.log(pred) - torch.log(gt)
    ratio = torch.max(pred / gt, gt / pred)
    si_log_variance = (log_diff**2).mean().item() - (log_diff.mean().item()) ** 2
    return {
        "abs_rel": (diff.abs() / gt).mean().item(),
        "sq_rel": ((diff**2) / gt).mean().item(),
        "rmse": (diff**2).mean().sqrt().item(),
        "rmse_log": (log_diff**2).mean().sqrt().item(),
        "si_log": si_log_variance,
        "delta1": (ratio < 1.25).float().mean().item(),
        "delta2": (ratio < 1.25**2).float().mean().item(),
        "delta3": (ratio < 1.25**3).float().mean().item(),
        # Eigen et al. (2014)'s own scale-invariant log error, sqrt(variance)*100.
        # ``si_log`` above returns the bare variance (this repo's pre-existing,
        # non-standard convention -- kept unchanged so no historical result shifts)
        # while every published SILog number this repo compares against uses this
        # scaling instead. New key so existing readers of ``si_log`` are unaffected.
        "si_log_pct": (max(si_log_variance, 0.0) ** 0.5) * 100.0,
    }

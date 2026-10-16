import math

import pytest
import torch

from metrics import METRIC_KEYS, compute_depth_metrics, eigen_crop_mask, garg_crop_mask


def test_garg_crop_mask_matches_known_kitti_bounds():
    mask = garg_crop_mask(375, 1242)
    ys, xs = torch.where(mask)
    assert ys.min().item() == int(0.40810811 * 375)
    assert ys.max().item() == int(0.99189189 * 375) - 1
    assert xs.min().item() == int(0.03594771 * 1242)
    assert xs.max().item() == int(0.96405229 * 1242) - 1
    assert mask.dtype == torch.bool
    assert mask.shape == (375, 1242)


def test_eigen_crop_mask_is_not_the_garg_crop():
    """The two crops are distinct regions, selected by a flag in every upstream
    script we compare against (DepthAnythingAC/evaluate_depth.py:162-172,
    unidepth_repo/unidepth/datasets/kitti.py:154-165). Conflating them silently
    scores a different image region than the published protocol did."""
    assert not torch.equal(eigen_crop_mask(375, 1242), garg_crop_mask(375, 1242))


def test_eigen_crop_mask_matches_known_kitti_bounds():
    mask = eigen_crop_mask(375, 1242)
    rows = torch.nonzero(mask.any(dim=1)).flatten()
    cols = torch.nonzero(mask.any(dim=0)).flatten()
    assert (int(rows[0]), int(rows[-1]) + 1) == (int(0.3324324 * 375), int(0.91351351 * 375))
    assert (int(cols[0]), int(cols[-1]) + 1) == (int(0.0359477 * 1242), int(0.96405229 * 1242))


def test_eigen_crop_mask_non_kitti_requires_nyu_shape():
    assert eigen_crop_mask(480, 640, dataset="nyu")[45:471, 41:601].all()
    with pytest.raises(ValueError, match="480x640"):
        eigen_crop_mask(375, 1242, dataset="nyu")


def test_crop_mask_excludes_borders():
    mask = garg_crop_mask(100, 200)
    assert not mask[0, 0]
    assert not mask[-1, -1]
    assert mask[60, 100]


def test_si_log_pct_is_correctly_scaled_and_si_log_is_unchanged():
    torch.manual_seed(0)
    gt = torch.rand(4, 4) * 10 + 1
    pred = gt * 1.1 + 0.05
    out = compute_depth_metrics(pred, gt)
    assert out["si_log_pct"] == pytest.approx(math.sqrt(max(out["si_log"], 0.0)) * 100.0)
    # Historical behaviour preserved: si_log is still the bare variance, not the
    # sqrt(.)*100-scaled quantity every published SILog number in this repo's
    # references uses.
    log_diff = torch.log(pred.flatten()) - torch.log(gt.flatten())
    expected_variance = (log_diff**2).mean().item() - (log_diff.mean().item()) ** 2
    assert out["si_log"] == pytest.approx(expected_variance)


def test_si_log_pct_present_in_nan_fallback():
    out = compute_depth_metrics(torch.zeros(2, 2), torch.zeros(2, 2))
    assert set(out.keys()) == set(METRIC_KEYS)
    assert math.isnan(out["si_log_pct"])

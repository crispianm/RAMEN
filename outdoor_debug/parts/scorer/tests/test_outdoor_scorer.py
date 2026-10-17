"""Unit tests (CPU, synthetic arrays) for outdoor_debug/parts/scorer/scorer.py pure functions."""
import sys, hashlib
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import scorer as S


def test_sentinel_excluded_and_bin_edges():
    gt = np.array([79.99, 80.0, 100.0, 654.99, 655.0, 655.35, np.nan, -1.0, 0.0, 1e4], np.float32)
    m = S.far_mask(gt)
    assert m.tolist() == [False, True, True, True, False, False, False, False, False, False]   # 80 inclusive, 655 / 655.35 / above excluded


def test_sentinel_does_not_move_the_ratio():
    gt = np.array([100.0, 200.0, 300.0, 655.35, 655.35, 655.35], np.float32)
    pred = np.array([90.0, 180.0, 240.0, 5.0, 5.0, 5.0], np.float32)       # sentinel px predicted wildly low
    r = pred[S.far_mask(gt)] / gt[S.far_mask(gt)]
    st = S.ratio_stats(r)
    assert st["n"] == 3 and st["pooled_median"] == pytest.approx(0.9) and st["abs_dev"] == pytest.approx(0.1)
    allr = S.ratio_stats(pred / gt)                                           # what leaving the sentinel in would give
    assert allr["pooled_median"] != pytest.approx(0.9)


def test_pooled_median_of_ratios_is_not_ratio_of_medians():
    gt = np.array([10.0, 100.0, 1000.0]); pred = np.array([200.0, 50.0, 600.0])        # ratios 20, 0.5, 0.6
    assert S.ratio_stats(pred / gt)["pooled_median"] == pytest.approx(0.6)             # the statistic used (median of ratios)
    assert S.ratio_of_medians(pred, gt) == pytest.approx(2.0)                          # median(pred)/median(gt): a different number, NOT used


def test_range_bins_and_pooled_absrel():
    gt = np.array([[5.0, 10.0, 10.5, 30.0, 30.1, 80.0, 80.1, 300.0, 0.05]], np.float32)
    pred = gt * 1.1
    rm, base = S.range_masks(gt)
    assert [int(rm[k].sum()) for k in ("0-10", "10-30", "30-80", ">80")] == [2, 2, 2, 2]     # (0.1,10], (10,30], (30,80], (80,inf); gt 0.05 invalid
    n, s = S.absrel_sums(pred, gt, rm["0-10"])
    assert n == 2 and s / n == pytest.approx(0.1, rel=1e-5)
    rm2, base2 = S.range_masks(gt, sky=gt >= 300)
    assert int(rm2[">80"].sum()) == 1                                                  # sky excluded from the >80 bin


def test_unaligned_absrel_arithmetic():
    gt = np.array([[[2.0, 4.0], [100.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]])
    pred = np.array([[[3.0, 2.0], [50.0, 7.0]], [[2.0, 0.0], [1.0, 1.0]]])
    # frame0 valid: 2,4 (100 >= 80 excluded, -1 excluded): |3-2|/2=.5, |2-4|/4=.5 -> .5 ; frame1: pred clamped to >=1e-3: |2-1|=1,|1e-3-1|/1=.999,0,0 -> mean .49975
    assert S.perframe_absrel_unaligned(pred, gt, 80.0, 0.1) == pytest.approx((0.5 + 0.49975) / 2)


def test_unaligned_matches_bench_joint_depth_errors():
    bj = pytest.importorskip("bench_joint")
    import torch
    rng = np.random.default_rng(1)
    gt = rng.uniform(0.5, 100, (3, 6, 6)); pred = rng.uniform(0.5, 100, (3, 6, 6))
    v = (gt > 0.1) & (gt < 80)
    ref = bj._depth_errors(torch.from_numpy(np.clip(pred, 1e-3, 80)), torch.from_numpy(gt), torch.from_numpy(v))["abs_rel"]
    assert S.perframe_absrel_unaligned(pred, gt, 80.0, 0.1) == pytest.approx(ref, rel=1e-9)


def test_sky_levels():
    f = [np.full(60, 100.0), np.full(60, 300.0), np.full(10, 1.0)[:0]]
    f = [v for v in f if v.size >= 50]
    s = S.sky_levels(f)
    assert s["frames"] == 2 and s["pooled_median"] == pytest.approx(200.0) and s["median_of_frame_medians"] == pytest.approx(200.0)
    assert s["frac_gt200"] == pytest.approx(0.5)


def test_keys_sha_format_matches_r3():
    keys = ["a", "b"]
    assert S.keys_sha(keys) == hashlib.sha256(b"a\nb").hexdigest()          # '\n'.join, no trailing newline (verified against R3_PIN)

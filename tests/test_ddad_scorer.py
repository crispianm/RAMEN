"""Synthetic-only tests for outdoor_debug/parts/LR_DEV/ddad_scorer.py (no DDAD data, no model, CPU)."""
import shutil, sys
from pathlib import Path
import numpy as np
import pytest

LR = Path(__file__).resolve().parents[1] / "outdoor_debug" / "parts" / "LR_DEV"
sys.path.insert(0, str(LR))
import ddad_scorer as DS
import ddad_lr_dataset as D

H, W = D.NATIVE_HW
LABS = DS.bin_labels()


def sparse(depths, rng):
    gt = np.full((H, W), np.nan, np.float32)
    idx = rng.choice(H * W, len(depths), replace=False)
    gt.flat[idx] = depths
    return gt


def depths_all_bins(rng, n=4000):
    return np.concatenate([rng.uniform(lo + 0.01, hi - 0.01, n) for lo, hi in DS.BINS_M]).astype(np.float32)


def run(frames, pred_fn, scene_fn=lambda i: "s0"):
    acc = DS.Accumulator()
    for i, gt in enumerate(frames):
        DS.score_frame(acc, pred_fn(gt, i), gt, scene_fn(i))
    return acc.result()


def test_pred_09_gt_every_bin():
    rng = np.random.default_rng(1)
    frames = [sparse(depths_all_bins(rng), rng) for _ in range(3)]
    r = run(frames, lambda gt, i: 0.9 * np.nan_to_num(gt))     # full-res pred = 0.9*GT at the sampled px (resize is identity at native)
    for lab in LABS:
        b = r["bins"][lab]
        assert b["median_pred_over_gt"] == pytest.approx(0.9, abs=1e-6), lab
        assert b["absrel_unaligned_pooled"] == pytest.approx(0.1, abs=1e-6), lab
        assert b["n_px"] == 3 * 4000 and b["n_frames"] == 3
    assert r["overall_absrel_0_200"] == pytest.approx(0.1, abs=1e-6)
    print("TEST1", {k: (v["median_pred_over_gt"], v["absrel_unaligned_pooled"]) for k, v in r["bins"].items()})


def _frames_framefactor(rng):
    return [sparse(depths_all_bins(rng, 500), rng) for _ in range(4)]


FACT = [0.5, 0.9, 1.3, 2.0]


def test_forced_fail_gt_resized():
    """(a) the correct scorer REFUSES a resized GT; (b) a metric computed on a bilinearly downsampled GT gives different numbers."""
    import torch, torch.nn.functional as F
    rng = np.random.default_rng(2)
    gt = sparse(depths_all_bins(rng, 500), rng)
    with pytest.raises(ValueError):
        DS.score_frame(DS.Accumulator(), 0.9 * np.nan_to_num(gt), gt[::2, ::2], "s")
    # wrong path: densify via nan->0, bilinearly downsample GT to half size, upsample back (a "resized-GT" metric) and score
    g0 = torch.from_numpy(np.nan_to_num(gt))[None, None]
    gs = F.interpolate(g0, size=(H // 2, W // 2), mode="bilinear", align_corners=False)
    gwrong = F.interpolate(gs, size=(H, W), mode="nearest")[0, 0].numpy()
    gwrong[gwrong == 0] = np.nan
    acc = DS.Accumulator(); p, g = D.sample_prediction(0.9 * np.nan_to_num(gt), gwrong); acc.add(p, g, "s")
    right = run([gt], lambda g, i: 0.9 * np.nan_to_num(g))
    wrong = acc.result()
    dm = max(abs(right["bins"][l]["median_pred_over_gt"] - wrong["bins"][l]["median_pred_over_gt"]) for l in LABS if wrong["bins"][l]["n_px"])
    dn = sum(right["bins"][l]["n_px"] for l in LABS) - sum(wrong["bins"][l]["n_px"] for l in LABS)
    print("TEST2a right median/absrel", [(right["bins"][l]["median_pred_over_gt"], round(right["bins"][l]["absrel_unaligned_pooled"], 4)) for l in LABS])
    print("TEST2a GT-resized median/absrel", [(wrong["bins"][l]["median_pred_over_gt"], round(wrong["bins"][l]["absrel_unaligned_pooled"], 4)) for l in LABS], "px lost", dn)
    assert dm > 1e-3 or dn != 0, "GT-resize variant is not detectable"


def test_forced_fail_alignment():
    """pred = 0.9*GT*factor(frame). Correct (unaligned) path != per-frame-median-scaled path != per-frame-LSQ-scaled path."""
    rng = np.random.default_rng(3)
    frames = _frames_framefactor(rng)
    pred_fn = lambda gt, i: 0.9 * FACT[i] * np.nan_to_num(gt)
    right = run(frames, pred_fn)
    # wrong: per-frame median scaling to GT before pooling
    acc = DS.Accumulator()
    for i, gt in enumerate(frames):
        p, g = D.sample_prediction(pred_fn(gt, i), gt)
        p = p * (np.median(g) / np.median(p)); acc.add(p, g, "s0")
    aligned = acc.result()
    for lab in LABS:
        assert right["bins"][lab]["median_pred_over_gt"] != pytest.approx(aligned["bins"][lab]["median_pred_over_gt"], abs=1e-3)
    assert right["overall_absrel_0_200"] != pytest.approx(aligned["overall_absrel_0_200"], abs=1e-3)
    # sanity: pooled unaligned median of ratios over equal-size frames is the factor mixture, here 0.9*{0.5,0.9,1.3,2.0}
    exp = np.median(np.repeat(0.9 * np.array(FACT), 500))
    assert right["bins"][LABS[0]]["median_pred_over_gt"] == pytest.approx(exp, rel=1e-6)
    print("TEST2b unaligned overall", right["overall_absrel_0_200"], "| median-aligned overall", aligned["overall_absrel_0_200"],
          "| unaligned bin medians", [round(right["bins"][l]["median_pred_over_gt"], 4) for l in LABS], "| aligned", [round(aligned["bins"][l]["median_pred_over_gt"], 4) for l in LABS])


def test_bin_edges():
    g = np.array([0.0, np.nan, -1.0, 1.0, 19.999, 20.0, 49.999, 50.0, 80.0, 119.999, 120.0, 200.0, 200.0001, 250.0], np.float64)
    v = DS.valid_mask(g)
    assert v.tolist() == [False, False, False, True, True, True, True, True, True, True, True, True, False, False]
    gv = g[v]; ms = DS.bin_masks(gv)
    where = lambda x: [i for i, m in enumerate(ms) if m[list(gv).index(x)]]
    assert where(20.0) == [1] and where(50.0) == [2] and where(80.0) == [3] and where(120.0) == [4] and where(200.0) == [4] and where(19.999) == [0]
    assert sum(m.sum() for m in ms) == len(gv)                       # every valid px in exactly one bin
    acc = DS.Accumulator(); acc.add(np.ones_like(g), g, "s")
    assert acc.result()["overall_n_px"] == int(v.sum())


def test_pooled_not_per_frame():
    rng = np.random.default_rng(4)
    # frame A: 1000 px at 10 m, pred 5 (AbsRel .5); frame B: 10 px at 10 m, pred 15 (AbsRel .5 -> use pred 10, AbsRel 0)
    gtA = sparse(np.full(1000, 10.0, np.float32), rng); gtB = sparse(np.full(10, 10.0, np.float32), rng)
    acc = DS.Accumulator()
    DS.score_frame(acc, np.full((H, W), 5.0, np.float32), gtA, "a")
    DS.score_frame(acc, np.full((H, W), 10.0, np.float32), gtB, "b")
    b = acc.result()["bins"][LABS[0]]
    assert b["absrel_unaligned_pooled"] == pytest.approx(500.0 / 1010.0)         # pooled; per-frame mean would be 0.25
    assert abs(b["absrel_unaligned_pooled"] - 0.25) > 0.2
    assert b["median_pred_over_gt"] == pytest.approx(0.5) and b["n_frames"] == 2 and b["n_scenes"] == 2


def test_sha_assertion_fires_on_altered_frame_list(tmp_path):
    alt = tmp_path / "frames.tsv"
    shutil.copy(LR / "ddad_val_frames.tsv", alt)
    pairs = tmp_path / "pairs.tsv"; pairs.write_text("x\n")
    # unaltered frame list passes the frame check; the (fake) pairs file fails the pairs check
    with pytest.raises(AssertionError, match="pairs sha"):
        DS.assert_pins(alt, pairs)
    # altered frame list fails the frame check first
    alt.write_text(alt.read_text() + "000150\t1\n")
    with pytest.raises(AssertionError, match="frame list sha"):
        DS.assert_pins(alt, pairs)


def test_pins_constants_and_nonfinite_visible():
    assert DS.FRAMES_SHA == D.FRAMES_SHA256 and DS.PAIRS_SHA == D.PAIRS_SHA256 and DS.BJ_SHA == DS.S.BJ_SHA and DS.SHORT_SIDE == 378
    acc = DS.Accumulator(); acc.add(np.array([np.nan, 5.0]), np.array([10.0, 10.0]), "s")
    assert acc.result()["n_nonfinite_pred_px"] == 1


def test_selftest_runs():
    r = DS.selftest()
    assert r["n_scored_frames"] == 6 and r["n_scenes"] == 2
    assert all(0.85 < b["median_pred_over_gt"] < 0.95 for b in r["bins"].values() if b["n_px"])

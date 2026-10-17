"""Synthetic regression test for the DRAFT TS v2 sky/valid recipe (outdoor_debug/parts/TA_sky/ts_v2_draft.py).
M-03: an 8000 m sky dome must give sky=1, mask=0 (the released d>1e4 rule gives sky=0, mask=1).
B2-03: a depth of exactly 10000.0 is consistent (not valid, not silently a half-state)."""
import importlib.util
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "outdoor_debug" / "parts" / "TA_sky" / "ts_v2_draft.py"
_s = importlib.util.spec_from_file_location("ts_v2_draft", _p)
m = importlib.util.module_from_spec(_s); _s.loader.exec_module(m)
SKY = 7


def _frame(vals, segs):
    return np.array([vals], np.float32), np.array([segs], np.uint8)


def test_m03_dome_8000m_is_sky_not_valid():
    d, s = _frame([8000.0, 20.0], [SKY, 3])
    sky, mask, unc = m.ts_v2_sky_valid(d, s, {SKY})
    assert sky.tolist() == [[True, False]] and mask.tolist() == [[False, True]] and not unc.any()
    # released rule for contrast: sky = ~finite | d>1e4 -> 0 ; valid = d<1e4 -> 1
    assert not (d[0, 0] > 1e4) and d[0, 0] < 1e4


def test_b2_03_exact_10000_edge():
    d, s = _frame([10000.0, 10000.0], [SKY, 3])
    sky, mask, unc = m.ts_v2_sky_valid(d, s, {SKY})
    assert sky.tolist() == [[True, False]]       # seg-sky wins
    assert unc.tolist() == [[False, True]]       # non-sky label but saturated -> uncertain
    assert not mask.any()                        # never valid at exactly max_depth


def test_65504_inf_and_nan_and_tree_quirk():
    d, s = _frame([65504.0, np.inf, np.nan, 65504.0], [SKY, SKY, SKY, 12])  # last = "tree" labelled non-sky
    sky, mask, unc = m.ts_v2_sky_valid(d, s, {SKY})
    assert sky.tolist() == [[True, True, True, False]]
    assert unc.tolist() == [[False, False, False, True]] and not mask.any()


def test_unlabeled_env_defaults_to_uncertain_and_flagged_fallback():
    d, s = _frame([65504.0, 50.0], [3, 3])
    sky, mask, unc = m.ts_v2_sky_valid(d, s, set())
    assert not sky.any() and unc.tolist() == [[True, False]] and mask.tolist() == [[False, True]]
    sky, mask, unc = m.ts_v2_sky_valid(d, s, set(), fallback_sat_is_sky=True)
    assert sky.tolist() == [[True, False]] and not unc.any()


def test_zero_and_negative_depth_not_valid():
    d, s = _frame([0.0, -1.0, 5.0], [3, 3, 3])
    _, mask, _ = m.ts_v2_sky_valid(d, s, {SKY})
    assert mask.tolist() == [[False, False, True]]


def test_id0_sky_only_where_saturated():
    # AmericanDiner-like: sky ids {0, 242}; id 0 saturated -> sky, id 0 finite -> uncertain, 242 -> sky at any depth
    d, s = _frame([65504.0, np.inf, 10000.0, 30.0, 300.0, 20.0], [0, 0, 0, 0, 242, 3])
    sky, mask, unc = m.ts_v2_sky_valid(d, s, {0, 242})
    assert sky.tolist() == [[True, True, True, False, True, False]]
    assert unc.tolist() == [[False, False, False, True, False, False]]
    assert mask.tolist() == [[False, False, False, False, False, True]]


def test_id0_hongkong_like_env_without_sky_label():
    d, s = _frame([65504.0, 30.0, 20.0], [0, 0, 3])
    sky, mask, unc = m.ts_v2_sky_valid(d, s, set())
    assert not sky.any() and unc.tolist() == [[True, False, False]] and mask.tolist() == [[False, True, True]]
    sky, mask, unc = m.ts_v2_sky_valid(d, s, set(), fallback_sat_is_sky=True)
    assert sky.tolist() == [[True, False, False]] and mask.tolist() == [[False, True, True]]


# ---- guarded variant (Auditor O12.3): ts_v2_guarded; ts_v2_sky_valid above is unchanged and still tested ----
def _g(vals, segs, **kw):
    d, s = _frame(vals, segs)
    return m.ts_v2_guarded(d, s, {SKY}, **kw)


def test_guard_far_dome_finite_seg_sky_below_9000_is_uncertain():
    # far-dome env: seg-sky at 8000 m (and 100 m) is NOT sky and NOT valid -> uncertain (released: sky 0, mask 1; v2: sky 1)
    sky, mask, unc = _g([8000.0, 100.0, 8999.0], [SKY, SKY, SKY], far_dome=True)
    assert not sky.any() and not mask.any() and unc.all()


def test_guard_far_dome_keeps_9km_plus_edge_and_saturation_as_sky():
    sky, mask, unc = _g([9000.0, 9440.0, 9999.0, 10000.0, 65504.0, np.inf, np.nan], [SKY] * 7, far_dome=True)
    assert sky.all() and not mask.any() and not unc.any()


def test_guard_far_dome_non_sky_label_unchanged_semantics():
    # non-sky label: finite valid stays valid, saturated -> uncertain (default), finite 9400 m non-sky stays valid
    sky, mask, unc = _g([20.0, 9400.0, 65504.0], [3, 3, 3], far_dome=True)
    assert not sky.any() and mask.tolist() == [[True, True, False]] and unc.tolist() == [[False, False, True]]


def test_guard_near_dome_flag_gates_seg_sky():
    # near-dome env: seg_sky_ok True -> seg-sky is sky at any depth (8000 m dome -> sky, not valid); False -> uncertain, NEVER sky
    sky, mask, unc = _g([8000.0, 300.0, 20.0], [SKY, SKY, 3], far_dome=False, seg_sky_ok=True)
    assert sky.tolist() == [[True, True, False]] and mask.tolist() == [[False, False, True]] and not unc.any()
    sky, mask, unc = _g([8000.0, 300.0, 20.0, 65504.0], [SKY, SKY, 3, SKY], far_dome=False, seg_sky_ok=False)
    assert not sky.any() and unc.tolist() == [[True, True, False, True]] and mask.tolist() == [[False, False, True, False]]


def test_guard_seg_sky_not_ok_also_applies_in_far_dome_env():
    sky, mask, unc = _g([65504.0, 9500.0, 50.0], [SKY, SKY, 3], far_dome=True, seg_sky_ok=False)
    assert not sky.any() and unc.tolist() == [[True, True, False]] and mask.tolist() == [[False, False, True]]


def test_guard_saturated_non_sky_flag():
    for far in (True, False):
        sky, mask, unc = _g([65504.0, 50.0], [3, 3], far_dome=far)                    # default: uncertain
        assert not sky.any() and unc.tolist() == [[True, False]] and mask.tolist() == [[False, True]]
        sky, mask, unc = _g([65504.0, 50.0], [3, 3], far_dome=far, sat_nonsky_is_sky=True)   # visual check passed
        assert sky.tolist() == [[True, False]] and not unc.any() and mask.tolist() == [[False, True]]


def test_guard_unlabelled_env_and_id0():
    d, s = _frame([65504.0, 30.0], [3, 3])
    sky, mask, unc = m.ts_v2_guarded(d, s, set(), far_dome=False, seg_sky_ok=False)
    assert not sky.any() and unc.tolist() == [[True, False]]
    sky, mask, unc = m.ts_v2_guarded(d, s, set(), far_dome=False, seg_sky_ok=False, fallback_sat_is_sky=True)
    assert sky.tolist() == [[True, False]] and not unc.any()
    d, s = _frame([65504.0, 30.0, 300.0], [0, 0, 242])      # AmericanDiner-like id-0 handling is kept
    sky, mask, unc = m.ts_v2_guarded(d, s, {0, 242}, far_dome=False)
    assert sky.tolist() == [[True, False, True]] and unc.tolist() == [[False, True, False]]


def test_guard_equals_v2_in_near_dome_ok_env_and_v2_unchanged():
    rng = np.random.default_rng(0)
    d = rng.choice([0.5, 20.0, 8000.0, 9500.0, 1e4, 65504.0, np.inf], size=(40, 40)).astype(np.float32)
    s = rng.choice([SKY, 3, 12], size=(40, 40)).astype(np.uint8)
    a = m.ts_v2_sky_valid(d, s, {SKY}); b = m.ts_v2_guarded(d, s, {SKY}, far_dome=False, seg_sky_ok=True)
    assert all((x == y).all() for x, y in zip(a, b))

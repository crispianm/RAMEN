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

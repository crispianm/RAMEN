"""Protocol v3 WP29: the FBASE causal depth post-filters (kal / oneeuro / flowema) and the released-VDA-L spec."""
from __future__ import annotations

import re

import numpy as np
import pytest
import torch

import bench_joint as bj
from evaluation import fbase_filters as fb

RNG = np.random.RandomState(0)


def _noisy(n=60, hw=(8, 8), level=2.0, sigma=0.1):
    return (level + sigma * RNG.randn(n, *hw)).astype(np.float32)


TOKENS = ["kal0.05", "oneeuro", "oneeuro1.0b1.0", "flowema0.7"]


@pytest.fixture(autouse=True)
def _zero_flow(monkeypatch):
    monkeypatch.setattr(fb, "FLOW_FN", lambda fr: torch.zeros(fr.shape[0] - 1, 2, *fr.shape[-2:]))


def _apply(tok, d):
    frames = torch.rand(d.shape[0], 3, *d.shape[1:])
    return bj.depth_post(tok)(d, {"i": frames, "r": frames})


# ------------------------------------------------------------------ tokens
def test_tokens_parse_and_are_post_filters():
    for t in TOKENS + ["kal1", "kal.2", "oneeuro0.5b2", "flowema1.0"]:
        assert bj.DEPTH_POST_RE.fullmatch(t), t
    for t in ("kal", "oneeuro1.0", "flowema", "flowema2", "kalman0.1"):
        assert not bj.DEPTH_POST_RE.fullmatch(t), t
    assert [bj.DEPTH_POST_RE.fullmatch(t).group(1) for t in ("ema0.7",)] == ["ema"]            # the old tokens keep their groups
    assert bj.DEPTH_POST_RE.fullmatch("tmed5").group(4) == "5" and bj.DEPTH_POST_RE.fullmatch("agf0.2t0.06").group(5)


# ------------------------------------------------------------------ causality + shape/dtype
@pytest.mark.parametrize("tok", TOKENS)
def test_filters_are_causal_and_keep_shape_dtype(tok):
    d = _noisy()
    y = _apply(tok, d)
    assert y.shape == d.shape and y.dtype == d.dtype and np.isfinite(y).all()
    k = 30
    d2 = d.copy()
    d2[k + 1:] += 5.0                                                     # change ONLY the future
    frames = torch.rand(d.shape[0], 3, 8, 8)
    f = bj.depth_post(tok)
    assert np.array_equal(f(d, {"i": frames})[:k + 1], f(d2, {"i": frames})[:k + 1])
    assert np.array_equal(y[0], d[0])                                      # first frame passes through


# ------------------------------------------------------------------ Kalman
def test_kalman_reduces_noise_on_a_constant_signal_and_has_the_steady_state_gain():
    d = _noisy(n=200)
    y = fb.depth_kalman(d, 0.05)
    assert y[100:].std() < 0.5 * d[100:].std() and abs(y[100:].mean() - 2.0) < 0.02
    # steady state: K = (-q + sqrt(q^2 + 4q)) / 2 = 0.2 for q = 0.05 (the documented default gain)
    imp = np.zeros((300, 1, 1), np.float64)
    imp[150] = 1.0
    g = fb.depth_kalman(imp)[150, 0, 0]                                    # the response to a unit impulse at t=150 is the gain at that time
    assert g == pytest.approx(0.2, abs=1e-6)
    with pytest.raises(ValueError):
        fb.depth_kalman(d, 0.0)


def test_kalman_tracks_a_step_with_a_bounded_lag():
    d = np.concatenate([np.full((40, 2, 2), 1.0), np.full((40, 2, 2), 3.0)]).astype(np.float32)
    y = fb.depth_kalman(d, 0.05)
    assert y[40, 0, 0] > 1.0 and abs(y[-1, 0, 0] - 3.0) < 1e-2 and np.all(np.diff(y[40:, 0, 0]) >= 0)


# ------------------------------------------------------------------ one-euro
def test_oneeuro_smooths_a_static_signal_and_follows_a_fast_ramp_better_with_beta():
    d = _noisy(n=200, sigma=0.05)
    y = fb.depth_oneeuro(d)
    assert y[100:].std() < 0.45 * d[100:].std()                             # 1 Hz at 30 fps would give ~0.30; the speed term widens the cutoff a little on noise
    ramp = np.linspace(1.0, 4.0, 60, dtype=np.float32)[:, None, None] * np.ones((1, 2, 2), np.float32)
    lag_lo = np.abs(fb.depth_oneeuro(ramp, 1.0, 0.0) - ramp)[-1].mean()
    lag_hi = np.abs(fb.depth_oneeuro(ramp, 1.0, 5.0) - ramp)[-1].mean()
    assert lag_hi < lag_lo                                                  # the speed-adaptive cutoff reduces the lag
    assert np.allclose(fb.depth_oneeuro(ramp, 1e6, 0.0), ramp, atol=1e-5)      # an enormous cutoff is the identity (alpha -> 1)


def test_oneeuro_is_scale_free():
    d = _noisy(n=80)
    assert np.allclose(fb.depth_oneeuro(7.0 * d), 7.0 * fb.depth_oneeuro(d), rtol=1e-4)    # relative speed: metres and disparity behave alike


# ------------------------------------------------------------------ flow-EMA
def test_flowema_alpha_one_is_the_identity_and_needs_frames_otherwise():
    d = _noisy(n=10)
    assert np.array_equal(fb.depth_flowema(d, None, 1.0), d)
    with pytest.raises(ValueError):
        fb.depth_flowema(d, None, 0.7)


def test_flowema_with_exact_flow_beats_the_plain_ema_on_a_moving_pattern(monkeypatch):
    n, h, w = 24, 16, 48
    base = np.cumsum(RNG.randn(w), 0).astype(np.float32)                   # a 1-D texture, shifted +1 px per frame (global translation)
    clean = np.stack([np.tile(np.roll(base, t)[None, :], (h, 1)) for t in range(n)])
    noisy = clean + 0.5 * RNG.randn(*clean.shape).astype(np.float32)
    # content moves +1 px/frame, so frame t's pixel p came from t-1 at p-1: the backward flow (t -> t-1) is (-1, 0)
    monkeypatch.setattr(fb, "FLOW_FN", lambda fr: torch.tensor([-1.0, 0.0]).view(1, 2, 1, 1).expand(fr.shape[0] - 1, 2, *fr.shape[-2:]).contiguous())
    frames = torch.rand(n, 3, h, w)
    y = fb.depth_flowema(noisy, frames, 0.4)
    plain = bj.depth_ema(noisy, 0.4)
    err = lambda a: np.abs(a[4:, :, 8:] - clean[4:, :, 8:]).mean()          # skip the warm-up and the wrapped border columns
    assert err(y) < 0.8 * err(noisy) and err(y) < 0.7 * err(plain)


def test_flowema_resizes_a_flow_of_another_resolution():
    d = _noisy(n=6, hw=(8, 12))
    fr = torch.rand(6, 3, 4, 6)
    fb.FLOW_FN = lambda f: torch.zeros(f.shape[0] - 1, 2, 4, 6)             # the autouse fixture restores it
    y = fb.depth_flowema(d, fr, 0.5)
    assert y.shape == d.shape and np.allclose(y, bj.depth_ema(d, 0.5), atol=1e-5)     # zero flow == the plain EMA


def test_a_spec_with_the_new_tokens_builds_the_post_chain():
    toks = ["kal0.05", "oneeuro", "flowema0.7"]
    assert all(bj.DEPTH_POST_RE.fullmatch(t) for t in toks)
    assert bj.split_spec("metric_vda_s+kal0.05")[2] == ["kal0.05"] and bj.split_spec("metric_vda_s+oneeuro0.5b2")[2] == ["oneeuro0.5b2"]


# ------------------------------------------------------------------ released VDA-L
def test_vda_l_spec_is_registered_with_a_path_on_both_hosts():
    assert "vda_l" in bj.DEPTH_MODELS and bj.DEPTH_MODELS["vda_l"] is bj._vda_l
    for host in bj._HOSTS.values():
        assert host["vda_l"].endswith("video_depth_anything_vitl.pth") and "metric" not in host["vda_l"]
    assert bj.split_spec("vda_l")[1] == "vda_l" and bj.split_spec("vda_l+agf0.2t0.06")[2] == ["agf0.2t0.06"]


def test_vda_l_loads_strictly_when_the_released_weights_exist():
    p = bj.PATHS["vda_l"]
    import os
    if not os.path.isfile(p) or not os.path.isdir(bj.PATHS["vda_repo"]):
        pytest.skip("released VDA-L weights / repo not on this host")
    out = bj._vda_l("cpu")
    assert out["metric"] is False and out["window"] == 32 and out["weights"] == {"vda_l": p}

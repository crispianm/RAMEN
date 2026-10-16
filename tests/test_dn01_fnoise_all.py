"""Like-for-like F_noise (audit/SKY_fnoise_all_20261001.md): the ADDITIVE keys f_noise_all / flicker_low_all / flicker_norm_all (ALL pixels, depth clamped to [0.1, 80] m, the same pairs); every existing key byte-identical;
a synthetic case where two models differ ONLY in a 'sky' region (the existing F_noise then differs in its VALID SET, the *_all keys compare like for like)."""
from __future__ import annotations

import numpy as np
import pytest

import scripts.dn01_bvi_fnoise as fn

OLD_KEYS = {"f_noise", "level", "flicker_low", "flicker_norm", "f_noise_offset", "corr", "valid_frac", "far_frac80_low", "median_depth_low", "median_depth_norm"}
NEW_KEYS = {"f_noise_all", "flicker_low_all", "flicker_norm_all"}
# GOLDEN: the values of the PRE-CHANGE fnoise() on this seeded input (computed at e47ce51 before the keys were added)
GOLDEN = {"f_noise": 0.74511865484, "level": 0.999642293099, "flicker_low": 1.721359424246, "flicker_norm": 2.227199728314, "f_noise_offset": 3.661293881349, "corr": 0.842210208303,
          "valid_frac": 0.483854166667, "far_frac80_low": 0.4609375, "median_depth_low": 82.728468945534, "median_depth_norm": 61.305384098905}


def _golden_input():
    rng = np.random.default_rng(123)
    N, H, W = 14, 12, 16
    dn = rng.uniform(0.5, 120, (N, H, W))
    dl = dn * rng.uniform(0.8, 1.2, (N, H, W)) + rng.normal(0, 0.5, (N, H, W))
    dl[:, :3] = rng.uniform(70, 200, (N, 3, W))
    return dl, dn


def test_existing_keys_are_unchanged_and_the_new_keys_are_purely_additive():
    dl, dn = _golden_input()
    r = fn.fnoise(dl, dn, t0=4)
    assert set(r) == OLD_KEYS | NEW_KEYS
    for k, v in GOLDEN.items():
        assert round(r[k], 12) == v, k


def test_all_keys_match_an_independent_formula_over_every_pixel_on_clamped_depth():
    dl, dn = _golden_input()
    r = fn.fnoise(dl, dn, t0=4)
    cl, cn = np.clip(dl, 0.1, 80.0), np.clip(dn, 0.1, 80.0)
    sl, sn = cl[4:] - cl[3:-1], cn[4:] - cn[3:-1]
    den = cn[4:]
    assert r["f_noise_all"] == pytest.approx(np.mean(np.abs(sl - sn) / den), rel=1e-12)
    assert r["flicker_low_all"] == pytest.approx(np.mean(np.abs(sl) / den), rel=1e-12)
    assert r["flicker_norm_all"] == pytest.approx(np.mean(np.abs(sn) / den), rel=1e-12)
    assert np.abs(sl - sn).size == (14 - 4) * 12 * 16                                    # ALL pixels of ALL pairs t = t0..N-1, no mask


def _scene(rng, N=16, H=20, W=24):
    """Bottom half = scene (identical for both models); top half = sky."""
    base = rng.uniform(5, 30, (H // 2, W))
    nl = rng.normal(0, 0.4, (N, H // 2, W))
    dn_scene = np.broadcast_to(base, (N, H // 2, W)).copy()
    dl_scene = dn_scene + nl                                                           # noise-induced flicker in the low-light run
    return dn_scene, dl_scene


def test_two_models_differing_only_in_the_sky_region():
    rng = np.random.default_rng(7)
    N, H, W = 16, 20, 24
    dn_scene, dl_scene = _scene(rng, N, H, W)
    sky_noise = rng.normal(0, 3.0, (N, H // 2, W))

    def build(sky_norm, sky_low):
        dn = np.concatenate([np.full((N, H // 2, W), sky_norm), dn_scene], axis=1)
        dl = np.concatenate([sky_low, dl_scene], axis=1)
        return dl, dn
    # model A: sky pushed FAR (200 m, constant: beyond the 80 m cap)   model B: sky pulled in to 30 m, noisy in the low-light run
    a_dl, a_dn = build(200.0, np.full((N, H // 2, W), 200.0))
    b_dl, b_dn = build(30.0, 30.0 + sky_noise)
    A, B = fn.fnoise(a_dl, a_dn, t0=4), fn.fnoise(b_dl, b_dn, t0=4)
    # the EXISTING F_noise differs in its VALID SET: A excludes the whole sky (D^norm beyond 80 m), B includes it
    assert A["valid_frac"] == pytest.approx(0.5) and B["valid_frac"] == pytest.approx(1.0)
    assert A["f_noise"] != pytest.approx(B["f_noise"], rel=0.05)
    # the *_all keys are over the SAME (all) pixels for both: the scene half is identical, so f_noise_all(A) = scene_contrib / 2 and B adds the sky's real change
    scene_only = fn.fnoise(dl_scene, dn_scene, t0=4)
    assert A["f_noise_all"] == pytest.approx(scene_only["f_noise"] / 2.0, rel=1e-9)         # a sky beyond the cap contributes exactly 0 change
    assert B["f_noise_all"] > A["f_noise_all"] * 1.5                                      # the pulled-in sky's flicker is counted
    sky_part = np.mean(np.abs(np.diff(np.clip(30.0 + sky_noise, 0.1, 80.0), axis=0))[3:] / 30.0)
    assert B["f_noise_all"] == pytest.approx(scene_only["f_noise"] / 2.0 + sky_part / 2.0, rel=1e-9)
    # and the existing F_noise would have PREFERRED model A only because it never scored A's sky: on the SAME pixels (the scene half) the two models are identical
    assert A["f_noise"] == pytest.approx(scene_only["f_noise"], rel=1e-9)


def test_identical_models_have_identical_all_keys_and_a_constant_offset_makes_no_flicker():
    dl, dn = _golden_input()
    assert fn.fnoise(dl, dn, t0=4)["f_noise_all"] == fn.fnoise(dl.copy(), dn.copy(), t0=4)["f_noise_all"]
    x = np.random.default_rng(1).uniform(2, 50, (12, 8, 8))
    r = fn.fnoise(x + 0.5, x, t0=4)
    assert r["f_noise_all"] < 1e-12 and r["flicker_norm_all"] > 0


def test_macro_aggregation_picks_up_the_new_keys_without_a_code_change():
    """run() builds by_scene / macro from the keys of the first sequence record: the new keys flow into the macro automatically."""
    import inspect
    src = inspect.getsource(fn.run)
    assert "next(iter(per.values()))" in src and "next(iter(by_scene.values()))" in src


def test_relative_mode_is_unchanged():
    rng = np.random.default_rng(0)
    q = rng.uniform(0.05, 1.0, (12, 6, 6))
    r = fn.fnoise_relative(q, q * 1.1 + 0.01, t0=4)
    assert not (NEW_KEYS & set(r))                                                     # descriptive relative path: additive keys are metric-only

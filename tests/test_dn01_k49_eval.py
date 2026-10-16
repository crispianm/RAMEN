"""DN-01 K49 eval functions (bench_joint): static clips, their degradation cells, and the static flicker metric (k)."""
from __future__ import annotations

import numpy as np
import torch

import bench_joint as bj


def _clean(n=5, h=28, w=42, seed=0):
    return torch.rand(n, 3, h, w, generator=torch.Generator().manual_seed(seed)) * 0.8 + 0.1


def test_static_clip_repeats_the_middle_frame_and_its_gt():
    c, g = _clean(), np.random.RandomState(0).uniform(1, 50, (5, 20, 30))
    sc, sg = bj.static_clip(c, g, n=7)
    assert sc.shape == (7, 3, 28, 42) and sg.shape == (7, 20, 30)
    assert all(torch.equal(sc[t], c[2]) for t in range(7)) and all(np.array_equal(sg[t], g[2]) for t in range(7))


def test_static_cells_parse_to_their_base_ll_cell():
    assert bj.parse_cell("st_llb3") == (bj.LowLight("llb3"), 0.0) and bj.parse_cell("st_ll")[0] == "ll"


def test_degrade_static_is_seeded_with_independent_noise_per_frame():
    sc, _ = bj.static_clip(_clean(), np.ones((5, 4, 4)), n=6)
    for cell in bj.STATIC_CELLS:
        a, b = bj.degrade_static(sc, cell, 7), bj.degrade_static(sc, cell, 7)
        assert torch.equal(a, b) and a.shape == sc.shape and a.min() >= 0 and a.max() <= 1
        assert not torch.equal(a[0], a[1])                          # a fresh draw per frame


def test_llvar_varies_the_level_per_frame():
    sc, _ = bj.static_clip(_clean(h=64, w=64), np.ones((5, 4, 4)), n=16)
    spread = {c: float(bj.degrade_static(sc, c, 3).mean((1, 2, 3)).std()) for c in ("st_llb3", "st_llvar")}
    assert spread["st_llvar"] > 3 * spread["st_llb3"]               # EV jitter moves each frame's level


def test_fnoise_cancels_motion_and_its_level_guard_sees_offsets():
    from scripts.dn01_bvi_fnoise import fnoise
    rng = np.random.RandomState(0)
    motion = 5.0 + np.cumsum(rng.uniform(-0.2, 0.2, (20, 6, 7)), 0)          # the camera moves: D^norm changes
    f = fnoise(motion, motion, t0=4)
    assert f["f_noise"] == 0.0 and f["level"] == 0.0 and f["flicker_norm"] > 0
    f = fnoise(motion + 0.5, motion, t0=4)                                    # a constant offset: no flicker ...
    assert f["f_noise"] < 1e-12 and f["level"] > 0.05                         # ... but the level guard sees it
    noisy = motion * (1 + 0.05 * rng.standard_normal(motion.shape))
    assert fnoise(noisy, motion, t0=4)["f_noise"] > 0.02
    far = motion.copy()
    far[:, 0, 0] = 500.0                                                      # D^norm out of [0.1, 80]: excluded
    assert fnoise(far * 1.0, far, t0=4)["valid_frac"] < 1.0


def test_fnoise_run_refuses_frames_that_differ_from_the_precheck(tmp_path):
    import pytest
    from PIL import Image
    from scripts.dn01_bvi_fnoise import checked_files, frames_digest, frame_files
    for kind in ("low", "normal"):
        d = tmp_path / "S99_x" / f"{kind}_light_10"
        d.mkdir(parents=True)
        for i in range(3):
            Image.fromarray(np.full((4, 4, 3), i * 10 + (kind == "low"), np.uint8)).save(d / f"{i:05d}.png")
    files = {k: frame_files(tmp_path, "S99_x", 10, k) for k in ("low", "normal")}
    rec = {"frames": frames_digest(files["low"], files["normal"])}
    assert checked_files(tmp_path, "S99_x__10", rec)["low"] == files["low"]      # N_FRAMES slices a 3-frame dir
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(files["low"][1])       # a different frame on this host
    with pytest.raises(SystemExit):
        checked_files(tmp_path, "S99_x__10", rec)


def test_flicker_static_definition():
    g = np.full((12, 6, 8), 5.0)
    g[:, 0, 0] = -1.0                                               # invalid GT pixel: ignored
    const = np.full((12, 6, 8), 5.0)
    f = bj.flicker_static(const, g, 80.0, 0.1, t0=4)
    assert f["flicker"] == 0.0 and f["abs_rel_static"] == 0.0
    alt = const * np.where(np.arange(12) % 2 == 0, 1.1, 0.9)[:, None, None]
    alt[:, 0, 0] = 1e6                                              # would dominate if not masked
    f = bj.flicker_static(alt, g, 80.0, 0.1, t0=4)
    assert abs(f["flicker"] - 1.0 / ((5 * 4.5 + 4 * 5.5) / 9)) < 1e-9   # |5.5 - 4.5| / mean_t D (frames 3..11)
    assert abs(f["abs_rel_static"] - 0.1) < 1e-9


def _moving_gt(n=20, h=12, w=16, seed=0):
    """A scene whose depth changes every frame (so a delay is visible), metres in (1, 40)."""
    rng = np.random.RandomState(seed)
    base = rng.uniform(5, 30, (h, w))
    return torch.from_numpy(np.stack([base * (1 + 0.3 * np.sin(0.7 * t + base / 7)) for t in range(n)]))


def test_lag_metrics_finds_the_delay():
    gt = _moving_gt()
    now = bj.lag_metrics(1.0 / gt, gt, 80.0)
    assert now["lag_k"] == 0.0 and now["lag_absrel0"] < 1e-9
    late = torch.cat([gt[:1].expand(2, -1, -1), gt[:-2]])              # depth = GT delayed by 2 frames
    got = bj.lag_metrics(1.0 / late, gt, 80.0)
    assert got["lag_k"] == 2.0 and got["lag_absrel0"] > 0.01
    assert bj.lag_metrics(3.0 / gt, gt, 80.0)["lag_k"] == 0.0         # scale / shift is fitted (relative models)
    assert np.isnan(bj.lag_metrics(1.0 / gt[:4], gt[:4], 80.0)["lag_k"])


def test_fnoise_diagnostics_offset_and_correlation():
    from scripts.dn01_bvi_fnoise import fnoise
    rng = np.random.RandomState(1)
    motion = 5.0 + np.cumsum(rng.uniform(-0.3, 0.3, (24, 6, 7)), 0)
    f = fnoise(motion, motion, t0=4)                                   # perfect cancellation
    assert f["f_noise"] == 0.0 and f["f_noise_offset"] > 0.01 and abs(f["corr"] - 1.0) < 1e-12
    still = np.full((24, 6, 7), 5.0)
    noisy = still * (1 + 0.05 * rng.standard_normal(still.shape))
    g = fnoise(noisy, still, t0=4)                                     # no motion: nothing to cancel
    assert abs(g["f_noise"] / g["f_noise_offset"] - 1.0) < 0.05


def test_fnoise_relative_mode_fits_one_affine_per_sequence():
    from scripts.dn01_bvi_fnoise import fnoise_relative
    rng = np.random.RandomState(2)
    q_norm = 1.0 / (5.0 + np.cumsum(rng.uniform(-0.3, 0.3, (20, 6, 7)), 0))
    r = fnoise_relative(2.0 * q_norm + 0.01, q_norm, t0=4)              # an affine-distorted relative output
    assert r["f_noise"] < 1e-9 and r["level"] < 1e-9 and abs(r["fit_scale"] - 0.5) < 1e-9
    noisy = (2.0 * q_norm + 0.01) * (1 + 0.05 * rng.standard_normal(q_norm.shape))
    assert fnoise_relative(noisy, q_norm, t0=4)["f_noise"] > 0.01


def test_fnoise_relative_drops_non_positive_aligned_disparity():
    from scripts.dn01_bvi_fnoise import fnoise_relative
    rng = np.random.RandomState(3)
    q_norm = rng.uniform(0.1, 1.0, (6, 7)) * (1 + np.cumsum(rng.uniform(-0.01, 0.01, (20, 6, 7)), 0))
    q_low = 0.5 * q_norm + 0.1                   # fits with a NEGATIVE shift: aligned = 2 q_low - 0.2
    q_low[10:, 0, 0] = 0.05                      # a collapsed dark pixel -> aligned disparity < 0 (no depth)
    r = fnoise_relative(q_low, q_norm, t0=4)
    assert np.isfinite(r["f_noise"]) and r["f_noise"] < 1.0 and r["level"] < 1.0   # was ~1e5 with the 1e-6 floor
    assert 0.0 < r["invalid_low_frac"] < 0.2
    assert fnoise_relative(0.5 * q_norm + 0.1, q_norm, t0=4)["invalid_low_frac"] == 0.0

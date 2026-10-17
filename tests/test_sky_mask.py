"""Synthetic-only tests for outdoor_debug/parts/confirm/seg/make_sky_mask.py (mocked probability maps; no model, no data)."""
import hashlib, sys
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip('torch')

SEG = Path(__file__).resolve().parents[1] / "outdoor_debug" / "parts" / "confirm" / "seg"
sys.path.insert(0, str(SEG))
import make_sky_mask as M
sys.path.insert(0, str(SEG.parent / 'av2'))
import make_panel as P

H, W = 100, 160   # short side 100 -> erosion radius round(8*100/900)=1; use explicit radii where needed


def maps(top=50):
    a = np.full((H, W), 0.05, np.float32); b = a.copy()
    a[:top] = 0.99; b[:top] = 0.99
    return a, b


def test_id2label_ok_and_tampered():
    ade = {i: f"c{i}" for i in range(150)}; ade[2] = "sky"
    assert M.assert_sky_id(ade, *M.ADE_EXPECT) == 2
    city = {i: f"c{i}" for i in range(19)}; city[10] = "sky"
    assert M.assert_sky_id(city, *M.CITY_EXPECT) == 10
    bad = dict(ade); bad[2] = "wall"; bad[7] = "sky"       # sky moved
    with pytest.raises(AssertionError):
        M.assert_sky_id(bad, *M.ADE_EXPECT)
    dup = dict(ade); dup[9] = "Sky"
    with pytest.raises(AssertionError):
        M.assert_sky_id(dup, *M.ADE_EXPECT)
    none = {i: f"c{i}" for i in range(150)}
    with pytest.raises(AssertionError):
        M.assert_sky_id(none, *M.ADE_EXPECT)


def test_agreement_requires_both():
    a, b = maps(); b[:, :80] = 0.89              # left half: B just below threshold
    m = M.certain_sky(a, b, radius=0)
    assert m[:50, 80:].all() and not m[:, :80].any()
    b[:, :80] = 0.90
    assert M.certain_sky(a, b, radius=0)[:50].all()   # >= 0.9 inclusive


def test_erosion_and_boundary_band():
    a, b = maps(50)
    m = M.certain_sky(a, b, radius=5)
    assert not m[45:50].any()                    # band within 5 px of the sky/non-sky edge removed
    assert m[:44].all()                          # image border is not a boundary
    assert m[44].all() or m[43].all()
    assert not m[50:].any()


def test_default_radius_scales():
    assert M.erosion_radius(900) == 8 and M.erosion_radius(450) == 4 and M.erosion_radius(10) == 1
    assert M.lidar_radius(900) == 5


def test_lidar_exclusion():
    a, b = maps(50)
    m = M.certain_sky(a, b, radius=0)
    uv = np.array([[80.0, 20.0], [-5, -5], [1e6, 3]])    # out-of-image points ignored
    m2 = M.apply_lidar(m, uv, radius=5)
    assert not m2[20, 80] and not m2[20, 84] and not m2[16, 80]
    assert m2[20, 86] and m2[20, 74] and m2[0, 0]
    assert M.apply_lidar(m, np.zeros((0, 2)), radius=5).sum() == m.sum()


def test_dropped_frames_rule_and_old_store_path_is_gone():
    a, b = maps(50)
    ea = np.full((H, W), 0.05, np.float32)
    tiny = ea.copy(); tiny[:1, :W] = .99          # 1 row
    ok = M.certain_sky(a, b); none = M.certain_sky(ea, ea); t = M.certain_sky(tiny, tiny)
    assert not M.is_dropped(ok) and M.is_dropped(none) and M.is_dropped(t)
    # LiDAR can push a frame below the limit
    uv = np.stack(np.meshgrid(np.arange(0, W), np.arange(0, 50)), -1).reshape(-1, 2)
    assert M.is_dropped(M.apply_lidar(ok, uv))
    # D16: the old path that REMOVED dropped frames from the store (contradicting R1 M4) no longer exists
    for gone in ("build_masks", "write_mask_file", "read_mask_file"):
        assert not hasattr(M, gone), gone
    with pytest.raises(SystemExit):
        M.main()


def test_determinism_of_masks():
    a, b = maps(); rs = np.random.RandomState(0)
    a2 = a + rs.rand(H, W).astype(np.float32) * 0.005
    m1 = M.certain_sky(a2, b); m2 = M.certain_sky(a2.copy(), b.copy())
    assert (m1 == m2).all() and m1.sum() > 0


def test_forced_fail_variants_differ():
    a, b = maps(50)
    b = b.copy(); b[:, :60] = 0.8; a = a.copy(); a[30:40, 100:] = 0.85
    ref = M.certain_sky(a, b, radius=5)
    assert (M.certain_sky(a, b, thresh=0.8, radius=5) != ref).any()          # lowered threshold
    assert (M.certain_sky(a, b, radius=5, require_both=False) != ref).any()  # one segmenter only
    assert (M.certain_sky(a, b, radius=0) != ref).any()                      # no erosion
    assert M.certain_sky(a, b, radius=0).sum() > ref.sum()


def test_work_size():
    assert M.work_size(900, 1600) == (512, 896)
    assert M.work_size(2048, 1550)[1] == 512 and M.work_size(2048, 1550)[0] % 32 == 0


def test_panel_smoke():
    rs = np.random.RandomState(1)
    frames = {f"L:{i:02d}": rs.randint(0, 255, (90, 160, 3), np.uint8) for i in range(30)}
    masks = {}
    for k in frames:
        c = np.zeros((90, 160), bool); c[:30] = True; a = c.copy(); a[:40] = True
        masks[k] = (c, a)
    g = P.panel(frames, masks, dropped={"L:03"}, tile=(480, 270))
    assert g.shape == (3 * 270, 4 * 480, 3)
    assert len(P.pick(list(frames))) == 12 and P.pick(list(frames))[0] == "L:00" and P.pick(list(frames))[-1] == "L:29"
    o = P.overlay(frames["L:00"], *masks["L:00"])
    assert (o[:30, :, 0] > o[:30, :, 2]).mean() > 0.5 and (o[30:40, :, 2] > o[30:40, :, 0]).mean() > 0.5       # certain = red, any-only = blue
    assert (o[60:] == frames["L:00"][60:]).all()


def test_oneformer_semantic_scores_normalised():
    g = torch.Generator().manual_seed(0)
    cl = torch.randn(2, 7, 20, generator=g)          # 19 classes + null
    mk = torch.randn(2, 7, 6, 9, generator=g)
    S = M.oneformer_semantic_scores(cl, mk)
    assert S.shape == (2, 19, 6, 9)
    assert torch.allclose(S.sum(1), torch.ones(2, 6, 9), atol=1e-5)
    # equals the HF formula divided by its per-pixel total
    raw = torch.einsum("bqc,bqhw->bchw", cl.softmax(-1)[..., :-1], mk.sigmoid())
    assert torch.allclose(S, raw / raw.sum(1, keepdim=True), atol=1e-6)
    # argmax identical to HF post_process_semantic_segmentation
    assert (S.argmax(1) == raw.argmax(1)).all()


def test_oneformer_sky_prob_mocked_peak():
    cl = torch.full((1, 2, 20), -20.0); cl[0, 0, 10] = 20.0; cl[0, 1, 3] = 20.0   # q0 = sky, q1 = other
    mk = torch.full((1, 2, 4, 4), -20.0); mk[0, 0, :2] = 20.0; mk[0, 1, 2:] = 20.0
    p = M.oneformer_sky_prob(cl, mk, 10)
    assert p.shape == (1, 1, 4, 4)
    assert (p[0, 0, :2] > 0.99).all() and (p[0, 0, 2:] < 0.01).all()
    # null-class-dominated query contributes nothing
    cl2 = cl.clone(); cl2[0, 0] = -20.0; cl2[0, 0, 19] = 20.0
    assert (M.oneformer_sky_prob(cl2, mk, 10)[0, 0, :2] < 0.5).all()


def test_oneformer_mocked_probs_into_mask_and_forced_fail():
    cl = torch.full((1, 2, 20), -20.0); cl[0, 0, 10] = 20.0; cl[0, 1, 3] = 20.0
    mk = torch.full((1, 2, 25, 40), -20.0); mk[0, 0, :12] = 20.0; mk[0, 1, 12:] = 20.0
    pb = M.oneformer_sky_prob(cl, mk, 10)
    pb = torch.nn.functional.interpolate(pb, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
    a, _ = maps(50)
    ref = M.certain_sky(a, pb, radius=1)
    assert ref[:40].any() and not ref[60:].any()
    assert (M.certain_sky(a, pb, radius=0) != ref).any()
    a2 = a.copy(); a2[:, :80] = 0.1                      # segmenter A disagrees on the left
    assert not M.certain_sky(a2, pb, radius=1)[:, :80].any()
    d1 = M.certain_sky(a, pb, radius=1); d2 = M.certain_sky(a, pb, radius=1)
    assert (d1 == d2).all()


def test_oneformer_id2label_tamper():
    city = {i: f"c{i}" for i in range(19)}; city[10] = "sky"
    assert M.assert_sky_id(city, *M.CITY_EXPECT) == 10
    moved = dict(city); moved[10] = "terrain"; moved[9] = "sky"
    dup = dict(city); dup[4] = "sky"
    missing = dict(city); missing[10] = "vegetation"
    for bad in (moved, dup, missing):
        with pytest.raises(AssertionError):
            M.assert_sky_id(bad, *M.CITY_EXPECT)

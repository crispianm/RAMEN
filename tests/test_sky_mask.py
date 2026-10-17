"""Synthetic-only tests for outdoor_debug/parts/confirm/seg/make_sky_mask.py (mocked probability maps; no model, no data)."""
import hashlib, sys
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip('torch')

SEG = Path(__file__).resolve().parents[1] / "outdoor_debug" / "parts" / "confirm" / "seg"
sys.path.insert(0, str(SEG))
import make_sky_mask as M
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


def test_dropped_frames():
    a, b = maps(50)
    ea = np.full((H, W), 0.05, np.float32)
    tiny = ea.copy(); tiny[:1, :W] = .99          # 1 row = 1% exactly -> kept after no erosion? radius forces drop
    pairs = {"ok": (a, b), "none": (ea, ea), "tiny": (tiny, tiny)}
    masks, dropped = M.build_masks(pairs)
    assert set(masks) == {"ok"} and sorted(dropped) == ["none", "tiny"]
    # LiDAR can push a frame below the limit
    masks2, dropped2 = M.build_masks({"ok": (a, b)}, lidar={"ok": np.stack(np.meshgrid(np.arange(0, W), np.arange(0, 50)), -1).reshape(-1, 2)})
    assert dropped2 == ["ok"]


def test_determinism_and_roundtrip(tmp_path):
    a, b = maps(); rs = np.random.RandomState(0)
    a2 = a + rs.rand(H, W).astype(np.float32) * 0.005; b2 = b.copy()
    pairs = {"f2": (a2, b2), "f1": (a, b)}
    shas = []
    for i in range(2):
        masks, dr = M.build_masks(pairs)
        shas.append(M.write_mask_file(tmp_path / f"m{i}.npz", masks, dr, {"x": 1}))
    assert shas[0] == shas[1]
    assert (tmp_path / "m0.npz").read_bytes() == (tmp_path / "m1.npz").read_bytes()
    assert hashlib.sha256((tmp_path / "m0.npz").read_bytes()).hexdigest() == shas[0]
    masks, dr = M.build_masks(pairs)
    got, meta = M.read_mask_file(tmp_path / "m0.npz")
    assert all((got[k] == masks[k]).all() for k in masks) and meta["rule"]["prob_thresh"] == 0.9


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


def test_panel_smoke(tmp_path):
    rs = np.random.RandomState(1)
    frames = {f"f{i:02d}": rs.randint(0, 255, (90, 160, 3), np.uint8) for i in range(30)}
    masks = {k: np.zeros((90, 160), bool) for k in list(frames)[::2]}
    for m in masks.values(): m[:30] = True
    g = P.panel(frames, masks)
    assert g.shape == (3 * 270, 4 * 480, 3)
    assert len(P.pick(list(frames))) == 12 and P.pick(list(frames))[0] == "f00" and P.pick(list(frames))[-1] == "f29"


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

"""SYNTHETIC-only tests for the R1 mask rules (build_av2_masks.py + make_sky_mask.py any/IoU + scorer mask-meta). No AV2 data, no model."""
import json, sys
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "seg"))
import av2_dataset as A
import av2_scorer as S
import build_av2_masks as B
import make_sky_mask as M

H, W = A.NATIVE_HW                      # 2048 x 1550 (portrait); short side 1550
SKY_ROWS = 600


def probs(sky_rows=SKY_ROWS, a_extra=None, b_extra=None):
    """A, B segmenters agree: rows < sky_rows sky (p .99, margin +.98), rest not sky (p .02, margin -.96)."""
    def one():
        p = np.full((H, W), 0.02, np.float32); m = np.full((H, W), -0.96, np.float32)
        p[:sky_rows] = 0.99; m[:sky_rows] = 0.98
        return p, m
    pa, ma = one(); pb, mb = one()
    return pa, ma, pb, mb


def disk_count(shape, pts, r):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    m = np.zeros(shape, bool)
    for x, y in pts:
        m |= (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    return m


def test_declared_radii_at_native_short_side():
    assert M.erosion_radius(1550) == 14 and M.lidar_radius(1550) == 9          # M5: 9 px, not 5


def test_certain_count_exact_and_lidar_exclusion_object_level():
    pa, ma, pb, mb = probs()
    r0 = B.frame_masks(pa, ma, pb, mb, None)
    # eroded: rows with Euclid distance to the first non-sky row (600) > 14 -> rows 0..585
    assert r0["n_certain"] == 586 * W
    uv = np.array([[700.0, 580.0], [100.0, 300.0]])                            # (x, y): one near the boundary, one deep inside
    r1 = B.frame_masks(pa, ma, pb, mb, uv)
    removed = r0["n_certain"] - r1["n_certain"]
    expect = int((disk_count((H, W), [(700, 580), (100, 300)], 9) & r0["certain"]).sum())
    assert removed == expect > 0
    # forced-fail: a 5 px radius would remove a different (smaller) count
    assert int((disk_count((H, W), [(700, 580), (100, 300)], 5) & r0["certain"]).sum()) != removed
    print("MASKED PIXELS synthetic frame: certain(no lidar)=%d, lidar-removed=%d (r_l=9), erosion r=14" % (r0["n_certain"], removed))


def test_any_layer_is_superset_and_has_either_segmenter():
    pa, ma, pb, mb = probs()
    pb2, mb2 = pb.copy(), mb.copy()
    pa2, ma2 = pa.copy(), ma.copy()
    pa2[1000:1100, :] = 0.7; ma2[1000:1100, :] = 0.4                            # only A says sky here (argmax) -> in `any`, not in `certain`
    r = B.frame_masks(pa2, ma2, pb2, mb2, None)
    assert r["any"][1050].all() and not r["certain"][1050].any()
    assert not (r["certain"] & ~r["any"]).any()
    # argmax-sky with p < 0.5 (margin >= 0) is in `any`
    pa3, ma3 = pa.copy(), ma.copy(); pa3[1500:1600] = 0.4; ma3[1500:1600] = 0.05
    assert B.frame_masks(pa3, ma3, pb, mb, None)["any"][1550].all()
    # not sky in either, p < .5 and margin < 0 -> not in any
    assert not r["any"][2000].any()


def test_subset_violation_is_caught(monkeypatch):
    pa, ma, pb, mb = probs()
    monkeypatch.setattr(M, "any_layer", lambda *a: np.zeros((H, W), bool))
    with pytest.raises(AssertionError, match="subset"):
        B.frame_masks(pa, ma, pb, mb, None)


def test_pooled_iou_gate():
    pa, ma, pb, mb = probs()
    good = B.frame_masks(pa, ma, pb, mb, None)
    assert M.check_agreement([(good["inter"], good["union"])]) == 1.0
    # B disagrees on half of A's sky -> IoU ~ 0.5
    pbx, mbx = pb.copy(), mb.copy(); pbx[:300] = 0.02; mbx[:300] = -0.96
    bad = B.frame_masks(pa, ma, pbx, mbx, None)
    assert bad["inter"] / bad["union"] == pytest.approx(0.5)
    with pytest.raises(M.MaskRejected):
        M.check_agreement([(bad["inter"], bad["union"])])
    # exactly at 0.85 passes, just below fails
    assert M.check_agreement([(85, 100)]) == 0.85
    with pytest.raises(M.MaskRejected):
        M.check_agreement([(84, 100)])


def test_assemble_rejects_and_writes_nothing(tmp_path):
    pa, ma, pb, mb = probs()
    pbx, mbx = pb.copy(), mb.copy(); pbx[:300] = 0.02; mbx[:300] = -0.96
    res = {"L:1": B.frame_masks(pa, ma, pbx, mbx, None)}
    out = tmp_path / "sky_masks.npz"
    with pytest.raises(M.MaskRejected):
        B.assemble(res, out, dict(segmenter_a_sha="a", segmenter_b_sha="b", frames_sha="f"))
    assert not out.exists() and not (tmp_path / "sky_masks.meta.json").exists()


def test_drop_rule_one_and_store_roundtrip(tmp_path):
    pa, ma, pb, mb = probs()
    pt = probs(sky_rows=20)                                                  # 20 rows sky -> after erosion 6 rows ~ 0.3% < 1% -> dropped
    res = {"L:1": B.frame_masks(pa, ma, pb, mb, None), "L:2": B.frame_masks(*pt, None)}
    assert not res["L:1"]["dropped"] and res["L:2"]["dropped"]
    out = tmp_path / "sky_masks.npz"
    meta = B.assemble(res, out, dict(segmenter_a_sha="a", segmenter_b_sha="b", frames_sha="f"))
    assert meta["dropped"] == ["L:2"] and meta["n_kept"] == 2 and meta["pooled_iou"] == 1.0
    st = A.MaskStore(out)
    c, a = st.get("L:1")
    assert c.sum() == res["L:1"]["n_certain"] and (c & ~a).sum() == 0
    c2, a2 = st.get("L:2")                                                       # O43 M4: a P-dropped frame STAYS in the store (the guards need its `any`)
    assert c2.sum() == res["L:2"]["n_certain"] and a2.any() and meta["n_kept"] == 2
    m = json.load(open(tmp_path / "sky_masks.meta.json")); S.check_mask_meta(m)
    assert S.is_mask_dropped("L:2", set(m["dropped"])) and not S.is_mask_dropped("L:1", set(m["dropped"]))


@pytest.mark.parametrize("bad", [dict(status="REJECTED"), dict(pooled_iou=0.8), dict(pooled_iou=None), dict(dropped=None), dict(status=None)])
def test_scorer_refuses_bad_mask_meta(bad):
    ok = dict(status="OK", pooled_iou=0.9, dropped=[])
    S.check_mask_meta(ok)
    with pytest.raises(RuntimeError):
        S.check_mask_meta({**ok, **bad})


def test_split_sky_margin_torch():
    torch = pytest.importorskip("torch")
    P = torch.tensor([0.1, 0.2, 0.6, 0.1]).reshape(1, 4, 1, 1)                  # sky id 2
    p, m = M.split_sky_margin(P, 2)
    assert float(p) == pytest.approx(0.6) and float(m) == pytest.approx(0.4)
    p, m = M.split_sky_margin(P, 1)
    assert float(m) == pytest.approx(0.2 - 0.6)


def test_wrong_segmenter_sha_raises(tmp_path):
    f = tmp_path / "pytorch_model.bin"; f.write_bytes(b"weights")
    import hashlib
    good = hashlib.sha256(b"weights").hexdigest()
    assert B.verify_seg_sha(f, good) == good
    with pytest.raises(RuntimeError, match="registered"):
        B.verify_seg_sha(f, "0" * 64)


def test_empty_union_is_rejected_not_perfect(tmp_path):
    with pytest.raises(M.MaskRejected, match="unevaluable"):
        M.check_agreement([(0, 0), (0, 0)])
    # a real frame with NO sky in either segmenter (all candidate frames) must not write a store
    p = np.full((H, W), 0.02, np.float32); m = np.full((H, W), -0.96, np.float32)
    res = {"L:1": B.frame_masks(p, m, p, m, None)}
    assert res["L:1"]["union"] == 0
    with pytest.raises(M.MaskRejected):
        B.assemble(res, tmp_path / "x.npz", dict(segmenter_a_sha="a", segmenter_b_sha="b", frames_sha="f"))
    assert not (tmp_path / "x.npz").exists()


# ------------------------------------------------------------------ second-review additions (D5-D8, D9, D12, D17, D18)
def _frame_with(pa_val, pb_val, ma=0.98, mb=0.98):
    """A: rows < 600 at p=pa_val; B: rows < 600 at p=pb_val (margins as given)."""
    pa, ma_, pb, mb_ = probs()
    pa = pa.copy(); pb = pb.copy(); ma_ = ma_.copy(); mb_ = mb_.copy()
    pa[:SKY_ROWS] = pa_val; pb[:SKY_ROWS] = pb_val; ma_[:SKY_ROWS] = ma; mb_[:SKY_ROWS] = mb
    return pa, ma_, pb, mb_


def test_prob_threshold_is_0p9_in_each_segmenter():
    r = B.frame_masks(*_frame_with(0.99, 0.89), None)            # B just below 0.9 -> certain EMPTY (but `any` is not: argmax margin >= 0)
    assert r["n_certain"] == 0 and r["any"].any()
    r = B.frame_masks(*_frame_with(0.89, 0.99), None)
    assert r["n_certain"] == 0
    r = B.frame_masks(*_frame_with(0.90, 0.90), None)            # exactly 0.9 in both: inclusive
    assert r["n_certain"] == 586 * W
    assert M.PROB_THRESH == 0.9


def test_both_segmenters_required_for_certain():
    pa, ma, pb, mb = probs()
    pb_off = np.full((H, W), 0.02, np.float32); mb_off = np.full((H, W), -0.96, np.float32)
    assert B.frame_masks(pa, ma, pb_off, mb_off, None)["n_certain"] == 0         # only A says sky
    assert B.frame_masks(pb_off, mb_off, pb, mb, None)["n_certain"] == 0         # only B says sky
    assert B.frame_masks(pa, ma, pb, mb, None)["n_certain"] == 586 * W
    # the A-only region is still `any`
    assert B.frame_masks(pa, ma, pb_off, mb_off, None)["any"][100].all()


def test_any_layer_p_ge_half_clause_without_argmax():
    """p_sky >= 0.5 puts a pixel in `any` even when another class has the larger probability (margin < 0)."""
    pa, ma, pb, mb = probs(sky_rows=0)
    pa = pa.copy(); ma = ma.copy()
    pa[1000:1100] = 0.6; ma[1000:1100] = -0.2                                 # sky 0.6 vs another class 0.8: argmax is not sky, p >= 0.5
    pa[1200:1300] = 0.49; ma[1200:1300] = -0.2                                # p just below 0.5 and not argmax: NOT in any
    r = B.frame_masks(pa, ma, pb, mb, None)
    assert r["any"][1050].all() and not r["any"][1250].any()
    assert M.any_sky(np.array([0.5], np.float32), np.array([-0.1], np.float32))[0] and not M.any_sky(np.array([0.4999], np.float32), np.array([-0.1], np.float32))[0]


def test_agreement_boundary_0p85_in_make_sky_mask():
    assert M.check_agreement([(8500, 10000)]) == 0.85
    with pytest.raises(M.MaskRejected):
        M.check_agreement([(8499, 10000)])
    assert M.IOU_MIN == 0.85


def test_scorer_meta_iou_boundary_0p85():
    S.check_mask_meta(dict(status="OK", pooled_iou=0.85, dropped=[]))
    with pytest.raises(RuntimeError):
        S.check_mask_meta(dict(status="OK", pooled_iou=0.8499, dropped=[]))
    assert S.IOU_MIN == M.IOU_MIN


def test_iou_is_pooled_BEFORE_the_drops(tmp_path):
    """A dropped frame (< 1% certain-sky) where the segmenters disagree must still count in the pooled IoU."""
    pa, ma, pb, mb = probs()
    good = B.frame_masks(pa, ma, pb, mb, None)
    pa2, ma2, pb2, mb2 = probs(sky_rows=20)                                    # both say sky in 20 rows (certain eroded to 6 rows -> dropped) ...
    pa2 = pa2.copy(); ma2 = ma2.copy()
    pa2[1000:1500] = 0.7; ma2[1000:1500] = 0.4                                 # ... and only A says sky in 500 more rows (in `any`, not certain)
    bad = B.frame_masks(pa2, ma2, pb2, mb2, None)
    assert bad["dropped"] and not good["dropped"]
    assert good["inter"] / good["union"] == 1.0                                # IoU over the NON-dropped frames alone would be 1.0 and pass
    with pytest.raises(M.MaskRejected):
        B.assemble({"L:1": good, "L:2": bad}, tmp_path / "x.npz", dict(segmenter_a_sha="a", segmenter_b_sha="b", frames_sha="f"))
    assert not (tmp_path / "x.npz").exists()
    i, u = good["inter"] + bad["inter"], good["union"] + bad["union"]
    assert i / u < 0.85


def test_segmenter_pins_are_constants_and_safetensors_refused(tmp_path, monkeypatch):
    assert B.SEG_A["bin_sha"] == "a3dbf2d7a9912203078f304d88df8cfee9fd752ca2d2413f82036bdb02c5e7df" and B.SEG_B["bin_sha"] == "876855252aa674c7ad6e920c268d6722add811a927806372811faa724f19856c"
    assert B.SEG_A["revision"] == "739f5d4692954e4a185eac280dec1ba5a7d52f1d" and B.SEG_B["revision"] == "8ed649db2cc1756b8bc36f0cad00b0812e333027"
    import inspect
    assert "seg_a_sha" not in inspect.getsource(B.main) and "seg-a-sha" not in inspect.getsource(B.main)       # no CLI-derived sha
    import hashlib
    da, db = tmp_path / "a", tmp_path / "b"
    for d, c in ((da, b"wa"), (db, b"wb")):
        d.mkdir(); (d / "pytorch_model.bin").write_bytes(c)
    pa = dict(repo="r", revision="v", bin_sha=hashlib.sha256(b"wa").hexdigest()); pb = dict(repo="r2", revision="v2", bin_sha=hashlib.sha256(b"wb").hexdigest())
    out = B.verify_segmenters(da, db, pa, pb)
    assert out["segmenter_a"] == dict(repo="r", revision="v", pytorch_model_bin_sha256=pa["bin_sha"]) and out["segmenter_b_sha"] == pb["bin_sha"]
    with pytest.raises(RuntimeError, match="registered"):
        B.verify_segmenters(da, db, dict(pa, bin_sha="0" * 64), pb)
    (db / "model.safetensors").write_bytes(b"x")                                # a safetensors file next to the verified .bin
    with pytest.raises(RuntimeError, match="safetensors"):
        B.verify_segmenters(da, db, pa, pb)


def test_seg_wrapper_passes_use_safetensors_false(monkeypatch):
    pytest.importorskip("torch"); tf = pytest.importorskip("transformers")
    from types import SimpleNamespace
    seen = []

    class FakeModel:
        def to(self, d): return self
        def eval(self): return self
    def fake_from_pretrained(*a, **kw):
        seen.append(kw); return FakeModel()
    id2 = {i: f"c{i}" for i in range(150)}; id2[2] = "sky"
    monkeypatch.setattr(tf.AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(id2label=id2))
    monkeypatch.setattr(tf.SegformerForSemanticSegmentation, "from_pretrained", fake_from_pretrained)
    M._Seg("dir", None, M.ADE_EXPECT, "segformer", "cpu", use_safetensors=False)
    assert seen and seen[0].get("use_safetensors") is False
    seen.clear(); M._Seg("dir", None, M.ADE_EXPECT, "segformer", "cpu")
    assert "use_safetensors" not in seen[0]


def test_lidar_exclusion_covers_returns_beyond_200m_and_just_outside_the_image():
    K = A.K_from_row(1000.0, 1000.0, W / 2, H / 2)
    pts = np.array([[0.0, -150.0, 300.0], [0.0, 0.0, -5.0], [0.0, 0.0, 450.0]])           # 300 m, behind the camera, 450 m
    uv = A.project_all_returns(pts, K)
    assert uv.shape == (2, 2) and np.allclose(uv[0], [W / 2, H / 2 - 500]) and np.allclose(uv[1], [W / 2, H / 2])
    assert np.isnan(A.sparse_gt(pts, K)).all()                                              # the capped GT map has none of them
    ex = M.lidar_exclusion((H, W), uv, 9)
    assert ex[int(H / 2 - 500), int(W / 2)] and ex[int(H / 2 - 500) + 9, int(W / 2)] and not ex[int(H / 2 - 500) + 10, int(W / 2)]
    out = M.lidar_exclusion((H, W), np.array([[-3.0, 100.0]]), 9)                              # 3 px left of the image: its disk still reaches into it
    assert out[100, 0] and out[100, 5] and not out[100, 7]
    assert not M.lidar_exclusion((H, W), np.array([[-12.0, 100.0]]), 9).any()


def test_declared_radii_in_docstrings_match():
    import re
    txt = open(Path(B.__file__)).read() + open(Path(A.__file__)).read()
    assert "12 px" not in txt and "within 5 px" not in txt

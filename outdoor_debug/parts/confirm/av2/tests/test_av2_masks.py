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

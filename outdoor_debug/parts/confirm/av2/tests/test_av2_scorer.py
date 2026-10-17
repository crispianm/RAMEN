"""SYNTHETIC-only tests for av2_dataset.py / av2_scorer.py (no AV2 payload exists or is read). Includes forced-fail cases."""
import math, sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av2_dataset as A
import av2_scorer as S

H, W = A.NATIVE_HW
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def sparse(field, n=30000, seed=0, region=None):
    rng = np.random.default_rng(seed)
    gt = np.full((H, W), np.nan, np.float32)
    idx = rng.choice(H * W, n, replace=False)
    gt.flat[idx] = field.flat[idx]
    if region is not None:
        gt[~region] = np.nan
    return gt


def ramp():
    return (1.0 + 199.0 * (XX / (W - 1))).astype(np.float32)      # 1..200 m across x: every bin populated


def test_pred_0p9_gt_every_bin():
    f = ramp(); gt = sparse(f)
    acc = S.Accumulator()
    S.score_frame_guard(acc, 0.9 * f, gt, "L", np.zeros((H, W), bool))
    r = acc.result()
    for lab, b in r["bins"].items():
        assert b["n_px"] > 0, lab
        assert b["absrel_unaligned_pooled"] == pytest.approx(0.1, abs=1e-4), lab
        assert b["median_pred_over_gt"] == pytest.approx(0.9, abs=1e-4), lab
    # forced fail: a wrong pred (1.2x) must NOT give 0.1
    acc2 = S.Accumulator(); S.score_frame_guard(acc2, 1.2 * f, gt, "L", None)
    assert acc2.result()["bins"]["[50,80)"]["absrel_unaligned_pooled"] == pytest.approx(0.2, abs=1e-4)


def test_lowres_prediction_bilinear_to_native():
    f = ramp(); gt = sparse(f)
    small = 0.9 * f[::4, ::4]                                   # model-grid-like prediction, A1 upsample is done by the scorer
    acc = S.Accumulator(); S.score_frame_guard(acc, small, gt, "L", None)
    assert acc.result()["bins"]["[20,50)"]["median_pred_over_gt"] == pytest.approx(0.9, abs=0.02)


def test_gt_resize_is_refused_and_detectable():
    f = ramp(); gt = sparse(f)
    acc = S.Accumulator()
    with pytest.raises(ValueError):
        S.score_frame_guard(acc, 0.9 * f, gt[::2, ::2], "L", None)          # a resized GT is refused
    # and a resized GT would change the result: nearest-downsampled GT keeps a different pixel set
    assert np.isfinite(gt[::2, ::2]).sum() != np.isfinite(gt).sum()


def test_sky_excluded_from_far_guard_and_variant_differs():
    f = ramp(); sky = np.zeros((H, W), bool); sky[: H // 3] = True
    gt = sparse(f, 60000)                                        # LiDAR returns also fall in the (any-)sky region (e.g. false sky labels / haze)
    pred = 0.9 * f; pred[sky] = 10.0 * f[sky]                    # sky region predicted wildly wrong
    a = S.Accumulator(); S.score_frame_guard(a, pred, gt, "L", sky)
    b = S.Accumulator(); S.score_frame_guard(b, pred, gt, "L", None)     # variant with sky included
    ra, rb = a.result(), b.result()
    assert ra["n_lidar_px_sky_excluded"] > 0
    for lab in ra["bins"]:
        assert ra["bins"][lab]["absrel_unaligned_pooled"] == pytest.approx(0.1, abs=1e-4)
        assert rb["bins"][lab]["absrel_unaligned_pooled"] > 0.5            # included-sky variant is detectably different
    # frac of excluded px is reported
    assert ra["n_lidar_px_valid_before_sky"] > ra["overall_n_px"]


def test_sky_median_primary_known_depth_and_mask_resize():
    sky = np.zeros((H, W), bool); sky[:600] = True
    pred = np.full((504, 378), 20.0, np.float32); pred[: int(np.ceil(600 * 504 / H))] = 450.0       # sky rows (model grid) at 450 m
    k = S.SkyAccumulator(); k.add(pred, sky)
    r = k.result()
    assert r["P_framebalanced_median"] == pytest.approx(450.0)
    assert r["frac_sky_px_gt200"] == 1.0 and r["frac_sky_px_gt300"] == 1.0
    # forced fail: a vertically flipped mask must NOT return 450
    k2 = S.SkyAccumulator(); k2.add(pred, sky[::-1])
    assert k2.result()["P_framebalanced_median"] != pytest.approx(450.0)


def test_frame_balanced_vs_pooled_differ():
    k = S.SkyAccumulator()
    for depth, rows in ((100.0, 300), (100.0, 300), (1000.0, 1800)):       # third frame has 6x the sky px
        sky = np.zeros((H, W), bool); sky[:rows] = True
        pred = np.full((504, 378), 5.0, np.float32); pred[: int(np.ceil(rows * 504 / H))] = depth
        k.add(pred, sky)
    r = k.result()
    assert r["P_framebalanced_median"] == pytest.approx(100.0)              # frame-balanced: median of the three frame medians
    assert r["pooled_median_description"] == pytest.approx(1000.0)           # pooled median: dominated by the big frame (detectably different)


def test_min_sky_px_drop_boundary():
    for npx, kept in ((1999, 0), (2000, 1)):
        m = np.zeros((504, 378), bool); m.ravel()[:npx] = True
        # native mask whose nearest resize to (504,378) is exactly m: cv2 nearest picks source index floor(i * src/dst)
        ys = np.floor(np.arange(504) * H / 504).astype(int); xs = np.floor(np.arange(378) * W / 378).astype(int)
        mn = np.zeros((H, W), bool); mn[np.ix_(ys, xs)] = m
        k = S.SkyAccumulator(); k.add(np.full((504, 378), 300.0, np.float32), mn)
        assert k.result()["n_frames_kept"] == kept, npx


def test_bin_edges():
    g = np.array([0.0, 19.999, 20.0, 49.999, 50.0, 79.999, 80.0, 119.999, 120.0, 200.0, 200.001], np.float64)
    ids = [[i for i, m in enumerate(S.bin_masks(g)) if m[j]] for j in range(len(g))]
    assert ids == [[0], [0], [1], [1], [2], [2], [3], [3], [4], [4], []]
    acc = S.Accumulator(); acc.add(np.ones_like(g), g, "L")                 # 0 and 200.001 are invalid
    assert acc.result()["overall_n_px"] == 9
    assert S.bin_labels() == ["[0,20)", "[20,50)", "[50,80)", "[80,120)", "[120,200]"]


def test_pins_fire():
    reg = {"a": "x", "b": "y"}
    assert S.assert_pins({"a": "x", "b": "y"}, reg)
    with pytest.raises(AssertionError):
        S.assert_pins({"a": "x", "b": "WRONG"}, reg)                          # wrong sha
    with pytest.raises(AssertionError):
        S.assert_pins({"a": "x"}, reg)                                        # missing
    with pytest.raises(AssertionError, match="UNSET"):
        S.assert_pins({}, {"a": S.UNSET})
    with pytest.raises(AssertionError, match="UNSET"):
        S.assert_pins({k: "z" for k in S.PINS})                               # the draft scorer has unset pins and refuses real runs


def test_decision_rule_O36():
    L = math.log(1.5)
    assert S.decide(L, True) == "CONFIRMED" and S.decide(L + 0.01, True) == "CONFIRMED"
    assert S.decide(L - 1e-9, True) == "NOT CONFIRMED" and S.decide(0.2, True) == "NOT CONFIRMED"
    assert S.decide(L + 1, False) == "NOT CONFIRMED"                          # big d but a failed guard
    assert S.decide(0.0, True) == "REFUTED" and S.decide(-0.3, True) == "REFUTED"
    assert S.decide(float("nan"), True).startswith("INVALID")


def _bins(abs_rel, n=50000):
    return {lab: dict(n_px=n, absrel_unaligned_pooled=abs_rel, median_pred_over_gt=1.0, n_frames=10, n_logs=5) for lab in S.bin_labels()}


def test_guard_5pct_relative_per_bin():
    t = _bins(0.10)
    h = _bins(0.105)                                                          # exactly +5%: passes (<=)
    assert all(v["status"] == "PASS" for v in S.guard_check(h, t).values())
    h["[120,200]"]["absrel_unaligned_pooled"] = 0.1051                        # forced fail in the last bin only
    g = S.guard_check(h, t)
    assert g["[120,200]"]["status"] == "FAIL" and g["[0,20)"]["status"] == "PASS"
    h["[120,200]"]["n_px"] = 100                                              # insufficient px: reported UNEVALUABLE, never silently PASS
    assert S.guard_check(h, t)["[120,200]"]["status"] == "UNEVALUABLE"


def test_compare_end_to_end_verdict():
    def run(P, absrel):
        return dict(sky=dict(P_framebalanced_median=P), bins=_bins(absrel))
    c = S.compare(run(600.0, 0.10), run(150.0, 0.10))
    assert c["verdict"] == "CONFIRMED" and c["d_ln_PH300_over_PTS2"] == pytest.approx(math.log(4))
    assert "SINGLE SEED" in c["single_seed_note"] and c["seed"] == 50
    assert S.compare(run(600.0, 0.12), run(150.0, 0.10))["verdict"] == "NOT CONFIRMED"      # guard fails
    assert S.compare(run(180.0, 0.10), run(150.0, 0.10))["verdict"] == "NOT CONFIRMED"      # 0<d<ln1.5
    assert S.compare(run(140.0, 0.10), run(150.0, 0.10))["verdict"] == "REFUTED"


def test_selection_rule_and_pairing():
    ids = [f"log{i:03d}" for i in range(150)][::-1]                           # unsorted input
    sel = A.select_logs(ids)
    assert len(sel) == 50 and sel[0] == "log000" and sel[1] == "log003" and sel == sorted(sel)
    cam = list(range(0, 300 * 50_000_000, 50_000_000))                        # 20 Hz, 300 frames
    lid = list(range(10_000_000, 150 * 100_000_000, 100_000_000))             # 10 Hz, offset 10 ms
    r = A.select_frames(cam[::-1], lid)
    assert len(r["clip"]) == 150 and r["clip"][1] == cam[2]
    assert [s[0] for s in r["scored"]] == list(range(0, 150, 10)) and not r["dropped"]
    assert all(s[3] <= A.MAX_PAIR_DT_NS for s in r["scored"])
    # tie rule: equidistant sweeps -> the earlier one
    assert A.nearest_sweep(150, np.array([100, 200]))[0] == 100
    # forced fail: no sweep within 50 ms -> dropped, counted
    r2 = A.select_frames(cam, [10**12])
    assert len(r2["scored"]) == 0 and len(r2["dropped"]) == 15
    # clip longer than the window is truncated
    r3 = A.select_frames(list(range(0, 400 * 50_000_000, 50_000_000)), lid)
    assert len(r3["clip"]) == A.MAX_LEN


def test_pose_chain_motion_compensation():
    K = A.K_from_row(1000.0, 1000.0, W / 2, H / 2)
    # camera looks along ego +x: cam z = ego x, cam x = -ego y, cam y = -ego z
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float).T          # columns = cam axes expressed in ego -> ego_R_cam
    ego_T_cam = np.eye(4); ego_T_cam[:3, :3] = R
    city_T_ego_lidar = np.eye(4)
    city_T_ego_cam = np.eye(4); city_T_ego_cam[0, 3] = 3.0              # the vehicle moved 3 m forward between sweep time and camera time
    p_ego = np.array([[50.0, 0.0, 0.0]])                                # static point 50 m ahead at lidar time
    pc = A.lidar_ego_to_camera(p_ego, city_T_ego_lidar, city_T_ego_cam, ego_T_cam)
    assert pc[0, 2] == pytest.approx(47.0) and abs(pc[0, 0]) < 1e-9
    pc0 = A.lidar_ego_to_camera(p_ego, np.eye(4), np.eye(4), ego_T_cam)  # forced fail: ignoring motion compensation gives 50
    assert pc0[0, 2] == pytest.approx(50.0) and pc0[0, 2] != pytest.approx(pc[0, 2])
    g = A.sparse_gt(pc, K)
    assert np.nanmax(g) == pytest.approx(47.0, abs=1e-4) and np.isfinite(g).sum() == 1 and g.shape == (H, W)
    # >200 m and behind-camera points are not valid; z-buffer keeps the nearest
    pts = np.array([[0.0, 0.0, 250.0], [0.0, 0.0, -5.0], [0.0, 0.0, 30.0], [0.0, 0.0, 20.0]])
    g2 = A.sparse_gt(pts, K)
    assert np.isfinite(g2).sum() == 1 and np.nanmin(g2) == pytest.approx(20.0)
    # exactly 200 m is valid
    assert np.isfinite(A.sparse_gt(np.array([[0.0, 0.0, 200.0]]), K)).sum() == 1


def test_mask_store_roundtrip_and_subset_rule(tmp_path):
    c = np.zeros((H, W), bool); c[:100] = True
    a = c.copy(); a[:150] = True
    p = tmp_path / "m.npz"
    A.save_mask_store(p, {"L:1": (c, a)})
    c2, a2 = A.MaskStore(p).get("L:1")
    assert (c2 == c).all() and (a2 == a).all()
    A.save_mask_store(p, {"L:2": (a, c)})                                      # forced fail: certain not a subset of any
    with pytest.raises(ValueError):
        A.MaskStore(p).get("L:2")


def test_selftest_runs():
    r = S.selftest()
    for b in r["bins"].values():
        assert b["median_pred_over_gt"] == pytest.approx(0.9, abs=0.02)
    assert r["sky"]["P_framebalanced_median"] == pytest.approx(450.0)


def test_stdlib_selector_equals_dataset_rule():
    import select_av2_frames as Sel
    rng = np.random.default_rng(1)
    for trial in range(20):
        cam = sorted(int(x) for x in rng.integers(0, 15 * 10**9, 300))
        lid = sorted(int(x) for x in rng.integers(0, 15 * 10**9, 150))
        clip, sc, dr = Sel.select(cam, lid)
        r = A.select_frames(cam, lid)
        assert clip == r["clip"] and sc == r["scored"] and dr == r["dropped"]


def test_unevaluable_guard_never_passes_and_pairing_is_25ms():
    import math
    ln15 = math.log(1.5)
    assert S.decide(ln15 + 0.1, True, ["[120,200]"]) == "NOT CONFIRMED ([120,200] guard unevaluable)"
    assert S.decide(0.0, True, ["[120,200]"]) == "REFUTED"
    assert S.decide(ln15 + 0.1, True) == "CONFIRMED"
    assert S.decide(ln15 - 0.1, True) == "NOT CONFIRMED"
    big = {"n_px": 10**6, "absrel_unaligned_pooled": 0.2}; small = {"n_px": 5, "absrel_unaligned_pooled": 0.2}
    bins_ok = {"[0,20)": big, "[120,200]": small}
    g = S.guard_check(bins_ok, bins_ok)
    assert g["[120,200]"]["status"] == "UNEVALUABLE" and g["[0,20)"]["status"] == "PASS"
    assert A.MAX_PAIR_DT_NS == 25_000_000

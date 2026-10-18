"""SYNTHETIC-only tests for av2_dataset.py / av2_scorer.py (no AV2 payload exists or is read). Includes forced-fail cases."""
import math, re, sys
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
        S.assert_pins({k: ("z" if v == S.UNSET else v) for k, v in S.PINS.items()})                               # the draft scorer has unset pins and refuses real runs
    assert "scorer_sha" not in S.PINS                                         # D1: the scorer's own sha is pinned OUTSIDE the file (--expect-scorer-sha)
    assert {"meta_sha", "clips_sha", "objects_sha", "frames_sha"} <= set(S.PINS)


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


STAGE1_CFG = Path(__file__).resolve().parents[2] / "stage1" / "configs"


def _cfg(pair, arm):
    import yaml
    return yaml.safe_load(open(STAGE1_CFG / f"arm_S1_{dict(s50='n46', s51='n47', s52='n48')[pair]}_{arm}.yaml"))


def _with_cfg(r):
    """Stage-1-labelled synthetic run: the declared stored cfg of (pair, arm) + hash + re-derived cfg_* facts."""
    m = r["meta"]
    if m["seed_label"] in S.START_N:
        c = _cfg(m["seed_label"], m["arm"])
        m.update(S.assert_ckpt_cfg(c, m["seed_label"], m["arm"]), cfg=c, cfg_sha256=S.cfg_sha256(c))
    return r


def _run(P, absrel, arm, seed="s50", stage=1, ckpt=None):
    return _with_cfg(_run0(P, absrel, arm, seed, stage, ckpt))


def _run0(P, absrel, arm, seed="s50", stage=1, ckpt=None):
    return dict(sky=dict(P_framebalanced_median=P, kept_frame_ids=["f1", "f2"], dropped_frame_ids=[], frame_mask_px=[5000, 6000]), bins=_bins(absrel),
                n_scored_frames=2, n_lidar_px_valid_before_sky=100, n_lidar_px_sky_excluded=10,
                meta=dict(arm=arm, ckpt_sha=ckpt or f"sha_{arm}", seed_label=seed, stage=stage, pins=dict(frames_sha="f"), scorer_sha="s" * 64, registry_sha="r" * 64))


def test_compare_end_to_end_verdict():
    def cmp(h, t, stage=2):
        return S.compare(h, t, stage=stage)
    H = lambda P, a: _run(P, a, "H300", stage=2, seed="stage2"); T = lambda P, a: _run(P, a, "TS2", stage=2, seed="stage2")
    c = cmp(H(600.0, 0.10), T(150.0, 0.10))
    assert c["verdict"] == "CONFIRMED" and c["d_ln_PH300_over_PTS2"] == pytest.approx(math.log(4))
    assert c["disclosure"] == S.STAGE2_NOTE and c["disclosure"].startswith("SINGLE SEED (50): one paired difference cannot size seed noise; the A0/A0b screen pair showed 26% spread on a guard. Sign and magnitude only.")
    assert "the guards carry the information" in S.STAGE2_NOTE and "NOT an accuracy claim" in S.STAGE2_NOTE
    assert cmp(H(600.0, 0.12), T(150.0, 0.10))["verdict"] == "NOT CONFIRMED"      # guard fails
    assert cmp(H(180.0, 0.10), T(150.0, 0.10))["verdict"] == "NOT CONFIRMED"      # 0<d<ln1.5
    assert cmp(H(140.0, 0.10), T(150.0, 0.10))["verdict"] == "REFUTED"
    c1 = S.compare(_run(600.0, 0.1, "H300"), _run(150.0, 0.1, "TS2"), stage=1)
    assert c1["disclosure"] == S.STAGE1_NOTE and "THREE" in c1["disclosure"]


def test_disclosure_strings_are_verbatim_and_stage_specific():
    for k in ("guards carry the information", "TS2+H300 was chosen after the screen", "NOT BETTER on its primary", "THREE paired", "26% spread on a guard"):
        assert k in S.STAGE1_NOTE, k
    assert "SINGLE SEED" not in S.STAGE1_NOTE and "THREE" not in S.STAGE2_NOTE
    assert "single seed" not in S.LABEL_STAGE1.lower() and "single seed 50" in S.LABEL_STAGE2
    reg = open(Path(__file__).resolve().parents[1] / "AV2_REGISTRATION.md").read()
    assert "SINGLE SEED (50): one paired difference cannot size seed noise; the A0/A0b screen pair showed 26% spread on a guard. Sign and magnitude only." in reg
    assert "P measures what the hinge enforces by design; a pass is near built-in and is NOT an accuracy claim; the guards carry the information." in reg
    assert "P measures what the hinge enforces by design; a pass is near built-in and is NOT an accuracy claim; the guards carry the information." in S.STAGE2_NOTE


def test_check_scorer_sha(tmp_path):
    f = tmp_path / "s.py"; f.write_text("x=1\n")
    good = A.sha256_file(f)
    assert S.check_scorer_sha(good, f) == good
    with pytest.raises(AssertionError, match="refusing"):
        S.check_scorer_sha("0" * 64, f)
    with pytest.raises(SystemExit):
        S.check_scorer_sha(None, f)
    with pytest.raises(SystemExit):
        S.check_scorer_sha("", f)


def test_registered_pins_match_constants():
    import build_av2_masks as B
    assert S.PINS["segmenter_a_sha"] == B.SEG_A["bin_sha"] == "a3dbf2d7a9912203078f304d88df8cfee9fd752ca2d2413f82036bdb02c5e7df"
    assert S.PINS["segmenter_b_sha"] == B.SEG_B["bin_sha"] == "876855252aa674c7ad6e920c268d6722add811a927806372811faa724f19856c"
    assert B.SEG_A["revision"] == "739f5d4692954e4a185eac280dec1ba5a7d52f1d" and B.SEG_B["revision"] == "8ed649db2cc1756b8bc36f0cad00b0812e333027"
    here = Path(__file__).resolve().parents[1]
    assert A.sha256_file(here / "av2_clips.tsv") == A.REGISTERED_SHAS["clips_sha"] == S.PINS["clips_sha"]
    assert A.sha256_file(here / "av2_val_logs_stage1_50.txt") == S.PINS["logs_sha"]


def test_compare_pairs_end_to_end():
    def run(P, absrel, arm, k):
        return _run(P, absrel, arm, seed=k, ckpt=f"sha_{arm}_{k}")
    def mk(**ov):
        d = {k: (run(600.0, 0.10, "H300", k), run(150.0, 0.10, "TS2", k)) for k in S.PAIRS}
        d.update(ov); return d
    assert S.compare_pairs(mk())["verdict"] == "CONFIRMED"
    r = S.compare_pairs(mk(s51=(run(170.0, 0.10, "H300", "s51"), run(150.0, 0.10, "TS2", "s51")))); assert r["verdict"] == "NOT CONFIRMED" and r["d_min"] < r["ln_margin"] <= r["d_mean"]
    assert S.compare_pairs(mk(s52=(run(140.0, 0.10, "H300", "s52"), run(150.0, 0.10, "TS2", "s52"))))["verdict"] == "REFUTED"
    assert S.compare_pairs(mk(s50=(run(600.0, 0.12, "H300", "s50"), run(150.0, 0.10, "TS2", "s50"))))["verdict"] == "NOT CONFIRMED"
    with pytest.raises(AssertionError):
        S.compare_pairs({k: v for k, v in mk().items() if k != "s52"})
    swapped = mk(); swapped["s50"] = (swapped["s50"][1], swapped["s50"][0])                      # TS2 given as H300
    with pytest.raises(AssertionError, match="argument order"):
        S.compare_pairs(swapped)


def test_route_frame_drop_changes_guards_not_P_and_reverse():
    """R1 M4 at the routing level: a mask-dropped frame is out of P only (guards see it); a kept frame is in both."""
    f = ramp(); gt = sparse(f, 20000); anys = np.zeros((H, W), bool)
    cert = np.zeros((H, W), bool); cert[:600] = True
    pred = np.full((504, 378), 20.0, np.float32); pred[:150] = 450.0
    acc, sky = S.Accumulator(), S.SkyAccumulator()
    assert S.route_frame(acc, sky, "L:1", pred, cert, anys, gt, "L", {"L:1"}) is True                 # dropped
    assert acc.scored_frames == 1 and acc.result()["overall_n_px"] > 0                                 # the guard accumulator CHANGED
    assert sky.seen == 0 and sky.result()["n_frames_kept"] == 0                                         # the sky accumulator did NOT
    assert S.route_frame(acc, sky, "L:2", pred, cert, anys, gt, "L", {"L:1"}) is False                 # not dropped
    assert acc.scored_frames == 2 and sky.seen == 1 and sky.kept_ids == ["L:2"]
    # the reverse: a frame that is not mask-dropped but fails the scorer's own <2000 px rule is out of P, still in the guards
    tiny = np.zeros((H, W), bool); tiny[:5] = True
    S.route_frame(acc, sky, "L:3", pred, tiny, anys, gt, "L", set())
    assert acc.scored_frames == 3 and sky.seen == 2 and sky.dropped == 1 and sky.dropped_ids == ["L:3"] and sky.kept_ids == ["L:2"]


@pytest.mark.parametrize("iou,ok", [(0.85, True), (0.8500001, True), (0.8499, False), (0.8, False), (1.0, True), (float("nan"), False), (True, False), (False, False),
                                    (float("inf"), False), (1.0000001, False), (-0.1, False), (None, False), ("0.9", False)])
def test_check_mask_meta_iou_boundary_and_garbage(iou, ok):
    m = dict(status="OK", pooled_iou=iou, dropped=[])
    if ok:
        S.check_mask_meta(m)
    else:
        with pytest.raises(RuntimeError):
            S.check_mask_meta(m)


def test_pairing_window_is_inclusive_25ms_both_selectors():
    """One camera frame (so the phase search cannot rescue it): |dt| = 25,000,000 ns is accepted, 25,000,001 ns (25.000001 ms) is dropped, on either side of the camera frame."""
    import select_av2_frames as Sel
    t0 = 10**15
    for sign in (+1, -1):
        for dt, scored in ((25_000_000, True), (25_000_001, False), (24_999_999, True)):
            r = A.select_frames([t0], [t0 + sign * dt]); s = Sel.select([t0], [t0 + sign * dt])
            assert (len(r["scored"]) == 1) == scored and (len(r["dropped"]) == 1) == (not scored), (sign, dt)
            assert s[1] == r["scored"] and s[2] == r["dropped"]
            if scored:
                assert r["scored"][0][3] == dt
            else:
                assert r["dropped"][0][2] == dt


def test_sky_accumulator_records_ids_and_mask_px():
    sky = np.zeros((H, W), bool); sky[:600] = True
    pred = np.full((504, 378), 300.0, np.float32)
    k = S.SkyAccumulator(); k.add(pred, sky, "a"); k.add(pred, np.zeros((H, W), bool), "b")
    r = k.result()
    assert r["kept_frame_ids"] == ["a"] and r["dropped_frame_ids"] == ["b"] and r["frame_mask_px"][1] == 0 and r["frame_mask_px"][0] > 2000


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
    # forced fail: no sweep within 25 ms -> dropped, counted
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
    bins_ok = {lab: (small if lab == "[120,200]" else big) for lab in S.bin_labels()}
    g = S.guard_check(bins_ok, bins_ok)
    assert g["[120,200]"]["status"] == "UNEVALUABLE" and g["[0,20)"]["status"] == "PASS"
    assert A.MAX_PAIR_DT_NS == 25_000_000


def test_stage1_pair_rule():
    import math
    ln15 = math.log(1.5)
    big = ln15 + 0.1
    assert S.decide_pairs([big, big, big], True) == "CONFIRMED"
    assert S.decide_pairs([big, big, ln15 - 0.01], True) == "NOT CONFIRMED"          # one pair below the margin even if the mean passes
    assert S.decide_pairs([big + 1, big + 1, ln15 - 0.01], True) == "NOT CONFIRMED"
    assert S.decide_pairs([big, big, 0.0], True) == "REFUTED"                          # any d_i <= 0
    assert S.decide_pairs([big, big, big], False) == "NOT CONFIRMED"                   # a failed guard
    assert S.decide_pairs([big, big, big], True, ["s51:[120,200]"]) == "NOT CONFIRMED (s51:[120,200] guard unevaluable)"
    assert S.decide_pairs([big, big], True).startswith("INVALID")
    assert S.PAIRS == ("s50", "s51", "s52")


def test_phase_rule_r1_picks_aligned_phase_and_ties_go_to_zero():
    cam = list(range(0, 300 * 50_000_000, 50_000_000))                        # 20 Hz
    aligned0 = list(range(10_000_000, 150 * 100_000_000, 100_000_000))         # sweeps 10 ms after even cam frames
    mis0 = list(range(60_000_000, 150 * 100_000_000, 100_000_000))             # sweeps 60 ms after even frames = 10 ms before ODD frames; phase 0 is 40 ms off
    r0 = A.select_frames(cam, aligned0); r1 = A.select_frames(cam, mis0)
    assert r0["phase"] == 0 and len(r0["scored"]) == 15
    assert r1["phase"] == 1 and len(r1["scored"]) == 15 and r1["clip"][0] == cam[1]       # forced-fail case: filed phase-0-only rule would score 0 frames
    # the filed (phase 0 only) rule on the same log scores nothing, so the test fails if the phase search is removed
    assert A._select_phase(sorted(cam), np.array(mis0, np.int64), 0)[1] == []
    # tie (both phases aligned: sweeps every 50 ms, i.e. one per camera frame) -> 0
    both = list(range(10_000_000, 300 * 50_000_000, 50_000_000))
    assert A.select_frames(cam, both)["phase"] == 0
    import select_av2_frames as Sel
    assert Sel.select(cam, mis0)[0][0] == cam[1] and Sel.select(cam, aligned0)[0][0] == cam[0]


# ------------------------------------------------------------------ runner (O45: expected scorer sha is a literal committed in the runbook)
def _runner_args(tmp_path):
    return ["--ckpt", "x.pt", "--pair", "s50", "--arm", "H300", "--out", str(tmp_path / "o.json"), "--root", "r", "--mask", "m", "--frames", "f", "--clips", "c", "--objects", "o", "--logs", "l", "--sums", "s"]


def test_runbook_placeholder_is_refused_by_the_runner(tmp_path):
    import run_score as R
    rb = Path(__file__).resolve().parents[1] / "RUNBOOK_STAGE1.md"
    assert "EXPECT_SCORER_SHA = <FILLED_AFTER_FINAL_COMMIT>" in rb.read_text() or R.read_expected_sha(rb.read_text())     # unfilled placeholder, or a filled 64-hex literal
    for bad in ("EXPECT_SCORER_SHA = <FILLED_AFTER_FINAL_COMMIT>", "EXPECT_SCORER_SHA = abc", "EXPECT_SCORER_SHA = " + "A" * 64, "no line here",
                "EXPECT_SCORER_SHA = " + "a" * 64 + "\nEXPECT_SCORER_SHA = " + "b" * 64):
        with pytest.raises(SystemExit):
            R.read_expected_sha(bad)
    assert R.read_expected_sha("x\nEXPECT_SCORER_SHA = " + "a" * 64 + "\ny") == "a" * 64
    f = tmp_path / "rb.md"; f.write_text("EXPECT_SCORER_SHA = <FILLED_AFTER_FINAL_COMMIT>\n")
    with pytest.raises(SystemExit):
        R.main([str(f), "--", *_runner_args(tmp_path)])


def test_runner_with_wrong_registered_sha_makes_the_scorer_refuse(tmp_path):
    import subprocess, run_score as R
    here = Path(__file__).resolve().parents[1]
    wrong = tmp_path / "wrong.md"; wrong.write_text("EXPECT_SCORER_SHA = " + "0" * 64 + "\n")
    p = subprocess.run([sys.executable, "-I", str(here / "run_score.py"), str(wrong), "--", *_runner_args(tmp_path)], capture_output=True, text=True)
    assert p.returncode != 0 and "scorer sha" in p.stderr and "refusing to run" in p.stderr
    good = tmp_path / "good.md"; good.write_text("EXPECT_SCORER_SHA = " + A.sha256_file(here / "av2_scorer.py") + "\n")
    p = subprocess.run([sys.executable, "-I", str(here / "run_score.py"), str(good), "--", *_runner_args(tmp_path)], capture_output=True, text=True)
    assert "refusing to run" not in p.stderr and p.returncode != 0          # passes the scorer-sha gate, then fails later on the dummy paths
    src = (here / "run_score.py").read_text()
    assert "sha256" not in src and "hashlib" not in src                      # the runner never hashes the scorer


# ---------------------------------------------------------------- N4 / pins: the in-file code pins equal the actual files; the RUNBOOK table agrees
def test_pins_equal_actual_file_shas():
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[3]
    actual = dict(dataset_module_sha=here / "av2_dataset.py", builder_sha=here / "build_av2_masks.py", make_sky_mask_sha=here.parent / "seg" / "make_sky_mask.py",
                  ddad_scorer_sha=repo / "outdoor_debug" / "parts" / "LR_DEV" / "ddad_scorer.py")
    for k, f in actual.items():
        assert S.PINS[k] == A.sha256_file(f), f"PINS.{k} is not the sha of {f}"
    assert S.PINS["ddad_scorer_sha"] == "8e71511b1192a1a15797a71482c40b547bac7d905641b7d2f145ad1da079a503"
    rb = (here / "RUNBOOK_STAGE1.md").read_text()
    for k in actual:
        assert S.PINS[k] in rb, f"RUNBOOK_STAGE1.md pin table does not carry PINS.{k}"
    for k in ("mask_sha", "meta_sha", "sums_sha"):
        assert S.PINS[k] == S.UNSET                                              # these fill after the mask build / download


def test_registered_rule_literal_equals_the_builders_rule():
    import json as _json, make_sky_mask as M
    assert S.REGISTERED_RULE == _json.loads(_json.dumps(M.RULE))
    assert (S.REGISTERED_RULE["prob_thresh"], S.REGISTERED_RULE["erode_ref_px"], S.REGISTERED_RULE["lidar_ref_px"], S.REGISTERED_RULE["drop_frac"], S.REGISTERED_RULE["any_p"],
            S.REGISTERED_RULE["iou_min"], S.REGISTERED_RULE["work_short"], S.REGISTERED_RULE["seed"]) == (0.9, 8, 5, 0.01, 0.5, 0.85, 512, 0)


def test_check_mask_provenance_unit():
    good = dict(rule=dict(S.REGISTERED_RULE), builder_sha="a" * 64, make_sky_mask_sha="b" * 64, dataset_module_sha="c" * 64)
    S.check_mask_provenance(good)
    for bad in (dict(good, rule=dict(S.REGISTERED_RULE, prob_thresh=0.8)), dict(good, rule=None), {k: v for k, v in good.items() if k != "builder_sha"}, dict(good, make_sky_mask_sha="short")):
        with pytest.raises(AssertionError):
            S.check_mask_provenance(bad)


def test_runbook_registry_table_is_present_and_unset_until_the_runs_exist():
    reg = S.parse_registry((Path(__file__).resolve().parents[1] / "av2_stage1_ckpts.txt").read_text())
    assert sorted(reg) == sorted((p, a) for p in S.PAIRS for a in ("H300", "TS2"))
    assert all(v == S.UNSET or len(v) == 64 for v in reg.values())
    if S.UNSET in reg.values():                                                  # as long as any is UNSET no run can be accepted
        with pytest.raises(AssertionError, match="UNSET"):
            S.assert_registered_ckpt(reg, "s50", "H300", "a" * 64)


def test_runner_passes_the_runbook_as_registry_and_the_literal_sha(tmp_path, monkeypatch):
    import subprocess, run_score as R
    rb = tmp_path / "rb.md"; rb.write_text("EXPECT_SCORER_SHA = " + "a" * 64 + "\n")
    seen = {}
    monkeypatch.setattr(subprocess, "call", lambda cmd: seen.setdefault("cmd", cmd) and 0)
    R.main([str(rb), "--", "--compare-pairs", "x"])
    cmd = seen["cmd"]
    assert cmd[cmd.index("--expect-scorer-sha") + 1] == "a" * 64 and cmd[cmd.index("--registry") + 1] == str(Path(R.__file__).resolve().parent / "av2_stage1_ckpts.txt")


# ---------------------------------------------------------------- D6: the guard table must be exactly the five registered bins
def test_D6_guard_check_refuses_empty_short_extra_or_renamed_bins():
    ok = _bins(0.1)
    assert len(S.guard_check(ok, ok)) == 5
    for bad in ({}, {k: v for k, v in list(ok.items())[:4]}, dict(ok, **{"[200,300]": ok["[0,20)"]}), {("[0,21)" if k == "[0,20)" else k): v for k, v in ok.items()}, list(ok.items())):
        for pair in ((bad, ok), (ok, bad)):
            with pytest.raises(AssertionError, match="five registered labels"):
                S.guard_check(*pair)
    with pytest.raises(AssertionError, match="five registered labels"):
        S.guard_check({}, {})                                                  # the vacuous all([]) pass
    h, t = _run(600.0, 0.1, "H300"), _run(150.0, 0.1, "TS2"); h["bins"] = {}; t["bins"] = {}
    with pytest.raises(AssertionError, match="five registered labels"):
        S.compare(h, t, stage=1)


# ---------------------------------------------------------------- D1: registry in its own pinned file
def test_D1_registry_is_its_own_file_pinned_by_sha_in_the_order_of_the_runbook():
    here = Path(__file__).resolve().parents[1]
    f = here / "av2_stage1_ckpts.txt"
    reg = S.parse_registry(f.read_text())
    assert not re.search(r"^CKPT_SHA\s", (here / "RUNBOOK_STAGE1.md").read_text(), re.M)      # the runbook holds no registry lines any more
    assert "registry_sha" in S.PINS
    if S.PINS["registry_sha"] == S.UNSET:
        pass
    else:                                                                       # filled pin: it is the file's sha and the registry is complete
        assert S.PINS["registry_sha"] == A.sha256_file(f) and S.UNSET not in reg.values()
    if S.UNSET in reg.values():                                                 # acyclic order: registry first, pin second
        assert S.PINS["registry_sha"] == S.UNSET
    rb = (here / "RUNBOOK_STAGE1.md").read_text()
    assert "train -> registry" in rb and "dryrun_ckpt_facts.py" in rb


def test_D1_check_registry_pin_refuses_unset_and_mismatch():
    S.check_registry_pin("a" * 64, {"registry_sha": "a" * 64})
    with pytest.raises(AssertionError, match="has not been filled"):
        S.check_registry_pin("a" * 64, {"registry_sha": S.UNSET})
    with pytest.raises(AssertionError, match="has not been filled"):
        S.check_registry_pin("a" * 64, {})
    with pytest.raises(AssertionError, match="pinned registry_sha"):
        S.check_registry_pin("a" * 64, {"registry_sha": "b" * 64})
    with pytest.raises(AssertionError, match="registry_sha is UNSET"):
        S.check_registry_pin("a" * 64)                                          # the real PINS: still UNSET


# ---------------------------------------------------------------- D2: cfg hash, recursive diff, forbidden keys
def test_D2_cfg_diff_is_recursive_and_exact():
    a = {"x": {"y": 1, "z": [1, 2]}, "k": 1}
    assert S.cfg_diff(a, {"x": {"y": 1, "z": [1, 2]}, "k": 1}) == set()
    assert S.cfg_diff(a, {"x": {"y": 2, "z": [1, 2]}, "k": 1}) == {"x.y"}
    assert S.cfg_diff(a, {"x": {"y": 1, "z": [1, 3]}, "k": 1}) == {"x.z"}
    assert S.cfg_diff(a, {"x": {"y": 1}, "k": 1, "n": 0}) == {"x.z", "n"}
    assert S.cfg_diff({"v": 1}, {"v": True}) == {"v"} and S.cfg_diff({"v": 1}, {"v": 1.0}) == {"v"}       # type changes count
    assert S.cfg_sha256(a) == S.cfg_sha256({"k": 1, "x": {"z": [1, 2], "y": 1}}) != S.cfg_sha256({"k": 2, "x": a["x"]})


def test_D2_declared_pair_cfgs_differ_only_in_hinge_keys_and_checkpoint_dir():
    for p in S.PAIRS:
        h, t = _run(600.0, 0.1, "H300", seed=p), _run(150.0, 0.1, "TS2", seed=p)
        assert S.check_pair_cfgs(h, t) == S.ALLOWED_PAIR_DIFF
        assert h["meta"]["cfg"]["loss"]["sky_vda_zero_disparity"] is True and t["meta"]["cfg"]["loss"]["sky_vda_zero_disparity"] is True


def test_D2_forced_fails_each_refuses():
    import copy
    def pair(fn_h=None, fn_t=None, rehash=True):
        h, t = _run(600.0, 0.1, "H300"), _run(150.0, 0.1, "TS2")
        for r, fn in ((h, fn_h), (t, fn_t)):
            if fn:
                fn(r["meta"]["cfg"])
                if rehash:
                    r["meta"]["cfg_sha256"] = S.cfg_sha256(r["meta"]["cfg"])
        return h, t
    # a pair that differs in anything else (learning rate, data seed ...)
    for fn, m in ((lambda c: c["training"].__setitem__("learning_rate", 0.5), "outside the declared"), (lambda c: c["model"].__setitem__("lora_rank", 8), "outside the declared"),
                  (lambda c: c["loss"].__setitem__("tgm_weight", 1.0), "outside the declared")):
        with pytest.raises(AssertionError, match=m):
            S.check_pair_cfgs(*pair(fn_t=fn))
    # forbidden keys (scorer-level facts, both arms)
    with pytest.raises(AssertionError, match="sky_hinge_max_depth"):
        S.check_pair_cfgs(*pair(fn_h=lambda c: c["loss"].__setitem__("sky_hinge_max_depth", 310.0)))
    with pytest.raises(AssertionError, match="sky_vda_zero_disparity"):
        S.check_pair_cfgs(*pair(fn_t=lambda c: c["loss"].__setitem__("sky_vda_zero_disparity", False)))
    with pytest.raises(AssertionError, match="sky_vda_zero_disparity"):
        S.check_pair_cfgs(*pair(fn_h=lambda c: c["loss"].pop("sky_vda_zero_disparity")))
    S.check_pair_cfgs(*pair(fn_h=lambda c: c["loss"].__setitem__("sky_hinge_max_depth", None)))                    # null is allowed
    # hash does not match the stored cfg
    with pytest.raises(AssertionError, match="cfg_sha256"):
        S.check_pair_cfgs(*pair(fn_t=lambda c: c["training"].__setitem__("learning_rate", 0.5), rehash=False))
    # no cfg stored at all
    h, t = pair(); del h["meta"]["cfg"]
    with pytest.raises(AssertionError, match="no stored cfg"):
        S.check_pair_cfgs(h, t)
    # two arms that do not differ in the hinge keys (e.g. one cfg under both labels) are refused as well
    h, t = _run(600.0, 0.1, "H300"), _run(150.0, 0.1, "TS2")
    t["meta"]["cfg"] = copy.deepcopy(h["meta"]["cfg"]); t["meta"]["cfg_sha256"] = h["meta"]["cfg_sha256"]
    with pytest.raises(AssertionError, match="arm TS2"):
        S.check_pair_cfgs(h, t)


# ---------------------------------------------------------------- D3: arm label re-asserted against the cfg; run json shas
def test_D3_forged_arm_label_contradicts_the_stored_cfg():
    h, t = _run(600.0, 0.1, "H300"), _run(150.0, 0.1, "TS2")
    h["meta"]["arm"], t["meta"]["arm"] = "TS2", "H300"                          # arm + ckpt labels swapped between the two jsons (attack A4); cfgs keep their hinge fields
    with pytest.raises(AssertionError, match="arm H300|arm TS2"):
        S.check_run_cfg(h)
    h2 = _run(600.0, 0.1, "H300"); h2["meta"]["cfg_sky_hinge_weight"] = 0.0     # a meta fact that contradicts the stored cfg
    with pytest.raises(AssertionError, match="re-derived"):
        S.check_run_cfg(h2)


def test_D3_run_shas_listing(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"; a.write_text("{}"); b.write_text("[]")
    ha, hb = A.sha256_file(a), A.sha256_file(b)
    lf = tmp_path / "l.txt"; lf.write_text(f"# c\n{ha}  a.json\n{hb} *b.json\n")
    L = S.load_run_shas(lf)
    S.check_run_shas([a, b], L, exact=True)
    S.check_run_shas([a], L)
    with pytest.raises(AssertionError, match="beyond/besides"):
        S.check_run_shas([a], L, exact=True)
    with pytest.raises(AssertionError, match="not in the committed"):
        S.check_run_shas([a, tmp_path / "c.json"], L)
    a.write_text("{ }")                                                         # edited after the sha was committed
    with pytest.raises(AssertionError, match="sha256"):
        S.check_run_shas([a, b], L)
    with pytest.raises(SystemExit, match="run-shas"):
        S.load_run_shas(None)
    for bad in (f"{ha}  a.json\n{ha}  a.json\n", "zz  a.json\n"):
        lf.write_text(bad)
        with pytest.raises(AssertionError, match="malformed or duplicate"):
            S.load_run_shas(lf)
    with pytest.raises(AssertionError, match="unique"):
        S.check_run_shas([a, tmp_path / "x" / "a.json"], {"a.json": ha})


# ---------------------------------------------------------------- D4: scorer sha and pins of every run
def test_D4_run_provenance():
    pins = dict(frames_sha="f", x="y")
    r = {"meta": dict(scorer_sha="s" * 64, pins=dict(pins))}
    S.check_run_provenance(r, "s" * 64, pins)
    with pytest.raises(AssertionError, match="scorer_sha"):
        S.check_run_provenance(r, "t" * 64, pins)
    for bad in (dict(frames_sha="OTHER", x="y"), dict(frames_sha="f"), dict(frames_sha="f", x="y", z="w"), None):
        with pytest.raises(AssertionError, match="pins|pin "):
            S.check_run_provenance({"meta": dict(scorer_sha="s" * 64, pins=bad)}, "s" * 64, pins)
    with pytest.raises(AssertionError, match="UNSET"):
        S.check_run_provenance({"meta": dict(scorer_sha="s" * 64, pins=dict(frames_sha=S.UNSET, x="y"))}, "s" * 64, dict(frames_sha=S.UNSET, x="y"))


def test_code_sha_pins_are_the_runbook_literals():
    """The in-file code-sha PINS equal the literals in RUNBOOK_STAGE1.md (independent of the file contents, so an edit of a pinned file is caught by its own dedicated test, not by this one)."""
    rb = (Path(__file__).resolve().parents[1] / "RUNBOOK_STAGE1.md").read_text()
    for k in ("dataset_module_sha", "builder_sha", "make_sky_mask_sha", "ddad_scorer_sha", "bench_joint_sha", "segmenter_a_sha", "segmenter_b_sha", "frames_sha", "clips_sha", "objects_sha", "logs_sha"):
        assert S.PINS[k] in rb, f"PINS.{k} is not the literal in RUNBOOK_STAGE1.md"

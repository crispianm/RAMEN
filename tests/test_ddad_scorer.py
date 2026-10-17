"""Synthetic-only tests for outdoor_debug/parts/LR_DEV/ddad_scorer.py (no DDAD data, no model, CPU)."""
import shutil, sys
from pathlib import Path
import numpy as np
import pytest

LR = Path(__file__).resolve().parents[1] / "outdoor_debug" / "parts" / "LR_DEV"
sys.path.insert(0, str(LR))
import ddad_scorer as DS
import ddad_lr_dataset as D

H, W = D.NATIVE_HW
LABS = DS.bin_labels()


def sparse(depths, rng):
    gt = np.full((H, W), np.nan, np.float32)
    idx = rng.choice(H * W, len(depths), replace=False)
    gt.flat[idx] = depths
    return gt


def depths_all_bins(rng, n=4000):
    return np.concatenate([rng.uniform(lo + 0.01, hi - 0.01, n) for lo, hi in DS.BINS_M]).astype(np.float32)


def run(frames, pred_fn, scene_fn=lambda i: "s0"):
    acc = DS.Accumulator()
    for i, gt in enumerate(frames):
        DS.score_frame(acc, pred_fn(gt, i), gt, scene_fn(i))
    return acc.result()


def test_pred_09_gt_every_bin():
    rng = np.random.default_rng(1)
    frames = [sparse(depths_all_bins(rng), rng) for _ in range(3)]
    r = run(frames, lambda gt, i: 0.9 * np.nan_to_num(gt))     # full-res pred = 0.9*GT at the sampled px (resize is identity at native)
    for lab in LABS:
        b = r["bins"][lab]
        assert b["median_pred_over_gt"] == pytest.approx(0.9, abs=1e-6), lab
        assert b["absrel_unaligned_pooled"] == pytest.approx(0.1, abs=1e-6), lab
        assert b["n_px"] == 3 * 4000 and b["n_frames"] == 3
    assert r["overall_absrel_0_200"] == pytest.approx(0.1, abs=1e-6)
    print("TEST1", {k: (v["median_pred_over_gt"], v["absrel_unaligned_pooled"]) for k, v in r["bins"].items()})


def _frames_framefactor(rng):
    return [sparse(depths_all_bins(rng, 500), rng) for _ in range(4)]


FACT = [0.5, 0.9, 1.3, 2.0]


def test_forced_fail_gt_resized():
    """(a) the correct scorer REFUSES a resized GT; (b) a metric computed on a bilinearly downsampled GT gives different numbers."""
    import torch, torch.nn.functional as F
    rng = np.random.default_rng(2)
    gt = sparse(depths_all_bins(rng, 500), rng)
    with pytest.raises(ValueError):
        DS.score_frame(DS.Accumulator(), 0.9 * np.nan_to_num(gt), gt[::2, ::2], "s")
    # wrong path: densify via nan->0, bilinearly downsample GT to half size, upsample back (a "resized-GT" metric) and score
    g0 = torch.from_numpy(np.nan_to_num(gt))[None, None]
    gs = F.interpolate(g0, size=(H // 2, W // 2), mode="bilinear", align_corners=False)
    gwrong = F.interpolate(gs, size=(H, W), mode="nearest")[0, 0].numpy()
    gwrong[gwrong == 0] = np.nan
    acc = DS.Accumulator(); p, g = D.sample_prediction(0.9 * np.nan_to_num(gt), gwrong); acc.add(p, g, "s")
    right = run([gt], lambda g, i: 0.9 * np.nan_to_num(g))
    wrong = acc.result()
    dm = max(abs(right["bins"][l]["median_pred_over_gt"] - wrong["bins"][l]["median_pred_over_gt"]) for l in LABS if wrong["bins"][l]["n_px"])
    dn = sum(right["bins"][l]["n_px"] for l in LABS) - sum(wrong["bins"][l]["n_px"] for l in LABS)
    print("TEST2a right median/absrel", [(right["bins"][l]["median_pred_over_gt"], round(right["bins"][l]["absrel_unaligned_pooled"], 4)) for l in LABS])
    print("TEST2a GT-resized median/absrel", [(wrong["bins"][l]["median_pred_over_gt"], round(wrong["bins"][l]["absrel_unaligned_pooled"], 4)) for l in LABS], "px lost", dn)
    assert dm > 1e-3 or dn != 0, "GT-resize variant is not detectable"


def test_forced_fail_alignment():
    """pred = 0.9*GT*factor(frame). Correct (unaligned) path != per-frame-median-scaled path != per-frame-LSQ-scaled path."""
    rng = np.random.default_rng(3)
    frames = _frames_framefactor(rng)
    pred_fn = lambda gt, i: 0.9 * FACT[i] * np.nan_to_num(gt)
    right = run(frames, pred_fn)
    # wrong: per-frame median scaling to GT before pooling
    acc = DS.Accumulator()
    for i, gt in enumerate(frames):
        p, g = D.sample_prediction(pred_fn(gt, i), gt)
        p = p * (np.median(g) / np.median(p)); acc.add(p, g, "s0")
    aligned = acc.result()
    for lab in LABS:
        assert right["bins"][lab]["median_pred_over_gt"] != pytest.approx(aligned["bins"][lab]["median_pred_over_gt"], abs=1e-3)
    assert right["overall_absrel_0_200"] != pytest.approx(aligned["overall_absrel_0_200"], abs=1e-3)
    # sanity: pooled unaligned median of ratios over equal-size frames is the factor mixture, here 0.9*{0.5,0.9,1.3,2.0}
    exp = np.median(np.repeat(0.9 * np.array(FACT), 500))
    assert right["bins"][LABS[0]]["median_pred_over_gt"] == pytest.approx(exp, rel=1e-6)
    print("TEST2b unaligned overall", right["overall_absrel_0_200"], "| median-aligned overall", aligned["overall_absrel_0_200"],
          "| unaligned bin medians", [round(right["bins"][l]["median_pred_over_gt"], 4) for l in LABS], "| aligned", [round(aligned["bins"][l]["median_pred_over_gt"], 4) for l in LABS])


def test_bin_edges():
    g = np.array([0.0, np.nan, -1.0, 1.0, 19.999, 20.0, 49.999, 50.0, 80.0, 119.999, 120.0, 200.0, 200.0001, 250.0], np.float64)
    v = DS.valid_mask(g)
    assert v.tolist() == [False, False, False, True, True, True, True, True, True, True, True, True, False, False]
    gv = g[v]; ms = DS.bin_masks(gv)
    where = lambda x: [i for i, m in enumerate(ms) if m[list(gv).index(x)]]
    assert where(20.0) == [1] and where(50.0) == [2] and where(80.0) == [3] and where(120.0) == [4] and where(200.0) == [4] and where(19.999) == [0]
    assert sum(m.sum() for m in ms) == len(gv)                       # every valid px in exactly one bin
    acc = DS.Accumulator(); acc.add(np.ones_like(g), g, "s")
    assert acc.result()["overall_n_px"] == int(v.sum())


def test_pooled_not_per_frame():
    rng = np.random.default_rng(4)
    # frame A: 1000 px at 10 m, pred 5 (AbsRel .5); frame B: 10 px at 10 m, pred 15 (AbsRel .5 -> use pred 10, AbsRel 0)
    gtA = sparse(np.full(1000, 10.0, np.float32), rng); gtB = sparse(np.full(10, 10.0, np.float32), rng)
    acc = DS.Accumulator()
    DS.score_frame(acc, np.full((H, W), 5.0, np.float32), gtA, "a")
    DS.score_frame(acc, np.full((H, W), 10.0, np.float32), gtB, "b")
    b = acc.result()["bins"][LABS[0]]
    assert b["absrel_unaligned_pooled"] == pytest.approx(500.0 / 1010.0)         # pooled; per-frame mean would be 0.25
    assert abs(b["absrel_unaligned_pooled"] - 0.25) > 0.2
    assert b["median_pred_over_gt"] == pytest.approx(0.5) and b["n_frames"] == 2 and b["n_scenes"] == 2


def test_sha_assertion_fires_on_altered_frame_list(tmp_path):
    alt = tmp_path / "frames.tsv"
    shutil.copy(LR / "ddad_val_frames.tsv", alt)
    pairs = tmp_path / "pairs.tsv"; pairs.write_text("x\n")
    # unaltered frame list passes the frame check; the (fake) pairs file fails the pairs check
    with pytest.raises(AssertionError, match="pairs sha"):
        DS.assert_pins(alt, pairs)
    # altered frame list fails the frame check first
    alt.write_text(alt.read_text() + "000150\t1\n")
    with pytest.raises(AssertionError, match="frame list sha"):
        DS.assert_pins(alt, pairs)


def test_pins_constants_and_nonfinite_visible():
    assert DS.FRAMES_SHA == D.FRAMES_SHA256 and DS.PAIRS_SHA == D.PAIRS_SHA256 and DS.BJ_SHA == DS.S.BJ_SHA and DS.SHORT_SIDE == 378
    acc = DS.Accumulator(); acc.add(np.array([np.nan, 5.0]), np.array([10.0, 10.0]), "s")
    assert acc.result()["n_nonfinite_pred_px"] == 1


def test_selftest_runs():
    r = DS.selftest()
    assert r["n_scored_frames"] == 6 and r["n_scenes"] == 2
    assert all(0.85 < b["median_pred_over_gt"] < 0.95 for b in r["bins"].values() if b["n_px"])


# ----------------------------------------------------------------------------- A3: release-spec branch
EXACT_LABEL = "released Metric-VDA-S at the registered 378 input (not its default 518), fp16, exploratory DEV, report-only"


def _fake_weights(tmp_path, monkeypatch, content=b"fake-weights"):
    w = tmp_path / "mvda.pth"; w.write_bytes(content)
    monkeypatch.setitem(DS.RELEASE["metric_vda_s"], "sha256", DS.S.sha256_file(w))
    return w


def test_a3_registered_constants():
    r = DS.RELEASE["metric_vda_s"]
    assert DS.RELEASE_LABEL == EXACT_LABEL == r["label"]
    assert r["sha256"] == "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33" and r["size"] == 116444063
    assert r["isambard_path"] == "/projects/b5dh/benchmark/weights/metric_video_depth_anything_vits.pth"


def test_a3_ours_path_meta_unchanged(tmp_path):
    torch = pytest.importorskip("torch")
    ck = tmp_path / "c.pt"
    torch.save(dict(trainable_ema={"w": torch.zeros(1)}, step=12000, cfg=dict(model=dict(arch="shared_temporal"))), ck)
    spec, mm = DS.model_meta(str(ck), None, {})
    assert spec is None                                              # main() then builds spec_prefix + ckpt exactly as before
    assert list(mm) == ["ckpt"] and list(mm["ckpt"]) == ["path", "sha256", "weights_key", "step", "arch"]
    assert mm["ckpt"] == dict(path=str(ck), sha256=DS.S.sha256_file(ck), weights_key="trainable_ema", step=12000, arch="shared_temporal")
    bad = tmp_path / "b.pt"; torch.save(dict(trainable={"w": torch.zeros(1)}), bad)
    with pytest.raises(AssertionError, match="trainable_ema"):
        DS.model_meta(str(bad), None, {})
    st = DS.selftest()
    assert st["label"] == DS.LABEL and "spec" not in st             # ours selftest label unchanged


def test_a3_release_meta_records_sha_path_label(tmp_path, monkeypatch):
    w = _fake_weights(tmp_path, monkeypatch)
    spec, mm = DS.model_meta(None, "metric_vda_s", {"metric_vda_s": str(w)})
    assert spec == "metric_vda_s" and "ckpt" not in mm
    rw = mm["release_weights"]
    assert rw["sha256"] == DS.S.sha256_file(w) and rw["path"] == str(w) and rw["size"] == w.stat().st_size and rw["key"] == "metric_vda_s"
    st = DS.selftest("metric_vda_s")
    assert st["label"] == EXACT_LABEL and st["spec"] == "metric_vda_s" and st["n_scored_frames"] == 6


def test_a3_forced_fail_release(tmp_path, monkeypatch):
    w = _fake_weights(tmp_path, monkeypatch)
    with pytest.raises(AssertionError, match="missing"):
        DS.model_meta(None, "metric_vda_s", {"metric_vda_s": str(tmp_path / "nope.pth")})
    w.write_bytes(b"tampered")                                         # wrong weights file (sha != registered)
    with pytest.raises(AssertionError, match="sha"):
        DS.model_meta(None, "metric_vda_s", {"metric_vda_s": str(w)})
    with pytest.raises(ValueError, match="unknown release spec"):
        DS.model_meta(None, "vda_s", {"vda_s": str(w)})
    with pytest.raises(AssertionError, match="ours"):                  # release spec + an 'ours' checkpoint
        DS.model_meta("ours.pt", "metric_vda_s", {"metric_vda_s": str(w)})
    with pytest.raises(ValueError):
        DS.release_meta("nope", {})
    w2 = _fake_weights(tmp_path, monkeypatch)
    monkeypatch.setitem(DS.RELEASE["metric_vda_s"], "label", EXACT_LABEL + " (518)")   # tampered label
    with pytest.raises(AssertionError, match="label"):
        DS.release_meta("metric_vda_s", {"metric_vda_s": str(w2)})


def test_a3_cli_rejects_spec_plus_ckpt(tmp_path):
    with pytest.raises(AssertionError, match="ours"):
        DS.main(["--spec", "metric_vda_s", "--ckpt", "x.pt", "--out", str(tmp_path / "o.json")])
    with pytest.raises(AssertionError, match="required"):
        DS.main(["--out", str(tmp_path / "o.json")])


def test_a3_shape_assert_fires_on_resampled_pipeline():
    torch = pytest.importorskip("torch")
    x = torch.zeros(4, 3, 378, 602)
    DS.assert_same_hw(torch.zeros(4, 378, 602), x)          # same HxW passes
    with pytest.raises(AssertionError, match="extra resample"):
        DS.assert_same_hw(torch.zeros(4, 400, 602), x)               # monkeypatched predict returned a different HxW


# ---------------------------------------------------------------- A4: release spec at 518 (REGISTRATION.md A4)
def test_a4_default_is_registered_378():
    assert DS.resolve_short_side(None, None) == 378
    assert DS.resolve_short_side("metric_vda_s", None) == 378
    assert DS.resolve_short_side("metric_vda_s", 378) == 378
    assert DS.resolve_short_side(None, 378) == 378        # ours at its own 378 is fine


def test_a4_release_may_use_518():
    assert DS.resolve_short_side("metric_vda_s", 518) == 518
    assert "518 input" in DS.RELEASE_LABEL_518 and "report-only" in DS.RELEASE_LABEL_518 and "registered 378" in DS.RELEASE_LABEL


@pytest.mark.parametrize("spec,ss", [(None, 518), (None, 448), ("metric_vda_s", 448), ("metric_vda_s", 0), ("unknown_spec", 518)])
def test_a4_forced_fail_ours_or_unregistered_cannot_change_input(spec, ss):
    with pytest.raises(ValueError):
        DS.resolve_short_side(spec, ss)


def test_a4_model_input_short_side_param_default_unchanged():
    import inspect
    assert inspect.signature(DS.model_input).parameters["short_side"].default == 378


def _fake_rgb(monkeypatch):
    monkeypatch.setattr(DS.D, "load_rgb", lambda p: np.full((1216, 1936, 3), 128, np.uint8))


@pytest.mark.parametrize("ss,hw", [(378, (378, 602)), (518, (518, 826))])
def test_a4_model_input_real_resize_measured_shape(monkeypatch, ss, hw):
    """Measured effect on the real bench_joint Resize (not a config key): the tensor that reaches the model has the size-dependent shape."""
    pytest.importorskip("torch")
    _fake_rgb(monkeypatch)
    x = DS.model_input([Path("a.png"), Path("b.png")], short_side=ss)
    assert tuple(x.shape) == (2, 3) + hw
    x0 = DS.model_input([Path("a.png")])                    # default arg == registered 378 path
    assert tuple(x0.shape[-2:]) == (378, 602)


def test_a4_forced_fail_wrong_expected_shape_is_detected(monkeypatch):
    """Control: the shape assertion used above must be able to fail (518 input is NOT the 378 shape)."""
    pytest.importorskip("torch")
    _fake_rgb(monkeypatch)
    x = DS.model_input([Path("a.png")], short_side=518)
    with pytest.raises(AssertionError):
        assert tuple(x.shape[-2:]) == (378, 602)


def _run_main_fake(monkeypatch, tmp_path, argv_extra, spec_args=("--spec", "metric_vda_s")):
    """Runs the REAL main() with a fake pipeline recording the shape of the tensor handed to run_pipeline. Needs CUDA only for .to('cuda')."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs cuda for the .to(device)/autocast asserts (tiny tensors only)")
    import bench_joint as bj
    seen = []
    monkeypatch.setattr(bj, "build_pipeline", lambda spec, dev: {"metric": True, "spec": spec})
    def fake_run(p, x, s, b, out_hw):
        seen.append(tuple(x.shape)); return torch.zeros(x.shape[0], *x.shape[-2:], device=x.device), None
    monkeypatch.setattr(bj, "run_pipeline", fake_run)
    monkeypatch.setattr(DS, "model_meta", lambda ckpt, spec, paths: (spec, dict(spec=spec)))
    monkeypatch.setattr(DS, "assert_pins", lambda *a, **k: {})
    monkeypatch.setattr(DS.D, "read_frame_list", lambda *a, **k: None)
    monkeypatch.setattr(DS.D, "read_pairs", lambda *a, **k: None)
    monkeypatch.setattr(DS.D, "load_clips", lambda lim=None: [("s0", [dict(image="a.png", scored=False), dict(image="b.png", scored=False)])])
    monkeypatch.setattr(DS.D, "load_rgb", lambda p: np.full((1216, 1936, 3), 128, np.uint8))
    out = tmp_path / "o.json"
    DS.main([*spec_args, "--out", str(out), "--device", "cuda", "--limit-scenes", "1", *argv_extra])
    import json
    return seen, json.loads(out.read_text())


@pytest.mark.parametrize("extra,hw,short,label", [
    ([], (378, 602), 378, "RELEASE_LABEL"),
    (["--release-short-side", "378"], (378, 602), 378, "RELEASE_LABEL"),
    (["--release-short-side", "518"], (518, 826), 518, "RELEASE_LABEL_518")])
def test_a4_main_instantiated_pipeline_sees_the_size_and_label(monkeypatch, tmp_path, capsys, extra, hw, short, label):
    seen, res = _run_main_fake(monkeypatch, tmp_path, extra)
    assert seen and all(s[-2:] == hw for s in seen), seen
    assert res["meta"]["short_side"] == short and res["label"] == res["meta"]["label"] == getattr(DS, label)
    assert f"model input {(2, 3) + hw}" in capsys.readouterr().out      # the first-run shape log line


def test_a4_main_refuses_518_for_ours_before_any_model_or_output(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(DS, "assert_pins", lambda *a, **k: called.append("pins") or {})
    out = tmp_path / "o.json"
    with pytest.raises(ValueError, match="refused"):
        DS.main(["--ckpt", "x.pt", "--release-short-side", "518", "--out", str(out)])
    with pytest.raises(ValueError, match="refused"):
        DS.main(["--spec", "metric_vda_s", "--release-short-side", "448", "--out", str(out)])
    with pytest.raises(ValueError, match="refused"):
        DS.main(["--spec", "unknown_spec", "--release-short-side", "518", "--out", str(out)])
    assert not called and not out.exists()

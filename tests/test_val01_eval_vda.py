"""scripts/val01_eval_vda.py reuses VDA's own benchmark/eval/eval.py + metric.py and
video_depth_anything.video_depth.VideoDepthAnything directly rather than reimplementing
them. These tests cover the pure logic (manifest parsing, path construction, KITTI-self-consistency
comparison arithmetic) hermetically, plus the run_official_eval/run_official_inference
wiring against a fully scripted fake eval/metric/video_depth module (same pattern as
tests/test_compute_loss_vda_mode.py). A separate, non-committed real-code check against
the actual vendored eval.py (available locally) verified the real alignment+clipping
behaviour directly -- see provenance/inventory/val01_eval_vda_alignment_check_20260827.json."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, "scripts")
import val01_eval_vda as ev  # noqa: E402


def test_infer_npy_path_matches_vda_convention():
    assert ev.infer_npy_path("/out", "kitti", "drive1/rgb/0000000005.png") == \
        "/out/kitti/drive1/rgb/0000000005.npy"
    assert ev.infer_npy_path("/out", "sintel", "scene/frame_01.jpg") == \
        "/out/sintel/scene/frame_01.npy"


def test_load_manifest_clips_parses_the_real_vda_format(tmp_path):
    manifest = {
        "kitti": [
            {"drive_a": [{"image": "drive_a/rgb/0.png", "gt_depth": "drive_a/depth/0.png", "factor": 256.0}]},
            {"drive_b": [{"image": "drive_b/rgb/0.png", "gt_depth": "drive_b/depth/0.png", "factor": 256.0}]},
        ]
    }
    path = tmp_path / "kitti_video.json"
    path.write_text(json.dumps(manifest))
    clips = ev.load_manifest_clips(str(path), "kitti")
    assert [c[0] for c in clips] == ["drive_a", "drive_b"]
    assert clips[0][1][0]["factor"] == 256.0


def test_check_kitti_self_consistency_pass():
    result = ev.check_kitti_self_consistency(measured_abs_rel=0.0931, measured_delta1=0.9333)
    assert result["overall"] == "PASS"
    assert result["abs_rel_result"] == "PASS"
    assert result["delta1_result"] == "PASS"


def test_check_kitti_self_consistency_fails_outside_abs_rel_tolerance():
    result = ev.check_kitti_self_consistency(measured_abs_rel=0.15, measured_delta1=0.9333)
    assert result["overall"] == "FAIL"
    assert result["abs_rel_result"] == "FAIL"


def test_check_kitti_self_consistency_fails_outside_delta1_tolerance():
    result = ev.check_kitti_self_consistency(measured_abs_rel=0.0931, measured_delta1=0.80)
    assert result["overall"] == "FAIL"
    assert result["delta1_result"] == "FAIL"


class _FakeEvalModule:
    eval_metrics = ["abs_relative_difference", "rmse_linear", "delta1_acc"]
    calls = []

    @staticmethod
    def eval_depthcrafter(infer_paths, depth_gt_paths, factors, args):
        _FakeEvalModule.calls.append(dict(infer_paths=list(infer_paths), depth_gt_paths=list(depth_gt_paths),
                                          factors=list(factors), max_eval_len=args.max_eval_len))
        return [0.1, 0.2, 0.9]   # one fixed metric triple per clip

    @staticmethod
    def get_gt(depth_gt_path, gt_factor, args):
        """Matches the real vendored eval.py::get_gt semantics used by
        run_unaligned_eval: np.load a raw array, divide by the factor."""
        return np.load(depth_gt_path) / gt_factor

    @staticmethod
    def get_infer(infer_path, args, target_size=None):
        return np.load(infer_path)


class _FakeMetricModule:
    pass


class _FakeVideoDepthAnything:
    init_kwargs = []

    def __init__(self, **kwargs):
        _FakeVideoDepthAnything.init_kwargs.append(kwargs)

    def load_state_dict(self, sd, strict=True):
        pass

    def to(self, device):
        return self

    def eval(self):
        return self

    def infer_video_depth(self, videos, target_fps, input_size, device, fp32):
        return np.zeros((videos.shape[0], 4, 4), dtype=np.float32), 1.0


@pytest.fixture
def fake_vda_repo(monkeypatch, tmp_path):
    _FakeEvalModule.calls.clear()
    _FakeVideoDepthAnything.init_kwargs.clear()
    pkg = types.ModuleType("video_depth_anything")
    pkg.__path__ = []
    vd_mod = types.ModuleType("video_depth_anything.video_depth")
    vd_mod.VideoDepthAnything = _FakeVideoDepthAnything
    monkeypatch.setitem(sys.modules, "video_depth_anything", pkg)
    monkeypatch.setitem(sys.modules, "video_depth_anything.video_depth", vd_mod)
    monkeypatch.setitem(sys.modules, "eval", _FakeEvalModule())
    monkeypatch.setitem(sys.modules, "metric", _FakeMetricModule())
    monkeypatch.setattr(ev, "git_commit", lambda repo: "fakecommit")
    return str(tmp_path)


def test_run_official_eval_aggregates_mean_over_clips(fake_vda_repo, tmp_path):
    clips = [("v0", [{"image": "v0/rgb/0.png", "gt_depth": "v0/depth/0.png", "factor": 1.0}]),
            ("v1", [{"image": "v1/rgb/0.png", "gt_depth": "v1/depth/0.png", "factor": 1.0}])]
    metrics = ev.run_official_eval(fake_vda_repo, str(tmp_path), "kitti", clips, str(tmp_path / "infer"))
    assert metrics == {"abs_relative_difference": 0.1, "rmse_linear": 0.2, "delta1_acc": 0.9}
    assert len(_FakeEvalModule.calls) == 2
    assert _FakeEvalModule.calls[0]["max_eval_len"] == ev.DATASET_PROTOCOL["kitti"]["max_eval_len"]


def test_run_official_inference_writes_one_npy_per_frame(fake_vda_repo, tmp_path):
    import cv2
    import torch
    bench_root = tmp_path / "bench"
    (bench_root / "v0" / "rgb").mkdir(parents=True)
    cv2.imwrite(str(bench_root / "v0" / "rgb" / "0.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    ckpt_path = tmp_path / "fake_ckpt.pth"
    torch.save({}, ckpt_path)   # _FakeVideoDepthAnything.load_state_dict ignores content
    clips = [("v0", [{"image": "v0/rgb/0.png", "gt_depth": "v0/depth/0.png", "factor": 1.0}])]
    infer_dir = str(tmp_path / "infer")
    ev.run_official_inference(fake_vda_repo, str(ckpt_path), "vits", clips, str(bench_root),
                              "kitti", infer_dir, device="cpu")
    expected = ev.infer_npy_path(infer_dir, "kitti", "v0/rgb/0.png")
    assert np.load(expected).shape == (4, 4)


def test_score_checkpoint_flags_self_consistency_only_for_the_released_checkpoint(fake_vda_repo, tmp_path, monkeypatch):
    import torch
    ckpt = tmp_path / "some_checkpoint.pth"
    torch.save({}, ckpt)   # a real (if empty) checkpoint -- not the released VDA-S's sha256
    manifest = {"kitti": [{"v0": [{"image": "v0/rgb/0.png", "gt_depth": "v0/depth/0.png", "factor": 1.0}]}]}
    manifest_path = tmp_path / "kitti_video.json"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "bench" / "v0" / "rgb").mkdir(parents=True)
    import cv2
    cv2.imwrite(str(tmp_path / "bench" / "v0" / "rgb" / "0.png"), np.zeros((4, 4, 3), dtype=np.uint8))

    artifact = ev.score_checkpoint(fake_vda_repo, str(ckpt), "vits", str(tmp_path / "bench"),
                                   str(manifest_path), "kitti", str(tmp_path / "infer"), device="cpu")
    assert "kitti_self_consistency_check" not in artifact
    assert artifact["checkpoint_sha256"] != ev.RELEASED_VDA_S_SHA256


def test_run_official_inference_passes_metric_flag_to_the_vendored_constructor(fake_vda_repo, tmp_path):
    """Opus's audit, 2026-08-27: this constructor call previously omitted metric= entirely,
    always building a relative-mode model regardless of which checkpoint was loaded."""
    import cv2
    import torch
    bench_root = tmp_path / "bench"
    (bench_root / "v0" / "rgb").mkdir(parents=True)
    cv2.imwrite(str(bench_root / "v0" / "rgb" / "0.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    ckpt_path = tmp_path / "fake_ckpt.pth"
    torch.save({}, ckpt_path)
    clips = [("v0", [{"image": "v0/rgb/0.png", "gt_depth": "v0/depth/0.png", "factor": 1.0}])]
    ev.run_official_inference(fake_vda_repo, str(ckpt_path), "vits", clips, str(bench_root),
                              "kitti", str(tmp_path / "infer"), device="cpu", metric=True)
    assert _FakeVideoDepthAnything.init_kwargs[-1]["metric"] is True


def test_run_official_inference_metric_mode_saves_inverted_disparity(fake_vda_repo, tmp_path, monkeypatch):
    """metric=True's raw model output is depth-like; run_official_inference must invert it
    to disparity before saving so run_official_eval's own scale+shift fit (written for
    disparity-space predictions -- verified directly against the real vendored eval.py,
    2026-08-27) still operates on the quantity it expects."""
    import cv2
    import torch
    bench_root = tmp_path / "bench"
    (bench_root / "v0" / "rgb").mkdir(parents=True)
    cv2.imwrite(str(bench_root / "v0" / "rgb" / "0.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    ckpt_path = tmp_path / "fake_ckpt.pth"
    torch.save({}, ckpt_path)
    monkeypatch.setattr(_FakeVideoDepthAnything, "infer_video_depth",
                        lambda self, videos, target_fps, input_size, device, fp32:
                            (np.full((videos.shape[0], 4, 4), 2.0, dtype=np.float32), 1.0))
    clips = [("v0", [{"image": "v0/rgb/0.png", "gt_depth": "v0/depth/0.png", "factor": 1.0}])]
    infer_dir = str(tmp_path / "infer")
    ev.run_official_inference(fake_vda_repo, str(ckpt_path), "vits", clips, str(bench_root),
                              "kitti", infer_dir, device="cpu", metric=True)
    saved = np.load(ev.infer_npy_path(infer_dir, "kitti", "v0/rgb/0.png"))
    np.testing.assert_allclose(saved, 0.5)   # 1 / depth(2.0) == 0.5


def test_run_unaligned_eval_reuses_vendored_get_gt_get_infer(fake_vda_repo, tmp_path):
    """RAMEN's own alignment-free path (metrics.py::compute_depth_metrics), kept separate
    from run_official_eval's aligned protocol -- reuses the vendored get_gt/get_infer for
    GT loading rather than reimplementing factor scaling."""
    infer_dir = tmp_path / "infer" / "kitti" / "v0" / "rgb"
    infer_dir.mkdir(parents=True)
    bench_root = tmp_path / "bench"
    (bench_root / "v0" / "depth").mkdir(parents=True)
    # disparity 0.5 -> depth 2.0, exactly matching a GT depth of 2.0 -> perfect unaligned score
    np.save(infer_dir / "0.npy", np.full((4, 4), 0.5, dtype=np.float32))
    np.save(bench_root / "v0" / "depth" / "0.npy", np.full((4, 4), 2.0, dtype=np.float32))
    clips = [("v0", [{"image": "v0/rgb/0.npy", "gt_depth": "v0/depth/0.npy", "factor": 1.0}])]
    result = ev.run_unaligned_eval(fake_vda_repo, str(bench_root), "kitti", clips, str(tmp_path / "infer"))
    assert result["abs_rel"] == pytest.approx(0.0, abs=1e-6)


def test_compute_alignment_stats_recovers_an_exact_known_fit(fake_vda_repo, tmp_path):
    """Opus's follow-up, 2026-08-27: recover the per-clip (scale, shift) eval_depthcrafter
    fits and discards, to turn the observed ~3x unaligned/aligned gap into a measured fact
    instead of an inference. Constructs disparity values with an EXACT known affine
    relationship to GT disparity (scale=2.0, shift=0.5, no noise) so lstsq must recover
    those exact numbers -- not just "a plausible-looking scale"."""
    infer_dir = tmp_path / "infer" / "kitti" / "v0" / "rgb"
    infer_dir.mkdir(parents=True)
    bench_root = tmp_path / "bench"
    (bench_root / "v0" / "depth").mkdir(parents=True)
    pred_disp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gt_disp = 2.0 * pred_disp + 0.5   # the exact affine relationship to recover
    gt_depth = 1.0 / gt_disp          # get_gt loads this directly (factor=1.0, no division)
    np.save(infer_dir / "0.npy", pred_disp)
    np.save(bench_root / "v0" / "depth" / "0.npy", gt_depth.astype(np.float32))
    clips = [("v0", [{"image": "v0/rgb/0.npy", "gt_depth": "v0/depth/0.npy", "factor": 1.0}])]

    stats = ev.compute_alignment_stats(fake_vda_repo, str(bench_root), "kitti", clips, str(tmp_path / "infer"))
    assert stats["num_clips"] == 1
    assert stats["scale"]["median"] == pytest.approx(2.0, abs=1e-4)
    assert stats["shift"]["median"] == pytest.approx(0.5, abs=1e-4)


def test_score_checkpoint_alignment_stats_flag_adds_the_key(fake_vda_repo, tmp_path):
    import cv2
    import torch
    bench_root = tmp_path / "bench"
    (bench_root / "v0" / "rgb").mkdir(parents=True)
    (bench_root / "v0" / "depth").mkdir(parents=True)
    cv2.imwrite(str(bench_root / "v0" / "rgb" / "0.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    np.save(bench_root / "v0" / "depth" / "0.npy", np.full((4, 4), 2.0, dtype=np.float32))
    ckpt = tmp_path / "some_checkpoint.pth"
    torch.save({}, ckpt)
    manifest = {"kitti": [{"v0": [{"image": "v0/rgb/0.png", "gt_depth": "v0/depth/0.npy", "factor": 1.0}]}]}
    manifest_path = tmp_path / "kitti_video.json"
    manifest_path.write_text(json.dumps(manifest))

    artifact = ev.score_checkpoint(fake_vda_repo, str(ckpt), "vits", str(bench_root),
                                   str(manifest_path), "kitti", str(tmp_path / "infer"),
                                   device="cpu", alignment_stats=True)
    assert "alignment_stats" in artifact
    assert "scale" in artifact["alignment_stats"]


def test_score_checkpoint_unaligned_flag_adds_unaligned_metrics_key(fake_vda_repo, tmp_path):
    import cv2
    import torch
    bench_root = tmp_path / "bench"
    (bench_root / "v0" / "rgb").mkdir(parents=True)
    (bench_root / "v0" / "depth").mkdir(parents=True)
    cv2.imwrite(str(bench_root / "v0" / "rgb" / "0.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    np.save(bench_root / "v0" / "depth" / "0.npy", np.full((4, 4), 2.0, dtype=np.float32))
    ckpt = tmp_path / "some_checkpoint.pth"
    torch.save({}, ckpt)
    manifest = {"kitti": [{"v0": [{"image": "v0/rgb/0.png", "gt_depth": "v0/depth/0.npy", "factor": 1.0}]}]}
    manifest_path = tmp_path / "kitti_video.json"
    manifest_path.write_text(json.dumps(manifest))

    artifact = ev.score_checkpoint(fake_vda_repo, str(ckpt), "vits", str(bench_root),
                                   str(manifest_path), "kitti", str(tmp_path / "infer"),
                                   device="cpu", unaligned=True)
    assert "unaligned_metrics" in artifact
    assert "abs_rel" in artifact["unaligned_metrics"]
    assert artifact["metric"] is False


def _effective_short_side(h, w, input_size):
    """Mirror of the vendored infer_video_depth aspect clamp (video_depth.py:72-75)."""
    ratio = max(h, w) / min(h, w)
    if ratio > 1.78:
        input_size = round(int(input_size * 1.777 / ratio) / 14) * 14
    return input_size


@pytest.mark.parametrize("short_side,expected", [(280, 524), (378, 707), (518, 969)])
def test_input_size_for_short_side_kitti_values(short_side, expected):
    """KITTI benchmark frames are 374x1242 (aspect 3.32): VDA's clamp turns 518 into 280,
    so pinning the short side needs a larger input_size (values checked on crispwork)."""
    n = ev.input_size_for_short_side(374, 1242, short_side)
    assert n == expected
    assert _effective_short_side(374, 1242, n) == short_side


@pytest.mark.parametrize("h,w", [(480, 640), (436, 776), (100, 100), (360, 640)])
def test_input_size_for_short_side_leaves_aspect_below_clamp_unchanged(h, w):
    for s in (280, 378, 518):
        assert ev.input_size_for_short_side(h, w, s) == s


def _record_input_sizes(monkeypatch):
    seen = []

    def infer(self, videos, target_fps, input_size, device, fp32):
        seen.append(input_size)
        return np.full((videos.shape[0],) + videos.shape[1:3], 2.0, dtype=np.float32), 1.0
    monkeypatch.setattr(_FakeVideoDepthAnything, "infer_video_depth", infer)
    return seen


def _wide_clip(tmp_path):
    import cv2
    import torch
    bench_root = tmp_path / "bench"
    (bench_root / "v0" / "rgb").mkdir(parents=True)
    cv2.imwrite(str(bench_root / "v0" / "rgb" / "0.png"), np.zeros((374, 1242, 3), dtype=np.uint8))
    ckpt_path = tmp_path / "fake_ckpt.pth"
    torch.save({}, ckpt_path)
    clips = [("v0", [{"image": "v0/rgb/0.png", "gt_depth": "v0/depth/0.png", "factor": 1.0}])]
    return bench_root, ckpt_path, clips


def test_short_side_default_none_is_unchanged_behaviour(fake_vda_repo, tmp_path, monkeypatch):
    """short_side=None must call the vendored model exactly as before (input_size=518) and
    save byte-identical predictions to an explicit pre-patch-style call."""
    seen = _record_input_sizes(monkeypatch)
    bench_root, ckpt_path, clips = _wide_clip(tmp_path)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    ev.run_official_inference(fake_vda_repo, str(ckpt_path), "vits", clips, str(bench_root),
                              "kitti", a, device="cpu", metric=True)
    ev.run_official_inference(fake_vda_repo, str(ckpt_path), "vits", clips, str(bench_root),
                              "kitti", b, device="cpu", metric=True, short_side=None)
    assert seen == [518, 518]
    pa, pb = (ev.infer_npy_path(d, "kitti", "v0/rgb/0.png") for d in (a, b))
    assert open(pa, "rb").read() == open(pb, "rb").read()


def test_short_side_is_passed_through_to_the_vendored_model(fake_vda_repo, tmp_path, monkeypatch):
    seen = _record_input_sizes(monkeypatch)
    bench_root, ckpt_path, clips = _wide_clip(tmp_path)
    ev.run_official_inference(fake_vda_repo, str(ckpt_path), "vits", clips, str(bench_root),
                              "kitti", str(tmp_path / "infer"), device="cpu", metric=True, short_side=378)
    assert seen == [707]


def test_score_checkpoint_records_short_side(fake_vda_repo, tmp_path, monkeypatch):
    _record_input_sizes(monkeypatch)
    bench_root, ckpt_path, clips = _wide_clip(tmp_path)
    manifest_path = tmp_path / "kitti_video.json"
    manifest_path.write_text(json.dumps({"kitti": [{"v0": clips[0][1]}]}))
    for s in (None, 378):
        art = ev.score_checkpoint(fake_vda_repo, str(ckpt_path), "vits", str(bench_root),
                                  str(manifest_path), "kitti", str(tmp_path / f"infer{s}"),
                                  device="cpu", short_side=s)
        assert art["short_side"] == s


def _unaligned_fixture(tmp_path, disp, gt_value):
    infer_dir = tmp_path / "infer" / "kitti" / "v0" / "rgb"
    infer_dir.mkdir(parents=True)
    bench_root = tmp_path / "bench"
    (bench_root / "v0" / "depth").mkdir(parents=True)
    np.save(infer_dir / "0.npy", disp.astype(np.float32))
    np.save(bench_root / "v0" / "depth" / "0.npy", np.full(disp.shape, gt_value, dtype=np.float32))
    return bench_root, [("v0", [{"image": "v0/rgb/0.npy", "gt_depth": "v0/depth/0.npy", "factor": 1.0}])]


def test_run_unaligned_eval_penalizes_zero_depth_pixels_like_bench_joint(fake_vda_repo, tmp_path):
    """ultrareview: half the pixels predict depth ~0 (saved disparity 1000 = 1/clip(depth, 1e-3)). They must COUNT as
    ~100% error (bench_joint clamps to 1e-3 and scores every GT pixel), not be dropped by invalid_pred='exclude'."""
    disp = np.full((4, 4), 0.5)
    disp[:, :2] = 1000.0
    bench_root, clips = _unaligned_fixture(tmp_path, disp, 2.0)
    r = ev.run_unaligned_eval(fake_vda_repo, str(bench_root), "kitti", clips, str(tmp_path / "infer"))
    assert r["abs_rel"] == pytest.approx(0.5 * (2.0 - 1e-3) / 2.0, abs=1e-4)      # exclude gave 0.0
    assert r["delta1"] == pytest.approx(0.5, abs=1e-6)                           # exclude gave 1.0


def test_run_unaligned_eval_clamps_far_pushes_to_the_dataset_cap_like_bench_joint(fake_vda_repo, tmp_path):
    """ultrareview: 150 m predicted on a 50 m GT pixel is capped at the KITTI 80 m cap (abs_rel 0.6), not left at 200%."""
    bench_root, clips = _unaligned_fixture(tmp_path, np.full((4, 4), 1.0 / 150.0), 50.0)
    r = ev.run_unaligned_eval(fake_vda_repo, str(bench_root), "kitti", clips, str(tmp_path / "infer"))
    assert r["abs_rel"] == pytest.approx(0.6, abs=1e-4)                          # uncapped gave 2.0
    assert r["delta1"] == pytest.approx(0.0, abs=1e-6)


def test_run_unaligned_eval_matches_bench_joint_depth_errors(fake_vda_repo, tmp_path):
    """The two unaligned definitions agree on one noisy prediction with zero and far pixels."""
    import torch
    import bench_joint as bj
    rng = np.random.default_rng(0)
    gt = rng.uniform(1.0, 60.0, (6, 8)).astype(np.float32)
    depth = gt * rng.uniform(0.6, 1.6, gt.shape)
    depth[0, :3] = 0.0                                                            # zero depth
    depth[5, :3] = 400.0                                                          # far push
    disp = 1.0 / np.clip(depth, 1e-3, None)                                       # what the saved .npy holds
    bench_root, clips = _unaligned_fixture(tmp_path, disp, 1.0)
    np.save(bench_root / "v0" / "depth" / "0.npy", gt)
    r = ev.run_unaligned_eval(fake_vda_repo, str(bench_root), "kitti", clips, str(tmp_path / "infer"))
    pd = torch.from_numpy(1.0 / np.clip(disp, 1e-3, None)).double()[None].clamp(1e-3, 80.0)
    g = torch.from_numpy(gt).double()[None]
    ref = bj._depth_errors(pd, g, (g > 0.1) & (g < 80.0))
    assert r["abs_rel"] == pytest.approx(ref["abs_rel"], rel=1e-5)
    assert r["delta1"] == pytest.approx(ref["delta1"], abs=1e-6)
    assert r["rmse"] == pytest.approx(ref["rmse"], rel=1e-5)

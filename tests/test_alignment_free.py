import json

import numpy as np
import pytest
import torch

from evaluation.alignment_free import (ContinuousStreamConsumer, DepthEvaluator, EvaluationMetadata,
                                       FrameManifest, _MicroAccumulator, canonical_result_payload,
                                       canonical_sha256, read_prediction_cache, validate_video_id_contiguity,
                                       write_frame_manifest, write_prediction_cache)
from metrics import METRIC_KEYS, compute_depth_metrics


def evaluator():
    return DepthEvaluator(EvaluationMetadata("model", "fixture", "test", min_depth=1e-3, max_depth=100.0))


def evaluator_with_scale_tracking():
    """TOOL-15: track_pixel_scale_logs defaults to False (memory-safety fix); tests
    that check the pixel-pooled scale statistics' actual values opt in explicitly."""
    return DepthEvaluator(EvaluationMetadata("model", "fixture", "test", min_depth=1e-3, max_depth=100.0),
                          track_pixel_scale_logs=True)


def test_perfect_metric_prediction():
    ev = evaluator_with_scale_tracking()
    gt = torch.tensor([[[1.0, 2.0], [4.0, 8.0]], [[2.0, 3.0], [5.0, 9.0]]])
    ev.add_video("v", gt, gt)
    out = ev.result()
    assert out["metrics"]["raw"]["frame_macro"]["abs_rel"] == 0
    assert out["metrics"]["raw"]["frame_macro"]["delta1"] == 1
    assert out["metrics"]["scale"]["scale_factor_to_gt"] == pytest.approx(1)
    assert out["counts"]["invalid_pred_fraction"] == 0


def test_constant_scale_stays_visible_in_raw_but_affine_reference_removes_it():
    ev = evaluator_with_scale_tracking()
    gt = torch.arange(1, 9, dtype=torch.float32).reshape(2, 2, 2)
    ev.add_video("v", 2 * gt, gt)
    out = ev.result()
    assert out["metrics"]["raw"]["frame_macro"]["abs_rel"] == pytest.approx(1)
    assert out["metrics"]["scale"]["median_log_scale_bias"] == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert out["metrics"]["scale"]["scale_factor_to_gt"] == pytest.approx(0.5)
    assert out["metrics"]["aligned_reference"]["frame_macro"]["abs_rel"] == pytest.approx(0, abs=1e-5)


def test_pixel_scale_logs_disabled_by_default_marks_fields_null_not_zero():
    """TOOL-15: 'not measured' and 'measured as zero' are different claims -- a
    reader must not mistake a deliberate default for an alarming empty-coverage
    finding. The marker is stated so an artifact is self-describing either way."""
    ev = evaluator()
    gt = torch.tensor([[[1.0, 2.0], [4.0, 8.0]], [[2.0, 3.0], [5.0, 9.0]]])
    ev.add_video("v", gt, gt)
    out = ev.result()
    scale = out["metrics"]["scale"]
    assert scale["pixel_scale_tracking"] == "disabled"
    assert scale["median_log_scale_bias"] is None
    assert scale["scale_factor_to_gt"] is None
    assert scale["median_abs_log_scale_error"] is None
    assert scale["coverage"]["pixels_with_valid_scale"] is None
    assert scale["coverage"]["pixel_fraction"] is None
    # frame_biases-derived fields are unaffected (bounded, always computed regardless)
    assert scale["coverage"]["frames_with_valid_scale"] == 2
    assert scale["coverage"]["frame_fraction"] == 1


def test_pixel_scale_logs_enabled_marks_the_field_positively():
    ev = evaluator_with_scale_tracking()
    gt = torch.tensor([[[1.0, 2.0], [4.0, 8.0]]])
    ev.add_video("v", gt, gt)
    out = ev.result()
    assert out["metrics"]["scale"]["pixel_scale_tracking"] == "enabled"


def test_invalid_predictions_are_counted_and_never_reward_delta():
    ev = evaluator()
    ev.add_frame("v", torch.tensor([[float("nan"), -1.0], [0.0, 1.0]]), torch.ones(2, 2))
    out = ev.result()
    raw = out["metrics"]["raw"]["frame_macro"]
    assert out["counts"]["invalid_pred_pixels"] == 3
    assert out["counts"]["invalid_pred_fraction"] == pytest.approx(0.75)
    assert raw["delta1"] == pytest.approx(0.25)


def test_frame_macro_and_pixel_micro_have_explicit_different_weights():
    ev = evaluator()
    ev.add_frame("v", torch.tensor([2.0]), torch.tensor([1.0]))       # AbsRel 1
    ev.add_frame("v", torch.ones(9), torch.ones(9))                  # AbsRel 0
    out = ev.result()["metrics"]["raw"]
    assert out["frame_macro"]["abs_rel"] == pytest.approx(0.5)
    assert out["pixel_micro"]["abs_rel"] == pytest.approx(0.1)


def test_per_video_fit_is_invariant_to_add_video_partitioning():
    gt = torch.arange(1, 9, dtype=torch.float32).reshape(2, 2, 2)
    pred = 3 * gt
    a, b = evaluator(), evaluator()
    a.add_video("same", pred, gt)
    b.add_frame("same", pred[0], gt[0])
    b.add_frame("same", pred[1], gt[1])
    assert a.result()["metrics"]["aligned_reference"]["pixel_micro"] == \
           b.result()["metrics"]["aligned_reference"]["pixel_micro"]


def test_continuous_consumer_resets_only_at_video_boundary():
    ev = evaluator()
    state = {"n": 0, "resets": 0}

    def reset():
        state["n"] = 0
        state["resets"] += 1

    def predict(_):
        state["n"] += 1
        return torch.tensor([float(state["n"])])
    videos = {"a": [(None, torch.tensor([1.0]), None), (None, torch.tensor([2.0]), None)],
              "b": [(None, torch.tensor([1.0]), None)]}
    ContinuousStreamConsumer(ev, predict, reset).consume(videos)
    assert state["resets"] == 2
    assert ev.result()["metrics"]["raw"]["frame_macro"]["abs_rel"] == 0


def test_json_schema_is_strict_json(tmp_path):
    ev = evaluator()
    ev.add_frame("v", torch.ones(2), torch.ones(2))
    path = tmp_path / "result.json"
    ev.write_json(path)
    saved = json.loads(path.read_text())
    assert saved["schema_version"] == "1.0.0"
    assert saved["aggregation"]["primary"] == "frame_macro"
    assert set(saved["metrics"]) == {"raw", "scale", "aligned_reference"}


def test_video_macro_gives_each_video_equal_weight():
    ev = evaluator()
    ev.add_frame("bad", torch.tensor([2.0]), torch.tensor([1.0]))
    ev.add_frame("good", torch.ones(1), torch.ones(1))
    ev.add_frame("good", torch.ones(1), torch.ones(1))
    out = ev.result()["metrics"]["raw"]
    assert out["frame_macro"]["abs_rel"] == pytest.approx(1 / 3)
    assert out["video_macro"]["abs_rel"] == pytest.approx(0.5)


def test_unsolved_affine_fit_and_scale_coverage_are_explicit():
    ev = evaluator_with_scale_tracking()
    ev.add_frame("empty", torch.tensor([float("nan")]), torch.tensor([1.0]))
    out = ev.result()
    assert out["metrics"]["aligned_reference"]["fit_counts"] == {
        "attempted": 1, "solved": 0, "unsolved": 1}
    assert out["metrics"]["scale"]["coverage"]["frame_fraction"] == 0
    assert out["metrics"]["scale"]["coverage"]["pixel_fraction"] == 0


def test_invalid_alignment_scope_and_mode_raise():
    with pytest.raises(ValueError, match="alignment_scope"):
        DepthEvaluator(EvaluationMetadata("model", "fixture", "test"), alignment_scope="bogus")
    with pytest.raises(ValueError, match="alignment_mode"):
        DepthEvaluator(EvaluationMetadata("model", "fixture", "test"), alignment_mode="bogus")


def test_alignment_mode_none_makes_aligned_reference_equal_raw():
    ev = DepthEvaluator(EvaluationMetadata("model", "fixture", "test"), alignment_mode="none")
    gt = torch.arange(1, 9, dtype=torch.float32).reshape(2, 2, 2)
    ev.add_video("v", 2 * gt, gt)
    out = ev.result()["metrics"]
    assert out["aligned_reference"]["frame_macro"] == out["raw"]["frame_macro"]
    assert out["aligned_reference"]["mode"] == "per_video_none"


def test_median_scale_mode_recovers_constant_scale_error():
    ev = DepthEvaluator(EvaluationMetadata("model", "fixture", "test"),
                         alignment_scope="per_video", alignment_mode="median_scale")
    gt = torch.arange(1, 9, dtype=torch.float32).reshape(2, 2, 2)
    ev.add_video("v", 3 * gt, gt)
    out = ev.result()["metrics"]["aligned_reference"]
    assert out["mode"] == "per_video_median_scale"
    assert out["frame_macro"]["abs_rel"] == pytest.approx(0, abs=1e-5)


def test_per_frame_scope_corrects_a_scale_error_that_varies_by_frame():
    # A single per-video fit cannot perfectly correct two frames with different
    # scale errors (2x and 5x) -- per_frame scope can, distinguishing it from
    # per_video scope.
    ev = DepthEvaluator(EvaluationMetadata("model", "fixture", "test"),
                         alignment_scope="per_frame", alignment_mode="median_scale")
    gt = torch.tensor([1.0, 2.0, 4.0, 8.0])
    ev.add_frame("v", 2 * gt, gt)
    ev.add_frame("v", 5 * gt, gt)
    out = ev.result()["metrics"]["aligned_reference"]
    assert out["frame_macro"]["abs_rel"] == pytest.approx(0, abs=1e-5)


def test_video_is_finalized_and_raw_tensors_dropped_on_boundary():
    ev = evaluator()
    ev.add_frame("a", torch.tensor([1.0]), torch.tensor([1.0]))
    ev.add_frame("b", torch.tensor([1.0]), torch.tensor([1.0]))  # boundary: "a" finalizes here
    assert "a" in ev._closed
    assert ev._current_video_id == "b"
    assert "raw_rows" in ev._closed["a"]
    # The finalized summary holds small metric rows, not the original frame tensors.
    assert not any(isinstance(v, torch.Tensor) for v in ev._closed["a"].values())


def test_reopening_a_finalized_video_raises():
    ev = evaluator()
    ev.add_frame("a", torch.tensor([1.0]), torch.tensor([1.0]))
    ev.add_frame("b", torch.tensor([1.0]), torch.tensor([1.0]))  # closes "a"
    with pytest.raises(ValueError, match="already finalised"):
        ev.add_frame("a", torch.tensor([1.0]), torch.tensor([1.0]))


def test_prediction_cache_round_trips_ragged_frames(tmp_path):
    rows = [
        ("a", np.array([[1.0, 2.0]], dtype=np.float32), np.array([[1.0, 2.0]], dtype=np.float32),
         np.array([[True, False]])),
        ("a", np.zeros((3, 2), dtype=np.float32), np.ones((3, 2), dtype=np.float32), None),
        ("b", np.full((2, 2), 5.0, dtype=np.float32), np.full((2, 2), 5.0, dtype=np.float32), None),
    ]
    path = tmp_path / "cache.npz"
    write_prediction_cache(path, rows)
    got = list(read_prediction_cache(path))
    assert [vid for vid, *_ in got] == ["a", "a", "b"]
    for (_, pred, gt, mask), (_, exp_pred, exp_gt, exp_mask) in zip(got, rows):
        assert pred.shape == exp_pred.shape
        assert np.array_equal(pred, exp_pred)
        assert np.array_equal(gt, exp_gt)
        exp_mask = np.ones(exp_gt.shape, dtype=bool) if exp_mask is None else exp_mask
        assert np.array_equal(mask, exp_mask)


def test_prediction_cache_rejects_pred_gt_shape_mismatch(tmp_path):
    with pytest.raises(ValueError, match="shape mismatch"):
        write_prediction_cache(tmp_path / "cache.npz",
                               [("a", np.zeros((2, 2)), np.zeros((3, 3)), None)])


def test_prediction_cache_validates_contiguity_on_read(tmp_path):
    path = tmp_path / "cache.npz"
    write_prediction_cache(path, [
        ("a", np.zeros((2, 2)), np.zeros((2, 2)), None),
        ("b", np.zeros((2, 2)), np.zeros((2, 2)), None),
        ("a", np.zeros((2, 2)), np.zeros((2, 2)), None),
    ])
    with pytest.raises(ValueError, match="non-contiguous"):
        read_prediction_cache(path)


def test_frame_manifest_round_trips_ragged_frames_and_channels(tmp_path):
    rows = [
        ("a", 0.0, np.zeros((2, 3, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32), None),
        ("a", 1.0, np.ones((4, 5, 3), dtype=np.float32), np.full((4, 5), 2.0, dtype=np.float32),
         np.array([[True] * 5] * 4)),
        ("b", 0.0, np.full((2, 2, 3), 7.0, dtype=np.float32), np.zeros((2, 2), dtype=np.float32), None),
    ]
    path = tmp_path / "manifest.npz"
    write_frame_manifest(path, rows)
    manifest = FrameManifest(path)

    assert manifest.video_id == ["a", "a", "b"]
    assert list(manifest.frame_index) == [0.0, 1.0, 0.0]
    assert len(manifest) == 3
    for row, (_, _, exp_frame, exp_gt, exp_mask) in enumerate(rows):
        assert manifest.frame(row).shape == exp_frame.shape
        assert np.array_equal(manifest.frame(row), exp_frame)
        assert np.array_equal(manifest.gt(row), exp_gt)
        exp_mask = np.ones(exp_gt.shape, dtype=bool) if exp_mask is None else exp_mask
        assert np.array_equal(manifest.mask(row), exp_mask)


def test_frame_manifest_rejects_mask_gt_shape_mismatch(tmp_path):
    with pytest.raises(ValueError, match="shape mismatch"):
        write_frame_manifest(tmp_path / "manifest.npz",
                             [("a", 0.0, np.zeros((2, 2, 3)), np.zeros((2, 2)), np.zeros((3, 3), dtype=bool))])


def test_validate_video_id_contiguity_accepts_grouped_videos():
    validate_video_id_contiguity(["a", "a", "b", "b", "b", "c"])  # no raise


def test_validate_video_id_contiguity_rejects_a_video_id_that_reappears():
    with pytest.raises(ValueError, match="non-contiguous"):
        validate_video_id_contiguity(["a", "b", "a"])


def test_streaming_accumulation_matches_multi_video_reference():
    # Three videos of different sizes, processed one at a time (as any real
    # streaming caller does) -- confirms frame_macro/pixel_micro/video_macro
    # are unaffected by videos being finalized and dropped as they complete.
    ev = evaluator()
    torch.manual_seed(0)
    videos = {"v1": (torch.rand(3, 4) * 5 + 1, torch.rand(3, 4) * 5 + 1),
              "v2": (torch.rand(1, 4) * 5 + 1, torch.rand(1, 4) * 5 + 1),
              "v3": (torch.rand(5, 4) * 5 + 1, torch.rand(5, 4) * 5 + 1)}
    for vid, (pred, gt) in videos.items():
        ev.add_video(vid, pred, gt)
    out = ev.result()
    assert out["counts"]["videos"] == 3
    assert out["counts"]["frames_scored"] == 3 + 1 + 5
    assert out["metrics"]["raw"]["pixel_micro"]["abs_rel"] >= 0
    assert out["metrics"]["raw"]["video_macro"]["abs_rel"] >= 0


def test_micro_accumulator_matches_pooled_compute_depth_metrics():
    torch.manual_seed(1)
    frames = [(torch.rand(4, 4) * 5 + 1, torch.rand(4, 4) * 5 + 1) for _ in range(3)]
    acc = _MicroAccumulator(1e-3, 80.0)
    mask = torch.ones(4, 4, dtype=torch.bool)
    for pred, gt in frames:
        acc.update(pred, gt, mask)
    pooled_pred = torch.cat([p.flatten() for p, _ in frames])
    pooled_gt = torch.cat([g.flatten() for _, g in frames])
    reference = compute_depth_metrics(pooled_pred, pooled_gt, invalid_pred="penalize", min_depth=1e-3, max_depth=80.0)
    streamed = acc.result()
    for key in METRIC_KEYS:
        assert streamed[key] == pytest.approx(reference[key], rel=1e-5)


def test_result_hash_excludes_timestamp_and_is_self_verifiable():
    one = evaluator().result()
    two = dict(one)
    two["created_utc"] = "different"
    assert canonical_sha256(canonical_result_payload(one)) == one["result_sha256"]
    assert canonical_sha256(canonical_result_payload(two)) == one["result_sha256"]

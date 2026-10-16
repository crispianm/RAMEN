from argparse import Namespace
import gc
import json
import weakref

import numpy as np
import pytest

from evaluation.alignment_free import write_frame_manifest
from evaluation.model_registry import build_ramen, build_vda
from scripts.eval_streaming_npz import (MODEL_REGISTRY, latency_summary, main, metadata,
                                        validate_chronological_manifest)


@pytest.mark.parametrize("indices", [[0, 0], [1, 0], [0, np.nan]])
def test_manifest_rejects_non_strict_or_nonfinite_order(indices):
    with pytest.raises(ValueError):
        validate_chronological_manifest(["a", "a"], indices)


def test_manifest_rejects_noncontiguous_video():
    with pytest.raises(ValueError, match="non-contiguous"):
        validate_chronological_manifest(["a", "b", "a"], [0, 0, 1])


def test_latency_summary_reports_tail_and_empty_sets():
    assert latency_summary([1.0, 2.0, 10.0])["p95_ms_per_frame"] > 2
    assert latency_summary([])["p50_ms_per_frame"] is None


def test_model_registry_has_ramen():
    assert MODEL_REGISTRY["ramen"] is build_ramen


def test_model_registry_has_vda():
    assert MODEL_REGISTRY["vda"] is build_vda


def test_build_vda_requires_vda_root_env_var(monkeypatch):
    monkeypatch.delenv("VDA_ROOT", raising=False)
    with pytest.raises(ValueError, match="VDA_ROOT"):
        build_vda("unused.pth", "cpu", "fp32", 32)


def test_metadata_model_name_comes_from_registry_selection(tmp_path):
    manifest_path = tmp_path / "manifest.npz"
    manifest_path.write_bytes(b"fixture")
    args = Namespace(model="ramen", dataset="d", split="s", manifest=str(manifest_path),
                     output_semantics="relative_depth", condition="clean", window=8)
    meta = metadata(args, checkpoint_hash="deadbeef", steady=False)
    assert meta.model_name == "RAMEN"
    assert meta.streaming["warmup_scored"] is True


def _write_manifest(path, video_lengths, shape=(2, 2)):
    def rows():
        for vid, length in video_lengths.items():
            for i in range(length):
                yield (vid, float(i), np.full(shape, float(i), dtype=np.float32),
                       np.full(shape, 1.0, dtype=np.float32), None)

    write_frame_manifest(path, rows())


def test_streaming_eval_does_not_retain_predictions_past_their_own_video(tmp_path, monkeypatch):
    """FIX-02 regression: main() must not buffer every prediction to fill steady_eval.

    Wraps each prediction in a tensor the test can weakly reference. Once a video
    other than the currently-open one has been processed, nothing outside the two
    DepthEvaluators should still be holding its predictions -- so their weakrefs
    must be collectible, which the pre-fix two-pass ``predictions`` list prevented.
    """
    manifest_path = tmp_path / "manifest.npz"
    _write_manifest(manifest_path, {"a": 3, "b": 4})
    out_path = tmp_path / "out.json"

    live_refs = []

    def build_identity(checkpoint, device, dtype_str, window):
        def predict_frame(tensor):
            pred = tensor.clone()
            live_refs.append(weakref.ref(pred))
            return pred

        return predict_frame, lambda: None, "0" * 64

    monkeypatch.setitem(MODEL_REGISTRY, "identity", build_identity)
    monkeypatch.setattr(
        "sys.argv",
        ["eval_streaming_npz.py", str(manifest_path), "--model", "identity", "--checkpoint", "unused",
         "--out", str(out_path), "--dataset", "d", "--split", "s", "--device", "cpu",
         "--window", "2", "--timing-warmup", "0"],
    )

    main()

    assert len(live_refs) == 7
    gc.collect()
    assert all(ref() is None for ref in live_refs), (
        "a prediction outlived both evaluators -- something is buffering the full run again"
    )

    result = json.loads(out_path.read_text())
    assert result["all_frames"]["counts"]["frames_scored"] == 7
    # steady state keeps frames at position >= window - 1 per video: 2 of 3, then 3 of 4
    assert result["steady_state"]["counts"]["frames_scored"] == 5

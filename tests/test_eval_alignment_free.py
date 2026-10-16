import json

import numpy as np
import pytest

from evaluation.alignment_free import write_prediction_cache
from scripts.eval_alignment_free import main


def _write_cache(path, video_ids, shape=(2, 2)):
    write_prediction_cache(
        path,
        ((vid, np.ones(shape, dtype=np.float32), np.ones(shape, dtype=np.float32), None)
         for vid in video_ids),
    )


def test_rejects_noncontiguous_cache_up_front(tmp_path, monkeypatch):
    """FIX-03 regression: a non-contiguous cache must fail fast with a clear message,
    not die mid-stream inside DepthEvaluator's own 'already finalised' check."""
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path, ["a", "b", "a"])
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        "sys.argv",
        ["eval_alignment_free.py", str(cache_path), "--out", str(out_path), "--model", "m",
         "--dataset", "d", "--split", "s"],
    )
    with pytest.raises(ValueError, match="non-contiguous"):
        main()
    assert not out_path.exists()


def test_accepts_a_contiguous_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path, ["a", "a", "b", "b", "b"])
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        "sys.argv",
        ["eval_alignment_free.py", str(cache_path), "--out", str(out_path), "--model", "m",
         "--dataset", "d", "--split", "s"],
    )
    main()
    result = json.loads(out_path.read_text())
    assert result["counts"]["videos"] == 2
    assert result["counts"]["frames_scored"] == 5


def test_scores_a_cache_with_frames_of_different_native_resolution(tmp_path, monkeypatch):
    """BENCH-04d regression: real KITTI drives don't all share one image size (370x1226
    vs 375x1242 etc., confirmed on crispwork's kitti_c_dark val set), so the cache must
    not assume a single stacked (N, H, W) shape across frames."""
    cache_path = tmp_path / "cache.npz"
    rows = [
        ("a", np.full((4, 5), 2.0, dtype=np.float32), np.full((4, 5), 2.0, dtype=np.float32), None),
        ("a", np.full((3, 3), 1.0, dtype=np.float32), np.full((3, 3), 1.0, dtype=np.float32), None),
        ("b", np.full((6, 2), 3.0, dtype=np.float32), np.full((6, 2), 3.0, dtype=np.float32), None),
    ]
    write_prediction_cache(cache_path, rows)
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        "sys.argv",
        ["eval_alignment_free.py", str(cache_path), "--out", str(out_path), "--model", "m",
         "--dataset", "d", "--split", "s"],
    )
    main()
    result = json.loads(out_path.read_text())
    assert result["counts"]["videos"] == 2
    assert result["counts"]["frames_scored"] == 3
    assert result["metrics"]["raw"]["frame_macro"]["abs_rel"] == pytest.approx(0)

"""Commit Ifig7: --dump-frames (input + restored frames, lossless) for the real-clip item (CPU, no torch)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
QUAL = REPO / "repro" / "qualitative"
sys.path.insert(0, str(QUAL))
CFG = yaml.safe_load((REPO / "configs" / "paper_results.yaml").read_text())["items"]


def test_only_real_item_requests_frames_dump():
    assert CFG["fig:teaser_many_real"]["params"]["dump_frames"] is True
    assert "dump_frames" not in CFG["fig:teaser_many_dev"]["params"] and "dump_frames" not in CFG["fig:teaser_many_test"]["params"]
    assert CFG["fig:teaser_many_real"]["params"]["dump_all_depth"] is True


def test_wrapper_passes_flag_only_when_set(monkeypatch, tmp_path):
    import repro.qualitative as Q
    seen = []
    monkeypatch.setattr(Q, "_run", lambda cmd, *a, **k: seen.append([str(c) for c in cmd]))
    Q._teaser_many({"params": {"device": "cpu", "dump_frames": True, "clips": [{"dataset": "bonn", "clip": "rgbd_bonn_crowd"}]}}, tmp_path / "a", "dev")
    assert "--dump-frames" in seen[-1]
    Q._teaser_many({"params": {"device": "cpu", "clips": [{"dataset": "bonn", "clip": "rgbd_bonn_crowd"}]}}, tmp_path / "b", "dev")
    assert "--dump-frames" not in seen[-1]


def test_renderer_dumps_input_and_restored_losslessly():
    src = (QUAL / "render_c3_figures.py").read_text()
    for s in ('"--dump-frames"', "teaser_frames_", 'res["c3_restored"]', 'frm = {"input": np.stack([u8(x[t_])', "np.stack([u8(rst[t_])"):
        assert s in src

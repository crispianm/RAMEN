"""Commit Ifig6: cumulative still change panels (--cumulative-still), full-depth archive for the many-clip items, real clips filled (CPU, no torch)."""
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


def test_many_items_request_cumulative_stills_and_archive_and_real_clips_filled():
    for k in ("fig:teaser_many_dev", "fig:teaser_many_test", "fig:teaser_many_real"):
        p = CFG[k]["params"]
        assert p["cumulative_still"] is True and p["dump_all_depth"] is True
    real = CFG["fig:teaser_many_real"]["params"]["clips"]
    assert len(real) == 11 and len({c["name"] for c in real}) == 11
    assert all(c["dir_gh200"].startswith("/projects/b5dh/data/teaser_real_20261015/") and c["start"] == 0 for c in real)
    assert {c["name"]: c["n"] for c in real}["horse"] == 51 and all(c["n"] == 64 for c in real if c["name"] != "horse")
    assert not any(c["name"].startswith("bvi_S16") or c["name"].startswith("bvi_S17") for c in real)       # face scenes excluded


def test_wrapper_passes_flags(monkeypatch, tmp_path):
    import repro.qualitative as Q
    seen = []
    monkeypatch.setattr(Q, "_run", lambda cmd, *a, **k: seen.append([str(c) for c in cmd]))
    Q._teaser_many({"params": {"device": "cpu", "cumulative_still": True, "dump_all_depth": True, "clips": [{"dataset": "bonn", "clip": "rgbd_bonn_crowd"}]}}, tmp_path / "a", "dev")
    assert "--cumulative-still" in seen[-1] and "--dump-all-depth" in seen[-1]
    Q._teaser_many({"params": {"device": "cpu", "clips": [{"dataset": "bonn", "clip": "rgbd_bonn_crowd"}]}}, tmp_path / "b", "dev")
    assert "--cumulative-still" not in seen[-1] and "--dump-all-depth" not in seen[-1]          # default off


def test_renderer_cumulative_rule_declared():
    src = (QUAL / "render_c3_figures.py").read_text()
    for s in ('"--cumulative-still"', "cum_vmax = float(np.percentile(np.concatenate([cum_ft.ravel(), cum_c3.ravel()]), 99))", '"Cumulative: FT", "Cumulative: RAMEN + TAG"', "cr[cum_t0 - 1:].sum(0)",
              'cum_t0 = 8 if N > 9 else 1', "still = comp(ft_, with_gt, cum=a.cumulative_still)"):
        assert s in src

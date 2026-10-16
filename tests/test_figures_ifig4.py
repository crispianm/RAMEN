"""Commit Ifig4: multi-clip teaser items (fig:teaser_many_dev / _test / _real) and the renderer's --clip / --video-dir options (CPU, no torch, no GPU)."""
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


def test_three_items_declared_qualitative_gh200_ft_and_ramen_only():
    for k in ("fig:teaser_many_dev", "fig:teaser_many_test", "fig:teaser_many_real"):
        it = CFG[k]
        assert it["kind"] == "qualitative" and it["host"] == "gh200" and it["models"] == ["ft_n46", "ramen_n46_tag"]
        assert it["params"]["depth_display"] == "v1_log_p2p98" and "relative" in it["params"]["display_disclosure"]
    assert len(CFG["fig:teaser_many_dev"]["params"]["clips"]) == 20 and len(CFG["fig:teaser_many_test"]["params"]["clips"]) == 5
    assert CFG["fig:teaser_many_dev"]["params"]["versions"] == ["withGT"] and CFG["fig:teaser_many_real"]["params"]["versions"] == ["noGT"]
    assert "exploratory" in CFG["fig:teaser_many_test"]["label"].lower() and CFG["fig:teaser_many_test"]["params"]["split"] == "test"


def test_dev_clips_unique_and_exclude_kitti_0047_and_test_clips_are_disjoint():
    dev = [(c["dataset"], c["clip"]) for c in CFG["fig:teaser_many_dev"]["params"]["clips"]]
    test = [(c["dataset"], c["clip"]) for c in CFG["fig:teaser_many_test"]["params"]["clips"]]
    assert len(set(dev)) == 20 and len(set(test)) == 5 and not set(dev) & set(test)
    assert not any("0047" in k for _, k in dev + test)


def test_wrapper_builds_cli_per_mode(monkeypatch, tmp_path):
    import repro.qualitative as Q
    seen = []
    monkeypatch.setattr(Q, "_run", lambda cmd, *a, **k: seen.append([str(c) for c in cmd]))
    ent = {"params": {"device": "cpu", "depth_display": "v1_log_p2p98", "clips": [{"dataset": "bonn", "clip": "rgbd_bonn_crowd"}]}}
    Q._teaser_many(ent, tmp_path / "d", "dev")
    c = seen[-1]
    assert c[c.index("--clip") + 1] == "bonn:rgbd_bonn_crowd" and c[c.index("--versions") + 1] == "withGT" and "--allow-test" not in c
    assert c[c.index("--depth-display") + 1] == "v1_log_p2p98" and (tmp_path / "d" / "bonn__rgbd_bonn_crowd").is_dir()
    Q._teaser_many(ent, tmp_path / "t", "test")
    assert "--allow-test" in seen[-1]
    real = {"params": {"device": "cpu", "clips": [{"name": "horse", "dir_gh200": "/g/horse", "dir_local": "/l/horse", "n": 51}]}}
    Q._teaser_many(real, tmp_path / "r", "real")
    c = seen[-1]
    assert c[c.index("--video-dir") + 1] == "/l/horse" and c[c.index("--versions") + 1] == "noGT" and c[c.index("--n") + 1] == "51" and "--clip" not in c


def test_renderer_declares_new_options_and_refuses_test_without_token():
    src = (QUAL / "render_c3_figures.py").read_text()
    for opt in ('"--clip"', '"--allow-test"', '"--video-dir"', '"--versions"'):
        assert opt in src
    assert 'a.allow_test and os.environ.get("RAMEN_AUDITOR_TOKEN")' in src and "KITTI drive_0047 is excluded" in src
    assert 'assert not (real and "abs_gt" in modes)' in src
    assert 'split_label = "test (exposed exploratory' in src


def test_default_path_strings_unchanged():
    src = (QUAL / "render_c3_figures.py").read_text()
    assert "dev (bench_joint.load_clips('dev'); allow_test never passed)" in src
    assert "shown = frames[clean.shape[0] // 2]" in src

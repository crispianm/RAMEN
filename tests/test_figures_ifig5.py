"""Commit Ifig5: --dump-all-depth + fig:teaser_depthdump (CPU, no torch)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
QUAL = REPO / "repro" / "qualitative"
sys.path.insert(0, str(QUAL))


def test_item_declared_and_not_a_paper_item():
    it = yaml.safe_load((REPO / "configs" / "paper_results.yaml").read_text())["items"]["fig:teaser_depthdump"]
    assert it["kind"] == "qualitative" and it["host"] == "gh200" and it["models"] == ["ft_n46", "ramen_n46_tag"] and it["reference"] is None and "none" in it["paper_ref"]


def test_wrapper_builds_cli(monkeypatch, tmp_path):
    import repro.qualitative as Q
    seen = []
    monkeypatch.setattr(Q, "_run", lambda cmd, *a, **k: seen.append([str(c) for c in cmd]))
    Q.render_fig_teaser_depthdump({"params": {"device": "cpu"}}, tmp_path)
    c = seen[-1]
    assert c[c.index("--only") + 1] == "99" and "--dump-all-depth" in c and "--no-teaser-chart" in c and c[c.index("--versions") + 1] == "noGT"


def test_renderer_declares_flag_and_dumps_the_panel_arrays():
    src = (QUAL / "render_c3_figures.py").read_text()
    assert '"--dump-all-depth"' in src and 'r["ft"].astype(np.float32)' in src and 'r["c3_tag"].astype(np.float32)' in src and "teaser_alldepth_" in src

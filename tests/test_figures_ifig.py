"""Commit Ifig: fig:teaser (noGT + withGT), fig:qual, fig:flicker_curve are FT n46 / RAMEN n46 only (CPU, no torch, no GPU)."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
QUAL = REPO / "repro" / "qualitative"
sys.path.insert(0, str(QUAL))

from repro import config as C  # noqa: E402
from repro import plan as P  # noqa: E402

CFG = C.load(REPO / "configs" / "paper_results.yaml")
FIG = json.loads((QUAL / "figures_config.json").read_text())
RAMEN_N46_SHA = "f5c7c8f98fe0c9f97bcac3586e577d04e914fbbe1e2ac41dbdbd8ce0501db1af"
FT_N46_SHA = "18c9d8caeab1ab88d52b92782e307bcb8d82897e8a8017aed172f06c26af92a0"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_figure_items_models_are_ft_n46_and_ramen_n46_only():
    assert CFG["items"]["fig:teaser"]["models"] == ["ft_n46", "ramen_n46_tag"]
    assert CFG["items"]["fig:qual"]["models"] == ["ft_n46", "ramen_n46", "ramen_n46_tag"]
    assert CFG["items"]["fig:flicker_curve"]["models"] == CFG["items"]["fig:qual"]["models"]
    for f in ("fig:teaser", "fig:qual", "fig:flicker_curve"):
        ms = set(CFG["items"][f]["models"])
        assert not (ms & {"metric_vda_s", "metric_vda_s_tag", "earlier_s45", "earlier_s45_tag"}), f
        assert CFG["items"][f]["kind"] == "qualitative" and CFG["items"][f]["host"] == "gh200"
    assert C.validate(CFG) == []


def test_figure_items_plan_no_scorer_units_and_stay_gh200():
    for f in ("fig:teaser", "fig:qual", "fig:flicker_curve"):
        units, items = P.plan(CFG, [f])
        assert units == [] and items == [f]                                  # renders are not scorer units: DRY unit count 0, renderers named in the plan
        assert P.unit_host(CFG["items"][f], False, "reference") == "gh200"


def test_teaser_declares_two_versions_and_outputs():
    p = CFG["items"]["fig:teaser"]["params"]
    assert p["versions"] == ["noGT", "withGT"] and p["item"] == 99 and p["split"] == "dev"
    assert {o.split(".")[0].replace("_still", "") for o in p["outputs"] if "meanchange" not in o} == {f"teaser_{m}{v}" for m in ("", "v1_", "v2_") for v in ("noGT", "withGT")}
    assert sum(o.endswith("_still.pdf") for o in p["outputs"]) == 6 and p["depth_display"] == "abs_gt,v1_log_p2p98,v2_disparity_p2p98"


def test_figures_config_pins_ramen_n46_and_ft_n46_and_hostmap_agrees():
    c3, ft = FIG["c3"], FIG["ft"][FIG["FT_CHOICE"]]
    assert c3["sha256"] == RAMEN_N46_SHA and ft["sha256"] == FT_N46_SHA and FIG["FT_CHOICE"] == "B"
    assert "c3_n46" in c3["ckpt"] and "c3_n46" in c3["ckpt_gh200"] and c3["ckpt_gh200"].startswith("/projects/b5dh/")
    gh = CFG["host_paths"]["gh200"]
    key = "checkpoints_v3/eval_20261012_c3rep/" + c3["ckpt"].split("eval_20261012_c3rep/")[1]
    assert gh[key] == c3["ckpt_gh200"]
    assert CFG["models"]["ramen_n46"]["ckpt"]["sha256"] == RAMEN_N46_SHA and CFG["models"]["ft_n46"]["ckpt"]["sha256"] == FT_N46_SHA
    assert FIG["teaser"] == {"dataset": "bonn", "cell": "st_llvar"} and FIG["tag"] == "agf0.2t0.06"


def test_selection_files_unchanged_and_teaser_clip_is_the_rule_clip():
    assert _sha(QUAL / "CLIP_RULE.md") == "085611d2869a7c9efc286c82bafeee1af20dfa7bba41cade39ca3c19dd30c10c"
    assert _sha(QUAL / "CLIP_SELECTION.json") == "206598653a07a927a5c9b7cec98d9dbf6cf44ead6ab9464fa4d8e7a0d5163788"
    sel = json.loads((QUAL / "CLIP_SELECTION.json").read_text())
    assert sel["teaser"]["clip"] == "rgbd_bonn_placing_obstructing_box" and sel["teaser"]["cell"] == "st_llvar"


@pytest.mark.parametrize("name", ["render_c3_figures.py", "render_flicker_curve.py"])
def test_renderers_have_no_released_model(name):
    src = (QUAL / name).read_text()
    assert not re.search(r'specs\s*=\s*\{[^}]*"released"', src) and 'cfg["released"]' not in src
    assert '"released"' not in src.replace('"released_', "")


def test_gt_panel_is_masked_sensor_gt_on_the_prediction_colour_scale():
    import render_c3_figures as R
    g = np.full((6, 8), -1.0)
    g[1:5, 2:7] = np.linspace(0.5, 9.0, 20).reshape(4, 5)
    g[0, 0] = 10.0                                                          # >= max_depth: invalid
    img, valid = R.gt_rgb(g, 1.0, 8.0, 10.0)
    assert valid.sum() == 20 and not valid[0, 0] and img.dtype == np.uint8 and img.shape == (6, 8, 3)
    assert (img[~valid] == (70, 70, 70)).all()                              # masked, not filled
    pred_style = R.depth_rgb(g, 1.0, 8.0)
    assert (img[valid] == pred_style[valid]).all()                          # same log-turbo scale as the depth panels
    assert not np.array_equal(img[valid], np.full_like(img[valid], 70))


def test_header_titles_are_number_free_and_teaser_titles_are_short():
    import render_c3_figures as R
    R.header(["Degraded input", "Ground truth", "Metric-VDA-S FT", "RAMEN + TAG", "Change: FT", "Change: RAMEN + TAG"], 100)
    with pytest.raises(AssertionError):
        R.header(["RAMEN C3"], 100)
    with pytest.raises(AssertionError):
        R.header(["FT seed 46"], 100)
    src = (QUAL / "render_c3_figures.py").read_text()
    for banned in ("sky", "Sky"):
        assert banned not in src


def test_save_pdf_writes_pdf(tmp_path):
    import render_c3_figures as R
    R.save_pdf(np.zeros((20, 30, 3), np.uint8), tmp_path / "a.pdf")
    assert (tmp_path / "a.pdf").read_bytes()[:5] == b"%PDF-"

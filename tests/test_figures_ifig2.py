"""Commit Ifig2: per-frame depth dump (<stem>_depth.npz) of the teaser/qual renderer + box_depth_stats helper (CPU, no torch)."""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

QUAL = Path(__file__).resolve().parents[1] / "repro" / "qualitative"
sys.path.insert(0, str(QUAL))
import render_c3_figures as R  # noqa: E402
import box_depth_stats as B  # noqa: E402


def _arrs():
    rng = np.random.default_rng(0)
    gt = rng.uniform(1.0, 4.0, (12, 16)).astype(np.float32)
    valid = np.ones_like(gt, bool)
    valid[0, :3] = False
    return {"gt": gt, "gt_valid": valid, "ft": gt * 1.1, "c3_tag": gt * 0.9, "c3_raw": gt.copy(), "color_range_m": np.array([1.37, 3.365]),
            "frame_index": np.int64(5), "panel_h": np.int64(240), "raw_hw": np.array([480, 640], np.int64)}


def test_npz_keys_shapes_dtypes_and_roundtrip(tmp_path):
    a = _arrs()
    rec = R.write_depth_npz(tmp_path / "x_depth.npz", a)
    z = np.load(tmp_path / "x_depth.npz", allow_pickle=False)
    assert sorted(z.files) == rec["keys"] == sorted(a)
    for k in a:
        assert np.array_equal(z[k], a[k]) and list(z[k].shape) == rec["shapes"][k]
    assert z["ft"].dtype == np.float32 and z["gt_valid"].dtype == bool and z["gt"].shape == z["ft"].shape == z["gt_valid"].shape
    assert rec["sha256"] == hashlib.sha256((tmp_path / "x_depth.npz").read_bytes()).hexdigest() and rec["file"] == "x_depth.npz"


def test_npz_is_deterministic_and_timestamp_free(tmp_path):
    R.write_depth_npz(tmp_path / "a.npz", _arrs())
    time.sleep(2.1)
    R.write_depth_npz(tmp_path / "b.npz", dict(reversed(list(_arrs().items()))))    # also independent of dict order
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes()
    import zipfile
    assert {i.date_time for i in zipfile.ZipFile(tmp_path / "a.npz").infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_box_stats_floor_ceiling_and_cli(tmp_path, capsys):
    a = _arrs()
    a["gt"][:] = 2.0
    a["gt_valid"][:] = True
    a["ft"][:] = 4.0                                                        # all at the ceiling
    a["c3_tag"][:] = 2.1
    a["c3_raw"][:] = 1.0                                                    # all at the floor
    R.write_depth_npz(tmp_path / "s_depth.npz", a)
    B.main([str(tmp_path / "s_depth.npz"), "--rect", "2", "2", "10", "8"])
    out = capsys.readouterr().out
    rows = {l.split()[0]: l.split() for l in out.splitlines()[2:]}
    assert float(rows["ft"][6]) == 1.0 and float(rows["ft"][5]) == 0.0      # @ceil
    assert float(rows["c3_raw"][5]) == 1.0 and float(rows["c3_raw"][6]) == 0.0
    assert abs(float(rows["gt"][2]) - 2.0) < 1e-6 and rows["gt"][1] == str(8 * 6)
    from PIL import Image
    m = np.zeros((12, 16), np.uint8)
    m[2:8, 2:10] = 255
    Image.fromarray(m).save(tmp_path / "m.png")
    B.main([str(tmp_path / "s_depth.npz"), "--mask", str(tmp_path / "m.png")])
    assert capsys.readouterr().out.splitlines()[0].split("px")[1].split()[0] == "48"


def test_renderer_wires_dump_default_on_and_sidecar_key():
    src = (QUAL / "render_c3_figures.py").read_text()
    assert "--no-depth-dump" in src and src.count('"depth_dump"') == 2 and "write_depth_npz(out" in src
    assert 'dd = {"depth_dump"' in src and "**dd, **disp" in src


def test_display_modes_declared_and_abs_gt_is_default():
    assert list(R.DISPLAY_MODES) == ["abs_gt", "v1_log_p2p98", "v2_disparity_p2p98"] and R.DISPLAY_MODES["abs_gt"][0] == ""
    src = (QUAL / "render_c3_figures.py").read_text()
    assert 'default="abs_gt"' in src and "--no-teaser-chart" in src
    assert 'colour(c / 0.03, "inferno")' in src                       # abs_gt change maps unchanged


def test_meanchange_chart_csv_pdf_png_deterministic_and_traceable(tmp_path):
    import time
    m_ft, m_c3 = np.array([0.5, 0.4, 0.6, 0.45]), np.array([0.05, 0.04, 0.06, 0.05])
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ra = R.write_meanchange_chart(tmp_path / "a", "c", m_ft, m_c3)
    time.sleep(1.1)
    rb = R.write_meanchange_chart(tmp_path / "b", "c", m_ft, m_c3)
    for ext in ("csv", "pdf", "png"):
        assert (tmp_path / "a" / f"c.{ext}").read_bytes() == (tmp_path / "b" / f"c.{ext}").read_bytes()
    assert (tmp_path / "a" / "c.pdf").read_bytes()[:5] == b"%PDF-"
    rows = (tmp_path / "a" / "c.csv").read_text().splitlines()
    assert rows[0] == "frame,time_s,FT n46,RAMEN n46 + TAG" and "released" not in rows[0].lower()
    assert rows[1] == "2,0.1,0.500000,0.050000" and len(rows) == 5          # frames 2..N, values = the plotted arrays
    assert ra["csv_sha256"] == rb["csv_sha256"] == hashlib.sha256((tmp_path / "a" / "c.csv").read_bytes()).hexdigest()


def test_depth_display_passthrough_and_default_unchanged(monkeypatch, tmp_path):
    import repro.qualitative as Q
    seen = []
    monkeypatch.setattr(Q, "_run", lambda cmd, *a, **k: seen.append([str(c) for c in cmd]))
    Q._figures({"params": {"device": "cpu"}}, tmp_path, 99)
    assert "--depth-display" not in seen[-1]                                  # default unchanged (renderer default abs_gt)
    Q._figures({"params": {"device": "cpu", "depth_display": "abs_gt,v1_log_p2p98"}}, tmp_path, 99)
    assert seen[-1][seen[-1].index("--depth-display") + 1] == "abs_gt,v1_log_p2p98"


def test_teaser_yaml_requests_three_display_modes():
    import yaml as Y
    cfg = Y.safe_load((QUAL.parents[1] / "configs" / "paper_results.yaml").read_text())
    p = cfg["items"]["fig:teaser"]["params"]
    assert p["depth_display"].split(",") == list(R.DISPLAY_MODES) and "relative per panel" in p["display_disclosure"]

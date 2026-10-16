"""Commit Ifig8: long-video item (fig:long_video): yaml, Bonn 500-frame manifest, renderer placeholder refusal, wrapper flags, DEV-only, template key contract (CPU, no torch)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
QUAL = REPO / "repro" / "qualitative"
sys.path.insert(0, str(QUAL))
sys.path.insert(0, str(QUAL / "long_video"))

CFG = yaml.safe_load((REPO / "configs" / "paper_results.yaml").read_text())
IT = CFG["items"]["fig:long_video"]
MAN = QUAL / "long_video" / "bonn_video_long500.json"
SCORED_BONN = Path("/mnt/DATA/DegradedDepth/RAMEN/results/vda_clean_repro/benchmark_data/bonn/bonn_video.json")
BONN_RAW = Path("/mnt/DATA/DegradedDepth/data/TEST/bonn")
TEMPLATE = Path("/mnt/DATA/DegradedDepth/qualitative/qual_redesign_20261015/scripts/render_P_long.py")


def test_config_valid_and_item_declared():
    import repro.config as C
    assert C.validate(CFG) == []
    p = IT["params"]
    assert IT["kind"] == "qualitative" and IT["host"] == "gh200" and p["split"] == "dev" and p["cell"] == "llvar"
    assert [(c["dataset"], c["clip"], c["start"], c["n"]) for c in p["clips"]] == [
        ("bonn", "rgbd_bonn_placing_obstructing_box", 0, 500), ("scannet_rgbfix", "scene0009_00", 0, 500), ("nyuv2", "study_0002", 0, 500)]
    assert p["models"] == ["ft", "c3_raw", "c3_tag", "dav2_ft", "daac_ft"] and IT["models"] == ["ft_n46", "ramen_n46", "ramen_n46_tag"]
    pins = p["model_pins"]       # Ifig9: the TRAIN checkpoints are resolved (step12000 of dn01_c3rep_pf_{dav2a,daacm}_n46; records AUD-FTONLY-TRAIN-20261015)
    assert not any("PLACEHOLDER" in str(pins[m][k]) for m in ("dav2_ft", "daac_ft") for k in ("ckpt", "ckpt_gh200", "sha256"))
    assert pins["dav2_ft"]["sha256"] == "e5cbf7ca420741aa39001ad7ed658a8666cd0892558901418f1f22776404b876" and pins["daac_ft"]["sha256"] == "665b59ff0ad824861394970fd792d5c1a54f712ad019abd3d3458857a4261fb0"
    assert pins["dav2_ft"]["ckpt_gh200"].endswith("runs/dn01_c3rep_pf_dav2a_n46/step12000.pt") and pins["daac_ft"]["ckpt_gh200"].endswith("runs/dn01_c3rep_pf_daacm_n46/step12000.pt")
    assert p["clips"][0]["manifest"]["sha256"] == hashlib.sha256(MAN.read_bytes()).hexdigest()


def test_bonn_manifest_500_frames_and_first_110_are_the_scored_clip():
    m = json.loads(MAN.read_text())["bonn"]
    assert len(m) == 1
    key, fr = next(iter(m[0].items()))
    assert key == "rgbd_bonn_placing_obstructing_box" and len(fr) == 500 and len({f["image"] for f in fr}) == 500 and all(f["factor"] == 5000.0 for f in fr)
    if SCORED_BONN.exists():
        sc = {k: v for e in json.loads(SCORED_BONN.read_text())["bonn"] for k, v in e.items()}[key]
        assert fr[:110] == sc                                  # same frames, same pairing as the scored 110-frame clip (so the llvar noise prefix is identical)


@pytest.mark.skipif(not BONN_RAW.exists(), reason="raw Bonn not on this host")
def test_bonn_manifest_is_reproducible_from_raw_names():
    import build_bonn_long_manifest as B
    assert B.build(BONN_RAW, ["rgbd_bonn_placing_obstructing_box"], 30, 500) == json.loads(MAN.read_text())


def test_placeholder_model_is_refused_loudly():
    import render_long_video as R
    pins = {m: {k: f"PLACEHOLDER_{m}_{k}" for k in ("ckpt", "ckpt_gh200", "sha256")} for m in ("dav2_ft", "daac_ft")}     # Ifig9: the yaml pins are resolved; a placeholder must still be refused
    R.resolve_models(["ft", "c3_raw", "c3_tag"], pins, "local")                     # 3 models: no pin needed
    for host in ("local", "gh200"):
        for extra in ("dav2_ft", "daac_ft"):
            with pytest.raises(AssertionError, match="PLACEHOLDER"):
                R.resolve_models(["ft", "c3_raw", "c3_tag", extra], pins, host)
    with pytest.raises(AssertionError):
        R.resolve_models(["ft", "c3_raw", "c3_tag", "dav2_ft"], {}, "local")        # no pin at all
    with pytest.raises(AssertionError):
        R.resolve_models(["ft", "c3_tag"], pins, "local")                           # template needs ft, c3_raw, c3_tag
    with pytest.raises(AssertionError):
        R.resolve_models(["ft", "c3_raw", "c3_tag", "released_vda"], pins, "local")


def test_real_pin_is_checked_against_sha(tmp_path):
    import render_long_video as R
    ck = tmp_path / "a.pt"
    ck.write_bytes(b"x")
    good = {"dav2_ft": {"ckpt": str(ck), "ckpt_gh200": str(ck), "sha256": hashlib.sha256(b"x").hexdigest()}}
    assert R.resolve_models(["ft", "c3_raw", "c3_tag", "dav2_ft"], good, "local")["dav2_ft"]["ckpt"] == str(ck)
    bad = {"dav2_ft": dict(good["dav2_ft"], sha256="0" * 64)}
    with pytest.raises(AssertionError, match="sha256 mismatch"):
        R.resolve_models(["ft", "c3_raw", "c3_tag", "dav2_ft"], bad, "local")


def test_wrapper_passes_flags(monkeypatch, tmp_path):
    import repro.qualitative as Q
    seen = []
    monkeypatch.setattr(Q, "_run", lambda cmd, *a, **k: seen.append([str(c) for c in cmd]))
    Q.render_fig_long_video({"params": {**IT["params"], "device": "cpu"}}, tmp_path)
    assert len(seen) == 3
    b, s, n = seen
    for cmd in seen:
        assert cmd[0].endswith("render_long_video.py") and "--models" in cmd and cmd[cmd.index("--models") + 1] == "ft,c3_raw,c3_tag,dav2_ft,daac_ft" and "--smoke" not in cmd
    assert b[b.index("--clip") + 1] == "bonn:rgbd_bonn_placing_obstructing_box" and b[b.index("--start") + 1] == "0" and b[b.index("--n") + 1] == "500"
    assert b[b.index("--manifest") + 1].endswith("long_video/bonn_video_long500.json") and "--manifest-sha256" in b
    assert "--root" not in b                                                         # local host: the default Bonn root
    assert s[s.index("--manifest") + 1].endswith("scannet_video_rgbfix_500.json")
    assert "--manifest" not in n and "--root" not in n                               # nyuv2: benchmark manifest is already 500 frames
    assert sorted(p.name for p in (tmp_path).iterdir()) == ["bonn__rgbd_bonn_placing_obstructing_box", "nyuv2__study_0002", "scannet_rgbfix__scene0009_00"]


def test_wrapper_gh200_root_and_five_models(monkeypatch, tmp_path):
    import repro.qualitative as Q
    import repro.config as C
    monkeypatch.setattr(C, "detected_host", lambda: "gh200")
    monkeypatch.delenv("DATA_ROOT", raising=False)
    seen = []
    monkeypatch.setattr(Q, "_run", lambda cmd, *a, **k: seen.append([str(c) for c in cmd]))
    p = {**IT["params"], "device": "cpu", "models": ["ft", "c3_raw", "c3_tag", "dav2_ft", "daac_ft"]}
    Q.render_fig_long_video({"params": p}, tmp_path)
    b, s, _ = seen
    assert b[b.index("--root") + 1] == "/projects/b5dh/data/test/benchmarks/bonn"
    assert s[s.index("--manifest") + 1] == "/projects/b5dh/data/test/benchmarks/scannet/scannet_video_rgbfix_500.json"
    assert b[b.index("--models") + 1] == "ft,c3_raw,c3_tag,dav2_ft,daac_ft" and "--model-pins" in b and "--host" in b and b[b.index("--host") + 1] == "gh200"


def test_renderer_is_dev_only_and_uses_scored_degradation():
    src = (QUAL / "render_long_video.py").read_text()
    assert "allow_test" not in src and "--allow-test" not in src
    for s in ("bj.load_clips(ds, \"dev\")", "assert key in dev_keys", "bj.llvar_draw(ds, key)", "bj.llvar_noise_seed(ds, key, cell)", "bj.degrade_llvar(clean, cell, drawn, seed)", "bj.build_pipeline(specs[nm], dev)",
              "bj.apply_variant(cfg[\"tag\"]", "RC.write_depth_npz"):
        assert s in src


@pytest.mark.skipif(not TEMPLATE.exists(), reason="redesign template not on this host")
def test_npz_keys_cover_what_the_template_reads():
    import re
    t = TEMPLATE.read_text() + (TEMPLATE.parent / "qual_common.py").read_text()
    read = {k for k in set(re.findall(r'd\["(\w+)"\]', t)) if not k.startswith("_")} - {"gt_single_frame_index"}   # "_..." keys are template-internal; gt_single_frame_index is optional in the template ("only reported")
    src = (QUAL / "render_long_video.py").read_text()
    for k in read:
        assert f'"{k}"' in src, f"template reads npz key {k!r} that the renderer does not write"
    assert {"ft", "c3_raw", "c3_tag", "gt", "gt_valid"} <= read


def test_each_clip_slice_of_its_manifest_entry_has_exactly_n_frames():
    """Ifig10: yaml start/n are slice bounds INTO the manifest entry (render_long_video: frames_all[start:start+n]); the long Bonn manifest already holds the 500 frames raw 30..529, so start must be 0 (Ifig9 had 30 -> 470 frames -> the GH200 job aborted)."""
    man = json.loads(MAN.read_text())["bonn"]
    key, fr = next(iter(man[0].items()))
    bc = IT["params"]["clips"][0]
    assert bc["clip"] == key and len(fr[bc["start"]:bc["start"] + bc["n"]]) == bc["n"] == 500
    sc = IT["params"]["clips"][1]
    scm = Path(sc["manifest"]["local"])
    if scm.exists():                                                         # local ScanNet 500 manifest (not in the repo)
        e = [x for x in json.loads(scm.read_text())["scannet"] if sc["clip"] in x][0][sc["clip"]]
        assert len(e[sc["start"]:sc["start"] + sc["n"]]) == sc["n"]

"""Auditor WP18 part 2: B4 (TEST log from the night-TEST and F_noise scorers) and B5 (resume refuses a mismatched limit_clips / bench_version; aggregators refuse smoke and unpinned llvar results)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

import bench_joint as bj

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dn01_bvi_fnoise as fn  # noqa: E402
import dn01_headline as hl  # noqa: E402
import dn01_headline_table as ht  # noqa: E402
import dn01_heldout_eval as he  # noqa: E402


class _Boom(RuntimeError):
    pass


def _boom(*a, **k):
    raise _Boom("model build reached")


# ------------------------------------------------------------------ B4
def test_night_test_scorer_logs_before_any_model_is_built(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("RAMEN_TEST_ACCESS_LOG", str(log))
    monkeypatch.setattr(bj, "build_pipeline", _boom)
    a = argparse.Namespace(model="metric_vda_s", dataset=["nuscenes_night_test"], cells=["real"], split="test", allow_test=True, limit_clips=None, out=str(tmp_path), device="cpu")
    with pytest.raises(_Boom):
        he.cmd_run(a)
    rec = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(rec) == 1 and rec[0]["datasets"] == ["nuscenes_night_test"] and rec[0]["split"] == "test"


def test_fnoise_logs_before_any_model_is_built(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("RAMEN_TEST_ACCESS_LOG", str(log))
    monkeypatch.setattr(bj, "build_pipeline", _boom)
    reg = tmp_path / "reg.json"
    reg.write_text("{}")
    a = argparse.Namespace(model="metric_vda_s", root=str(tmp_path), registration=str(reg), device="cpu", relative=False, limit_clips=None, grid=None, out=str(tmp_path / "o"))
    with pytest.raises(_Boom):
        fn.run(a)
    assert len(log.read_text().splitlines()) == 1 and json.loads(log.read_text())["model"] == "metric_vda_s"


# ------------------------------------------------------------------ B5: resume
def _art(tmp_path, **kw):
    p = tmp_path / f"r{len(list(tmp_path.glob('r*.json')))}.json"
    p.write_text(json.dumps({"bench_version": bj.BENCH_VERSION, "limit_clips": None, **kw}))
    return p


def test_bench_joint_resume_refuses_a_smoke_file_for_a_full_run(tmp_path):
    p = _art(tmp_path, limit_clips=2)
    with pytest.raises(SystemExit, match="REFUSED to resume"):
        bj.check_resumable(p, None)
    with pytest.raises(SystemExit):
        bj.check_resumable(_art(tmp_path), 2)                      # and the other way round
    bj.check_resumable(p, 2)
    bj.check_resumable(_art(tmp_path), None)
    bj.check_resumable(tmp_path / "missing.json", None)           # nothing to resume: fine
    with pytest.raises(SystemExit):
        bj.check_resumable(_art(tmp_path, bench_version=1), None)


def test_wrapper_resume_refuses_mismatched_limit_and_version(tmp_path):
    st = {"expected": [str(_art(tmp_path, limit_clips=3))], "limit": None, "cmd": ["true"]}
    with pytest.raises(RuntimeError, match="REFUSED to resume"):
        hl.run_stage(st)
    assert hl.run_stage({**st, "limit": 3}) == "resumed"
    with pytest.raises(RuntimeError):
        hl.run_stage({"expected": [str(_art(tmp_path, bench_version=1))], "limit": None, "cmd": ["true"]})
    fnf = tmp_path / "f.json"                                      # F_noise files carry limit_clips only when set: absent = None
    fnf.write_text(json.dumps({"macro": {}}))
    assert hl.run_stage({"expected": [str(fnf)], "limit": None, "cmd": ["true"]}) == "resumed"
    with pytest.raises(RuntimeError):
        hl.run_stage({"expected": [str(fnf)], "limit": 2, "cmd": ["true"]})


def test_plan_stages_carry_the_limit():
    st = hl.plan("metric_vda_s", Path("/o"), ["kitti"], False, 4, "/b")
    assert all(s.get("limit") == 4 for s in st) and all(s.get("limit") is None for s in hl.plan("metric_vda_s", Path("/o"), ["kitti"], False, None, "/b"))


# ------------------------------------------------------------------ Table addendum (e): ckpt_sha256 in the log and the manifest
def test_ckpt_info_hashes_the_file_without_loading_a_model(tmp_path):
    import hashlib
    ck = tmp_path / "step4000.pt"
    ck.write_bytes(b"weights")
    want = hashlib.sha256(b"weights").hexdigest()
    for spec in (f"ours:{ck}", f"ours:{ck}+agf0.2t0.06", f"pocketdvdnet:{ck}+metric_vda_s"):
        i = bj.ckpt_info(spec)
        assert i["ckpt_sha256"] == want and i["ckpt_basename"] == "step4000.pt" and i["ckpts"] == {"step4000.pt": want}
    assert bj.ckpt_info("metric_vda_s+ema0.7") == {"ckpt_sha256": None, "ckpt_basename": None, "ckpts": {}}


def test_the_access_log_records_the_checkpoint_sha(tmp_path):
    ck = tmp_path / "c.pt"
    ck.write_bytes(b"x")
    log = tmp_path / "l.jsonl"
    a = argparse.Namespace(model=f"ours:{ck}", dataset=["kitti"], cells=["llvar"], split="all", allow_test=True, limit_clips=None, out="/o")
    bj.log_test_access(a, log)
    bj.log_test_access(argparse.Namespace(**{**vars(a), "model": "metric_vda_s"}), log)
    r = [json.loads(x) for x in log.read_text().splitlines()]
    assert r[0]["ckpt_sha256"] == bj.ckpt_info(a.model)["ckpt_sha256"] and r[0]["ckpt_basename"] == "c.pt"
    assert r[1]["ckpt_sha256"] is None and r[1]["ckpt_basename"] is None


# ------------------------------------------------------------------ B5: aggregators
def test_results_refuse_smoke_and_unpinned_llvar(tmp_path):
    good = {"bench": "bench_joint", "dataset": "kitti", "cell": "llvar", "model": "m", "split": "all", "llvar_draws_sha256": "ab", "limit_clips": None, "video_macro": {}}
    (tmp_path / "a__kitti__llvar__all__v3.json").write_text(json.dumps(good))
    assert len(bj._results(str(tmp_path))) == 1
    (tmp_path / "a__kitti__llvar__all__v3.json").write_text(json.dumps({**good, "llvar_draws_sha256": None}))
    with pytest.raises(ValueError, match="null llvar_draws_sha256"):
        bj._results(str(tmp_path))
    (tmp_path / "a__kitti__llvar__all__v3.json").write_text(json.dumps({**good, "limit_clips": 2}))
    with pytest.raises(ValueError, match="smoke"):
        bj._results(str(tmp_path))
    bj.check_aggregable({"cell": "b3s25", "llvar_draws_sha256": None, "limit_clips": None})      # a non-llvar cell has no pin: fine


def test_headline_table_refuses_smoke_and_unpinned_llvar(tmp_path):
    base = {"bench": "bench_joint", "dataset": "kitti", "cell": "llvar", "model": "m", "llvar_draws_sha256": "ab", "limit_clips": None}
    (tmp_path / "x.json").write_text(json.dumps(base))
    assert ("bench", "kitti", "llvar", "m") in ht.load_results(tmp_path)
    (tmp_path / "x.json").write_text(json.dumps({**base, "limit_clips": 1}))
    with pytest.raises(ValueError, match="smoke"):
        ht.load_results(tmp_path)
    (tmp_path / "x.json").write_text(json.dumps({**base, "llvar_draws_sha256": None}))
    with pytest.raises(ValueError, match="llvar_draws_sha256"):
        ht.load_results(tmp_path)

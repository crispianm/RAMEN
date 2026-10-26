"""CPU-only tests (no GPU, no scorer import, no model). Run:
  python -B -m pytest repro/efficiency/test_efficiency.py --basetemp <tmpdir> -p no:cacheprovider
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.efficiency import common as C  # noqa: E402
from repro.efficiency import launcher as LA  # noqa: E402
from repro.efficiency import legs as L  # noqa: E402


def test_array_sha_native_dtype_sensitive():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert C.array_sha256(a) == C.array_sha256(a.copy())
    assert C.array_sha256(a) != C.array_sha256(a.astype(np.float64))           # dtype is part of the digest: a cast can never pass
    b = a.copy(); b[1, 1] = np.nextafter(b[1, 1], np.float32(10))                              # one-ulp change is detected
    assert C.array_sha256(a) != C.array_sha256(b)
    assert C.array_sha256(np.asfortranarray(a)) == C.array_sha256(a)               # layout-independent


def test_cache_path_unique_and_safe():
    p1, p2 = C.cache_path(Path("/x"), "kitti", "llvar", "a/b"), C.cache_path(Path("/x"), "kitti", "llvar", "a_b")
    assert p1 != p2 and "/" not in p1.name


def test_leg_table_complete_and_unique():
    legs = L.build_legs()
    ids = [l.id for l in legs]
    assert len(ids) == len(set(ids)) == 24
    assert {"ramen-S0-fp32", "ramen-S0-fp32-rep1", "ramen-S2-fp32-rep1", "mvda-S2-fp32-rep1", "ramen-S4-fp16"} <= set(ids)
    assert all(l.alone == ("cudnn" in l.flags) for l in legs)                 # T6 legs and only they run alone
    assert not any(l.agf_backend != "cpu" for l in legs if l.stack != "S4")
    assert not any(l.model == "mvda" and l.stack in ("X", "S4") for l in legs)
    assert next(l for l in legs if l.id == "ramen-X-fp32").flags == ["hookoff", "fp16build"]


def test_exclude_toggles_reform_stacks():
    legs = {l.id: l for l in L.build_legs(exclude=("T8", "T7"))}
    assert "dense_group" not in legs["ramen-S3-fp32"].flags and "chlast" not in legs["ramen-S3-fp32"].flags and "bnfold" in legs["ramen-S3-fp32"].flags


def test_comparisons_cover_increments_and_primary():
    cs = L.comparisons(L.build_legs(), "S3")
    pairs = {(c["test"], c["ref"], c["gate"]) for c in cs}
    assert ("ramen-X-fp32", "ramen-S0-fp32", "bitwise") in pairs
    assert ("ramen-S4-fp16", "ramen-S3-fp16", "fp16") in pairs
    assert ("ramen-S3-fp16", "ramen-S0-fp32", "fp16") in pairs and ("ramen-S3-fp32", "ramen-S0-fp32", "fp32") in pairs      # C16 / C32
    assert ("mvda-S1-fp32", "mvda-S0-fp32", "fp32") in pairs


def test_estimate_scales_with_N():
    legs = L.build_legs()
    assert L.estimate(legs, 8)["est_wall_h"] < L.estimate(legs, 2)["est_wall_h"]


def test_dry_run_prints_table_and_starts_nothing(capsys, monkeypatch):
    called = []
    monkeypatch.setattr(LA.subprocess, "Popen", lambda *a, **k: called.append(a))
    assert LA.main(["plan", "--dry-run", "--workers", "4"]) == 0
    out = capsys.readouterr().out
    assert "ramen-S3-fp16" in out and "estimated wall" in out and "nothing started" in out and not called
    assert LA.main(["run", "--dry-run", "--workers", "3"]) == 0 and not called


def test_run_refuses_without_token_or_verified_cache(tmp_path):
    with pytest.raises(SystemExit):
        LA.main(["run", "--workers", "2", "--cache", str(tmp_path), "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        LA.main(["run", "--workers", "2", "--cache", str(tmp_path), "--out", str(tmp_path / "o"), "--token", "X"])        # no verify_report PASS


def _fake_leg(root: Path, lid: str, depth_seed=0, metric=0.1, mask_bad=False, rows=3):
    d = root / "legs" / lid
    d.mkdir(parents=True)
    shas, csvrows = {}, []
    for i in range(rows):
        arr = np.random.RandomState(depth_seed + i).rand(4, 4).astype(np.float32)
        shas[f"kitti|llvar|c{i}"] = {"depth_sha256": C.array_sha256(arr), "depth_finite_mask_sha256": C.array_sha256(np.isfinite(arr) & (not mask_bad or i != 0)),
                                     "depth_all_finite": True, "input_sha256": "in"}
        csvrows.append({"leg": lid, "dataset": "kitti", "cell": "llvar", "variant": "none", "clip": f"c{i}", "abs_rel": str(metric), "psnr": "30.0", "ssim": "0.9", "tof": "1.0"})
    (d / "depth_shas.json").write_text(json.dumps(shas))
    macro = {"kitti|llvar|none": {"num_clips": rows, "video_macro": {"abs_rel": metric, "psnr": 30.0, "ssim": 0.9, "tof": 1.0}}}
    (d / "macro.json").write_text(json.dumps({"macro": macro}))
    (d / "toggles.json").write_text("{}")
    import csv
    with open(d / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csvrows[0]))
        w.writeheader(); w.writerows(csvrows)


def test_compare_gates(tmp_path):
    _fake_leg(tmp_path, "a", 0, 0.1000)
    _fake_leg(tmp_path, "b", 0, 0.1000)                       # identical
    _fake_leg(tmp_path, "c", 7, 0.1000)                       # different depth, same metrics
    _fake_leg(tmp_path, "d", 0, 0.1002)                       # 2e-3 rel metric shift
    _fake_leg(tmp_path, "e", 0, 0.1000, mask_bad=True)
    ld = lambda x: LA.load_leg(tmp_path, x)                   # noqa: E731
    assert LA.compare_pair(ld("b"), ld("a"), "bitwise")["PASS"]
    assert not LA.compare_pair(ld("c"), ld("a"), "bitwise")["PASS"]                  # one differing bit in depth => not bitwise
    assert LA.compare_pair(ld("c"), ld("a"), "fp32")["PASS"]
    assert not LA.compare_pair(ld("d"), ld("a"), "fp32")["PASS"] and LA.compare_pair(ld("d"), ld("a"), "fp16")["PASS"]
    assert not LA.compare_pair(ld("e"), ld("a"), "fp16")["PASS"]                     # finite-mask change = FAIL


def test_missing_row_is_fail(tmp_path):
    _fake_leg(tmp_path, "a", 0, rows=3)
    _fake_leg(tmp_path, "b", 0, rows=2)
    assert not LA.compare_pair(LA.load_leg(tmp_path, "b"), LA.load_leg(tmp_path, "a"), "fp32")["PASS"]


def test_identical_control(tmp_path):
    _fake_leg(tmp_path, "f", 0); _fake_leg(tmp_path, "g", 0); _fake_leg(tmp_path, "h", 1)
    assert LA.identical_control(LA.load_leg(tmp_path, "f"), LA.load_leg(tmp_path, "g"))["PASS"]
    assert not LA.identical_control(LA.load_leg(tmp_path, "f"), LA.load_leg(tmp_path, "h"))["PASS"]


def test_failed_leg_is_rerun_not_dropped(tmp_path, monkeypatch):
    """First attempt of a leg exits 1 (OOM), second succeeds: the launcher must re-run it and report DONE; a leg failing every attempt must be FAIL."""
    attempts = {}

    class P:
        def __init__(self, ok, out):
            self.ok, self.out = ok, out
        def poll(self):
            if self.ok:
                (self.out / "DONE.json").write_text("{}")
            return 0 if self.ok else 1

    def fake_start(leg, out, a, wid, source):
        n = attempts[leg.id] = attempts.get(leg.id, 0) + 1
        d = out / "legs" / leg.id; d.mkdir(parents=True, exist_ok=True)
        return P((leg.id == "flaky" and n >= 2) or leg.id == "good", d)

    monkeypatch.setattr(LA, "start_leg", fake_start)
    monkeypatch.setattr(LA.time, "sleep", lambda s: None)
    legs = [L.Leg(id=i, model="ramen", stack="S0", precision="fp32") for i in ("good", "flaky", "bad")]
    st = LA.run_phase(legs, tmp_path, None, 2, "cache")
    assert st == {"good": "DONE", "flaky": "DONE", "bad": "FAIL"} and attempts["bad"] == LA.MAX_ATTEMPTS
    assert len((tmp_path / "launcher_log.jsonl").read_text().splitlines()) >= 5


def test_untouched_record_and_sums(tmp_path):
    (tmp_path / "x.txt").write_text("hi"); (tmp_path / "y.txt").write_text("yo")
    s = C.write_sha256sums(tmp_path).read_text().splitlines()
    assert len(s) == 2 and s[0].endswith("x.txt")


def test_no_timing_fields_in_worker_source():
    src = (Path(__file__).with_name("leg_worker.py")).read_text()
    for banned in ("time.time", "perf_counter", "elapsed", "cuda.Event", "synchronize"):
        assert banned not in src


def test_toggle_fields_declared():
    assert set(C.TOGGLE_FIELDS) >= {"attn_backend", "precision", "bn_folded", "cudnn_benchmark", "channels_last", "compiled", "refine_hook", "input_staging", "agf_backend",
                                    "venv", "torch", "cudnn_allow_tf32", "matmul_allow_tf32", "float32_matmul_precision"}
    assert C.PINNED_NUMERICS == {"cudnn_allow_tf32": True, "matmul_allow_tf32": False, "float32_matmul_precision": "highest"}


def test_gpu_gate_inert_and_exclusive(tmp_path, monkeypatch):
    import fcntl
    from repro.efficiency.gpu_gate import gate
    monkeypatch.delenv("EFF_GATE_DIR", raising=False)
    with gate(True):
        pass
    monkeypatch.setenv("EFF_GATE_DIR", str(tmp_path))
    with gate(False):
        with gate(False):                           # shared holders coexist
            pass
        f = open(tmp_path / "gpu.lock", "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise AssertionError("exclusive lock must fail while a shared holder exists")
        except BlockingIOError:
            pass
        f.close()
    with gate(True):
        f = open(tmp_path / "gpu.lock", "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
            raise AssertionError("shared lock must fail while exclusive")
        except BlockingIOError:
            pass
        f.close()

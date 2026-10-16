"""Stub-only tests of the bridge scheduling/compare (no GPU, no scorer, no model). Run:
  /home/wg19671/FocusMamba/.venv/bin/python -B -m pytest repro/efficiency/test_bridge.py --basetemp /mnt/DATA/DegradedDepth/claude_outputs/scratch/paper_repro/pt_bridge -p no:cacheprovider
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.efficiency import bridge as B  # noqa: E402
from repro.efficiency import common as C  # noqa: E402
from repro.efficiency import launcher as LA  # noqa: E402
from repro.efficiency import legs as L  # noqa: E402

VARS = ("none", "agf0.2t0.06", "agf0.5t0.12")


def write_leg(out: Path, lid: str, scale: float = 1.0, sha: str = "a", psnr: float = 30.0, variants=VARS, finite=True):
    d = out / "legs" / lid
    d.mkdir(parents=True)
    macro = {f"kitti|llvar|{v}": {"num_clips": 1, "video_macro": {"abs_rel": 0.1 * scale, "tae": 0.2 * scale, "psnr": psnr, "ssim": 0.9, "tof": 1.0}} for v in variants}
    (d / "macro.json").write_text(json.dumps({"macro": macro}))
    (d / "depth_shas.json").write_text(json.dumps({"kitti|llvar|c1": {"depth_sha256": sha, "input_sha256": "i", "depth_all_finite": finite, "depth_finite_mask_sha256": "m"}}))
    with open(d / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "cell", "variant", "clip", "abs_rel"])
        w.writeheader()
        for v in variants:
            w.writerow({"dataset": "kitti", "cell": "llvar", "variant": v, "clip": "c1", "abs_rel": 0.1 * scale})
    (d / "toggles.json").write_text("{}")


def pfile(tmp_path, text="stack=S4\n"):
    p = tmp_path / "efficiency_primary.txt"
    p.write_text(text)
    return p


def test_primary_file_absent_refuses(tmp_path):
    with pytest.raises(SystemExit, match="absent"):
        B.read_primary(tmp_path / "nope.txt")
    with pytest.raises(SystemExit, match="stack must be"):
        B.read_primary(pfile(tmp_path, "stack=S0\n"))
    with pytest.raises(SystemExit, match="unknown toggle"):
        B.read_primary(pfile(tmp_path, "stack=S3\nexclude=T99\n"))


def test_legs_s4_and_exclusions(tmp_path):
    prim = B.read_primary(pfile(tmp_path, "stack=S4\n"))
    lg = B.build_bridge_legs(prim, which=B.parse_which("B1,B2,B3,B4"))
    assert lg["B1"].toggles == [] and lg["B1"].precision == "fp16" and lg["B1"].agf_backend == "cpu"
    assert lg["B2"].agf_backend == "fast_triton_v3" and "T10" in lg["B2"].toggles
    assert lg["B4"].model == "mvda" and lg["B4"].toggles == ["T3", "T4", "T6", "T9"] and lg["B3"].toggles == []
    prim2 = B.read_primary(pfile(tmp_path, "stack=S4\nexclude=T11,T10\n"))
    b2 = B.build_bridge_legs(prim2)["B2"]
    assert b2.agf_backend == "cpu" and "T10" not in b2.toggles
    x = B.build_bridge_legs(B.read_primary(pfile(tmp_path, "stack=X\n")))
    assert set(x) == {"B1", "B2"} and x["B2"].toggles == ["T1", "T2"]
    with pytest.raises(SystemExit):
        B.build_bridge_legs(B.read_primary(pfile(tmp_path, "stack=X\n")), which=B.parse_which("B1,B2,B3,B4"))


def test_which_validation():
    assert B.parse_which(None) == ("B1", "B2")
    for bad in ("B1", "B2,B3", "B1,B2,B3", "B1,B2,B9"):
        with pytest.raises(SystemExit):
            B.parse_which(bad)


def test_leg_subset_filter():
    legs = L.build_legs()
    sub = L.select_legs(legs, "ramen-S4-fp16,ramen-S0-fp16")
    assert [l.id for l in sub] == ["ramen-S0-fp16", "ramen-S4-fp16"]       # original launch order
    assert L.select_legs(legs, None) is legs
    with pytest.raises(SystemExit):
        L.select_legs(legs, "nope")


def test_compare_pass_and_gated_fail(tmp_path):
    prim = B.read_primary(pfile(tmp_path))
    lg = B.build_bridge_legs(prim)
    out = tmp_path / "o"
    write_leg(out, lg["B1"].id)
    write_leg(out, lg["B2"].id, scale=1.001, sha="b")           # 1e-3 rel < 5e-3, shas differ (fine for fp16)
    rep = B.evaluate(out, B.bridge_comparisons(lg, prim))
    assert len(rep) == 3 and all(r["PASS"] for r in rep) and all(r["host"] == "4090" for r in rep)
    assert rep[1]["tag_backends"]["test"] == "fast_triton_v3" and rep[1]["tag_backends"]["ref"] == "numpy_reference"
    assert B.verdict(rep)["paired"]
    out2 = tmp_path / "o2"                                        # a TAG-only failure must fail the bridge
    write_leg(out2, lg["B1"].id)
    write_leg(out2, lg["B2"].id, scale=1.0, sha="b")
    mj = out2 / "legs" / lg["B2"].id / "macro.json"
    m = json.loads(mj.read_text())
    m["macro"]["kitti|llvar|agf0.2t0.06"]["video_macro"]["tae"] *= 1.02
    mj.write_text(json.dumps(m))
    rep = B.evaluate(out2, B.bridge_comparisons(lg, prim))
    v = B.verdict(rep)
    assert not v["paired"] and v["failed"] == ["bridge_tag_agf0.2t0.06_B2_vs_B1"] and "UNPAIRED" in v["latency_4090"]


def test_psnr_tof_tolerances(tmp_path):
    prim = B.read_primary(pfile(tmp_path, "stack=S3\n"))
    lg = B.build_bridge_legs(prim)
    out = tmp_path / "o"
    write_leg(out, lg["B1"].id)
    write_leg(out, lg["B2"].id, psnr=30.03)
    assert not B.verdict(B.evaluate(out, B.bridge_comparisons(lg, prim)))["PASS"]
    assert B.bridge_comparisons(lg, prim)[1]["tag_backends"]["test"] == "numpy_reference"     # no T11 -> numpy


def test_bitwise_for_X(tmp_path):
    prim = B.read_primary(pfile(tmp_path, "stack=X\n"))
    lg = B.build_bridge_legs(prim)
    comps = B.bridge_comparisons(lg, prim)
    assert {c["gate"] for c in comps} == {"bitwise"}
    out = tmp_path / "o"
    write_leg(out, lg["B1"].id)
    write_leg(out, lg["B2"].id, sha="a")
    assert B.verdict(B.evaluate(out, comps))["PASS"]
    out2 = tmp_path / "o2"
    write_leg(out2, lg["B1"].id)
    write_leg(out2, lg["B2"].id, sha="b")
    assert not B.verdict(B.evaluate(out2, comps))["PASS"]


def test_missing_leg_and_unit_mismatch_and_nonfinite(tmp_path):
    prim = B.read_primary(pfile(tmp_path))
    lg = B.build_bridge_legs(prim)
    out = tmp_path / "o"
    write_leg(out, lg["B1"].id)
    assert not B.verdict(B.evaluate(out, B.bridge_comparisons(lg, prim)))["PASS"]          # B2 missing
    write_leg(out, lg["B2"].id, variants=("none",), finite=False)
    rep = B.evaluate(out, B.bridge_comparisons(lg, prim))
    assert not any(r["PASS"] for r in rep)


def test_lock_protocol(tmp_path):
    lk = tmp_path / "GPU_LOCK"
    lk.write_text("")
    B.lock_acquire(lk)
    assert lk.read_text().startswith("bridge-r1 ")
    with pytest.raises(SystemExit, match="held"):
        B.lock_acquire(lk)
    B.lock_release(lk)
    assert lk.read_text() == ""
    lk.write_text("latency ladder 1\n")
    B.lock_release(lk)                                            # never clears someone else's lock
    assert lk.read_text().strip() == "latency ladder 1"


class Args:
    pass


def mk_args(tmp_path, pf, out, lock, repeat=False, legs="B1,B2"):
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    (cache / "verify_report.json").write_text(json.dumps({"summary": {"PASS": True}}))
    a = Args()
    a.__dict__.update(primary_file=pf, legs=legs, cache=cache, out=out, source="cache", token="TOK", lock=lock, repeat_binding=repeat, patches_dir=None)
    return a


def test_run_first_pass_then_single_binding_repeat(tmp_path, monkeypatch):
    pf = pfile(tmp_path)
    lock = tmp_path / "GPU_LOCK"
    lock.write_text("")
    out = tmp_path / "out"
    calls = []
    state = {"fail_first": True}

    def fake_phase(legs, o, a, n, source):
        assert n == 1 and source == "cache" and lock.read_text().startswith("bridge-r1")       # lock held while legs run
        for l in legs:
            calls.append(l.id)
            bad = state["fail_first"] and l.id.startswith("B2") and "rep1" not in l.id
            write_leg(o, l.id, scale=1.02 if bad else 1.0, sha="a")
        return {l.id: "DONE" for l in legs}
    monkeypatch.setattr(LA, "run_phase", fake_phase)
    monkeypatch.setattr(LA, "gpu_other_procs", lambda *a_, **k: [])
    monkeypatch.setattr(C, "untouched_record", lambda label: {"label": label, "x": 1})
    assert LA.main is not None
    a = mk_args(tmp_path, pf, out, lock)
    assert B.cmd_bridge_run(a) == 1
    r = json.loads((out / "BRIDGE_RESULT.json").read_text())
    assert r["host"] == "4090" and not r["verdict"]["paired"] and lock.read_text() == ""
    assert (out / "BEFORE.json").exists() and (out / "AFTER.json").exists()
    with pytest.raises(SystemExit):                              # no second first-pass
        B.cmd_bridge_run(mk_args(tmp_path, pf, out, lock))
    state["fail_first"] = False
    assert B.cmd_bridge_run(mk_args(tmp_path, pf, out, lock, repeat=True)) == 0
    fin = json.loads((out / "BRIDGE_FINAL.json").read_text())
    assert not fin["first_pass"]["verdict"]["PASS"] and fin["binding_repeat"]["verdict"]["PASS"] and fin["host"] == "4090"
    assert (out / "BEFORE_rep1.json").exists() and (out / "BEFORE.json").exists()
    assert [c for c in calls if "rep1" in c] == ["B1_ramen-S0-fp16-rep1", "B2_ramen-S4-fp16-rep1"]
    with pytest.raises(SystemExit, match="exactly one"):         # never a second repeat
        B.cmd_bridge_run(mk_args(tmp_path, pf, out, lock, repeat=True))


def test_run_refuses(tmp_path, monkeypatch):
    lock = tmp_path / "GPU_LOCK"
    lock.write_text("")
    with pytest.raises(SystemExit, match="absent"):
        B.cmd_bridge_run(mk_args(tmp_path, tmp_path / "missing.txt", tmp_path / "o", lock))
    lock.write_text("someone 1\n")
    with pytest.raises(SystemExit, match="held"):
        B.cmd_bridge_run(mk_args(tmp_path, pfile(tmp_path), tmp_path / "o2", lock))
    lock.write_text("")
    a = mk_args(tmp_path, pfile(tmp_path), tmp_path / "o3", lock)
    a.token = None
    with pytest.raises(SystemExit, match="token"):
        B.cmd_bridge_run(a)
    a = mk_args(tmp_path, pfile(tmp_path), tmp_path / "o4", lock)
    (a.cache / "verify_report.json").write_text(json.dumps({"summary": {"PASS": False}}))
    with pytest.raises(SystemExit, match="verified cache"):
        B.cmd_bridge_run(a)
    monkeypatch.setattr(LA, "gpu_other_procs", lambda *a_, **k: ["123, python, 9000"])
    with pytest.raises(SystemExit, match="not idle"):
        B.cmd_bridge_run(mk_args(tmp_path, pfile(tmp_path), tmp_path / "o5", lock))
    assert lock.read_text() == ""                                # lock released after refusal


def test_cli_wiring(tmp_path, capsys):
    pf = pfile(tmp_path)
    assert LA.main(["bridge", "plan", "--primary-file", str(pf), "--legs", "B1,B2,B3,B4"]) == 0
    o = capsys.readouterr().out
    assert "host=4090" in o and "bridge_raw_B4_vs_B3" in o
    assert LA.main(["plan", "--dry-run", "--legs", "ramen-S0-fp16,ramen-S4-fp16"]) == 0


# ---- L (Addendum 16): possible later bridge with PRIMARY C or AZ ---------------------------------------------------------------------------------------
def _pf(tmp_path, text):
    f = tmp_path / "p.txt"
    f.write_text(text)
    return f


def test_primary_c_and_az_accepted_cz_and_bz_refused(tmp_path):
    for st in ("C", "AZ"):
        assert B.read_primary(_pf(tmp_path, f"stack={st}\n"))["stack"] == st
    for st in ("CZ", "BZ", "A", "B", "S0"):
        with pytest.raises(SystemExit):
            B.read_primary(_pf(tmp_path, f"stack={st}\n"))


def test_bridge_legs_for_c_and_az(tmp_path):
    for st, mv in (("C", "S1"), ("AZ", "S3")):
        legs = B.build_bridge_legs(B.read_primary(_pf(tmp_path, f"stack={st}\n")), which=("B1", "B2", "B3", "B4"))
        assert legs["B1"].stack == "S0" and legs["B2"].stack == st and legs["B2"].precision == "fp16" and legs["B2"].agf_backend == "cpu"
        assert legs["B2"].toggles == L.STACK_TOGGLES["ramen"][st] and legs["B2"].id == f"B2_ramen-{st}-fp16"
        assert legs["B4"].stack == mv and legs["B4"].toggles == L.STACK_TOGGLES["mvda"][mv]               # Metric-VDA-S matching stack
        assert legs["B2"].alone == (st == "AZ")                                                              # T6 only in AZ

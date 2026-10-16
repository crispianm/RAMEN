"""Unit-level tests for generate_results.py / repro/ (no GPU, no torch)."""
from __future__ import annotations

import copy
import json
import os
import math
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_results as G  # noqa: E402
from repro import aggregate as A  # noqa: E402
from repro import config as C  # noqa: E402
from repro import plan as P  # noqa: E402
from repro import verify as V  # noqa: E402
from repro.util import slug  # noqa: E402

CFG = yaml.safe_load("""
schema_version: 1
tag: {alpha: 0.2, tau: 0.06, token: agf0.2t0.06}
models:
  ramen: {label: RAMEN, spec: "ours:{ckpt}", ckpt: {path: /x/step12000.pt, sha256: c9decd90}}
  ramen_tag: {label: RAMEN+TAG, spec: "ours:{ckpt}", tag: true, ckpt: {path: /x/step12000.pt, sha256: c9decd90}}
  mvda: {label: MVDA, spec: metric_vda_s}
datasets:
  kitti: {n_clips: 3, manifest_sha256: 7024ba29}
  bonn: {n_clips: 2}
  kitti_minus_0047: {n_clips: 2, derive_from: {base: kitti, drop: drive_0047}}
items:
  tab:static:
    kind: bench
    models: [ramen, ramen_tag, mvda]
    datasets: [kitti, bonn, kitti_minus_0047]
    cells: [st_llvar]
    split: all
    temporal: true
    metrics: [abs_rel_aligned, flicker, tae_gtvalid, psnr]
    aggregation: {mean_over: [kitti, bonn], tae_datasets: [kitti]}
  tab:dynamic:
    kind: bench
    models: [ramen, mvda]
    datasets: [kitti]
    cells: [st_llvar, llvar]
    split: all
    temporal: true
    metrics: [abs_rel_aligned]
  tab:clean:
    kind: bench
    models: [mvda]
    datasets: [kitti]
    cells: [b0s0]
    split: dev
    metrics: [abs_rel_aligned]
  fig:qual: {kind: qualitative, models: [ramen]}
""")


def art(ds, cell, vals, split="all"):
    pv = {f"c{i}": dict(v) for i, v in enumerate(vals)}
    keys = vals[0].keys()
    return {"dataset": ds, "cell": cell, "split": split, "limit_clips": None, "bench_version": 3, "num_clips": len(vals), "per_video": pv,
            "video_macro": {k: sum(v[k] for v in vals) / len(vals) for k in keys}}


def test_schema_valid_and_unknown_keys():
    assert C.validate(CFG) == []
    bad = copy.deepcopy(CFG)
    bad["items"]["tab:static"]["bogus"] = 1
    bad["items"]["tab:static"]["models"].append("nope")
    bad["tag"]["token"] = "agf0.3t0.06"
    errs = C.validate(bad)
    assert any("bogus" in e for e in errs) and any("nope" in e for e in errs) and any("tag.token" in e for e in errs)


def test_ckpt_required_when_spec_uses_placeholder():
    bad = copy.deepcopy(CFG)
    del bad["models"]["ramen"]["ckpt"]
    assert any("ckpt.path" in e for e in C.validate(bad))


def test_spec_resolution_and_tag():
    assert C.model_spec(CFG, "ramen") == "ours:/x/step12000.pt"
    assert C.model_spec(CFG, "ramen_tag") == "ours:/x/step12000.pt+agf0.2t0.06"
    assert C.model_spec(CFG, "mvda") == "metric_vda_s"


def test_slug_matches_bench_joint():
    assert slug("ours:/projects/b5dh/RAMEN/x/step12000.pt+agf0.2t0.06") == "ours-_projects_b5dh_RAMEN_x_step12000+agf0.2t0.06"
    src = (Path(__file__).resolve().parents[1] / "bench_joint.py").read_text()
    assert 'spec.replace("ours_up:", "oursup-").replace("ours:", "ours-").replace("/", "_").replace(".pt", "")' in src


def test_plan_dedup_and_derived_base():
    units, items = P.plan(CFG)
    keys = [u.key for u in units]
    assert len(keys) == len(set(keys))
    shared = [u for u in units if u.model_id == "ramen" and u.dataset == "kitti" and u.cell == "st_llvar"]
    assert len(shared) == 1 and set(shared[0].items) == {"tab:static", "tab:dynamic"}
    assert any(u.derive_from for u in units)
    assert units[-1].derive_from is not None                  # derived last
    # only fig:qual selected -> no units
    assert P.plan(CFG, ["fig:qual"])[0] == []


def test_only_normalisation():
    assert P.normalise_only(CFG, ["tab_static"]) == ["tab:static"]
    with pytest.raises(C.ConfigError):
        P.normalise_only(CFG, ["tab:nope"])


def test_command_shape():
    u = next(u for u in P.plan(CFG, ["tab:static"])[0] if u.model_id == "ramen_tag" and u.dataset == "kitti")
    cmd = u.command(Path("/o"), py="py", smoke=True)
    assert cmd[1].endswith("scripts/dn01_headline.py") and cmd[2] == "--bj" and "--allow-test" in cmd and "--temporal" in cmd
    assert cmd[cmd.index("--limit-clips") + 1] == "1" and cmd[cmd.index("--model") + 1].endswith("+agf0.2t0.06")
    dev = next(u for u in P.plan(CFG, ["tab:clean"])[0])
    assert "--allow-test" not in dev.command(Path("/o"))


def test_test_gate():
    units, _ = P.plan(CFG, ["tab:static"])
    assert "TEST" in P.test_gate(units, False, {})
    assert "RAMEN_AUDITOR_TOKEN" in P.test_gate(units, True, {})
    assert P.test_gate(units, True, {"RAMEN_AUDITOR_TOKEN": "AUD-1"}) is None
    assert P.test_gate(P.plan(CFG, ["tab:clean"])[0], False, {}) is None


def test_dry_run_estimate_text(tmp_path):
    units, items = P.plan(CFG)
    txt = P.format_plan(units, items, CFG, tmp_path)
    assert "TOTAL" in txt and "derive" in txt and "tab:static" in txt


def test_aggregation_and_derive():
    base = art("kitti", "st_llvar", [{"abs_rel_aligned": 0.1, "flicker": 0.01, "tae_gtvalid": 1.0, "psnr": 20.0},
                                      {"abs_rel_aligned": 0.3, "flicker": 0.03, "tae_gtvalid": 3.0, "psnr": 22.0},
                                      {"abs_rel_aligned": 0.5, "flicker": 0.05, "tae_gtvalid": 5.0, "psnr": 24.0}])
    base["per_video"] = {"a_drive_0047": base["per_video"]["c0"], "b": base["per_video"]["c1"], "c": base["per_video"]["c2"]}
    d = A.derive_alias_art(base, "kitti_minus_0047", "drive_0047")
    assert d["num_clips"] == 2 and d["video_macro"]["abs_rel_aligned"] == pytest.approx(0.4)
    bon = art("bonn", "st_llvar", [{"abs_rel_aligned": 0.2, "flicker": 0.02, "tae_gtvalid": 9.0, "psnr": 30.0}] * 2)
    res = A.aggregate_item("tab:static", CFG["items"]["tab:static"], {("ramen", "kitti", "st_llvar"): base, ("ramen", "bonn", "st_llvar"): bon})
    mean = res["mean"][0]["metrics"]
    assert mean["abs_rel_aligned"] == pytest.approx((0.3 + 0.2) / 2)
    assert mean["tae_gtvalid"] == pytest.approx(3.0)           # tae only over kitti
    assert all(not r["macro_mismatch"] for r in res["rows"])


def test_write_item_csv(tmp_path):
    a = art("kitti", "b0s0", [{"abs_rel_aligned": 0.1}])
    res = A.aggregate_item("tab:clean", CFG["items"]["tab:clean"], {("mvda", "kitti", "b0s0"): a})
    jp, cp = A.write_item(res, tmp_path, "tab_clean")
    assert cp.read_text().splitlines()[0] == "model,dataset,cell,split,n_clips,abs_rel_aligned"
    assert json.loads(jp.read_text())["rows"][0]["metrics"]["abs_rel_aligned"] == 0.1


def _res(vals):
    return {"rows": [{"model": m, "dataset": "kitti", "cell": "llvar", "metrics": v} for m, v in vals.items()], "mean": []}


def test_verify_tolerances():
    ref = _res({"a": {"abs_rel_aligned": 0.100, "psnr": 20.0, "tof_native": 1.0}, "b": {"abs_rel_aligned": 0.200, "psnr": 25.0, "tof_native": 2.0}})
    ok = _res({"a": {"abs_rel_aligned": 0.1004, "psnr": 20.015, "tof_native": 1.012}, "b": {"abs_rel_aligned": 0.200, "psnr": 25.0, "tof_native": 2.0}})
    assert V.compare_results(ok, ref, same_host=False)["status"] == "PASS"
    assert V.compare_results(ok, ref, same_host=True)["status"] == "FAIL"
    bad = copy.deepcopy(ok)
    bad["rows"][0]["metrics"]["psnr"] = 20.05
    assert V.compare_results(bad, ref, False)["status"] == "FAIL"


def test_verify_missing_extra_and_flip():
    ref = _res({"a": {"abs_rel_aligned": 0.10}, "b": {"abs_rel_aligned": 0.20}})
    assert V.compare_results(_res({"a": {"abs_rel_aligned": 0.10}}), ref, False)["status"] == "FAIL"
    assert V.compare_results(_res({"a": {"abs_rel_aligned": 0.10}, "b": {"abs_rel_aligned": 0.2}, "c": {"abs_rel_aligned": 1}}), ref, False)["status"] == "FAIL"
    ref2 = _res({"a": {"abs_rel_aligned": 0.1000}, "b": {"abs_rel_aligned": 0.1008}})      # 0.8% apart, each candidate within 0.5% but the order flips
    flip = _res({"a": {"abs_rel_aligned": 0.1004}, "b": {"abs_rel_aligned": 0.1003}})
    r = V.compare_results(flip, ref2, False)
    assert r["status"] == "FLAG" and r["flags"][0]["why"] == "rank/sign flip"


def test_latency_flag_not_fail():
    ref = _res({"a": {"per_frame_ms_p50": 10.0}})
    cand = _res({"a": {"per_frame_ms_p50": 12.0}})
    assert V.compare_results(cand, ref, False, kind="latency")["status"] == "FLAG"


def test_finished_refuses_smoke_mixup(tmp_path):
    from repro import execute as X
    u = P.plan(CFG, ["tab:clean"])[0][0]
    p = u.out_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"limit_clips": 1, "bench_version": 3}))
    assert X.finished(u, tmp_path, smoke=True)[0]
    with pytest.raises(RuntimeError):
        X.finished(u, tmp_path, smoke=False)


def test_dirty_tree_refusal(tmp_path, monkeypatch, capsys):
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(yaml.safe_dump(CFG))
    from repro import manifest as M
    monkeypatch.setattr(M, "frozen_state", lambda p: {"header_problems": [], "tracked": True, "modified": False, "commit": "x", "ok": True})
    monkeypatch.setattr(M, "git_state", lambda out=None: {"commit": "x", "branch": "b", "dirty": True, "dirty_files": [" M foo.py"]})
    rc = G.main(["--config", str(cfgp), "--out", str(tmp_path / "o"), "--only", "tab:clean"])
    assert rc == 3 and "dirty" in capsys.readouterr().err


def test_dry_run_cli(tmp_path, capsys):
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(yaml.safe_dump(CFG))
    assert G.main(["--config", str(cfgp), "--out", str(tmp_path / "o"), "--dry-run"]) == 0
    o = capsys.readouterr().out
    assert "TOTAL" in o and "MISSING INPUT" in o and "TEST gate" in o


def test_end_to_end_aggregate_only(tmp_path, monkeypatch):
    """Fake scorer outputs on disk -> aggregate-only run writes csv/json/MANIFEST with output hashes."""
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(yaml.safe_dump(CFG))
    out = tmp_path / "o"
    units, _ = P.plan(CFG, ["tab:clean"])
    for u in units:
        p = u.out_path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(art("kitti", "b0s0", [{"abs_rel_aligned": 0.1}, {"abs_rel_aligned": 0.2}], "dev")))
    from repro import manifest as M
    monkeypatch.setattr(M, "frozen_state", lambda p: {"header_problems": [], "tracked": True, "modified": False, "commit": "x", "ok": True})
    monkeypatch.setattr(M, "git_state", lambda out=None: {"commit": "x", "branch": "b", "dirty": False, "dirty_files": []})
    assert G.main(["--config", str(cfgp), "--out", str(out), "--only", "tab:clean", "--aggregate-only"]) == 0
    man = json.loads((out / "MANIFEST.json").read_text())
    assert "tab_clean.csv" in man["output_sha256"] and man["git"]["commit"] == "x"
    assert G.main(["--config", str(cfgp), "--out", str(out), "--only", "tab:clean", "--verify", str(out), "--same-host"]) == 0


# ---- follow-up features: per-model references, wrapper units, seed mean, hosts, scorer clone, tag backend ----
CFG2 = copy.deepcopy(CFG)
CFG2["models"]["init"] = {"label": "INIT", "spec": "fidelity_init:temp2", "runner": "fidelity_init", "ckpt": {"path": "/x/temp2.pt", "sha256": "96d35f08"}}
CFG2["models"]["s46"] = {"label": "s46", "spec": "metric_vda_s"}
CFG2["models"]["s47"] = {"label": "s47", "spec": "dav2_s"}
CFG2["items"]["tab:p1"] = {"kind": "bench", "host": "4090", "models": ["ramen"], "datasets": ["kitti"], "cells": ["llvar"], "split": "dev", "temporal": True,
                           "metrics": ["abs_rel_aligned"], "override_script": "bench_overrides",
                           "arms": {"full": {"coupling": 1.0, "stage2": 1.0}, "coup0": {"coupling": 0.0, "stage2": 1.0}}}
CFG2["items"]["tab:fid"] = {"kind": "bench", "models": ["s46", "s47", "init"], "datasets": ["kitti", "bonn"], "cells": ["llvar"], "split": "all", "metrics": ["psnr_native"],
                            "seed_mean": [{"name": "MEAN", "members": ["s46", "s47"]}],
                            "reference": {"default": "refA", "by_model": {"s47": "refB"}}}
CFG2["smoke"] = {"models": ["ramen", "mvda", "s46"], "datasets": ["kitti", "bonn"], "cells": ["llvar", "st_llvar", "b0s0"]}
CFG2["items"]["tab:latency"] = {"kind": "latency", "models": ["mvda"], "metrics": ["per_frame_ms_p50"],
                                "harness": {"runner": "ladder", "stack": "S0", "rows": ["mvda_bracket", "ramen_raw"], "iters": 30, "samples": 30}}


def test_schema_extensions_valid_and_errors():
    assert C.validate(CFG2) == []
    bad = copy.deepcopy(CFG2)
    bad["items"]["tab:p1"]["arms"]["x"] = {"bogus": 1}
    bad["items"]["tab:fid"]["seed_mean"][0]["members"].append("ramen")
    bad["items"]["tab:fid"]["reference"] = {"by_model": {"nope": "r"}}
    bad["items"]["tab:fid"]["tag_backend"] = "turbo"
    e = C.validate(bad)
    assert len(e) >= 4


def test_reference_dir_per_model():
    assert C.reference_dir(CFG2, "tab:fid", "s46").name == "refA"
    assert C.reference_dir(CFG2, "tab:fid", "s47").name == "refB"
    assert C.reference_dir(CFG2, "tab:clean", "mvda") is None


def test_override_arm_units_and_command():
    units, _ = P.plan(CFG2, ["tab:p1"])
    assert {u.row_id for u in units} == {"ramen@full", "ramen@coup0"}
    assert len({u.rel_dir() for u in units}) == 2 and all(u.host == "4090" for u in units)
    u = next(u for u in units if u.arm["name"] == "coup0")
    cmd = u.command(Path("/o"), py="py", scorer_clone=Path("/clone"))
    assert cmd[1].endswith("repro/vendored/bench_overrides.py") and cmd[cmd.index("--coupling") + 1] == "0.0" and "--" in cmd and "--grid" in cmd
    assert u.cwd(Path("/clone")) == Path("/clone")


def test_runner_unit_fidelity_init():
    units, _ = P.plan(CFG2, ["tab:fid"])
    r = [u for u in units if u.runner]
    assert len(r) == 2
    cmd = r[0].command(Path("/o"), py="py", scorer_clone=Path("/clone"))
    assert cmd[1].endswith("vendored/fidelity_init_score.py") and cmd[cmd.index("--bench-root") + 1] == "/clone" and cmd[cmd.index("--expect-sha") + 1] == "96d35f08"
    assert r[0].filename() == "restorer_temp2_init__kitti__llvar__all__v3.json"
    bad = copy.deepcopy(CFG2)
    bad["items"]["tab:fid"]["cells"] = ["llvar", "st_llvar"]
    assert any("fidelity_init" in e for e in C.validate(bad))


def test_seed_mean_rows():
    mk = lambda v: art("kitti", "llvar", [{"psnr_native": v}, {"psnr_native": v}])           # noqa: E731
    got = {("s46", "kitti", "llvar"): mk(10.0), ("s47", "kitti", "llvar"): mk(14.0), ("init", "kitti", "llvar"): mk(1.0)}
    res = A.aggregate_item("tab:fid", CFG2["items"]["tab:fid"], got)
    m = next(r for r in res["rows"] if r["model"] == "MEAN")
    assert m["metrics"]["psnr_native"] == 12.0 and m["n_seeds"] == 2 and m["seed_range"]["psnr_native"] == [10.0, 14.0]


def test_mean_rows_include_seed_group():
    cfg = copy.deepcopy(CFG2["items"]["tab:fid"])
    cfg["aggregation"] = {"mean_over": ["kitti", "bonn"]}
    got = {}
    for d, v in (("kitti", 10.0), ("bonn", 20.0)):
        got[("s46", d, "llvar")] = art(d, "llvar", [{"psnr_native": v}])
        got[("s47", d, "llvar")] = art(d, "llvar", [{"psnr_native": v + 2}])
    res = A.aggregate_item("tab:fid", cfg, got)
    mm = next(r for r in res["mean"] if r["model"] == "MEAN")
    assert mm["metrics"]["psnr_native"] == pytest.approx(16.0)


def test_host_split_and_hours():
    units, items = P.plan(CFG2)
    hrs = P.hours_by_host(units)
    assert set(hrs) <= {"gh200", "4090", "any"} and "gh200" in hrs and "4090" in hrs
    su, _ = P.plan(CFG2, smoke=True)
    assert {u.host for u in su} == {"4090"}
    txt = P.format_plan(su, items, CFG2, Path("/o"), smoke=True, qual_present={"fig:qual": False})
    assert "GPU hours per host" in txt and "fig:qual[MISSING]" in txt and "SMOKE" in txt
    fu, _ = P.plan(CFG2, ["tab:fid"], tag_backend="fast")
    assert {u.host for u in fu} == {"4090"} and all(u.tag_backend == "fast" for u in fu)


def test_tag_backend_env_and_key():
    from repro import execute as X
    ref = P.plan(CFG2, ["tab:fid"])[0][0]
    fast = P.plan(CFG2, ["tab:fid"], tag_backend="fast")[0][0]
    assert ref.key != fast.key
    assert os.environ.get("RAMEN_AGF_BACKEND") or "RAMEN_AGF_BACKEND" not in X.unit_env(ref, Path("/o"))
    assert X.unit_env(fast, Path("/o"))["RAMEN_AGF_BACKEND"] == "gpu"


def test_shared_unit_with_conflicting_hosts_is_any():
    c = copy.deepcopy(CFG2)
    c["items"]["tab:dynamic"]["host"] = "4090"
    units, _ = P.plan(c, ["tab:static", "tab:dynamic"])
    shared = next(u for u in units if u.model_id == "ramen" and u.dataset == "kitti" and u.cell == "st_llvar")
    assert shared.host == "any"


def test_vendored_scripts_byte_identical():
    from repro import execute as X
    assert X.verify_vendored() == []
    vj = json.loads((Path(__file__).resolve().parents[1] / "repro" / "vendored" / "VENDORED.json").read_text())["files"]
    assert vj["bench_overrides.py"]["commit"] == "0c6d07b"


def test_ensure_scorer_clone(tmp_path):
    import subprocess
    from repro import execute as X
    r = tmp_path / "repo"
    r.mkdir()
    g = lambda *a: subprocess.run(["git", "-C", str(r), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True, text=True).stdout.strip()   # noqa: E731
    g("init", "-q")
    (r / "bench_joint.py").write_text("x=1\n")
    g("add", "."); g("commit", "-qm", "scorer")
    c0 = g("rev-parse", "HEAD")
    (r / "new_repro_file.py").write_text("y\n")
    g("add", "."); g("commit", "-qm", "later")
    d = X.ensure_scorer_clone(tmp_path / "clone", c0, r)
    assert (d / "bench_joint.py").exists() and not (d / "new_repro_file.py").exists()
    assert X.ensure_scorer_clone(tmp_path / "clone", c0, r) == d                   # idempotent
    (r / "bench_joint.py").write_text("x=2\n")                                       # scorer edited -> refuse
    with pytest.raises(RuntimeError, match="differ"):
        X.ensure_scorer_clone(tmp_path / "clone2", c0, r)


def test_smoke_is_dev_only_with_counts_and_one_arm():
    c = copy.deepcopy(CFG2)
    c["smoke"] = {"models": ["ramen", "mvda"], "datasets": ["kitti"], "cells": ["llvar", "st_llvar", "b0s0"]}
    c["items"]["tab:p1"]["models"] = ["ramen"]
    stats: dict = {}
    units, items = P.plan(c, smoke=True, stats=stats)
    assert all(u.split == "dev" or u.kind == "latency" for u in units)
    assert not P.needs_test(units) and P.test_gate(units, False, {}) is None
    assert stats["dropped_test"] > 0 and stats["dropped_arms"] == 1
    arms = {u.arm["name"] for u in units if u.arm}
    assert arms == {"full"}
    assert {u.dataset for u in units if u.kind == "bench"} <= {"kitti"}
    txt = P.format_plan(units, items, c, Path("/o"), smoke=True, stats=stats)
    assert f"Dropped {stats['dropped_test']} TEST-split" in txt and "time basis" in txt and "timing" in txt


def test_latency_label_is_timing_not_test():
    units, items = P.plan(CFG2, ["tab:latency"])
    assert units and P.needs_test(units) == []
    assert " timing " in P.format_plan(units, items, CFG2, Path("/o"))


def test_plan_json_roundtrip():
    units, items = P.plan(CFG2, ["tab:p1", "tab:latency"], smoke=True)
    d = P.plan_json(units, items, CFG2, Path("/o"), True, {}, {})
    json.dumps(d)
    assert {u["label"] for u in d["units"]} == {"dev", "timing"} and d["units"][0]["command"]


def test_compare_sha_list(tmp_path):
    ref, out = tmp_path / "ref", tmp_path / "out"
    ref.mkdir(); out.mkdir()
    (ref / "a.png").write_bytes(b"A"); (ref / "b.gif").write_bytes(b"B"); (ref / "c.png").write_bytes(b"C")
    import hashlib
    (ref / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((ref / n).read_bytes()).hexdigest()}  {n}\n" for n in ("a.png", "b.gif", "c.png")))
    (out / "a.png").write_bytes(b"A"); (out / "b.gif").write_bytes(b"different")
    r = V.compare_sha_list(ref, out)
    assert r["byte_identical"] == ["a.png"] and r["bytes_differ"] == ["b.gif"] and r["missing"] == ["c.png"] and r["status"] == "FAIL"


POST_TOKEN = __import__("re").compile(r"^(ema|tmed|agf|sgf|rgf|dvp|kal|oneeuro|flowema)")


def _model_arg(cmd):
    return cmd[cmd.index("--model") + 1] if "--model" in cmd else None


def test_no_grid_invocation_carries_a_depth_post_token():
    for cfgx, kw in ((CFG, {}), (CFG2, {}), (CFG2, {"smoke": True})):
        units, _ = P.plan(cfgx, **kw)
        n_grid = 0
        for u in units:
            cmd = u.command(Path("/o"), py="py", scorer_clone=Path("/c")) if not u.run_with else None
            if cmd and "--grid" in cmd:
                n_grid += 1
                assert not any(POST_TOKEN.match(t) for t in _model_arg(cmd).split("+")[1:]), cmd
                assert not any(POST_TOKEN.match(t) for t in _model_arg(cmd).split(":")[-1].split("+")[1:]), cmd
        assert n_grid or cfgx is CFG


def test_raw_and_tag_share_one_grid_pass():
    units, _ = P.plan(CFG, ["tab:static"])
    raw = next(u for u in units if u.model_id == "ramen" and u.dataset == "kitti")
    tag = next(u for u in units if u.model_id == "ramen_tag" and u.dataset == "kitti")
    assert raw.grid == ["none", "agf0.2t0.06"] and tag.run_with == raw.key and tag.group == [] and raw.group == [tag]
    assert raw.rel_dir() == tag.rel_dir()                                   # one directory; files named as single-token runs
    cmd = raw.command(Path("/o"), py="py")
    assert cmd[cmd.index("--model") + 1] == "ours:/x/step12000.pt" and cmd[cmd.index("--grid") + 1:cmd.index("--grid") + 3] == ["none", "agf0.2t0.06"]
    assert raw.filename().startswith("ours-_x_step12000__kitti") and tag.filename().startswith("ours-_x_step12000+agf0.2t0.06__kitti")
    assert tag.est_min == 0.0
    lone = next(u for u in P.plan(CFG, ["tab:clean"])[0])                    # lone raw: plain run, no --grid
    assert "--grid" not in lone.command(Path("/o"))


def test_finished_requires_whole_grid_pass(tmp_path):
    from repro import execute as X
    units, _ = P.plan(CFG, ["tab:static"])
    raw = next(u for u in units if u.model_id == "ramen" and u.dataset == "kitti")
    p = raw.out_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"limit_clips": None, "bench_version": 3}))
    assert X.finished(raw, tmp_path, False)[0] is False
    raw.group[0].out_path(tmp_path).write_text(json.dumps({"limit_clips": None, "bench_version": 3}))
    assert X.finished(raw, tmp_path, False)[0] is True


def _fnoise_art(macro):
    return {"split": "test", "per_sequence": {"A__10": {"f_noise": 1.0}, "A__20": {"f_noise": 3.0}, "B__20": {"f_noise": 8.0}},
            "per_scene": {"A": {"f_noise": 2.0}, "B": {"f_noise": 8.0}}, "macro": {"f_noise": macro}}


def test_fnoise_uses_per_scene_macro_and_mismatch_fails_verify():
    item = {"metrics": ["f_noise"], "aggregation": {}}
    good = A.aggregate_item("tab:rep_fnoise", item, {("m", "bvi", "fnoise"): _fnoise_art(5.0)})     # scene macro (2+8)/2 = 5; the sequence mean would be 4
    row = good["rows"][0]
    assert row["metrics"]["f_noise"] == pytest.approx(5.0) and row["macro_mismatch"] == []
    bad = A.aggregate_item("tab:rep_fnoise", item, {("m", "bvi", "fnoise"): _fnoise_art(4.0)})      # a stored macro the per-scene mean does not reproduce
    assert bad["rows"][0]["macro_mismatch"] == ["f_noise"]
    rep = V.compare_results(bad, good, True)
    assert rep["status"] == "FAIL" and any("macro_mismatch (cand)" in f["why"] for f in rep["fails"])
    assert V.compare_results(good, good, True)["status"] == "PASS"
    rep_ref = V.compare_results(good, bad, True)                                                    # a defective reference aggregation fails too
    assert rep_ref["status"] == "FAIL" and any("macro_mismatch (ref)" in f["why"] for f in rep_ref["fails"])


def test_latency_rows_carry_protocol_iters_for_samples_complete():
    art_row = {"per_frame_ms_p50": 5.0, "per_frame_ms_p95": 6.0, "protocol_iters": 30, "samples": {"n_samples": 30}}
    res = A.aggregate_item("tab:latency", {"metrics": ["per_frame_ms_p50"], "aggregation": {}}, {("ramen_raw", "-", "-"): art_row})
    r = res["rows"][0]
    assert r["protocol_iters"] == 30 and G._n_samples(r.get("samples")) == r["protocol_iters"]

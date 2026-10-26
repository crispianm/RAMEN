"""Freeze commit F: ladder latency item, conditional item, FROZEN header, checkpoint re-hash, pre-declared readings, host rules, flicker-curve definition (CPU only, no torch)."""
from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import generate_results as G  # noqa: E402
from repro import aggregate as A  # noqa: E402
from repro import config as C  # noqa: E402
from repro import execute as X  # noqa: E402
from repro import manifest as M  # noqa: E402
from repro import plan as P  # noqa: E402
from repro import readings as R  # noqa: E402
from repro import verify as V  # noqa: E402

CFG = C.load(REPO / "configs" / "paper_results.yaml")
# The clean-frame non-inferiority RULE (Auditor R1 ruling 4, vs the RELEASED Metric-VDA-S) is no longer attached to tab:clean (commit I: descriptive, no verdict); readings.clean_verdict stays unit-tested on this local copy.
CLEAN_READING = {"kind": "clean_noninferiority", "reference_model": "metric_vda_s", "subjects": ["earlier_s45", "earlier_s45_tag"], "datasets": ["kitti", "bonn", "scannetpp"], "rel_tol": 0.05,
                 "ratio_metrics": ["abs_rel_aligned", "abs_rel_unaligned", "tae_gtvalid"], "abs_metrics": {"delta1_aligned": 0.01}, "report_only": ["tof"],
                 "sentence_preserved": "clean-frame accuracy is preserved (within 5% of Metric-VDA-S; development clips)", "sentence_differs": "differs on clean frames"}


def _clean_item():
    return {**CFG["items"]["tab:clean"], "models": ["earlier_s45", "earlier_s45_tag", "metric_vda_s", "ft_n46"], "reading": CLEAN_READING,
            "metrics": ["abs_rel_aligned", "abs_rel_unaligned", "delta1_aligned", "tae_gtvalid", "tof"]}


# ---------------------------------------------------------------- header / freeze
def test_real_config_header_is_frozen_mark():
    assert C.frozen_header_problems((REPO / "configs" / "paper_results.yaml").read_text()) == []


def test_header_refusals():
    assert any("DRAFT" in e for e in C.frozen_header_problems("# x  DRAFT for token\nschema_version: 1\n"))
    assert any("FROZEN" in e for e in C.frozen_header_problems("# x\nschema_version: 1\n"))
    assert C.frozen_header_problems("# a\n# FROZEN (this commit)\nschema_version: 1\n") == []
    assert C.frozen_header_problems("schema_version: 1\n# FROZEN (this commit)\n") != []        # the mark must be in the leading comment block


def test_real_run_refuses_draft_config(tmp_path, capsys):
    p = tmp_path / "c.yaml"
    p.write_text((REPO / "configs" / "paper_results.yaml").read_text().replace("FROZEN (this commit)", "DRAFT", 1))
    assert G.main(["--config", str(p), "--out", str(tmp_path / "o"), "--only", "tab:latency"]) == 3
    assert "not a frozen one" in capsys.readouterr().err


def test_real_run_refuses_untracked_config(tmp_path, capsys):
    p = tmp_path / "c.yaml"
    p.write_text((REPO / "configs" / "paper_results.yaml").read_text())
    assert M.frozen_state(p)["ok"] is False
    assert G.main(["--config", str(p), "--out", str(tmp_path / "o"), "--only", "tab:latency"]) == 3


def test_fast_tag_backend_and_skip_hash_refused_for_real_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(M, "frozen_state", lambda p: {"header_problems": [], "tracked": True, "modified": False, "commit": "x", "ok": True})
    assert G.main(["--out", str(tmp_path / "o"), "--only", "tab:static", "--tag-backend", "fast", "--allow-dirty"]) == 3
    assert "reference (numpy)" in capsys.readouterr().err
    assert G.main(["--out", str(tmp_path / "o"), "--only", "tab:clean", "--skip-hash", "--allow-dirty"]) == 3
    assert "--skip-hash" in capsys.readouterr().err


# ---------------------------------------------------------------- checkpoint re-hash
def test_mismatch_is_not_skippable(tmp_path):
    f = tmp_path / "step.pt"
    f.write_bytes(b"abc")
    cfg = {"models": {"m": {"spec": "ours:{ckpt}", "ckpt": {"path": str(f), "sha256": "0" * 64}}}, "host_paths": {}}
    u = P.Unit("bench", "m", "ours:x", "kitti", "b0s0", "dev", False)
    info, problems = X.check_inputs(cfg, [u])
    assert info["m"]["status"] == "MISMATCH" and X.mismatches(info) and problems
    cfg["models"]["m"]["ckpt"]["sha256"] = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    info, problems = X.check_inputs(cfg, [u])
    assert info["m"]["status"] == "ok" and not X.mismatches(info) and not problems


def test_ft_n47_n48_are_hash_pinned_in_config():
    for m, sha in (("ft_n47", "ec38c317"), ("ft_n48", "d769e513")):
        assert CFG["models"][m]["ckpt"]["sha256"].startswith(sha) and CFG["models"][m + "_tag"]["ckpt"]["sha256"].startswith(sha)


# ---------------------------------------------------------------- latency ladder item
def test_latency_item_is_the_session2_s0_rows_with_samples():
    h = CFG["items"]["tab:latency"]["harness"]
    assert h["stack"] == "S0" and h["iters"] == 30 and h["samples"] == 30 and h["warmup"] == 5
    assert h["rows"] == ["mvda_bracket", "ramen_raw", "ramen_tag_v3_agf0.2t0.06", "ramen_tag_v3_agf0.5t0.12", "ramen_tag_numpy_agf0.2t0.06", "ramen_tag_numpy_agf0.5t0.12"]
    assert CFG["items"]["tab:latency"]["reference"].endswith("latency_ladder/session_2")


def test_latency_unit_command_and_smoke(monkeypatch):
    monkeypatch.delenv("RAMEN_LADDER_PYTHON", raising=False)
    monkeypatch.delenv("RAMEN_GPU_LOCK", raising=False)
    units, items = P.plan(CFG, ["tab:latency"])
    assert len(units) == 1 and units[0].harness["stack"] == "S0" and units[0].host == "4090"
    cmd = units[0].command(Path("/o"))
    assert cmd[1].endswith("repro/efficiency/latency/ladder_samples.py") and ["--stacks", "S0"] == cmd[2:4] and "--skip-standalone" in cmd
    assert cmd[cmd.index("--warmup") + 1] == "5" and cmd[cmd.index("--iters") + 1] == "30"
    sc = units[0].command(Path("/o"), smoke=True)
    assert sc[sc.index("--warmup") + 1] == "1" and sc[sc.index("--iters") + 1] == "3"
    monkeypatch.setenv("RAMEN_LADDER_PYTHON", "/x/py")
    assert units[0].command(Path("/o"))[0] == "/x/py"


def test_ladder_wrapper_leaves_run_ladder_untouched_and_stores_samples(tmp_path):
    sys.path.insert(0, str(REPO / "repro" / "efficiency" / "latency"))
    import ladder_samples as LS
    import run_ladder as RL
    assert RL.time_stages is LS._time_stages and RL.Session.run_item is LS._run_item
    written = {}

    class FakeCommon:
        @staticmethod
        def write_json(path, rec):
            written[Path(path).name] = rec

    class FakeSession:
        out = tmp_path
        common = FakeCommon
        rows: list = []

    def fake_orig(self, item):
        LS._LAST["res"] = [{"total": [1.0] * 30}, {"total": [2.0] * 30}]
        self.rows.append({"id": item["id"]})
    LS._orig_run_item = fake_orig
    LS._run_item(FakeSession(), {"id": "S0__ramen_raw"})
    rec = written["S0__ramen_raw.json"]
    assert rec["samples"]["n_samples"] == 30 and len(rec["samples"]["window1_total_ms"]) == 30 and len(rec["samples"]["window2_total_ms"]) == 30


def _ladder_session(root: Path, stack="S0", with_samples=True, scale=1.0):
    (root / "rows").mkdir(parents=True)

    def row(i, w1, kind="row", post=None, setting=None):
        r = {"id": f"{stack}__{i}", "kind": kind, "window1": {"ms_per_frame_p50": w1 * scale, "ms_per_frame_p95": w1 * scale * 1.01, "window_ms_p50": 32 * w1 * scale},
             "window2": {"ms_per_new_frame_p50": 1.45 * w1 * scale}, "mem": {"peak_alloc_GiB": 3.2}, "protocol": {"iters": 30}}
        if with_samples:
            r["samples"] = {"window1_total_ms": [32 * w1] * 30, "window2_total_ms": [22 * w1] * 30, "n_samples": 30}
        (root / "rows" / f"{stack}__{i}.json").write_text(json.dumps(r))
    for k, v in enumerate([5.73, 5.74, 5.75, 5.751, 5.752, 5.76]):
        row(f"mvda_bracket{k}", v, "bracket")
    for rid, v in (("ramen_raw", 9.5), ("ramen_tag_v3_agf0.2t0.06", 9.56)):
        row(rid, v)


def test_ladder_rows_bracket_upper_median_and_all_samples(tmp_path):
    _ladder_session(tmp_path / "s")
    got = A.ladder_rows(tmp_path / "s", "S0", ["mvda_bracket", "ramen_raw", "ramen_tag_v3_agf0.2t0.06", "ramen_tag_numpy_agf0.2t0.06"])
    assert set(k[0] for k in got) == {"metric_vda_s", "ramen_raw", "ramen_tag_v3_agf0.2t0.06"}          # a row not in the session is simply absent
    assert got[("metric_vda_s", "-", "-")]["per_frame_ms_p50"] == 5.751                               # upper median of the six runs (sorted[3])
    assert got[("metric_vda_s", "-", "-")]["samples"]["per_run_n_samples"] == [30] * 6
    assert len(got[("ramen_raw", "-", "-")]["samples"]["window1_total_ms"]) == 30
    res = A.aggregate_item("tab:latency", {"metrics": ["per_frame_ms_p50", "per_frame_ms_p95", "window_ms_p50", "ms_per_new_frame_w2_p50"], "aggregation": {"per_dataset": "macro"}}, got)
    r = {x["model"]: x for x in res["rows"]}
    assert r["ramen_raw"]["metrics"]["window_ms_p50"] == pytest.approx(32 * 9.5) and r["ramen_raw"]["samples"]["n_samples"] == 30


def test_samples_complete_flag_and_verify_flag_not_fail(tmp_path):
    _ladder_session(tmp_path / "ref")
    _ladder_session(tmp_path / "cand", scale=1.2)
    rows = ["mvda_bracket", "ramen_raw"]
    cand = A.aggregate_item("l", {"metrics": ["per_frame_ms_p50"], "aggregation": {}}, A.ladder_rows(tmp_path / "cand", "S0", rows))
    ref = A.aggregate_item("l", {"metrics": ["per_frame_ms_p50"], "aggregation": {}}, A.ladder_rows(tmp_path / "ref", "S0", rows))
    assert V.compare_results(cand, ref, False, kind="latency")["status"] == "FLAG"                      # 20% > 10%: flag, never fail
    assert G._n_samples(A.ladder_rows(tmp_path / "cand", "S0", rows)[("ramen_raw", "-", "-")]["samples"]) == 30
    assert G._n_samples(None) is None


def test_conditional_latency_eff(tmp_path, monkeypatch):
    f = tmp_path / "efficiency_primary.txt"
    monkeypatch.setenv("RAMEN_EFFICIENCY_PRIMARY", str(f))
    st = {}
    units, items = P.plan(CFG, ["tab:latency", "tab:latency_eff"], stats=st)
    assert items == ["tab:latency"] and "SKIPPED tab:latency_eff" in st["conditions"]["tab:latency_eff"]
    f.write_text("# released PRIMARY\nS3\n")
    st = {}
    units, items = P.plan(CFG, ["tab:latency", "tab:latency_eff"], stats=st)
    assert items == ["tab:latency", "tab:latency_eff"] and sorted(u.harness["stack"] for u in units) == ["S0", "S3"]
    f.write_text("nonsense\n")
    with pytest.raises(C.ConfigError):
        P.plan(CFG, ["tab:latency_eff"])


def test_latency_without_harness_or_with_sample_mismatch_is_invalid():
    bad = copy.deepcopy(CFG)
    del bad["items"]["tab:latency"]["harness"]
    assert any("ladder harness" in e for e in C.validate(bad))
    bad = copy.deepcopy(CFG)
    bad["items"]["tab:latency"]["harness"]["samples"] = 29
    assert any("samples" in e for e in C.validate(bad))


# ---------------------------------------------------------------- hosts
def test_static_diag_shares_tab_static_units_on_gh200():
    units, _ = P.plan(CFG, ["tab:static", "tab:static_diag"])
    assert {u.host for u in units} == {"gh200"}
    shared = [u for u in units if {"tab:static", "tab:static_diag"} <= set(u.items)]
    assert shared and all(u.host == "gh200" for u in shared)
    assert CFG["items"]["tab:static_diag"].get("host") == "gh200"


def test_one_host_per_item_check():
    a = P.Unit("bench", "m", "m", "kitti", "c", "dev", False, ["t"], host="gh200")
    b = P.Unit("bench", "m2", "m2", "kitti", "c", "dev", False, ["t"], host="4090")
    with pytest.raises(C.ConfigError):
        P.one_host_per_item([a, b], ["t"])
    b.host = "any"
    P.one_host_per_item([a, b], ["t"])


def test_run_label_per_host(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert "Isambard-AI GH200" in G.run_label("gh200", "abcdef0123", None, False) and "Slurm job(s) 123" in G.run_label("gh200", "abcdef0123", None, False)
    assert "one RTX 4090" in G.run_label("4090", "abcdef0123", None, False) and "Slurm" not in G.run_label("4090", "abcdef0123", None, False)
    assert "SMOKE" in G.run_label("any", "abcdef0123", "4090", True)


def test_data_root_local_and_repo_prefix(monkeypatch):
    cfg = {"host_paths": {"DATA_ROOT": "/default"}}
    monkeypatch.delenv("DATA_ROOT", raising=False)
    assert C.data_root(cfg) == Path("/default") and C.resolve_path(cfg, "a/b") == Path("/default/a/b")
    monkeypatch.setenv("DATA_ROOT", "/cli")
    assert C.resolve_path(cfg, "a/b") == Path("/cli/a/b")
    assert C.resolve_path(cfg, "repo:provenance/x.json") == REPO / "provenance/x.json"
    assert CFG["host_paths"]["DATA_ROOT"] == "/mnt/DATA/DegradedDepth" and "DATA_ROOT_gh200" not in CFG["host_paths"]


GH = CFG["host_paths"]["gh200"]


def test_gh200_map_complete_vs_config_path_keys():
    keys = C.path_keys(CFG)
    lo = set(CFG["host_paths"]["gh200_local_only"])
    assert keys - set(GH) - lo == set(), "a path key neither mapped nor local-only would silently fall back to a local path on the GH200"
    assert set(GH) <= keys and lo <= keys
    assert all(v and v.startswith("/projects/b5dh/") for v in GH.values())                  # every value comes from layout_table.tsv, none null/guessed
    assert lo == {"claude_outputs/scratch/paper_repro/latency_ladder/session_2", "qualitative/12b_ramen_figures", "efficiency_primary.txt", "claude_outputs/reports/P1_READ_20261012/P1_TABLE.md", "checkpoints_release/ramen.pt", "checkpoints_release/ramen_l.pt"}
    assert len(GH) == 20                                                                          # 19 at H; results_v3/bfix_s12 dropped in I (no item references the seed-45 read any more); +2 at I2: the TRAIN outputs dav2a_n46 / daacm_n46 (not in layout_table.tsv: pinned by sha256 in the yaml, Auditor Add.22)


def test_gh200_map_values_match_layout_table():
    tsv = Path("/mnt/DATA/DegradedDepth/claude_outputs/scratch/paper_repro/isambard_layout/layout_table.tsv")
    if not tsv.exists():
        pytest.skip("layout table not on this host")
    remote = {ln.split("\t")[1]: ln.split("\t")[2] for ln in tsv.read_text().splitlines()[1:]}
    for k, v in GH.items():
        if k.startswith("results_v3/"):
            assert v.startswith("/projects/b5dh/RAMEN_v3_results/")
        elif k.startswith("claude_outputs/scratch/paper_repro/train_pf/"):                       # commit I2: TRAIN outputs (AUD-FTONLY-TRAIN-20261015), absent from the 10-14 layout table
            assert v == "/projects/b5dh/RAMEN_train_pf_20261015/runs/" + k.split("/runs/")[1]
        else:
            assert remote[k] == v, k


def test_gh200_resolution_through_map_and_refusal(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.setenv("RAMEN_HOST", "gh200")
    k = "checkpoints_v3/eval_20261009/RAMEN_train_dn01_v3_bfix_C3_B/runs/dn01_v3_bfix_C3_B/step12000.pt"
    assert C.resolve_path(CFG, k) == Path(GH[k]) and C.resolve_path(CFG, "qualitative/12b_ramen_figures") == Path("/mnt/DATA/DegradedDepth/qualitative/12b_ramen_figures")
    with pytest.raises(C.UnmappedPath):
        C.resolve_path(CFG, "some/unmapped/file.pt")
    cfg = copy.deepcopy(CFG)
    cfg["host_paths"]["gh200"][k] = None
    with pytest.raises(C.UnmappedPath):
        C.resolve_path(cfg, k)
    assert any("absolute" in e for e in C.validate(cfg)) is False                       # null is allowed in the map (refused at use), not silently local
    cfg2 = copy.deepcopy(CFG)
    del cfg2["host_paths"]["gh200"][k]
    assert any("neither mapped" in e for e in C.validate(cfg2))
    cfg2["models"]["earlier_s45"]["ckpt"]["path"] = "somewhere/else.pt"
    with pytest.raises(C.UnmappedPath):
        P.plan(cfg2, ["tab:p1"])                                                         # a gh200 unit with an unmapped path key is refused at plan time
    monkeypatch.setenv("DATA_ROOT", "/cli")
    assert C.resolve_path(CFG, k) == Path("/cli") / k                                   # --data-root / DATA_ROOT keep working (local style)
    monkeypatch.setenv("RAMEN_HOST", "4090")
    monkeypatch.delenv("DATA_ROOT")
    assert C.resolve_path(CFG, k) == Path("/mnt/DATA/DegradedDepth") / k


def test_every_item_plans_through_the_map_on_gh200(monkeypatch, tmp_path):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.setenv("RAMEN_EFFICIENCY_PRIMARY", str(tmp_path / "efficiency_primary.txt"))      # absent: independent of the efficiency_primary.txt at DATA_ROOT (tab:latency_eff skipped)
    monkeypatch.setenv("RAMEN_HOST", "gh200")
    units, items = P.plan(CFG)
    assert len(units) == 689                                                                      # 627 at Ifig10 -> 689 at I2 (+62: tab:pf_* items, see test_s2_pf_items_*); 309 at H: released-row units dropped, +24 derived kitti_minus_0047, +156 cascade-FT units (C1/C2), +240 noise-ladder units (tab:noise; 288 before commit I3 dropped the llb3kappa1 pair), +8 gate-smoke DEV units (tab:casc_gate, commit I'; the cascade split adds none)
    for u in units:
        if u.runner == "fidelity_init":
            assert u.extra["ckpt_path"].startswith("/projects/b5dh/")
    m = [u.spec for u in units if "ours:" in u.spec]
    assert m and all("ours:/projects/b5dh/" in x for x in m)


def test_bvi_root_default_from_map(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.delenv("BVI_ROOT", raising=False)
    monkeypatch.setenv("RAMEN_HOST", "gh200")
    assert C.resolve_path(CFG, CFG["host_paths"]["bvi_root"]) == Path("/projects/b5dh/data/train/bvi_rlv/BVI-Lowlight-videos")


def test_default_tag_backend_is_reference_for_every_item():
    assert CFG["tag_backend"] == "reference" and all(it.get("tag_backend", "reference") == "reference" for it in CFG["items"].values())


# ---------------------------------------------------------------- readings
def _rows(vals):
    return {"rows": [{"model": m, "dataset": d, "cell": "b0s0", "n_clips": 1, "metrics": mm} for (m, d), mm in vals.items()], "mean": []}


def _clean_vals(ramen_scale=1.0, d1=0.0, tag_scale=None):
    base = {"abs_rel_aligned": 0.10, "abs_rel_unaligned": 0.20, "tae_gtvalid": 1.0, "delta1_aligned": 0.90, "tof": 0.5}
    v = {}
    for d in ("kitti", "bonn", "scannetpp"):
        v[("metric_vda_s", d)] = dict(base)
        v[("earlier_s45", d)] = {**{k: x * ramen_scale for k, x in base.items() if k != "delta1_aligned"}, "delta1_aligned": 0.90 + d1, "tof": 0.9}
        v[("earlier_s45_tag", d)] = {**{k: x * (tag_scale or ramen_scale) for k, x in base.items() if k != "delta1_aligned"}, "delta1_aligned": 0.90 + d1, "tof": 0.1}
        v[("ft_n46", d)] = dict(base)
    return v


def test_clean_reading_preserved_boundary_and_tof_ignored():
    item = _clean_item()
    res = _rows(_clean_vals(ramen_scale=1.05, d1=-0.01))                       # exactly at the limits (tOF very different: not part of the sentence)
    R.apply_reading(item, res)
    sv = res["verdict"]["subjects"]
    assert sv["earlier_s45"]["preserved"] and sv["earlier_s45_tag"]["preserved"]
    assert sv["earlier_s45"]["sentence"] == "clean-frame accuracy is preserved (within 5% of Metric-VDA-S; development clips)"
    assert not any(sv["earlier_s45"]["improves"].values())
    v = {(r["model"], r["dataset"]): r["verdict"] for r in res["rows"]}
    assert v[("earlier_s45", "kitti")] == "within-tolerance" and v[("metric_vda_s", "kitti")] == "reference" and v[("ft_n46", "kitti")].startswith("comparator")


def test_clean_reading_differs_one_dataset_one_metric_and_tag_separately():
    item = _clean_item()
    vals = _clean_vals()
    vals[("earlier_s45", "bonn")]["tae_gtvalid"] = 1.0501                          # just over 1.05x on one dataset
    vals[("earlier_s45_tag", "scannetpp")]["delta1_aligned"] = 0.8899              # just under -0.01
    res = _rows(vals)
    R.apply_reading(item, res)
    sv = res["verdict"]["subjects"]
    assert not sv["earlier_s45"]["preserved"] and sv["earlier_s45"]["sentence"].startswith("differs on clean frames") and "bonn/tae_gtvalid" in sv["earlier_s45"]["sentence"]
    assert not sv["earlier_s45_tag"]["preserved"] and "scannetpp/delta1_aligned" in sv["earlier_s45_tag"]["failing"][0]
    assert any(r["verdict"].startswith("differs: tae_gtvalid") for r in res["rows"] if (r["model"], r["dataset"]) == ("earlier_s45", "bonn"))


def test_clean_reading_improves_needs_all_datasets_and_nan_fails():
    item = _clean_item()
    res = _rows(_clean_vals(ramen_scale=0.9, d1=0.02))
    R.apply_reading(item, res)
    assert all(res["verdict"]["subjects"]["earlier_s45"]["improves"].values())
    vals = _clean_vals(ramen_scale=0.9, d1=0.02)
    vals[("earlier_s45", "kitti")]["abs_rel_aligned"] = 0.0951                      # 4.9% better on one dataset only
    vals[("earlier_s45", "bonn")]["tae_gtvalid"] = float("nan")
    res = _rows(vals)
    R.apply_reading(item, res)
    s = res["verdict"]["subjects"]["earlier_s45"]
    assert not s["improves"]["abs_rel_aligned"] and not s["preserved"]


def test_eps_d_reading_every_dataset_and_disclosure(tmp_path):
    item = CFG["items"]["tab:static_diag"]
    ds = item["datasets"]
    vals = {}
    for d in ds:
        vals[("A", d)] = {"eps_d": 2.0}
        vals[("B", d)] = {"eps_d": 1.0}
        vals[("C", d)] = {"eps_d": 1.0 if d != ds[0] else 3.0}
    res = _rows(vals)
    R.apply_reading(item, res)
    pairs = {(p["a"], p["b"]): p["result"] for p in res["verdict"]["pairs"]}
    assert pairs[("A", "B")] == "A flickers more than B by the alignment-invariant diagnostic eps_d"
    assert pairs[("A", "C")] == "mixed" and pairs[("B", "C")] == "mixed"
    assert "chosen after the main results" in res["disclosure"] and "not pre-registered" in res["disclosure"]
    res.update({"metrics": ["eps_d"], "item": "tab:static_diag"})
    for r in res["rows"]:
        r["metrics"].setdefault("eps_d", float("nan"))
    files = A.write_item(res, tmp_path, "tab_static_diag")
    pc = next(f for f in files if f.name.endswith("_pairs.csv"))
    assert all("chosen after the main results" in row["disclosure"] for row in csv.DictReader(open(pc)))


def test_verdict_column_in_csv(tmp_path):
    item = _clean_item()
    res = _rows(_clean_vals())
    R.apply_reading(item, res)
    res.update({"metrics": item["metrics"], "item": "tab:clean"})
    A.write_item(res, tmp_path, "tab_clean")
    rows = list(csv.DictReader(open(tmp_path / "tab_clean.csv")))
    assert "verdict" in rows[0] and {r["verdict"] for r in rows} >= {"reference", "within-tolerance"}


def test_real_config_clean_models_and_labels():
    it = CFG["items"]["tab:clean"]
    assert it["models"] == ["earlier_s45", "earlier_s45_tag", "ft_n46"] and it["cells"] == ["b0s0"] and it["split"] == "dev"      # released Metric-VDA-S removed (Add.38a)
    assert "reading" not in it                                                                                               # descriptive: no verdict (Auditor AUD-FTONLY-S0 item 4)
    assert "development clips (seen during development)" in it["label"] and "seed 46" in it["label"] and "DESCRIPTIVE" in it["label"] and "no verdict" in it["label"]
    res = {"rows": [{"model": "earlier_s45", "dataset": "kitti", "cell": "b0s0", "n_clips": 1, "metrics": {m: 0.1 for m in it["metrics"]}, "macro_mismatch": []}], "mean": []}
    R.apply_reading(it, res)
    assert "verdict" not in res and "verdict" not in res["rows"][0]


# ---------------------------------------------------------------- BVI + clip list identity
def test_bvi_clip_list_pin_and_per_file_shas(tmp_path):
    assert CFG["datasets"]["bvi_rlv_test"]["clip_list_sha256"].startswith("8a093838")
    from repro.util import sha256_file
    assert sha256_file(P.REGISTRATION).startswith("8a093838")
    reg = {"sequences": {"S01__10": {"included": True, "frames": {"low": ["a.png"], "normal": ["b.png"]}}, "S02__10": {"included": False, "frames": {"low": ["c.png"], "normal": ["d.png"]}}}}
    rp = tmp_path / "reg.json"
    rp.write_text(json.dumps(reg))
    for rel in ("S01/low_light_10/a.png", "S01/normal_light_10/b.png"):
        (tmp_path / "root" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "root" / rel).write_bytes(b"x")
    out = M.bvi_file_shas(tmp_path / "root", rp)
    assert sorted(out) == ["S01/low_light_10/a.png", "S01/normal_light_10/b.png"] and all(len(v) == 64 for v in out.values())


# ---------------------------------------------------------------- fig:flicker_curve definition
def test_flicker_curve_definition_matches_manual_and_scorer_decomposition():
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(REPO / "repro" / "qualitative"))
    import render_flicker_curve as F
    rng = np.random.default_rng(0)
    N, H, W = 64, 6, 8
    depth = 5 + rng.random((N, H, W))
    gt = np.full((N, H, W), 4.0)
    gt[:, :2, :] = 0.0                                                          # invalid pixels (outside min_depth..max_depth)
    c = F.c_series(depth, gt, 80.0, 0.001)
    assert c.shape == (56,)
    valid = gt[0] > 0.001
    dbar = depth[7:64].mean(0)
    for t in (8, 30, 63):
        assert c[t - 8] == pytest.approx(float((np.abs(depth[t] - depth[t - 1]) / dbar)[valid].mean()))
    # the scorer's static flicker is the mean of c_t over t = 8..63 (bench_joint.flicker_static, t0 = 8)
    assert float(c.mean()) == pytest.approx(float(np.mean([(np.abs(depth[t] - depth[t - 1]) / dbar)[valid].mean() for t in range(8, 64)])))
    cur = F.curve_over_clips([c, c * 2])
    assert cur["t"][0] == 8 and cur["t"][-1] == 63 and cur["n_clips"] == 2 and cur["mean"][0] == pytest.approx(1.5 * c[0])
    p = CFG["items"]["fig:flicker_curve"]
    assert p["models"] == CFG["items"]["fig:qual"]["models"] and p["params"]["t_first"] == 8 and p["params"]["t_last"] == 63 and p["params"]["split"] == "dev"


# ---------------------------------------------------------------- Addendum 30: GH200 for all numbers, 4090 timing only, MacBook slot
def test_addendum30_hosts(monkeypatch):
    monkeypatch.delenv("RAMEN_EFFICIENCY_PRIMARY", raising=False)
    units, items = P.plan(CFG)
    host_of = {i: {u.host for u in units if i in u.items} for i in items}
    for i in items:
        k = CFG["items"][i]["kind"]
        if k == "latency":
            assert host_of[i] == {"4090"}
        elif k != "qualitative":
            assert host_of[i] == {"gh200"}, i
        else:
            assert P.unit_host(CFG["items"][i], False, "reference") == "gh200"
    assert {u.host for u in units if u.kind != "latency"} == {"gh200"}
    assert CFG["items"]["tab:clean"]["previous_host"] == "4090" and CFG["items"]["tab:p1"]["previous_host"] == "4090"


def test_plan_output_has_hosts_nhr_delta_and_smoke_statement():
    units, items = P.plan(CFG)
    txt = P.format_plan(units, items, CFG, Path("/o"))
    assert "was" in txt and "Isambard NHR estimate" in txt and "Addendum 30 delta" in txt and "NHR" in txt
    su, si = P.plan(CFG, smoke=True)
    assert "plumbing only" in P.format_plan(su, si, CFG, Path("/o"), smoke=True)


def test_mac_slot_has_no_executor(tmp_path, monkeypatch):
    for i in ("tab:latency", "tab:latency_eff"):
        assert CFG["items"][i]["host_slots"]["mac"] == {"label": "MacBook (Apple silicon)", "value": "not measured", "authorised": False}
    bad = copy.deepcopy(CFG)
    bad["items"]["tab:latency"]["host_slots"]["mac"]["authorised"] = True
    assert any("host_slots" in e for e in C.validate(bad))
    _ladder_session(tmp_path / "s")
    got = A.ladder_rows(tmp_path / "s", "S0", ["ramen_raw"])
    res = A.aggregate_item("tab:latency", {"metrics": ["per_frame_ms_p50"], "aggregation": {}}, got)
    res["host_slots"] = CFG["items"]["tab:latency"]["host_slots"]
    for r in res["rows"]:
        r["host_slots"] = {"mac": "not measured"}
    A.write_item(res, tmp_path, "tab_latency")
    row = next(csv.DictReader(open(tmp_path / "tab_latency.csv")))
    assert row["MacBook (Apple silicon)"] == "not measured"
    # host=mac: no executor, nothing is launched, the unit is skipped as 'not measured'
    import subprocess
    mu = P.Unit("latency", "m", "ladder:S0", "-", "-", "-", False, ["t"], host="mac", harness={"stack": "S0", "rows": [], "warmup": 5, "iters": 30, "script": "x"})
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("launched")))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("launched")))
    assert X.run_unit(mu, tmp_path, False, False, None, tmp_path / "logs") == "not_measured"


def test_p1_cross_host_verify_and_candidate_figure_labels():
    assert CFG["items"]["tab:p1"]["verify_class"] == "cross_host" and CFG["items"]["tab:p1"]["host"] == "gh200"
    for f in ("fig:teaser", "fig:qual"):
        assert CFG["items"][f]["label"].startswith("candidate (unreleased)") and CFG["items"][f]["reference"] == "qualitative/12b_ramen_figures"
    bad = copy.deepcopy(CFG)
    bad["items"]["tab:p1"]["verify_class"] = "same_host"
    assert any("verify_class" in e for e in C.validate(bad))


# ---------------------------------------------------------------- F': tab:p1 by-hand reference, tab:rep label
P1T = Path("/mnt/DATA/DegradedDepth/claude_outputs/reports/P1_READ_20261012/P1_TABLE.md")


def test_p1_by_hand_reference_declared_and_rep_label():
    bh = CFG["items"]["tab:p1"]["by_hand_reference"]
    assert bh["sha256"] == "a46e02c3a3aee51184a2335edfbc3b44e6e6ffb382b5073f2f889a50fd381666" and bh["compare"] == "p1_table" and CFG["items"]["tab:p1"]["verify_class"] == "cross_host"
    lab = CFG["items"]["tab:rep"]["label"]
    assert "4090 table values" not in lab and "regenerated table values shown only if all 30 per-seed rule outcomes are identical to the released ones (Auditor R2)" in lab


@pytest.mark.skipif(not P1T.exists(), reason="released P1 read not on this host")
def test_p1_cross_host_comparison_table(tmp_path):
    from repro.util import sha256_file
    assert sha256_file(P1T) == CFG["items"]["tab:p1"]["by_hand_reference"]["sha256"]
    tab = V.parse_p1_table(P1T.read_text())
    assert len(tab) == 2 * 3 * 3 and tab[("raw", "kitti", "full")]["AbsRel_al"] == (0.0924, 4)
    rows = []
    for (var, ds, arm), cols in tab.items():
        mid = ("earlier_s45_tag" if var == "tag" else "earlier_s45") + "@" + arm
        for cell in ("llvar", "st_llvar"):
            m = {name: cols[c][0] for c, (name, cl) in V.P1_COLS.items() if cl == cell}
            rows.append({"model": mid, "dataset": ds, "cell": cell, "metrics": m})
    res = {"rows": rows}
    rep = V.compare_p1(res, tab)
    assert rep["status"] == "PASS" and rep["n_compared"] > 100
    next(r for r in rows if r["model"] == "earlier_s45@full" and r["dataset"] == "kitti" and r["cell"] == "llvar")["metrics"]["abs_rel_aligned"] *= 1.02      # 2% > cross-host 0.5%
    assert V.compare_p1(res, tab)["status"] == "FAIL"


# ---------------------------------------------------------------- commit I: FT-only comparison items (Addendum 38/38a)
RELEASED = {"metric_vda_s", "metric_vda_s_tag", "metric_vda_l", "dav2_s", "daac"}
MAIN = ("tab:dynamic", "tab:dynamic_night", "tab:static", "tab:static_diag")
SEEDS_ALL = ["ramen_n46", "ramen_n46_tag", "ramen_n47", "ramen_n47_tag", "ramen_n48", "ramen_n48_tag", "ft_n46", "ft_n46_tag", "ft_n47", "ft_n47_tag", "ft_n48", "ft_n48_tag"]


def test_comparison_items_have_no_released_model_and_no_seed45():
    for iid in MAIN + ("tab:clean", "tab:casc", "tab:casc_tag", "tab:casc_night", "tab:casc_night_tag"):
        ms = set(CFG["items"][iid]["models"])
        assert not (ms & RELEASED), iid
        assert not (ms & {"earlier_s45", "earlier_s45_tag"}) or iid == "tab:clean", iid
        for m in ms:
            assert "released" not in CFG["models"][m]["label"].lower() or m in ("metric_vda_s",), (iid, m)
    for iid in MAIN:
        assert CFG["items"][iid]["models"] == SEEDS_ALL


def test_main_items_share_the_tab_rep_units_and_add_none():
    rep, _ = P.plan(CFG, ["tab:rep", "tab:rep_night"])
    new, _ = P.plan(CFG, list(MAIN))
    key = lambda u: (u.model_id, u.dataset, u.cell, u.split, u.kind, u.temporal)                       # noqa: E731
    base = {key(u) for u in rep}
    assert {key(u) for u in new if not u.derive_from} <= base                                         # no new scoring unit
    assert {key(u) for u in new if u.derive_from and u.dataset == "kitti_minus_0047"}                  # kitti_minus_0047 derived from the kitti unit (no GPU)
    assert all(u.host == "gh200" for u in new)
    nf = next(u for u in new if u.model_id == "ft_n47" and u.dataset == "kitti" and u.cell == "llvar")
    assert "tab:dynamic" in nf.items


def test_seed_mean_groups_are_matched_unweighted_and_in_every_main_item():
    for iid in MAIN:
        sm = {g["name"]: g["members"] for g in CFG["items"][iid]["seed_mean"]}
        assert sm == {"RAMEN (3-seed mean)": ["ramen_n46", "ramen_n47", "ramen_n48"], "RAMEN + TAG (3-seed mean)": ["ramen_n46_tag", "ramen_n47_tag", "ramen_n48_tag"],
                      "Metric-VDA-S FT (3-seed mean)": ["ft_n46", "ft_n47", "ft_n48"], "Metric-VDA-S FT + TAG (3-seed mean)": ["ft_n46_tag", "ft_n47_tag", "ft_n48_tag"]}, iid
    item = CFG["items"]["tab:static"]
    arts = {}
    vals = {"ramen_n46": 1.0, "ramen_n47": 2.0, "ramen_n48": 6.0}                                       # unweighted: 3.0 (not weighted by clips)
    for m, v in vals.items():
        arts[(m, "kitti", "st_llvar")] = {"split": "all", "video_macro": {"flicker": v}, "per_video": {"c1": {"flicker": v}}, "num_clips": 1}
    r = A.aggregate_item("tab:static", {**item, "metrics": ["flicker"], "aggregation": {}}, arts)
    m = next(x for x in r["rows"] if x["model"] == "RAMEN (3-seed mean)")
    assert m["metrics"]["flicker"] == 3.0 and m["n_seeds"] == 3 and m["seed_range"]["flicker"] == [1.0, 6.0]


def test_static_diag_reading_redeclared_post_hoc_for_ft_rows():
    rd = CFG["items"]["tab:static_diag"]["reading"]
    assert rd["kind"] == "eps_d_every_dataset" and rd["metric"] == "eps_d"
    assert "re-declared" in rd["disclosure"] and "post hoc" in rd["disclosure"] and "chosen after the main results" in rd["disclosure"] and "not pre-registered" in rd["disclosure"]
    assert CFG["items"]["tab:static_diag"]["datasets"] == CFG["items"]["tab:static"]["datasets"] and CFG["items"]["tab:static_diag"]["host"] == "gh200"


def test_tab_clean_descriptive_has_no_verdict_column(tmp_path):
    it = CFG["items"]["tab:clean"]
    res = _rows({(m, "kitti"): {k: 0.1 for k in it["metrics"]} for m in it["models"]})
    res.update({"metrics": it["metrics"], "item": "tab:clean"})
    A.write_item(res, tmp_path, "tab_clean")
    assert "verdict" not in next(csv.DictReader(open(tmp_path / "tab_clean.csv")))


CASC_BENCH = ["tab:casc", "tab:casc_static", "tab:casc_tag", "tab:casc_tag_static"]


def test_cascade_ft_items_spec_pins_and_tag_split():
    for k, pre in (("fdvd", "fastdvdnet_blind"), ("pock", "pocketdvdnet")):
        for s, sha in ((46, "18c9d8ca"), (47, "ec38c317"), (48, "d769e513")):
            for tag in (False, True):
                m = CFG["models"][f"casc_{k}_ft_n{s}" + ("_tag" if tag else "")]
                assert m["spec"] == pre + "+metric_vda_s_ft:{ckpt}" and m["tag"] is tag and m["ckpt"]["sha256"].startswith(sha)
                assert m["ckpt"] == CFG["models"][f"ft_n{s}"]["ckpt"]                                    # the SAME FT checkpoint pins as ft_n46/47/48
        assert next(iter(CFG["models"][f"casc_{k}_ft_n46"]["weights"].values())).startswith("8118974a" if k == "fdvd" else "78320cda")
    assert CFG["items"]["tab:casc"]["models"] == [f"casc_{k}_ft_n{s}" for k in ("fdvd", "pock") for s in (46, 47, 48)]
    assert CFG["items"]["tab:casc_tag"]["models"] == [f"casc_{k}_ft_n{s}_tag" for k in ("fdvd", "pock") for s in (46, 47, 48)]
    assert CFG["items"]["tab:casc"]["cells"] == ["llvar"] and CFG["items"]["tab:casc_night"]["kind"] == "night" and CFG["items"]["tab:casc_night"]["split"] == "test"
    assert CFG["items"]["tab:casc"]["datasets"] == CFG["items"]["tab:dynamic"]["datasets"]
    units, _ = P.plan(CFG, CASC_BENCH)
    nu = [u for u in units if not u.derive_from]
    assert len(nu) == 6 * 2 * 5 * 2 and sum(1 for u in units if u.derive_from) == 6 * 2 * 2          # 6 models(raw,tag) x2 variants x5 ds x2 cells; kitti_minus_0047 derived
    ex = [u for u in nu if u.grid]                                                                   # raw+TAG share ONE scoring pass per model/dataset/cell
    assert len(ex) == 3 * 2 * 5 * 2 and all(u.variant == "none" and u.grid == ["none", "agf0.2t0.06"] for u in ex)
    cmd = ex[0].command(Path("/o"), py="python")
    assert cmd[cmd.index("--model") + 1].startswith(("fastdvdnet_blind+metric_vda_s_ft:", "pocketdvdnet+metric_vda_s_ft:")) and "+agf" not in cmd[cmd.index("--model") + 1]


def test_cascade_units_need_no_new_gh200_path_keys_and_plan_through_map(monkeypatch):
    monkeypatch.setenv("RAMEN_HOST", "gh200")
    monkeypatch.delenv("DATA_ROOT", raising=False)
    units, _ = P.plan(CFG, ["tab:casc", "tab:casc_tag", "tab:casc_night", "tab:casc_night_tag"])
    assert units and all(str(u.spec).replace("+agf0.2t0.06", "").split("metric_vda_s_ft:")[1].startswith("/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n4") for u in units if "metric_vda_s_ft:" in u.spec)
    assert {u.host for u in units} == {"gh200"}


def test_gh200_map_has_no_unreferenced_key_after_released_rows_removed():
    assert "results_v3/bfix_s12" not in CFG["host_paths"]["gh200"] and "results_v3/c3rep_s12" in CFG["host_paths"]["gh200"]


def test_noise_ladder_item_is_dev_only_with_the_released_ladder_cells():
    it = CFG["items"]["tab:noise"]
    assert it["split"] == "dev" and it["datasets"] == ["kitti", "bonn"] and it["reference"] is None
    assert it["cells"] == ["llb3x1", "llb3x2", "llb3x4", "llb3x8", "llb3x16", "st_llb3x1", "st_llb3x2", "st_llb3x4", "st_llb3x8", "st_llb3x16"]   # I3: llb3kappa1 / st_llb3kappa1 dropped
    assert it["models"] == SEEDS_ALL and "seed_mean" not in it                                                  # G is computed per seed outside the generator, never from seed_mean rows
    units, _ = P.plan(CFG, ["tab:noise"])
    assert len(units) == 240 and sum(1 for u in units if u.grid) == 120 and {u.split for u in units} == {"dev"} and not P.needs_test(units)
    cmd = next(u for u in units if u.grid).command(Path("/o"), py="python")
    assert "--allow-test" not in cmd and "--split" in cmd and cmd[cmd.index("--split") + 1] == "dev"


# ---------------------------------------------------------------- commit I': S0 gate smoke block + llvar/st_llvar split of the cascade items
def test_cascade_split_llvar_temporal_st_llvar_not_temporal():
    for raw, st, tag in (("tab:casc", "tab:casc_static", False), ("tab:casc_tag", "tab:casc_tag_static", True)):
        a, b = CFG["items"][raw], CFG["items"][st]
        assert a["cells"] == ["llvar"] and a["temporal"] is True and b["cells"] == ["st_llvar"] and b["temporal"] is False
        for k in ("models", "datasets", "split", "seed_mean", "kind", "host"):
            assert a.get(k) == b.get(k), (raw, k)
        assert b["metrics"] == ["flicker", "abs_rel_unaligned", "abs_rel_aligned", "delta1_aligned", "abs_rel_static", "eps_d"]
    # the static item is protocol-identical to tab:static (R1 precedent): non-temporal static cell
    assert CFG["items"]["tab:static"]["cells"] == ["st_llvar"]


def test_cascade_split_unit_counts_unchanged_and_static_units_have_no_temporal_flag():
    units, _ = P.plan(CFG, CASC_BENCH)
    nu = [u for u in units if not u.derive_from]
    assert len(nu) == 120 and sum(1 for u in units if u.derive_from) == 24 and sum(1 for u in nu if u.grid) == 60   # 60 executing + 60 TAG members + 24 derived = 144
    assert all(u.temporal == (u.cell == "llvar") for u in units)
    for u in nu:
        cmd = u.command(Path("/o"), py="python")
        assert ("--temporal" in cmd) == (u.cell == "llvar"), (u.model_id, u.dataset, u.cell)
    raw, _ = P.plan(CFG, ["tab:casc", "tab:casc_static"])
    assert len([u for u in raw if not u.derive_from]) == 60 and sum(1 for u in raw if u.derive_from) == 12
    night, _ = P.plan(CFG, ["tab:casc_night", "tab:casc_night_tag"])
    assert len(night) == 12
    assert len(P.plan(CFG, ["tab:casc_night"])[0]) == 6


CASC_SMOKE_MODELS = ["casc_fdvd_ft_n46", "casc_fdvd_ft_n46_tag", "casc_pock_ft_n46", "casc_pock_ft_n46_tag"]
NOISE_SMOKE_MODELS = ["ramen_n46", "ramen_n46_tag", "ft_n46", "ft_n46_tag"]
PF_SMOKE_MODELS = ["dav2_ft_n46", "dav2_ft_n46_tag", "daac_ft_n46", "daac_ft_n46_tag"]            # commit I2


def test_smoke_block_is_the_union_of_cascade_gate_and_ladder_gate_selections():
    sm = C.smoke_cfg(CFG)
    assert sm["models"] == CASC_SMOKE_MODELS + NOISE_SMOKE_MODELS + PF_SMOKE_MODELS                                        # I3: union of the S0 cascade gate models and the S3 ladder gate models
    assert sm["datasets"] == ["kitti"] and sm["cells"] == ["llvar", "st_llvar", "b0s0", "llb3x1", "llb3x16", "st_llb3x1"]
    assert {k: v for k, v in sm.items() if k not in ("models", "datasets", "cells")} == {k: v for k, v in C.DEFAULT_SMOKE.items() if k not in ("models", "datasets", "cells")}
    assert set(CFG["smoke"]) == {"models", "datasets", "cells"} and all(m in CFG["models"] for m in sm["models"])
    assert set(C.DEFAULT_SMOKE["cells"]) <= set(sm["cells"])


def test_cascade_gate_item_plans_8_dev_files_in_smoke_and_is_not_a_paper_item():
    it = CFG["items"]["tab:casc_gate"]
    assert it["split"] == "dev" and it["datasets"] == ["kitti"] and it["cells"] == ["llvar", "st_llvar"] and it["models"] == CASC_SMOKE_MODELS
    units, _ = P.plan(CFG, ["tab:casc_gate"], smoke=True)
    assert len(units) == 8 and not P.needs_test(units) and not any(u.derive_from for u in units)
    assert {(u.model_id, u.cell) for u in units} == {(m, c) for m in CASC_SMOKE_MODELS for c in ("llvar", "st_llvar")}
    assert all("--limit-clips" in u.command(Path("/o"), py="python", smoke=True) for u in units)


# ---------------------------------------------------------------- commit I3: ladder gate smoke (union smoke block) + llb3kappa1 pair dropped
def test_ladder_gate_smoke_plans_12_dev_kitti_files_and_cascade_gate_stays_8():
    units, _ = P.plan(CFG, ["tab:noise"], smoke=True)
    assert len(units) == 12 and sum(1 for u in units if u.grid) == 6 and not P.needs_test(units) and not any(u.derive_from for u in units)
    assert {(u.model_id, u.cell) for u in units} == {(m, c) for m in NOISE_SMOKE_MODELS for c in ("llb3x1", "llb3x16", "st_llb3x1")}
    assert {u.dataset for u in units} == {"kitti"} and {u.split for u in units} == {"dev"}
    assert len(P.plan(CFG, ["tab:casc_gate"], smoke=True)[0]) == 8                                   # S0 gate unchanged by the union


def test_noise_ladder_has_no_kappa1_pair_and_every_remaining_cell_is_a_ladder_rung():
    units, _ = P.plan(CFG, ["tab:noise"])
    assert not any("kappa" in u.cell for u in units) and "kappa" not in " ".join(CFG["items"]["tab:noise"]["cells"])
    assert {u.cell for u in units} == set(CFG["items"]["tab:noise"]["cells"]) and len(CFG["items"]["tab:noise"]["cells"]) == 10
    assert len(units) == 12 * 2 * 10                                                                  # 12 models x 2 datasets x 10 cells


# ---------------------------------------------------------------- commit I2 (S2): DAv2-S FT / DA-AC FT n46, NEW items only
PF_ITEMS = ["tab:pf_gate", "tab:pf_static", "tab:pf_static_tag", "tab:pf_dynamic", "tab:pf_dynamic_tag", "tab:pf_night", "tab:pf_night_tag", "tab:pf_fnoise"]
PF_SHA = {"dav2_ft_n46": "e5cbf7ca420741aa39001ad7ed658a8666cd0892558901418f1f22776404b876", "daac_ft_n46": "665b59ff0ad824861394970fd792d5c1a54f712ad019abd3d3458857a4261fb0"}
PF_SPEC = {"dav2_ft_n46": "dav2_s_ft:", "daac_ft_n46": "daac_ft:"}


def test_s2_pf_models_pin_exactly_the_two_train_checkpoints(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.setenv("RAMEN_HOST", "gh200")
    for mid, sha in PF_SHA.items():
        for tag in (False, True):
            m = CFG["models"][mid + ("_tag" if tag else "")]
            assert m["ckpt"]["sha256"] == sha and m["tag"] is tag and m["spec"] == PF_SPEC[mid] + "{ckpt}"
            assert C.model_spec(CFG, mid + ("_tag" if tag else "")).startswith(PF_SPEC[mid] + "/projects/b5dh/RAMEN_train_pf_20261015/runs/dn01_c3rep_pf_")
            assert C.model_spec(CFG, mid + ("_tag" if tag else "")).endswith("_n46/step12000.pt" + ("+agf0.2t0.06" if tag else ""))
    assert C.validate(CFG) == []


def test_s2_pf_items_unit_counts_and_scope(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.setenv("RAMEN_HOST", "gh200")
    for iid in PF_ITEMS:
        it = CFG["items"][iid]
        assert it.get("reference") is None and "host" not in it and set(it["models"]) <= set(PF_SHA) | {m + "_tag" for m in PF_SHA}
    st, _ = P.plan(CFG, ["tab:pf_static", "tab:pf_static_tag"])
    assert len(st) == 24 and sum(1 for u in st if u.derive_from) == 4 and sum(1 for u in st if u.grid) == 10 and {u.cell for u in st} == {"st_llvar"} and not any(u.temporal for u in st)
    dy, _ = P.plan(CFG, ["tab:pf_dynamic", "tab:pf_dynamic_tag"])
    assert len(dy) == 24 and sum(1 for u in dy if u.derive_from) == 4 and sum(1 for u in dy if u.grid) == 10 and all(u.temporal for u in dy)
    ni, _ = P.plan(CFG, ["tab:pf_night", "tab:pf_night_tag", "tab:pf_fnoise"])
    assert len(ni) == 6 and {u.kind for u in ni} == {"night", "fnoise"} and sum(1 for u in ni if u.kind == "night") == 4
    assert {u.split for u in st + dy} == {"all"} and {u.dataset for u in st} == {"kitti", "kitti_minus_0047", "bonn", "scannetpp", "nyuv2", "sintel_rgbfix"}
    allpf, _ = P.plan(CFG, PF_ITEMS)
    assert len(allpf) == 8 + 24 + 24 + 6                                                              # 62 new units; no unit shared with any earlier item
    earlier, _ = P.plan(CFG, [i for i in CFG["items"] if i not in PF_ITEMS])
    assert len(earlier) == 627 and not ({u.key for u in allpf} & {u.key for u in earlier})


def test_s2_pf_gate_plans_8_dev_files_in_smoke():
    it = CFG["items"]["tab:pf_gate"]
    assert it["split"] == "dev" and it["datasets"] == ["kitti"] and it["cells"] == ["llvar", "st_llvar"] and it["models"] == PF_SMOKE_MODELS
    units, _ = P.plan(CFG, ["tab:pf_gate"], smoke=True)
    assert len(units) == 8 and sum(1 for u in units if u.grid) == 4 and not P.needs_test(units) and not any(u.derive_from for u in units)
    assert {(u.model_id, u.cell) for u in units} == {(m, c) for m in PF_SMOKE_MODELS for c in ("llvar", "st_llvar")}
    assert all("--limit-clips" in u.command(Path("/o"), py="python", smoke=True) for u in units)
    assert len(P.plan(CFG, ["tab:casc_gate"], smoke=True)[0]) == 8 and len(P.plan(CFG, ["tab:noise"], smoke=True)[0]) == 12      # earlier gates unchanged by the larger smoke block


def test_s2_pf_commands_use_the_perframe_spec_with_the_isambard_path(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.setenv("RAMEN_HOST", "gh200")
    units, _ = P.plan(CFG, ["tab:pf_static", "tab:pf_static_tag"])
    ex = [u for u in units if u.grid]
    cmd = ex[0].command(Path("/o"), py="python")
    assert cmd[cmd.index("--model") + 1].startswith(("dav2_s_ft:/projects/b5dh/RAMEN_train_pf_20261015/", "daac_ft:/projects/b5dh/RAMEN_train_pf_20261015/")) and "agf" not in cmd[cmd.index("--model") + 1]
    assert cmd[cmd.index("--grid") + 1:cmd.index("--device")] == ["none", "agf0.2t0.06"] and "--temporal" not in cmd


def test_s2_no_earlier_item_or_model_changed_vs_ifig10():
    import subprocess, yaml
    try:
        old = yaml.safe_load(subprocess.run(["git", "show", "d75ccc3f5d616d471abdde1b882e659fb923fba3:configs/paper_results.yaml"], cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True, check=True).stdout.replace("ramen_s45", "earlier_s45"))   # ramen_s45 -> earlier_s45 rename (release port)
    except Exception:
        pytest.skip("Ifig10 commit not available in this checkout")
    nolabel = lambda k, v: {a: b for a, b in v.items() if a != "label"} if k.startswith("earlier_s45") else v    # the earlier_s45 labels were reworded
    noclean = lambda k, v: {a: b for a, b in v.items() if a not in ("title", "label")} if k == "tab:clean" else v    # tab:clean title/label reworded to "earlier checkpoint (seed 45)"
    for sec in ("items", "models", "datasets", "tag", "vendored"):
        assert all((nolabel(k, CFG[sec][k]) == nolabel(k, v)) if sec == "models" else (noclean(k, CFG[sec][k]) == noclean(k, v)) if sec == "items" else CFG[sec][k] == v for k, v in old[sec].items()), sec
    assert set(CFG["items"]) - set(old["items"]) == set(PF_ITEMS) and set(CFG["models"]) - set(old["models"]) == set(PF_SHA) | {m + "_tag" for m in PF_SHA} | {"ramen", "ramen_l"}
    assert {k: v for k, v in CFG["host_paths"]["gh200"].items() if k not in old["host_paths"]["gh200"]}.keys().__len__() == 2
    assert {k: v for k, v in old["host_paths"]["gh200"].items()} == {k: CFG["host_paths"]["gh200"][k] for k in old["host_paths"]["gh200"]}
    assert CFG["smoke"]["models"][:len(old["smoke"]["models"])] == old["smoke"]["models"] and CFG["smoke"]["models"][len(old["smoke"]["models"]):] == PF_SMOKE_MODELS
    assert CFG["smoke"]["datasets"] == old["smoke"]["datasets"] and CFG["smoke"]["cells"] == old["smoke"]["cells"] and CFG["scorer_commit"] == old["scorer_commit"]

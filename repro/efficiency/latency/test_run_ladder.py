"""Stub-only tests for run_ladder.py (no GPU, no model, no nvidia-smi)."""
import contextlib, json, os, sys, types
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import pytest
import run_ladder as RL


def test_plan_rows_and_brackets():
    plan = RL.build_plan()
    for st in RL.LADDER_DEFAULT:
        p = [i for i in plan if i["stack"] == st]
        assert sorted(i["k"] for i in p if i["kind"] == "bracket") == list(range(6))            # own six-run bracket per stack
        rows = [i["id"] for i in p if i["kind"] == "row"]
        assert f"{st}__ramen_raw" in rows
        assert sum("tag_v3" in r for r in rows) == 2                                             # V3 x2 settings in EVERY stack
        assert sum("tag_numpy" in r for r in rows) == (2 if st == "S0" else 0)                   # numpy only in S0
        assert p[0]["kind"] == "bracket" and p[-1]["kind"] == "bracket"                          # before ... after
    assert len({i["id"] for i in plan}) == len(plan)
    assert plan[0]["stack"] == "S0" and plan[-1]["stack"] == "S4"


def test_brackets_interleaved_between_rows():
    p = [i["kind"][0] for i in RL.build_plan(["S0"])]
    assert "".join(p) == "brbrbrbrbrb"


def test_stack_flags_match_run_variant():
    import run_variant as RV
    for st in ("X", "S1", "S2", "S3"):
        assert RL.STACK_FLAGS[st] == RV.VARIANTS[st]
    assert RL.STACK_FLAGS["S4"] == RL.STACK_FLAGS["S3"] and RL.BUILD_FROM["S4"] == "S3"


def test_mvda_shared_toggles_only_t3_t4_t6_t9():
    assert RL.mvda_flags("S0") == [] and RL.mvda_flags("X") == []
    assert RL.mvda_flags("S1") == ["sdpa", "sdpa_motion"]
    assert RL.mvda_flags("S2") == ["sdpa", "sdpa_motion", "cudnn"]
    assert RL.mvda_flags("S3") == ["sdpa", "sdpa_motion", "cudnn", "c_enc", "c_head"]
    # every global (process-wide) toggle that RAMEN gets is also applied to Metric-VDA-S
    for st in RL.STACK_ORDER:
        assert {f for f in RL.STACK_FLAGS[st] if f in ("sdpa", "sdpa_motion", "cudnn")} == {f for f in RL.mvda_flags(st) if f in ("sdpa", "sdpa_motion", "cudnn")}


def test_dry_run_table_and_estimate(capsys):
    assert RL.main(["--dry-run"]) == 0
    o = capsys.readouterr().out
    assert "S4__ramen_tag_v3_agf0.5t0.12" in o and "estimated:" in o and "min" in o
    assert RL.estimate_minutes(RL.build_plan())["total_min"] > 10


def test_bracket_upper_median_and_flags():
    b = RL.bracket_stats([5.72, 5.73, 5.725, 5.731, 5.724, 5.726])
    assert b["reference_upper_median"] == sorted([5.72, 5.73, 5.725, 5.731, 5.724, 5.726])[3] and b["flag"] == "OK" and b["n"] == 6
    assert RL.bracket_stats([5, 6, 5, 6, 5, 6])["flag"] == "FLAG"


def fake_rows():
    def r(st, id, kind, ms, post=None, setting=None):
        return {"stack": st, "id": id, "kind": kind, "post": post, "setting": setting, "window1": {"ms_per_frame_p50": ms}, "window2": {"ms_per_new_frame_p50": ms * 1.45},
                "mem": {"peak_alloc_GiB": 3.0, "peak_reserved_GiB": 4.0}}
    rows = [r("S0", f"S0__mvda_bracket{k}", "bracket", 5.7) for k in range(6)]
    rows += [r("S0", "S0__ramen_raw", "row", 9.6), r("S0", "S0__ramen_tag_numpy_agf0.2t0.06", "row", 13.7, "numpy", "agf0.2t0.06")]
    rows += [r("X", f"X__mvda_bracket{k}", "bracket", 5.0) for k in range(6)] + [r("X", "X__ramen_raw", "row", 8.0)]
    return rows


def test_summary_ratio_is_within_stack_and_gates():
    s = RL.summarize(fake_rows(), {"ramen_raw": 9.55, "metric_vda_s": 5.7})
    assert abs(s["stacks"]["S0"]["rows"][0]["ratio_vs_same_stack_bracket"] - 9.6 / 5.7) < 1e-9
    assert abs(s["stacks"]["X"]["rows"][0]["ratio_vs_same_stack_bracket"] - 8.0 / 5.0) < 1e-9      # never against another stack's bracket
    g = s["gates"]
    assert g["S0_vs_released_ramen_raw"]["flag"] == "OK" and g["harness_check_ramen_raw"]["flag"] == "OK"
    bad = fake_rows(); bad[6]["window1"]["ms_per_frame_p50"] = 12.0
    assert RL.summarize(bad)["gates"]["S0_vs_released_ramen_raw"]["flag"].startswith("FLAG")
    assert RL.summarize(fake_rows(), {"ramen_raw": 8.0})["gates"]["harness_check_ramen_raw"]["flag"] == "FLAG"
    assert "S0" in RL.markdown_tables(s)


def test_other_heavy_ignores_compositor_and_own_pid():
    apps = ["157, /usr/bin/kwin_wayland, 157", "900, python, 8000", "901, python, 2000"]
    assert RL.other_heavy(apps, own_pid=900) == ["901, python, 2000"]
    assert RL.other_heavy(apps[:1]) == []


def test_lock_must_be_empty_then_taken(tmp_path):
    lk = tmp_path / "GPU_LOCK"; lk.write_text("")
    RL.lock_acquire(lk, "t"); assert "run_ladder" in lk.read_text()
    with pytest.raises(SystemExit): RL.lock_acquire(lk, "t")
    RL.lock_release(lk); assert lk.read_text() == ""
    RL.lock_acquire(lk, "t")


def test_code_record_has_commit_and_shas():
    c = RL.code_record()
    assert set(c["sha256"]) >= {"latency_copy.py", "patches.py", "tag_fast_patch.py"} and len(c["HEAD"]) == 40 and "min_commit_is_ancestor" in c


class FakeEvent:
    t = 0.0
    def __init__(self, enable_timing=True): pass
    def record(self): FakeEvent.t += 1.0; self.at = FakeEvent.t
    def elapsed_time(self, o): return o.at - self.at


def fake_torch():
    cuda = types.SimpleNamespace(Event=FakeEvent, synchronize=lambda: None)
    return types.SimpleNamespace(cuda=cuda, float16=0, autocast=lambda *a, **k: contextlib.nullcontext())


def test_time_stages_two_windows_events_after_last_op():
    calls = []
    stages = [("m", lambda y, s, b: (calls.append("m"), y)[1]), ("post", lambda y, s, b: (calls.append("p"), y)[1])]
    res, wall = RL.time_stages(fake_torch(), stages, ["w1", "w2"], 2, 3)
    assert len(res) == 2 and len(res[0]["total"]) == 3 and len(wall) == 2 and calls.count("p") == 2 * 5
    assert res[0]["total"][0] == 2.0                                    # ev0 .. ev(last): 2 stages -> 2 event ticks apart (end event recorded after last op)
    w1, w2, wu = RL.row_stats(res, wall, 22)
    assert w1["ms_per_frame_p50"] == 2.0 / 32 and w2["ms_per_new_frame_p50"] == 2.0 / 22


def test_only_new_files_under_latency():
    import subprocess
    st = subprocess.run(["git", "status", "--porcelain", "--", "repro/efficiency/latency"], cwd=RL.REPO_ROOT, capture_output=True, text=True).stdout
    assert all(l.startswith("??") for l in st.splitlines())          # nothing tracked was modified under latency/


# ---- L (Addendum 16): stacks C and AZ -------------------------------------------------------------------------------------------------------------------
def test_default_plan_is_unchanged_by_c_and_az():
    assert RL.LADDER_DEFAULT == ["S0", "X", "S1", "S2", "S3", "S4"] and RL.STACK_ORDER == RL.LADDER_DEFAULT + ["C", "AZ"]
    assert {i["stack"] for i in RL.build_plan()} == set(RL.LADDER_DEFAULT)           # the pre-L default plan (and session_2's) has no C / AZ item
    assert RL.build_plan() == RL.build_plan(RL.LADDER_DEFAULT)


def test_c_az_flags_equal_the_xprime_leg_toggles():
    from repro.efficiency import legs as LG
    for st in ("C", "AZ"):
        want = {f for t in LG.STACK_TOGGLES["ramen"][st] for f in LG.TOGGLE_FLAGS[t]}
        assert set(RL.STACK_FLAGS[st]) == want and len(RL.STACK_FLAGS[st]) == len(want)
    assert LG.STACK_TOGGLES["ramen"]["C"] == ["T1", "T2", "T3", "T4", "T5"]
    assert LG.STACK_TOGGLES["ramen"]["AZ"] == ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T9", "T10"]   # = X' legs.py 2ce36e8d
    assert set(RL.STACK_FLAGS["C"]) == set(RL.STACK_FLAGS["S1"]) | {"bnfold"}
    assert set(RL.STACK_FLAGS["AZ"]) == set(RL.STACK_FLAGS["S3"]) - {"dense_group"}                      # S3 without T8 (AZ has no dense_group)


def test_c_az_mvda_matching_stacks():
    assert RL.mvda_flags("C") == RL.mvda_flags("S1") == ["sdpa", "sdpa_motion"]                         # Metric-VDA-S matching stack of C is S1
    assert RL.mvda_flags("AZ") == RL.mvda_flags("S3") == ["sdpa", "sdpa_motion", "cudnn", "c_enc", "c_head"]
    from repro.efficiency import legs as LG
    assert LG.MVDA_MATCH == {"C": "S1", "AZ": "S3"}


def test_session_plan_s0_s1_c_az_rows_and_brackets():
    plan = RL.build_plan([s for s in RL.STACK_ORDER if s in ["S0", "S1", "C", "AZ"]])
    assert [s for i, s in enumerate(i["stack"] for i in plan) if i == 0 or plan[i - 1]["stack"] != s] == ["S0", "S1", "C", "AZ"]
    for st in ("S0", "S1", "C", "AZ"):
        p = [i for i in plan if i["stack"] == st]
        assert sorted(i["k"] for i in p if i["kind"] == "bracket") == list(range(6))              # own six-run Metric-VDA-S bracket per stack
        rows = [i["id"] for i in p if i["kind"] == "row"]
        assert f"{st}__ramen_raw" in rows and sum("tag_v3" in r for r in rows) == 2               # raw + TAG V3 x2 settings
        assert sum("tag_numpy" in r for r in rows) == (2 if st == "S0" else 0)
        assert p[0]["kind"] == "bracket" and p[-1]["kind"] == "bracket"


def test_s0_s1_plan_items_identical_with_and_without_c_az():
    full = RL.build_plan(RL.STACK_ORDER)
    for st in ("S0", "S1"):
        assert [i for i in full if i["stack"] == st] == [i for i in RL.build_plan([st])]


def test_plan_reader_accepts_c_and_az_in_the_same_order_as_run_ladder():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    from repro import plan as P
    assert tuple(P.LADDER_STACKS) == tuple(RL.STACK_ORDER) or set(P.LADDER_STACKS) == set(RL.STACK_ORDER)
    assert {"C", "AZ"} <= set(P.LADDER_STACKS)

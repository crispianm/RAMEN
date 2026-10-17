"""Tests for l1_readout.py (synthetic score files only): G0 paths (pass / retry at 2000 / STOP at 4000 / boundary), every verdict label, the S1 guard exemption, the Branch-B refusal, the CLI end to end,
and FORCED-FAIL MUTANTS of verdict(): each deliberately wrong implementation must be caught by the case table. Run: ~/.venvs/ramen/bin/python -m pytest outdoor_debug/parts/far_diag/l1_final/test_l1_readout.py -q"""
import copy, json, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l1_readout as R

def S_(xv, kitti=0.14, nyu=0.10, sc=0.12, tk=2.0, ts=1.75):
    g = lambda a, t=None: {"video_macro": {"abs_rel_unaligned": a, **({"tae_gtvalid": t} if t is not None else {})}}
    return {"primary": {"vkitti_far_80_655": {"legacy_subsample_median": xv, "pooled_median": xv, "per_clip_median_mean": xv, "ratio_of_medians": xv, "bin_absrel": 0.2}},
            "guards": {"kitti": g(kitti, tk), "nyuv2": g(nyu), "scannetpp": g(sc, ts)}}
def D_(x120, x80=0.74):
    b = lambda m: {"median_pred_over_gt": m, "absrel_unaligned_pooled": 0.3, "n_px": 1}
    return {"bins": {"[0,20)": b(1.2), "[20,50)": b(1.1), "[50,80)": b(0.94), "[80,120)": b(x80), "[120,200]": b(x120)}, "overall_absrel_0_200": 0.2}
def base():   # G0-passing baselines: B0 xV .85 (dev .15) / B0b .86 (.14); B0 xD .58 (.42) / B0b .60 (.40); 80-bin .74 both
    return {"B0": S_(0.85), "B0b": S_(0.86)}, {"B0": D_(0.58), "B0b": D_(0.60)}
CASES = {   # name -> (arm, S override, D override, S1?) -> expected label prefix
    "fix": ("S3", dict(xv=0.93), dict(x120=0.75, x80=0.80), "FIX-CANDIDATE"),
    "cause_guard_fail": ("S3", dict(xv=0.93, kitti=0.20), dict(x120=0.75, x80=0.80), "CAUSE-IMPLICATED"),
    "primary_only": ("S2", dict(xv=0.93), dict(x120=0.58), "PRIMARY-ONLY"),
    "ddad_only": ("S4", dict(xv=0.85), dict(x120=0.75, x80=0.80), "DDAD-ONLY"),
    "not": ("S5", dict(xv=0.85), dict(x120=0.58), "NOT"),
    "ddad_leg_blocked_by_80bin_growth": ("S6", dict(xv=0.93), dict(x120=0.75, x80=0.50), "PRIMARY-ONLY"),
    "vkitti_below_NF": ("S2", dict(xv=0.855), dict(x120=0.75, x80=0.80), "DDAD-ONLY"),
    "ddad_below_NF": ("S2", dict(xv=0.93), dict(x120=0.595, x80=0.80), "PRIMARY-ONLY"),
    "vkitti_moves_away": ("S4", dict(xv=0.75), dict(x120=0.58), "NOT"),
    "S1_nyu_guard_reported_not_gated": ("S1", dict(xv=0.93, nyu=0.20), dict(x120=0.75, x80=0.80), "FIX-CANDIDATE"),
    "S1_scannetpp_tae_reported_not_gated": ("S1", dict(xv=0.93, ts=2.5), dict(x120=0.75, x80=0.80), "FIX-CANDIDATE"),
    "S1_kitti_guard_still_gated": ("S1", dict(xv=0.93, kitti=0.25), dict(x120=0.75, x80=0.80), "CAUSE-IMPLICATED"),
    "nonS1_nyu_guard_gated": ("S3", dict(xv=0.93, nyu=0.20), dict(x120=0.75, x80=0.80), "CAUSE-IMPLICATED"),
    "tae_guard_gated": ("S5", dict(xv=0.93, tk=2.5), dict(x120=0.75, x80=0.80), "CAUSE-IMPLICATED"),
}
def build(case):
    arm, so, do, _ = CASES[case]; S, D = base()
    S[arm] = S_(**so); D[arm] = D_(**do); return S, D, arm
def table_mismatches(fn):
    bad = []
    for c, (arm, _, _, want) in CASES.items():
        S, D, arm = build(c); got = fn(S, D, arm)["label"]
        if not got.startswith(want): bad.append((c, got, want))
    return bad
def test_case_table_with_the_real_verdict():
    assert table_mismatches(R.verdict) == []
def test_s1_reported_guards_listed_but_not_gating():
    S, D, arm = build("S1_nyu_guard_reported_not_gated"); v = R.verdict(S, D, arm)
    assert v["guards_worse"] == [] and v["guards_reported_only"] and v["guards_reported_only"][0][0] == "nyuv2"

# ---- forced-fail mutants of verdict(): each MUST produce at least one wrong label on the case table ----
def _patched(**kw):
    def v(S, D, arm):
        src = dict(R=R); return _mut_verdict(S, D, arm, **kw)
    return v
def _mut_verdict(S, D, arm, flip_v=False, no_nf=False, ignore_80=False, flip_guard=False, gate_s1=False, use_80_bin=False, drop_tae=False, drop_ddad=False):
    xd = (lambda d, b="[120,200]": R.xd(d, "[80,120)")) if use_80_bin else R.xd
    nfp = 0.0 if no_nf else R.nf_rel(R.dev(R.xv(S["B0"])), R.dev(R.xv(S["B0b"])))
    nfd = 0.0 if no_nf else max(abs(R.dev(xd(D["B0"])) - R.dev(xd(D["B0b"]))), 0.01)
    chV = R.dev(R.xv(S["B0"])) - R.dev(R.xv(S[arm])); chV = -chV if flip_v else chV
    chD = R.dev(xd(D["B0"])) - R.dev(xd(D[arm])); ch80 = 0.0 if ignore_80 else R.dev(R.xd(D["B0"], "[80,120)")) - R.dev(R.xd(D[arm], "[80,120)"))
    legV, legD = chV > nfp, (chD > nfd and ch80 >= 0.0) and not drop_ddad
    worse = []
    for ds, k in R.GUARDS:
        if drop_tae and k == "tae_gtvalid": continue
        nf = R.nf_rel(R.gd(S["B0"], ds, k), R.gd(S["B0b"], ds, k)); w = R.gd(S[arm], ds, k) - R.gd(S["B0"], ds, k); w = -w if flip_guard else w
        if w > nf and not (arm == "S1" and ds in ("nyuv2", "scannetpp") and not gate_s1): worse.append(ds)
    label = ("FIX-CANDIDATE" if not worse else "CAUSE-IMPLICATED") if (legV and legD) else ("PRIMARY-ONLY" if legV else ("DDAD-ONLY" if legD else "NOT"))
    return dict(label=label)
@pytest.mark.parametrize("mut", [dict(flip_v=True), dict(no_nf=True), dict(ignore_80=True), dict(flip_guard=True), dict(gate_s1=True), dict(use_80_bin=True), dict(drop_tae=True), dict(drop_ddad=True)])
def test_forced_fail_mutants_are_caught(mut):
    assert table_mismatches(lambda S, D, arm: _mut_verdict(S, D, arm, **mut)) != [], f"mutant {mut} slipped through the case table"
def test_unmutated_reference_reimplementation_agrees():
    assert table_mismatches(lambda S, D, arm: _mut_verdict(S, D, arm)) == []

# ---- G0 ----
def test_g0_pass_and_boundaries():
    S, D = base(); assert R.g0(S, D)[0]
    S["B0"] = S_(0.93); D["B0b"] = D_(0.75); assert R.g0(S, D)[0], "boundary values 0.93 / 0.75 pass (<=)"
    S["B0"] = S_(0.9301); assert not R.g0(S, D)[0]
def test_g0_each_leg_and_each_run_matters():
    for who, S_o, D_o in (("B0", S_(0.95), None), ("B0b", S_(0.95), None), ("B0", None, D_(0.80)), ("B0b", None, D_(0.80))):
        S, D = base()
        if S_o: S[who] = S_o
        if D_o: D[who] = D_o
        assert not R.g0(S, D)[0], (who, "must fail G0")
def _write(root, S, D):
    for a in S:
        d = Path(root) / a; d.mkdir(parents=True, exist_ok=True); json.dump(S[a], open(d / "scores.json", "w")); json.dump(D[a], open(d / "ddad_scores.json", "w"))
def test_cli_paths(tmp_path, capsys):
    S, D, arm = build("fix"); _write(tmp_path, S, D)
    assert R.main(["--scores-dir", str(tmp_path), "--stage", "2000", "--ddad-gate", "A", "--arms", f"B0,B0b,{arm}"]) == 0
    assert "FIX-CANDIDATE" in capsys.readouterr().out
    S["B0"] = S_(0.95); _write(tmp_path, S, D)
    assert R.main(["--scores-dir", str(tmp_path), "--stage", "2000", "--ddad-gate", "A", "--arms", f"B0,B0b,{arm}"]) == 10
    out = capsys.readouterr().out; assert "AUTOMATIC RETRY" in out and "FIX-CANDIDATE" not in out and "NOT" not in out.split("G0: FAIL")[1]
    assert R.main(["--scores-dir", str(tmp_path), "--stage", "4000", "--ddad-gate", "A", "--arms", f"B0,B0b,{arm}"]) == 11
    out = capsys.readouterr().out; assert "STOP at 4000" in out and "FIX-CANDIDATE" not in out
def test_branch_b_and_garbage_are_refused(tmp_path, capsys):
    for g in ("B", "a", ""):
        assert R.main(["--scores-dir", str(tmp_path), "--stage", "2000", "--ddad-gate", g]) == 2
    with pytest.raises(SystemExit): R.main(["--scores-dir", str(tmp_path), "--stage", "2000"])   # flag is mandatory
def test_missing_baselines_not_evaluable(tmp_path, capsys):
    S, D, arm = build("fix"); _write(tmp_path, {arm: S[arm]}, {arm: D[arm]})
    assert R.main(["--scores-dir", str(tmp_path), "--stage", "2000", "--ddad-gate", "A", "--arms", f"B0,B0b,{arm}"]) == 3

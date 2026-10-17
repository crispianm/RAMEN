"""Tests for l1_readout.py (synthetic score files only): G0 paths (pass / retry at 2000 / STOP at 4000 / boundary), every verdict label, the S1 guard exemption, the Branch-B refusal, the CLI end to end,
and FORCED-FAIL MUTANTS of verdict(): each deliberately wrong implementation must be caught by the case table. Run: ~/.venvs/ramen/bin/python -m pytest outdoor_debug/parts/far_diag/l1_final/test_l1_readout.py -q"""
import copy, json, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l1_readout as R

def S_(xv, kitti=0.14, nyu=0.10, sc=0.12, tk=2.0, ts=1.75, step=2000, sha="a" * 64):
    g = lambda a, t=None: {"video_macro": {"abs_rel_unaligned": a, **({"tae_gtvalid": t} if t is not None else {})}}
    return {"meta": {"ckpt": {"step": step, "sha256": sha}}, "primary": {"vkitti_far_80_655": {"legacy_subsample_median": xv, "pooled_median": xv, "per_clip_median_mean": xv, "ratio_of_medians": xv, "bin_absrel": 0.2}},
            "guards": {"kitti": g(kitti, tk), "nyuv2": g(nyu), "scannetpp": g(sc, ts)}}
def D_(x120, x80=0.74, step=2000, sha="a" * 64):
    b = lambda m: {"median_pred_over_gt": m, "absrel_unaligned_pooled": 0.3, "n_px": 1}
    return {"meta": {"ckpt": {"step": step, "sha256": sha}}, "bins": {"[0,20)": b(1.2), "[20,50)": b(1.1), "[50,80)": b(0.94), "[80,120)": b(x80), "[120,200]": b(x120)}, "overall_absrel_0_200": 0.2}
def base():   # G0-passing baselines: B0 xV .85 (dev .15) / B0b .86 (.14); B0 xD .58 (.42) / B0b .60 (.40); 80-bin .74 both
    return {"B0": S_(0.85), "B0b": S_(0.86)}, {"B0": D_(0.58), "B0b": D_(0.60)}
CASES = {   # name -> (arm, S override, D override, S1?) -> expected label prefix
    "fix": ("S3", dict(xv=0.93), dict(x120=0.75, x80=0.80), "FIX-CANDIDATE"),
    "cause_guard_fail": ("S3", dict(xv=0.93, kitti=0.20), dict(x120=0.75, x80=0.80), "CAUSE-IMPLICATED"),
    "primary_only": ("S2", dict(xv=0.93), dict(x120=0.58), "PRIMARY-ONLY"),
    "ddad_only": ("S4", dict(xv=0.85), dict(x120=0.75, x80=0.80), "DDAD-ONLY"),
    "not": ("S5", dict(xv=0.85), dict(x120=0.58), "NOT"),
    "ddad_80bin_growth_within_NF_b": ("S3", dict(xv=0.93), dict(x120=0.75, x80=0.731), "FIX-CANDIDATE"),
    "ddad_80bin_growth_beyond_NF_b": ("S3", dict(xv=0.93), dict(x120=0.75, x80=0.729), "PRIMARY-ONLY"),
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
def _mut_verdict(S, D, arm, flip_v=False, no_nf=False, ignore_80=False, flip_guard=False, gate_s1=False, use_80_bin=False, drop_tae=False, drop_ddad=False, no_nfb=False):
    xd = (lambda d, b="[120,200]": R.xd(d, "[80,120)")) if use_80_bin else R.xd
    nfp = 0.0 if no_nf else R.nf_rel(R.dev(R.xv(S["B0"])), R.dev(R.xv(S["B0b"])))
    nfd = 0.0 if no_nf else max(abs(R.dev(xd(D["B0"])) - R.dev(xd(D["B0b"]))), 0.01)
    nfb = 0.0 if no_nfb else max(abs(R.dev(R.xd(D["B0"], "[80,120)")) - R.dev(R.xd(D["B0b"], "[80,120)"))), 0.01)
    chV = R.dev(R.xv(S["B0"])) - R.dev(R.xv(S[arm])); chV = -chV if flip_v else chV
    chD = R.dev(xd(D["B0"])) - R.dev(xd(D[arm])); ch80 = 0.0 if ignore_80 else R.dev(R.xd(D["B0"], "[80,120)")) - R.dev(R.xd(D[arm], "[80,120)"))
    legV, legD = chV > nfp, (chD > nfd and ch80 >= -nfb) and not drop_ddad
    worse = []
    for ds, k in R.GUARDS:
        if drop_tae and k == "tae_gtvalid": continue
        nf = R.nf_rel(R.gd(S["B0"], ds, k), R.gd(S["B0b"], ds, k)); w = R.gd(S[arm], ds, k) - R.gd(S["B0"], ds, k); w = -w if flip_guard else w
        if w > nf and not (arm == "S1" and ds in ("nyuv2", "scannetpp") and not gate_s1): worse.append(ds)
    label = ("FIX-CANDIDATE" if not worse else "CAUSE-IMPLICATED") if (legV and legD) else ("PRIMARY-ONLY" if legV else ("DDAD-ONLY" if legD else "NOT"))
    return dict(label=label)
@pytest.mark.parametrize("mut", [dict(flip_v=True), dict(no_nf=True), dict(ignore_80=True), dict(flip_guard=True), dict(gate_s1=True), dict(use_80_bin=True), dict(drop_tae=True), dict(drop_ddad=True), dict(no_nfb=True)])
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
    for k_ in S: S[k_]["meta"]["ckpt"]["step"] = 4000
    for k_ in D: D[k_]["meta"]["ckpt"]["step"] = 4000
    _write(tmp_path, S, D)
    assert R.main(["--scores-dir", str(tmp_path), "--stage", "4000", "--ddad-gate", "A", "--arms", f"B0,B0b,{arm}"]) == 11
    out = capsys.readouterr().out; assert "STOP at 4000" in out and "FIX-CANDIDATE" not in out
def test_branch_b_and_garbage_are_refused(tmp_path, capsys):
    for g in ("B", "a", ""):
        assert R.main(["--scores-dir", str(tmp_path), "--stage", "2000", "--ddad-gate", g]) == 2
    with pytest.raises(SystemExit): R.main(["--scores-dir", str(tmp_path), "--stage", "2000"])   # flag is mandatory
def test_missing_baselines_not_evaluable(tmp_path, capsys):
    S, D, arm = build("fix"); _write(tmp_path, {arm: S[arm]}, {arm: D[arm]})
    assert R.main(["--scores-dir", str(tmp_path), "--stage", "2000", "--ddad-gate", "A", "--arms", f"B0,B0b,{arm}"]) == 3


def test_nf_b_is_the_80bin_noise_floor_and_boundaries():
    S, D, arm = build("fix")                                   # baselines' [80,120) medians are equal -> NF_b = the 0.01 floor
    assert R.verdict(S, D, arm)["nfb"] == 0.01
    D["S3"] = D_(0.75, x80=0.731); assert R.verdict(S, D, "S3")["legD"] is True        # growth 0.009 <= 0.01
    D["S3"] = D_(0.75, x80=0.729); assert R.verdict(S, D, "S3")["legD"] is False       # growth 0.011 > 0.01
    # a noisier pair of baselines widens the tolerance: |B0 - B0b| on the [80,120) dev = 0.03 -> growth 0.025 is inside
    D["B0b"] = D_(0.60, x80=0.77); v = R.verdict(S, D, "S3")
    assert v["nfb"] == pytest.approx(0.03)
    D["S3"] = D_(0.75, x80=0.715); assert R.verdict(S, D, "S3")["legD"] is True        # growth 0.025 <= 0.03
    D["S3"] = D_(0.75, x80=0.705); assert R.verdict(S, D, "S3")["legD"] is False       # growth 0.035 > 0.03


# ================= second review (S5, S6, N1): integrity, missing arms, ties/floors/guard-better, SOURCE mutants =================
ARGS = lambda d, arms, stage=2000: ["--scores-dir", str(d), "--stage", str(stage), "--ddad-gate", "A", "--arms", arms]
def _full(tmp_path, arms=("S3",)):
    S, D, _ = build("fix"); 
    for a in arms:
        if a not in S: S[a] = S_(0.93); D[a] = D_(0.75, x80=0.80)
    _write(tmp_path, S, D); return S, D
def test_s5_step_must_equal_stage(tmp_path, capsys):
    S, D = _full(tmp_path); assert R.main(ARGS(tmp_path, "B0,B0b,S3")) == 0
    S["S3"] = S_(0.93, step=4000); S["S3"]["meta"]["ckpt"]["step"] = 4000; _write(tmp_path, S, D)
    assert R.main(ARGS(tmp_path, "B0,B0b,S3")) == 4 and "INTEGRITY FAIL" in capsys.readouterr().out
    S, D = _full(tmp_path); D["B0"] = D_(0.58, step=4000); _write(tmp_path, S, D)       # the DDAD file of a baseline scored at the wrong step
    assert R.main(ARGS(tmp_path, "B0,B0b,S3")) == 4
    S, D = _full(tmp_path); _write(tmp_path, S, D)
    assert R.main(ARGS(tmp_path, "B0,B0b,S3", stage=4000)) == 4        # k2000 scores read as stage 4000 (the former rc 11 mix-up)
def test_s5_ckpt_sha_must_match_between_the_two_score_files(tmp_path, capsys):
    S, D = _full(tmp_path); D["S3"] = D_(0.75, x80=0.80, sha="b" * 64); _write(tmp_path, S, D)
    assert R.main(ARGS(tmp_path, "B0,B0b,S3")) == 4 and "sha256 differs" in capsys.readouterr().out
def test_s5_missing_meta_is_an_integrity_failure(tmp_path):
    S, D = _full(tmp_path); del S["B0b"]["meta"]; _write(tmp_path, S, D)
    assert R.main(ARGS(tmp_path, "B0,B0b,S3")) == 4
def test_s5_missing_declared_arm_is_nonzero_not_silent(tmp_path, capsys):
    S, D = _full(tmp_path)
    assert R.main(ARGS(tmp_path, "B0,B0b,S3,S4,S5")) == 5
    out = capsys.readouterr().out; assert "missing S4" in out and "INCOMPLETE" in out and "S3" in out
    assert R.main(ARGS(tmp_path, "B0,B0b,S3")) == 0       # exactly the declared arms: complete
def test_s5_missing_arm_does_not_hide_a_g0_failure(tmp_path):
    S, D = _full(tmp_path); S["B0"] = S_(0.95); _write(tmp_path, S, D)
    assert R.main(ARGS(tmp_path, "B0,B0b,S3,S4")) == 10

# ---- S6/N1: ties, the 0.01 floor, guard-better, from the second review's independent hand-computed table (readout_cases.py) ----
def _S2(xv, k=.14, n=.10, s=.12, tk=2.0, ts=1.75): return S_(xv, kitti=k, nyu=n, sc=s, tk=tk, ts=ts)
def _D2(x120, x80): return D_(x120, x80=x80)
def _base2():   # devV .15/.13 -> NF_p .02 ; devD .45/.42 -> NF_d .03 ; dev80 .28/.25 -> NF_b .03 ; kitti abs_rel .140/.141 -> guard NF max(.001, .0014) = .0014
    return {"B0": _S2(.85), "B0b": _S2(.87, k=.141)}, {"B0": _D2(.55, .72), "B0b": _D2(.58, .75)}
C2 = [
 ("both legs, guards equal", "S3", dict(xv=.88), (.60, .72), "FIX-CANDIDATE"),
 ("leg V exactly = NF_p (strict >) -> no", "S3", dict(xv=.87), (.60, .72), "DDAD-ONLY"),
 ("leg V just above NF_p", "S3", dict(xv=.8701), (.60, .72), "FIX-CANDIDATE"),
 ("leg D exactly = NF_d (strict >) -> no", "S3", dict(xv=.88), (.58, .72), "PRIMARY-ONLY"),
 ("leg D just above NF_d", "S3", dict(xv=.88), (.5801, .72), "FIX-CANDIDATE"),
 ("[80,120) grows by exactly NF_b=.03 (<= -> inside)", "S3", dict(xv=.88), (.60, .69), "FIX-CANDIDATE"),
 ("[80,120) grows by .0299 (inside)", "S3", dict(xv=.88), (.60, .6901), "FIX-CANDIDATE"),
 ("[80,120) grows by .0301 (outside)", "S3", dict(xv=.88), (.60, .6899), "PRIMARY-ONLY"),
 ("[80,120) overshoots 1 (x80 1.2: dev .2 < .28)", "S3", dict(xv=.88), (.60, 1.2), "FIX-CANDIDATE"),
 ("x_V overshoots (1.10: dev .10)", "S3", dict(xv=1.10), (.60, .72), "FIX-CANDIDATE"),
 ("x_V overshoots too far (1.20: dev .20 > .15)", "S3", dict(xv=1.20), (.60, .72), "DDAD-ONLY"),
 ("kitti guard worse by exactly NF (.0014) -> within", "S3", dict(xv=.88, k=.1414), (.60, .72), "FIX-CANDIDATE"),
 ("kitti guard worse by .0015 -> CAUSE-IMPLICATED", "S3", dict(xv=.88, k=.1415), (.60, .72), "CAUSE-IMPLICATED"),
 ("kitti tae worse (2.0 -> 2.03, NF .02)", "S3", dict(xv=.88, tk=2.03), (.60, .72), "CAUSE-IMPLICATED"),
 ("scannetpp tae worse, non-S1 gated", "S2", dict(xv=.88, ts=1.80), (.60, .72), "CAUSE-IMPLICATED"),
 ("S1 nyu abs_rel worse: reported only", "S1", dict(xv=.88, n=.2), (.60, .72), "FIX-CANDIDATE"),
 ("S1 scannetpp abs_rel+tae worse: reported only", "S1", dict(xv=.88, s=.2, ts=2.5), (.60, .72), "FIX-CANDIDATE"),
 ("S1 kitti abs_rel worse: gated", "S1", dict(xv=.88, k=.2), (.60, .72), "CAUSE-IMPLICATED"),
 ("guard better (lower) is fine", "S4", dict(xv=.88, k=.10, n=.05), (.60, .72), "FIX-CANDIDATE"),
 ("guard much better (abs would flag): kitti .01", "S4", dict(xv=.88, k=.01, tk=0.5), (.60, .72), "FIX-CANDIDATE"),
 ("PRIMARY-ONLY with a guard failure stays PRIMARY-ONLY", "S5", dict(xv=.88, k=.2), (.55, .72), "PRIMARY-ONLY"),
 ("DDAD-ONLY", "S6", dict(xv=.85), (.60, .72), "DDAD-ONLY"),
 ("both worse -> NOT", "S6", dict(xv=.80), (.50, .60), "NOT"),
 ("D improves but 80-bin collapses -> leg D off, V off -> NOT", "S2", dict(xv=.85), (.70, .50), "NOT"),
]
def table2_mismatches(vfn):
    bad = []
    for name, arm, sk, (x120, x80), want in C2:
        S, D = _base2(); S[arm] = _S2(**sk); D[arm] = _D2(x120, x80)
        got = vfn(S, D, arm)["label"]
        if not got.startswith(want): bad.append((name, got[:30], want))
    # exact tie at the 0.01 floor: equal baseline 80-bins, the arm's [80,120) grows by exactly 0.01 (0.72 -> 0.71): inside (N1)
    S, D = _base2(); D["B0b"] = _D2(.58, .72); S["S3"] = _S2(.88); D["S3"] = _D2(.60, .71)
    if not vfn(S, D, "S3")["legD"]: bad.append(("floor tie 0.72->0.71", "legD False", "legD True"))
    # a floor-sized growth of 0.0101 is outside
    D["S3"] = _D2(.60, .7099)
    if vfn(S, D, "S3")["legD"]: bad.append(("floor +0.0001", "legD True", "legD False"))
    # guard NF floor (1% relative of B0): baselines equal on kitti abs_rel .14 -> NF = .0014; worse by .0013 inside, .0015 outside
    S, D = _base2(); S["B0b"] = _S2(.87, k=.14); S["S3"] = _S2(.88, k=.1413); D["S3"] = _D2(.60, .72)
    if not vfn(S, D, "S3")["label"].startswith("FIX-CANDIDATE"): bad.append(("guard floor inside", vfn(S, D, "S3")["label"][:20], "FIX"))
    S["S3"] = _S2(.88, k=.1415)
    if not vfn(S, D, "S3")["label"].startswith("CAUSE"): bad.append(("guard floor outside", vfn(S, D, "S3")["label"][:20], "CAUSE"))
    # NF_d floor 0.01 when the baselines' D devs are equal: change .0099 -> no leg, .0101 -> leg
    S, D = _base2(); D["B0b"] = _D2(.55, .75); S["S3"] = _S2(.88); D["S3"] = _D2(.5599, .72)
    if vfn(S, D, "S3")["legD"]: bad.append(("NF_d floor .0099", "legD True", "False"))
    D["S3"] = _D2(.5601, .72)
    if not vfn(S, D, "S3")["legD"]: bad.append(("NF_d floor .0101", "legD False", "True"))
    # NF_p 1% relative floor: equal baselines on V (dev .15), arm moves by .001 < the 1% floor (.0015): no leg
    S, D = _base2(); S["B0b"] = _S2(.85); S["S3"] = _S2(.851); D["S3"] = _D2(.60, .72)
    if vfn(S, D, "S3")["legV"]: bad.append(("NF_p 1% floor", "legV True", "False"))
    S["S3"] = _S2(.8516)
    if not vfn(S, D, "S3")["legV"]: bad.append(("NF_p 1% floor +", "legV False", "True"))
    # EXACT guard tie (binary-exact numbers): B0 2.0, B0b 2.5 -> NF .5; arm 2.5 -> worse by exactly .5: within (strict >)
    S, D = _base2(); S["B0"] = _S2(.85, k=2.0); S["B0b"] = _S2(.87, k=2.5); S["S3"] = _S2(.88, k=2.5); D["S3"] = _D2(.60, .72)
    if not vfn(S, D, "S3")["label"].startswith("FIX-CANDIDATE"): bad.append(("exact guard tie", vfn(S, D, "S3")["label"][:20], "FIX"))
    return bad
def test_independent_case_table_and_ties_with_the_real_verdict():
    assert table2_mismatches(R.verdict) == []

# SOURCE mutants of l1_readout.py: the mutated source is exec'd and run through BOTH case tables + the G0 boundaries; every mutant MUST produce a mismatch (a vacuous table cannot pass)
SRC = Path(R.__file__).read_text()
SRC_MUTANTS = {
 "bin >= -> > (growth exactly NF_b is outside; with the slack also removed)": ("ch80 >= -nfb - 1e-12", "ch80 > -nfb"),
 "N1 slack removed (float tie at the 0.01 floor)": ("ch80 >= -nfb - 1e-12", "ch80 >= -nfb"),
 "leg V > -> >=": ("legV = chV > nfp", "legV = chV >= nfp"),
 "leg D > -> >=": ("chD > nfd and", "chD >= nfd and"),
 "guard NF 1% relative floor dropped": ("def nf_rel(a, b): return max(abs(a - b), 0.01 * abs(a))", "def nf_rel(a, b): return abs(a - b)"),
 "guard NF |B0 - B0b| dropped (floor only)": ("def nf_rel(a, b): return max(abs(a - b), 0.01 * abs(a))", "def nf_rel(a, b): return 0.01 * abs(a)"),
 "guard worse test uses abs()": ("w = gd(S[arm], ds, k) - gd(S[\"B0\"], ds, k)".replace('\\"', '"'), "w = abs(gd(S[arm], ds, k) - gd(S[\"B0\"], ds, k))".replace('\\"', '"')),
 "guard w > nf -> >=": ("if w > nf:", "if w >= nf:"),
 "NF_d 0.01 floor dropped": ("nfd = max(abs(dev(xd(D[\"B0\"])) - dev(xd(D[\"B0b\"]))), 0.01)".replace('\\"', '"'), "nfd = abs(dev(xd(D[\"B0\"])) - dev(xd(D[\"B0b\"])))".replace('\\"', '"')),
 "NF_b 0.01 floor dropped": ("nfb = max(abs(dev(xd(D[\"B0\"], \"[80,120)\")) - dev(xd(D[\"B0b\"], \"[80,120)\"))), 0.01)".replace('\\"', '"'), "nfb = abs(dev(xd(D[\"B0\"], \"[80,120)\")) - dev(xd(D[\"B0b\"], \"[80,120)\")))".replace('\\"', '"')),
 "NF_p 1% relative floor dropped": ("nfp = nf_rel(dev(xv(S[\"B0\"])), dev(xv(S[\"B0b\"])))".replace('\\"', '"'), "nfp = abs(dev(xv(S[\"B0\"])) - dev(xv(S[\"B0b\"])))".replace('\\"', '"')),
 "S1 exemption widened to kitti": ("(\"nyuv2\", \"scannetpp\")".replace('\\"', '"'), "(\"nyuv2\", \"scannetpp\", \"kitti\")".replace('\\"', '"')),
 "S1 exemption dropped": ("arm == \"S1\" and ds in".replace('\\"', '"'), "False and ds in"),
 "dev() replaced by signed 1 - x": ("def dev(x): return abs(1.0 - x)", "def dev(x): return 1.0 - x"),
 "G0 V <= -> <": ("okV=xv(S[a]) <= G0_V", "okV=xv(S[a]) < G0_V"),
 "G0 D <= -> <": ("okD=xd(D[a]) <= G0_D", "okD=xd(D[a]) < G0_D"),
}
def _load_mutant(old, new):
    assert SRC.count(old) == 1, f"mutation pattern found {SRC.count(old)}x: {old!r}"
    ns = {"__name__": "mutant", "__file__": R.__file__}; exec(compile(SRC.replace(old, new), "l1_readout_mutant", "exec"), ns); return ns
def _g0_boundary_ok(ns):
    S, D = base(); S["B0"] = S_(0.93); D["B0b"] = D_(0.75)
    return ns["g0"](S, D)[0]
@pytest.mark.parametrize("name", sorted(SRC_MUTANTS))
def test_source_mutants_of_the_readout_are_killed(name):
    ns = _load_mutant(*SRC_MUTANTS[name])
    killed = bool(table2_mismatches(ns["verdict"])) or bool(table_mismatches(ns["verdict"])) or not _g0_boundary_ok(ns)
    assert killed, f"SURVIVING MUTANT: {name}"
def test_unmutated_source_passes_everything():
    ns = _load_mutant("def dev(x): return abs(1.0 - x)", "def dev(x): return abs(1.0 - x)")
    assert table2_mismatches(ns["verdict"]) == [] and table_mismatches(ns["verdict"]) == [] and _g0_boundary_ok(ns)

# ---- S5 mutants (integrity / missing arm): the mutated main() must fail at least one of the CLI expectations ----
S5_MUTANTS = {
 "step == stage check removed (scores.json)": ('if m.get("step") != stage: bad.append', 'if False: bad.append'),
 "ckpt sha cross-file compare removed": ('if ms.get("sha256") != md.get("sha256"): bad.append', 'if False: bad.append'),
 "sha presence check removed": ('if not m.get("sha256"): bad.append', 'if False: bad.append'),
 "integrity failure returns 0 (continues)": ('print("INTEGRITY FAIL (no verdicts): " + "; ".join(bad)); return 4', 'print("INTEGRITY FAIL (no verdicts): " + "; ".join(bad))'),
 "missing declared arm returns 0": ('print(f"INCOMPLETE: declared arm(s) {miss} not scored: rc 5 (this is not a complete readout)"); return 5', 'print("INCOMPLETE")'),
 "ddad file step not checked": ('for nm, m in (("scores.json", ms), ("ddad_scores.json", md)):', 'for nm, m in (("scores.json", ms),):'),
}
def _s5_expectations_hold(ns, tmp_path):
    main = ns["main"]; res = []
    S, D = _full(tmp_path); res.append(main(ARGS(tmp_path, "B0,B0b,S3")) == 0)
    S2_, D2_ = _full(tmp_path); S2_["S3"] = S_(0.93, step=4000); _write(tmp_path, S2_, D2_); res.append(main(ARGS(tmp_path, "B0,B0b,S3")) == 4)
    S2_, D2_ = _full(tmp_path); D2_["B0"] = D_(0.58, step=4000); _write(tmp_path, S2_, D2_); res.append(main(ARGS(tmp_path, "B0,B0b,S3")) == 4)
    S2_, D2_ = _full(tmp_path); D2_["S3"] = D_(0.75, x80=0.80, sha="b" * 64); _write(tmp_path, S2_, D2_); res.append(main(ARGS(tmp_path, "B0,B0b,S3")) == 4)
    S2_, D2_ = _full(tmp_path); S2_["S3"] = S_(0.93, sha=""); D2_["S3"] = D_(0.75, x80=0.80, sha=""); _write(tmp_path, S2_, D2_); res.append(main(ARGS(tmp_path, "B0,B0b,S3")) == 4)
    _full(tmp_path); res.append(main(ARGS(tmp_path, "B0,B0b,S3,S4")) == 5)
    return all(res)
@pytest.mark.parametrize("name", sorted(S5_MUTANTS))
def test_s5_source_mutants_are_killed(name, tmp_path_factory):
    ns = _load_mutant(*S5_MUTANTS[name]); assert not _s5_expectations_hold(ns, tmp_path_factory.mktemp("m")), f"SURVIVING MUTANT: {name}"
def test_s5_expectations_hold_on_the_real_source(tmp_path):
    assert _s5_expectations_hold(_load_mutant("def dev(x): return abs(1.0 - x)", "def dev(x): return abs(1.0 - x)"), tmp_path)

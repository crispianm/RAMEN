"""Builds stop-check fixtures from the REAL from-init log of job 7226119 (c3rep n46, seed=46000, no [ts2]) truncated at step 120, edited to look like the confirm arms (seed 50000, 4x [ts2]
lines copied verbatim from the screen TS2 log, synthetic metrics.jsonl), then runs check_first_run_confirm.sh on positives and on mutated copies (negative controls). Fixtures are SYNTHETIC
(labelled in the file name); they test the checker, they are not run output. Also runs the checker on the UNEDITED screen continuation logs (must FAIL)."""
import re, json, shutil, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent; FX = H / "stopcheck_fixtures"; FX.mkdir(exist_ok=True)
SCR = Path("/home/wg19671/.claude/jobs/a0d391cb/tmp/scr_logs")
real = (SCR / "dn01_c3rep_20261009_c3_n46_7226119.out").read_text(errors="replace").splitlines()
ts2 = [l for l in (SCR / "outdoor_screen_TS2_H300_7504902.out").read_text(errors="replace").splitlines() if l.startswith("[ts2]")]
assert len(ts2) == 4
cut = next(i for i, l in enumerate(real) if re.search(r"step140/12000", l))
body = real[:cut]
def synth(arm, seed=50000, lr_scale=1.0):
    L = []
    for l in body:
        if l.startswith("[data] train clips"): L += ts2 + [l.replace("seed=46000", f"seed={seed}")]; continue
        if lr_scale != 1.0:
            l = re.sub(r"lr=(\S+)", lambda m: f"lr={float(m.group(1))*lr_scale:.2e}", l)
        L.append(l)
    return L
def metrics(d, hinge):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "metrics.jsonl", "w") as f:
        for s in range(10, 101, 10):
            r = {"event": "loss_terms", "step": s, "silog": 0.3}
            if hinge: r["sky_hinge_weighted"] = 0.2
            f.write(json.dumps(r) + "\n")
def write(name, lines): (FX / name).write_text("\n".join(lines) + "\n"); return FX / name
runs = FX / "runs"
if runs.exists(): shutil.rmtree(runs)
metrics(runs / "confirm_C_TS2", False); metrics(runs / "confirm_C_TS2_H300", True)
P_CTL = write("SYNTH_C_TS2.out", synth("C_TS2")); P_H = write("SYNTH_C_TS2_H300.out", synth("C_TS2_H300"))
def run(args, label, want_fail, runs_dir=runs):
    r = subprocess.run(["bash", str(H / "check_first_run_confirm.sh"), "-r", str(runs_dir), *args], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if "FAIL" in l or "RESULT" in l or "WARN" in l]
    ok = (r.returncode == 1) == want_fail
    print(f"## {label}: rc={r.returncode} expected {'FAIL' if want_fail else 'no FAIL'} -> {'OK' if ok else 'WRONG'}")
    for l in lines[:6]: print("   ", l[:230])
    return ok
allok = True
allok &= run([f"C_TS2={P_CTL}", f"C_TS2_H300={P_H}"], "POSITIVE: both synthetic fixtures (real n46 log + edits)", False)
# negative controls
base_c, base_h = synth("C_TS2"), synth("C_TS2_H300")
def mut(name, f, arm_log="h"):
    ls = f(list(base_h if arm_log == "h" else base_c)); return write(name, ls)
N = []
N.append(("seed 49000 (wrong seed)", [f"C_TS2={P_CTL}", f"C_TS2_H300={write('NEG_seed.out', synth('x', seed=49000))}"]))
N.append(("only 3 [ts2] lines", [f"C_TS2={P_CTL}", f"C_TS2_H300={mut('NEG_ts2x3.out', lambda L: (L.remove(ts2[0]), L)[1])}"]))
N.append(("[ts2] with wrong sha", [f"C_TS2={P_CTL}", f"C_TS2_H300={mut('NEG_tssha.out', lambda L: [l.replace('f19d0b83', '00000000') for l in L])}"]))
N.append(("no [ts2] lines in the control", [f"C_TS2={mut('NEG_nots2_ctl.out', lambda L: [l for l in L if not l.startswith('[ts2]')], 'c')}", f"C_TS2_H300={P_H}"]))
N.append(("a [resume] line present", [f"C_TS2={P_CTL}", f"C_TS2_H300={mut('NEG_resume.out', lambda L: L + ['[resume] /x/step12000.pt epoch=3 step=12000 best=0.3485'])}"]))
N.append(("lr scaled x2 (wrong schedule)", [f"C_TS2={P_CTL}", f"C_TS2_H300={write('NEG_lr.out', synth('x', lr_scale=2.0))}"]))
N.append(("vkitti bound 656", [f"C_TS2={P_CTL}", f"C_TS2_H300={mut('NEG_bound.out', lambda L: [l.replace('vkitti=655', 'vkitti=656') for l in L])}"]))
N.append(("[data] line differs between arms", [f"C_TS2={P_CTL}", f"C_TS2_H300={mut('NEG_data.out', lambda L: [l.replace('irs=80', 'irs=81') for l in L])}"]))
N.append(("non-finite warning", [f"C_TS2={P_CTL}", f"C_TS2_H300={mut('NEG_nf.out', lambda L: L + ['[warn] non-finite loss at step 60'])}"]))
for lab, a in N: allok &= run(a, "NEG " + lab, True)
# hinge-key negatives (metrics)
r2 = FX / "runs_neg1"; 
for d in (r2,):
    if d.exists(): shutil.rmtree(d)
metrics(r2 / "confirm_C_TS2", True); metrics(r2 / "confirm_C_TS2_H300", True)      # control has the hinge key
allok &= run([f"C_TS2={P_CTL}", f"C_TS2_H300={P_H}"], "NEG hinge key present in the control", True, r2)
r3 = FX / "runs_neg2"
if r3.exists(): shutil.rmtree(r3)
metrics(r3 / "confirm_C_TS2", False); metrics(r3 / "confirm_C_TS2_H300", False)     # H300 lacks the hinge key
allok &= run([f"C_TS2={P_CTL}", f"C_TS2_H300={P_H}"], "NEG hinge key missing in H300", True, r3)
# unedited screen continuation logs (resume line, seed 45000, lr 12020...)
allok &= run([f"C_TS2={SCR/'outdoor_screen_TS2_7504898.out'}", f"C_TS2_H300={SCR/'outdoor_screen_TS2_H300_7504902.out'}"], "NEG unedited screen continuation logs", True, runs)
print("\nNEGATIVE-CONTROL SUITE:", "ALL AS EXPECTED" if allok else "UNEXPECTED RESULT")
sys.exit(0 if allok else 1)

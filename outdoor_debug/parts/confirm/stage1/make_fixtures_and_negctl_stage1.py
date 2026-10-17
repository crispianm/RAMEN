"""Stage 1 stop-check fixtures + negative controls. Fixtures are SYNTHETIC (derived from the REAL screen logs of TS2 7504898 / TS2_H300 7504902 by editing seed, resume path, config line, plus a synthetic metrics.jsonl);
they test the checker, they are not run output. Also runs the checker on the UNEDITED screen logs (must FAIL). Run: python3 make_fixtures_and_negctl_stage1.py"""
import re, json, shutil, subprocess, sys, hashlib
from pathlib import Path
H = Path(__file__).resolve().parent; FX = H / "stopcheck_fixtures"; FX.mkdir(exist_ok=True)
SCR = Path("/home/wg19671/.claude/jobs/a0d391cb/tmp/scr_logs")
LOG = {"TS2": (SCR / "outdoor_screen_TS2_7504898.out").read_text(errors="replace").splitlines(), "H300": (SCR / "outdoor_screen_TS2_H300_7504902.out").read_text(errors="replace").splitlines()}
SEED = {"46": 50, "47": 51, "48": 52}
def csha(name): return hashlib.sha256((H / f"configs/arm_{name}.yaml").read_bytes()).hexdigest()
def synth(n, a, seed=None, start_n=None, csha_override=None):
    name = f"S1_n{n}_{a}"; seed = SEED[n] * 1000 if seed is None else seed; sn = start_n or n; L = []
    for l in LOG[a]:
        l = l.replace("seed=45000", f"seed={seed}").replace("dn01_c3rep_20261009_c3_n46", f"dn01_c3rep_20261009_c3_n{sn}")
        if re.match(r"\s*config : ", l): l = f" config : outdoor_debug/parts/confirm/stage1/configs/arm_{name}.yaml sha256 {csha_override or csha(name)} OK"
        L.append(l)
    return L
def write(name, L): (FX / name).write_text("\n".join(L) + "\n"); return FX / name
def metrics(root, name, hinge, smoke=False):
    d = root / (f"confirm1_smoke50_{name}" if smoke else f"confirm1_{name}"); d.mkdir(parents=True, exist_ok=True)
    with open(d / "metrics.jsonl", "w") as f:
        for s in range(12010, 12051, 10):
            r = {"event": "loss_terms", "step": s, "silog": 0.3}
            if hinge: r["sky_hinge_weighted"] = 0.2
            f.write(json.dumps(r) + "\n")
runs = FX / "runs"
if runs.exists(): shutil.rmtree(runs)
NAMES = [(n, a) for n in SEED for a in ("TS2", "H300")]
for n, a in NAMES: metrics(runs, f"S1_n{n}_{a}", a == "H300")
P = {(n, a): write(f"SYNTH_S1_n{n}_{a}.out", synth(n, a)) for n, a in NAMES}
def specs(over=None):
    return [f"S1_n{n}_{a}={(over or {}).get((n, a), P[(n, a)])}" for n, a in NAMES]
def run(args, label, want_fail, runs_dir=runs, cfgd=None):
    cmd = ["bash", str(H / "check_first_run_stage1.sh"), "-r", str(runs_dir)] + (["-c", str(cfgd)] if cfgd else []) + args
    r = subprocess.run(cmd, capture_output=True, text=True); ok = (r.returncode == 1) == want_fail
    print(f"## {label}: rc={r.returncode} expected {'FAIL' if want_fail else 'no FAIL'} -> {'OK' if ok else 'WRONG'}")
    for l in [l for l in r.stdout.splitlines() if "FAIL" in l or "RESULT" in l or "WARN" in l or "REPORT" in l][:5]: print("   ", l[:240])
    return ok
allok = run(specs(), "POSITIVE: six synthetic fixtures (real screen logs, edited)", False)
def neg(label, over): return run(specs({k: write(f"NEG_{i}.out", v) for i, (k, v) in enumerate(over.items(), 1)}), "NEG " + label, True)
allok &= neg("n47 TS2 wrong seed (45000 instead of 51000)", {("47", "TS2"): synth("47", "TS2", seed=45000)})
allok &= neg("n48 H300 seed of another pair (51000)", {("48", "H300"): synth("48", "H300", seed=51000)})
allok &= neg("n47 H300 resumed from the n46 checkpoint", {("47", "H300"): synth("47", "H300", start_n="46")})
allok &= neg("n48 TS2 resumed from the n47 checkpoint", {("48", "TS2"): synth("48", "TS2", start_n="47")})
allok &= neg("config line sha != yaml sha", {("46", "H300"): synth("46", "H300", csha_override="0" * 64)})
t3 = "\n".join(synth("47", "TS2")).replace("[ts2] TartanAir TS2 sky/mask ON", "[ts2-removed]", 1).split("\n")
allok &= neg("only 3 [ts2] lines (n47 TS2)", {("47", "TS2"): t3})
allok &= neg("[ts2] wrong sha", {("48", "H300"): [l.replace("f19d0b83", "00000000") for l in synth("48", "H300")]})
allok &= neg("no [ts2] lines in a control", {("46", "TS2"): [l for l in synth("46", "TS2") if not l.startswith("[ts2]")]})
allok &= neg("lr x2 in n47 H300", {("47", "H300"): [re.sub(r"lr=(\S+)", lambda m: f"lr={float(m.group(1))*2:.2e}", l) for l in synth("47", "H300")]})
def nodecay(l):
    m = re.search(r"step(\d+)/14000 loss=\S+ lr=\S+", l)
    return re.sub(r"lr=\S+", "lr=1.00e-04", l) if m and int(m.group(1)) > 12100 else l
allok &= neg("lr never decays (constant 1e-4 after step 12100)", {("46", "TS2"): [nodecay(l) for l in synth("46", "TS2")]})
allok &= neg("vkitti bound 656 (n48 TS2)", {("48", "TS2"): [l.replace("vkitti=655", "vkitti=656") for l in synth("48", "TS2")]})
allok &= neg("[data] clip counts differ inside a pair (n47 H300)", {("47", "H300"): [l.replace("train clips=94581", "train clips=94580") for l in synth("47", "H300")]})
allok &= neg("non-finite warning", {("47", "H300"): synth("47", "H300") + ["[warn] non-finite loss at step 60"]})
allok &= neg("no resume line (from-scratch log)", {("48", "H300"): [l for l in synth("48", "H300") if not l.startswith("[resume]")]})
# config-side negatives: edited config dir copies (resume_reseed missing / != data.seed)
cfgd = FX / "cfg_neg"
for lab, edit in (("resume_reseed missing in n47 TS2 config", lambda t: re.sub(r"^  resume_reseed: 51\n", "", t, flags=re.M)), ("resume_reseed 50 != data.seed 51 in n47 TS2 config", lambda t: t.replace("  resume_reseed: 51", "  resume_reseed: 50"))):
    if cfgd.exists(): shutil.rmtree(cfgd)
    shutil.copytree(H / "configs", cfgd); p = cfgd / "arm_S1_n47_TS2.yaml"; p.write_text(edit(p.read_text()))
    allok &= run(specs(), "NEG " + lab, True, cfgd=cfgd)
shutil.rmtree(cfgd)
# metrics negatives
r1 = FX / "runs_neg1"; r2 = FX / "runs_neg2"
for d in (r1, r2):
    if d.exists(): shutil.rmtree(d)
for n, a in NAMES: metrics(r1, f"S1_n{n}_{a}", True); metrics(r2, f"S1_n{n}_{a}", False)
allok &= run(specs(), "NEG hinge key present in the controls", True, r1)
allok &= run(specs(), "NEG hinge key missing in the H300 runs", True, r2)
# unedited screen logs under the Stage 1 names
allok &= run([f"S1_n46_TS2={SCR/'outdoor_screen_TS2_7504898.out'}", f"S1_n46_H300={SCR/'outdoor_screen_TS2_H300_7504902.out'}"], "NEG unedited screen logs (seed 45000)", True)
# smoke mode on a truncated fixture (log cut after step 12060)
for n, a in (("47", "TS2"), ("48", "H300")):
    L = synth(n, a); cut = next(j for j, l in enumerate(L) if re.search(r"step12080/14000", l)); write(f"SMOKE_S1_n{n}_{a}.out", [l.replace("/14000", "/12050") for l in L[:cut]]); metrics(FX / "runs_smoke", f"S1_n{n}_{a}", a == "H300", True)
def run_s(args, label, rd):
    r = subprocess.run(["bash", str(H / "check_first_run_stage1.sh"), "-s", "-r", str(rd)] + args, capture_output=True, text=True); ok = r.returncode == 0
    print(f"## {label}: rc={r.returncode} expected no FAIL -> {'OK' if ok else 'WRONG'}"); [print("   ", l[:200]) for l in r.stdout.splitlines() if "FAIL" in l or "RESULT" in l or "WAIT" in l][:6]; return ok
allok &= run_s([f"S1_n47_TS2={FX/'SMOKE_S1_n47_TS2.out'}", f"S1_n48_H300={FX/'SMOKE_S1_n48_H300.out'}"], "POSITIVE smoke mode (-s), truncated at step 12060 (n47 TS2 + n48 H300)", FX / "runs_smoke")
print("\nNEGATIVE-CONTROL SUITE:", "ALL AS EXPECTED" if allok else "UNEXPECTED RESULT"); sys.exit(0 if allok else 1)

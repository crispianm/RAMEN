"""L1 stop-check fixtures + NEGATIVE CONTROLS for check_first_run_l1.sh. Fixtures are SYNTHETIC: derived from the REAL log of the c3_n46 from-release 12k run (job 7226119: same recipe and seed 46; its rows 20..K are kept,
`/12000` -> `/K`, every lr rewritten from the closed form, snapshot list, seed, config line and metrics.jsonl synthesised). They test the CHECKER, they are not run output.
Every check item is shown to FAIL on a mutated fixture. Run:  python3 make_fixtures_and_negctl_l1.py [path/to/dn01_c3rep_20261009_c3_n46_7226119.out]
(the real log is read from the R+D scratch dir by default; it is NOT committed; the fixtures it produces are)."""
import hashlib, json, math, re, shutil, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent; FX = H / "stopcheck_fixtures"; FX.mkdir(exist_ok=True)
REAL = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/wg19671/.claude/jobs/a0d391cb/tmp/scr_logs/dn01_c3rep_20261009_c3_n46_7226119.out")
L0 = REAL.read_text(errors="replace").splitlines()
ARMS = ["B0", "B0b", "S1", "S2", "S3", "S4", "S5", "S6"]; SEED = {a: (49 if a == "B0b" else 46) for a in ARMS}
i0 = next(i for i, l in enumerate(L0) if l.startswith("[params]")); BODY = L0[i0:]
def scale(g, K):
    s = g - 1; return (s + 1) / 500 if s < 500 else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - 500) / (K - 500))))
def csha(K, a): return hashlib.sha256((H / f"configs/k{K}/arm_{a}.yaml").read_bytes()).hexdigest()
def synth(arm, K, smoke=0, seed=None, mut=None):
    total = smoke or K; seed = SEED[arm] * 1000 if seed is None else seed; L = [f" config : outdoor_debug/parts/far_diag/l1_final/configs/k{K}/arm_{arm}.yaml sha256 {csha(K, arm)} OK"]
    for l in BODY:
        m = re.match(r"(e\d+) step(\d+)/12000 loss=(\S+) lr=\S+ (.*)$", l)
        if m:
            g = int(m.group(2))
            if g > total: continue
            l = f"{m.group(1)} step{g}/{total} loss={m.group(3)} lr={1e-4 * scale(g, K):.2e} {m.group(4)}"
        mm = re.match(r"\[monitor\] step(\d+) ", l)
        if mm and int(mm.group(1)) > total: continue
        if re.match(r"\[val|  -> new best", l): continue
        l = l.replace("seed=46000", f"seed={seed}").replace("[6000, 8000, 12000]", f"[{total}]")
        if arm == "S1":
            l = l.replace(", nyu_train=10, scannetpp_train=10", "").replace("train clips=94581", "train clips=61234")
        L.append(l)
    cut = next((j for j, l in enumerate(L) if re.match(r"e\d+ step\d+/%d " % total, l) and int(re.search(r"step(\d+)/", l).group(1)) == total), None)
    if cut is not None: L = L[:cut + 1]
    return mut(L) if mut else L
def write(name, L): p = FX / name; p.write_text("\n".join(L) + "\n"); return p
def metrics(root, arm, K, smoke=0, drop=None, add=None, far=0.014):
    d = root / (f"l1_smoke_{arm}_k{K}" if smoke else f"l1_{arm}_k{K}"); d.mkdir(parents=True, exist_ok=True)
    with open(d / "metrics.jsonl", "w") as f:
        for s in range(100, (smoke or K) + 1, 100):
            r = {"event": "loss_terms", "step": s, "silog": 0.3}
            if arm == "S6": r["silog_far_px_frac"] = far
            if drop: r.pop(drop, None)
            if add: r[add] = 0.01
            f.write(json.dumps(r) + "\n")
def mk_runs(K, smoke=0, name="runs"):
    root = FX / name
    if root.exists(): shutil.rmtree(root)
    for a in ARMS: metrics(root, a, K, smoke)
    return root
def run(label, args, want_fail, K=2000, smoke=0, runs=None, cfgd=None, final=True, item=None, chk=None):
    """item=(ARM, N): additionally require the line `[ARM] item N: FAIL` (want_fail) -- so a NEG cannot pass because some OTHER item failed (isolated NEG).  chk: checker script (default the shipped one)."""
    cmd = ["bash", str(chk or H / "check_first_run_l1.sh"), "-K", str(K)] + (["-F"] if final else []) + (["-S", str(smoke)] if smoke else []) + ["-r", str(runs or FX / f"runs_k{K}")] + (["-c", str(cfgd)] if cfgd else []) + args
    r = subprocess.run(cmd, capture_output=True, text=True); ok = (r.returncode == 1) == want_fail
    if item is not None: ok = ok and any(l.startswith(f"[{item[0]}] item {item[1]}: FAIL") for l in r.stdout.splitlines())
    print(f"## {label}: rc={r.returncode} expected {'FAIL' if want_fail else 'no FAIL'} -> {'OK' if ok else 'WRONG'}")
    for l in [l for l in r.stdout.splitlines() if "FAIL" in l or "RESULT" in l or "REPORT" in l][:4]: print("   ", l[:250])
    if r.returncode not in (0, 1): print("   STDERR:", r.stderr[-300:])
    return ok
def specs(K, over=None, smoke=0, arms=ARMS):
    return [f"{a}={(over or {}).get(a, FX / (f'SYNTH_{a}_k{K}' + (f'_smoke{smoke}' if smoke else '') + '.out'))}" for a in arms]
for K in (2000, 4000):
    mk_runs(K, 0, f"runs_k{K}")
    for a in ARMS: write(f"SYNTH_{a}_k{K}.out", synth(a, K))
mk_runs(2000, 120, "runs_smoke")
for a in ARMS: write(f"SYNTH_{a}_k2000_smoke120.out", synth(a, 2000, smoke=120))
allok = True
def P(label, **kw): global allok; allok &= run(label, **kw)
# positives
P("POSITIVE K=2000: 8 synthetic fixtures (real c3_n46 rows 20..2000, lr rewritten)", args=specs(2000), want_fail=False)
P("POSITIVE K=4000: 8 synthetic fixtures", args=specs(4000), want_fail=False, K=4000)
P("POSITIVE smoke mode (-S 120) on 8 truncated fixtures", args=specs(2000, smoke=120), want_fail=False, smoke=120, runs=FX / "runs_smoke")
cut = lambda a, K, n: [l for l in synth(a, K) if not re.match(r"e\d+ step(\d+)/", l) or int(re.search(r"step(\d+)/", l).group(1)) <= n]
P("POSITIVE-WAIT: a log truncated at step 400 gives WAIT lines, no FAIL", args=[f"B0={write('TRUNC_B0.out', cut('B0', 2000, 400))}"], want_fail=False, runs=FX / "runs_k2000_empty", final=False)
P("NEG the same truncated log under -F (WAIT becomes FAIL)", args=[f"B0={FX / 'TRUNC_B0.out'}"], want_fail=True, runs=FX / "runs_k2000_empty")
# negatives: each mutates ONE aspect of ONE fixture and must FAIL
def neg(label, over, **kw): global allok; allok &= run("NEG " + label, args=specs(kw.pop("K", 2000), {k: write(f"NEG_{i}.out", v) for i, (k, v) in enumerate(over.items(), 1)}, kw.get("smoke", 0)), want_fail=True, **kw)
def relr(L, f): return [re.sub(r"(step(\d+)/\d+ loss=\S+ lr=)(\S+)", lambda m: m.group(1) + f(int(m.group(2)), float(m.group(3))), l) for l in L]
neg("(1) lr never decays (constant 1e-4 after warmup)", {"B0": relr(synth("B0", 2000), lambda g, v: "1.00e-04" if g > 500 else f"{v:.2e}")})
neg("(1) lr hits 0 at step 1600", {"S3": relr(synth("S3", 2000), lambda g, v: "0.00e+00" if g >= 1600 else f"{v:.2e}")})
neg("(1) lr x2 everywhere (peak 2e-4)", {"S2": relr(synth("S2", 2000), lambda g, v: f"{2 * v:.2e}")})
neg("(1) the UNEDITED 12000-step lr values (real cosine of the 12k run) under a K=2000 label", {"B0": [re.sub(r"step(\d+)/12000 ", lambda m: f"step{m.group(1)}/2000 ", l) for l in BODY[:0]] or relr(synth("B0", 2000), lambda g, v: f"{1e-4 * (0.5 * (1 + math.cos(math.pi * max(0, g - 500) / 11500))) if g > 500 else v:.2e}")})
neg("(1) warmup missing (lr starts at 1e-4)", {"B0b": relr(synth("B0b", 2000), lambda g, v: "1.00e-04" if g <= 500 else f"{v:.2e}")})
neg("(1) restart-style warmup of 100 steps (lr reaches 1e-4 at 100)", {"S4": relr(synth("S4", 2000), lambda g, v: f"{1e-4 * min(1.0, g / 100) if g <= 500 else v:.2e}")})
neg("(1) a 4000-step schedule in a K=2000 run (cosine too slow)", {"S5": relr(synth("S5", 2000), lambda g, v: f"{1e-4 * scale(g, 4000):.2e}")})
neg("(2) a [resume] line is present (restart from a checkpoint)", {"B0": synth("B0", 2000, mut=lambda L: L[:9] + ["[resume] /projects/b5dh/x/step12000.pt epoch=3 step=12000 best=0.3485"] + L[9:])})
neg("(2) total printed 12000 (max_steps not overridden)", {"S1": [l.replace("/2000 ", "/12000 ") for l in synth("S1", 2000)]})
neg("(2) snapshot list [12000]", {"S2": [l.replace(f"[2000]", "[12000]") for l in synth("S2", 2000)]})
neg("(2) [params] shows an unfrozen backbone", {"S3": [l.replace("unfrozen_backbone=False", "unfrozen_backbone=True") for l in synth("S3", 2000)]})
neg("(3) start probe: monitor step0 ratio 0.58 (collapsed start, e.g. wrong init)", {"B0": [l.replace("step0 outdoor median(pred)/median(GT) = 1.006", "step0 outdoor median(pred)/median(GT) = 0.580") for l in synth("B0", 2000)]})
neg("(3) start probe: no monitor step0 line", {"S4": [l for l in synth("S4", 2000) if not l.startswith("[monitor] step0 ")]})
neg("(4) severity always 1 (no ramp)", {"S5": [re.sub(r"sev=\S+", "sev=1", l) for l in synth("S5", 2000)]})
neg("(5) B0b log carries B0's seed 46000", {"B0b": synth("B0b", 2000, seed=46000)})
neg("(5) B0 log carries B0b's seed 49000", {"B0": synth("B0", 2000, seed=49000)})
neg("(5) config sha line != yaml sha", {"S6": [re.sub(r"sha256 [0-9a-f]{64}", "sha256 " + "0" * 64, l) if re.match(r"\s*config :", l) else l for l in synth("S6", 2000)]})
neg("(5) config line names the wrong arm yaml", {"S2": [l.replace("arm_S2.yaml", "arm_S3.yaml") for l in synth("S2", 2000)]})
neg("(5) config line names the k4000 yaml", {"S1": [l.replace("/k2000/", "/k4000/") for l in synth("S1", 2000)]})
neg("(5) config line missing", {"B0": [l for l in synth("B0", 2000) if not l.strip().startswith("config :")]})
neg("(6) S1 keeps nyu_train/scannetpp_train in the bound line", {"S1": [l.replace("irs=80, spring=80, kitti_raw=80", "irs=80, spring=80, kitti_raw=80, nyu_train=10, scannetpp_train=10") for l in synth("S1", 2000)]})
neg("(6) S1 clip total equals the full mix (94581)", {"S1": [l.replace("train clips=61234", "train clips=94581") for l in synth("S1", 2000)]})
neg("(6) non-S1 clip total differs (94580)", {"S3": [l.replace("train clips=94581", "train clips=94580") for l in synth("S3", 2000)]})
neg("(6) a non-S1 arm lacks nyu_train in the bound line", {"S4": [l.replace(", nyu_train=10, scannetpp_train=10", "") for l in synth("S4", 2000)]})
neg("(6) world size 2", {"S2": [l.replace("world_size=4", "world_size=2") for l in synth("S2", 2000)]})
neg("(7) [ts2] line present", {"S5": synth("S5", 2000, mut=lambda L: L[:8] + ["[ts2] TartanAir TS2 sky/mask ON env_table_sha256=f19d0b838252c9a2409498b35f46738d99b3646afba984d6a82ffea3cd610b76"] + L[8:])})
neg("(8) non-finite loss warning", {"S6": synth("S6", 2000) + ["[warn] non-finite loss at step 60"]})
neg("(10) B0 loss 2x the c3_n46 reference", {"B0": [re.sub(r"loss=(\S+)", lambda m: f"loss={2 * float(m.group(1)):.4f}", l) if re.match(r"e\d+ step", l) else l for l in synth("B0", 2000)]})
neg("(10) B0b loss 0.5x the reference", {"B0b": [re.sub(r"loss=(\S+)", lambda m: f"loss={0.5 * float(m.group(1)):.4f}", l) if re.match(r"e\d+ step", l) else l for l in synth("B0b", 2000)]})
neg("(10) B0b loss 1.3x (above the 1.18 upper band) the reference", {"B0b": [re.sub(r"loss=(\S+)", lambda m: f"loss={1.3 * float(m.group(1)):.4f}", l) if re.match(r"e\d+ step", l) else l for l in synth("B0b", 2000)]})
neg("K mismatch: the K=2000 fixtures checked with -K 4000 (via -K 4000 on K=2000 logs)", {}, K=4000, runs=FX / "runs_k4000") if False else None
# -K mismatch is a NEG on the whole set (all logs are K=2000, checker told 4000)
allok &= run("NEG -K 4000 applied to the K=2000 fixtures", specs(2000), True, K=4000, runs=FX / "runs_k4000")
allok &= run("NEG -K 2000 applied to the K=4000 fixtures", specs(4000), True, K=2000, runs=FX / "runs_k2000")
# metrics negatives (log fixtures are fine, metrics mutated)
def runs_mut(name, K, f):
    root = mk_runs(K, 0, name); f(root); return root
r = runs_mut("runs_neg_a", 2000, lambda root: metrics(root, "S6", 2000, drop="silog_far_px_frac")); allok &= run("NEG (9) S6 metrics lack silog_far_px_frac", specs(2000), True, runs=r)
r = runs_mut("runs_neg_b", 2000, lambda root: metrics(root, "B0", 2000, add="silog_far_px_frac")); allok &= run("NEG (9) B0 metrics carry silog_far_px_frac", specs(2000), True, runs=r)
r = runs_mut("runs_neg_c", 2000, lambda root: metrics(root, "S4", 2000, drop="silog")); allok &= run("NEG (9) S4 metrics lack silog", specs(2000), True, runs=r)
r = runs_mut("runs_neg_d", 2000, lambda root: metrics(root, "S6", 2000, far=0.0)); allok &= run("NEG (9) S6 far-pixel fraction 0.0 (far weighting measured to cover no pixel)", specs(2000), True, runs=r)
r = runs_mut("runs_neg_e", 2000, lambda root: metrics(root, "S6", 2000, far=0.6)); allok &= run("NEG (9) S6 far-pixel fraction 0.6 (implausible)", specs(2000), True, runs=r)
# config-side negatives: edited copies of the yaml dir
cfgd = FX / "cfg_neg"
for lab, arm, edit in (("resume_reseed present in the B0b yaml", "B0b", lambda t: t.replace("  resume_from: null", "  resume_from: null\n  resume_reseed: 49")),
                       ("resume_from set in the B0 yaml", "B0", lambda t: t.replace("  resume_from: null", "  resume_from: /x.pt")),
                       ("data.seed 47 in the S2 yaml", "S2", lambda t: t.replace("\n  seed: 46\n", "\n  seed: 47\n")),
                       ("lr_restart key in the S4 yaml", "S4", lambda t: t.replace("  resume_from: null", "  resume_from: null\n  lr_restart_step: 12000"))):
    if cfgd.exists(): shutil.rmtree(cfgd)
    shutil.copytree(H / "configs/k2000", cfgd); p = cfgd / f"arm_{arm}.yaml"; before = p.read_text(); p.write_text(edit(before)); assert p.read_text() != before, "edit did not apply: " + lab
    allok &= run("NEG (5) " + lab, specs(2000), True, cfgd=cfgd)
shutil.rmtree(cfgd)
# the unedited REAL 12k log under an L1 name (it IS a from-release log, but with the 12000-step schedule and snapshot list): must FAIL
allok &= run("NEG the unedited REAL 12000-step log passed as B0 (K=2000)", [f"B0={REAL}"], True)

# ===================== second review (S2, S3): extra negatives, ISOLATED (each must FAIL on the NAMED item), mid-run versions, sub-check isolation =====================
def I(label, over, item, **kw):   # isolated NEG: complete-log mode (-F) unless final=False
    global allok; allok &= run("NEG[iso] " + label, args=specs(kw.pop("K", 2000), {k: write(f"IN_{i}.out", v) for i, (k, v) in enumerate(over.items(), 1)}), want_fail=True, item=item, **kw)
cutn = lambda L, n: [l for l in L if not re.match(r"e\d+ step(\d+)/", l) or int(re.search(r"step(\d+)/", l).group(1)) <= n]
# --- near-boundary LR (beyond 1.5%, far inside the old-mutant 50%)
I("(1) lr x1.05 everywhere (beyond the 1.5% tolerance)", {"S2": relr(synth("S2", 2000), lambda g, v: f"{1.05 * v:.2e}")}, ("S2", 1))
I("(1) lr x0.97 everywhere", {"S2": relr(synth("S2", 2000), lambda g, v: f"{0.97 * v:.2e}")}, ("S2", 1))
I("(1) warmup 450 (peak reached at step 450)", {"S4": relr(synth("S4", 2000), lambda g, v: f"{1e-4 * min(1.0, g / 450) if g <= 500 else v:.2e}")}, ("S4", 1))
I("(1) lr 0 only at the last printed row (step 2000)", {"S3": relr(synth("S3", 2000), lambda g, v: "0.00e+00" if g == 2000 else f"{v:.2e}")}, ("S3", 1))
# --- step rows that no longer parse (S2): format drift must FAIL, with and without -F, never "ok"
nosev = [re.sub(r" sev=\S+", "", l) for l in synth("B0", 2000)]
I("(S2) B0 step rows lack ' sev=' (full row regex matches nothing), -F", {"B0": nosev}, ("B0", 1))
I("(S2) the same, MID-RUN (no -F): FAIL, not WAIT/ok", {"B0": cutn(nosev, 400)}, ("B0", 1), final=False, runs=FX / "runs_k2000")
half = [re.sub(r" sev=\S+", "", l) if re.match(r"e\d+ step(\d+)/", l) and int(re.search(r"step(\d+)/", l).group(1)) % 40 == 0 else l for l in synth("B0", 2000)]
I("(S2) only some step rows lack ' sev=' (counts differ), MID-RUN", {"B0": cutn(half, 400)}, ("B0", 1), final=False, runs=FX / "runs_k2000")
# header only (no step rows): -F FAIL; mid-run WAIT and NEVER an item-1 'ok' line
hdr = [l for l in synth("B0", 2000) if not re.match(r"e\d+ step", l)]
I("(S2) header-only log (no step rows), -F", {"B0": hdr}, ("B0", 1))
r = subprocess.run(["bash", str(H / "check_first_run_l1.sh"), "-K", "2000", "-r", str(FX / "runs_k2000_empty"), f"B0={write('HDR_B0.out', hdr)}"], capture_output=True, text=True)
okw = r.returncode == 0 and not any(l.startswith("[B0] item 1: ok") for l in r.stdout.splitlines()) and any(l.startswith("[B0] item 1: WAIT") for l in r.stdout.splitlines())
print(f"## POSITIVE-WAIT (S2): header-only log mid-run: rc=0, item 1 is WAIT and never 'ok' -> {'OK' if okw else 'WRONG'}"); allok &= okw
# --- yaml edits WITH the log's sha line updated (only the yaml-content check can fail)
for lab, arm, edit in (("resume_reseed in the yaml", "B0b", lambda s: s.replace("  resume_from: null", "  resume_from: null\n  resume_reseed: 49")),
                       ("resume_from set in the yaml", "B0", lambda s: s.replace("  resume_from: null", "  resume_from: /x.pt")),
                       ("data.seed 47 in the yaml", "S2", lambda s: s.replace("\n  seed: 46\n", "\n  seed: 47\n")),
                       ("lr_restart key in the yaml", "S4", lambda s: s.replace("  resume_from: null", "  resume_from: null\n  lr_restart_step: 12000"))):
    cd2 = FX / "cfg_neg2"
    if cd2.exists(): shutil.rmtree(cd2)
    shutil.copytree(H / "configs/k2000", cd2); pp = cd2 / f"arm_{arm}.yaml"; b4 = pp.read_text(); pp.write_text(edit(b4)); assert pp.read_text() != b4
    sh = hashlib.sha256(pp.read_bytes()).hexdigest()
    L = [re.sub(r"sha256 [0-9a-f]{64}", "sha256 " + sh, l) if re.match(r"\s*config :", l) else l for l in synth(arm, 2000)]
    I(f"(5) {lab} (log sha line updated to the edited yaml: only the content check can fail)", {arm: L}, (arm, 5), cfgd=cd2)
shutil.rmtree(FX / "cfg_neg2")
# --- B0 (the REFERENCE) clip counts / world size on every arm / extra [data] line
I("(6) B0 (the reference) clip total 94580", {"B0": [l.replace("train clips=94581", "train clips=94580") for l in synth("B0", 2000)]}, ("B0", 6))
allok &= run("NEG[iso] (6) world_size=2 on EVERY arm (no arm differs from the reference)", [f"{a}={write(f'WS_{a}.out', [l.replace('world_size=4', 'world_size=2') for l in synth(a, 2000)])}" for a in ARMS], True, item=("B0", 6))
I("(6) S3 carries an extra [data] line that B0 lacks (clip counts and bound line equal)", {"S3": synth("S3", 2000, mut=lambda L: L[:16] + ["[data] sampler=weighted"] + L[16:])}, ("S3", 6))
# --- missing metrics / no loss_terms rows under -F
emp = FX / "runs_empty_x"; emp.mkdir(exist_ok=True)
allok &= run("NEG[iso] (9) complete logs, NO metrics.jsonl at all, -F", specs(2000), True, runs=emp, item=("B0", 9))
nolt = mk_runs(2000, 0, "runs_nolt")
for a in ARMS: (nolt / f"l1_{a}_k2000/metrics.jsonl").write_text(json.dumps({"event": "val", "step": 2000}) + "\n")
allok &= run("NEG[iso] (9) complete logs, metrics.jsonl with no loss_terms rows, -F", specs(2000), True, runs=nolt, item=("B0", 9))
# --- item 10 with fewer than 3 reference rows (under -F WAIT->FAIL on item 10 itself) and item 3 mid-run
short = [l for l in synth("B0", 2000) if not re.match(r"e\d+ step", l) or int(re.search(r"step(\d+)/", l).group(1)) <= 40]
I("(10) only 2 reference-comparable loss rows, -F: item 10 itself FAILs", {"B0": short}, ("B0", 10))
mr = mk_runs(2000, 0, "runs_mid")
for a in ARMS:
    pp = mr / f"l1_{a}_k2000/metrics.jsonl"; pp.write_text("".join(l for l in pp.read_text().splitlines(True) if json.loads(l)["step"] <= 400))
def M(label, over, want_fail, item=None):   # MID-RUN mode: logs cut at step 400, NO -F
    global allok
    sp = [f"{a}={write(f'MID_{a}.out', cutn(over.get(a, synth(a, 2000)), 400))}" for a in ARMS]
    allok &= run("MID " + label, sp, want_fail, runs=mr, final=False, item=item)
M("POSITIVE: all 8 cut at 400, no FAIL", {}, False)
M("lr x2", {"S2": relr(synth("S2", 2000), lambda g, v: f"{2 * v:.2e}")}, True, ("S2", 1))
M("lr x1.05", {"S2": relr(synth("S2", 2000), lambda g, v: f"{1.05 * v:.2e}")}, True, ("S2", 1))
M("warmup 100", {"S4": relr(synth("S4", 2000), lambda g, v: f"{1e-4 * min(1.0, g / 100) if g <= 500 else v:.2e}")}, True, ("S4", 1))
M("no monitor step0 line (item 3 FAIL once >= 2 step rows exist, not WAIT)", {"S4": [l for l in synth("S4", 2000) if not l.startswith("[monitor] step0 ")]}, True, ("S4", 3))
M("seed swapped (B0b log carries 46000)", {"B0b": synth("B0b", 2000, seed=46000)}, True, ("B0b", 5))
M("[resume] line", {"B0": synth("B0", 2000, mut=lambda L: L[:9] + ["[resume] /x/step12000.pt epoch=3 step=12000 best=0.3"] + L[9:])}, True, ("B0", 2))
M("total printed 12000", {"S1": [l.replace("/2000 ", "/12000 ") for l in synth("S1", 2000)]}, True, ("S1", 2))
M("B0 loss 2x", {"B0": [re.sub(r"loss=(\S+)", lambda m: f"loss={2 * float(m.group(1)):.4f}", l) if re.match(r"e\d+ step", l) else l for l in synth("B0", 2000)]}, True, ("B0", 10))
M("B0 (the reference) clip total 94580", {"B0": [l.replace("train clips=94581", "train clips=94580") for l in synth("B0", 2000)]}, True, ("B0", 6))
# --- D1 (Manager): a log read while the trainer is mid-write of its LAST row (no trailing newline) must not FAIL (= scancel a healthy run); complete bad rows still must
def writeraw(name, text): p = FX / name; p.write_text(text); return p
def midspecs(trunc=None, midbad=False, a="B0"):
    sp = []
    for arm in ARMS:
        L = cutn(synth(arm, 2000), 400)
        if arm == a and midbad:
            L = [re.sub(r" sev=\S+", "", l) if re.match(r"e\d+ step(\d+)/", l) and int(re.search(r"step(\d+)/", l).group(1)) == 100 else l for l in L]
        txt = "\n".join(L) + "\n"
        if arm == a and trunc: txt = txt + trunc          # appended partial row WITHOUT a trailing newline
        sp.append(f"{arm}={writeraw(f'D1_{arm}.out', txt)}")
    return sp
allok &= run("D1(a) POSITIVE: all 8 cut at 400 + a truncated final row 'e0 step420/2000 loss=0.7385 lr=1.0' (no newline), mid-run: no FAIL", midspecs(trunc="e0 step420/2000 loss=0.7385 lr=1.0"), False, runs=mr, final=False)
allok &= run("D1(b) NEG: the same truncated final row PLUS a complete unparsable row (no ' sev=') at step 100 in the middle: still FAIL", midspecs(trunc="e0 step420/2000 loss=0.7385 lr=1.0", midbad=True), True, runs=mr, final=False, item=("B0", 1))
bf = list(synth("B0", 2000)); last_idx = max(i for i, l in enumerate(bf) if re.match(r"e\d+ step", l)); bf[last_idx] = re.sub(r" sev=\S+", "", bf[last_idx])
I("D1(c') -F at end of run: the final complete step row (newline-terminated) is unparsable", {"B0": bf}, ("B0", 1))
fp = "\n".join(synth("B0", 2000)) + "\ne0 step2001/2000 loss=0.7"      # a complete log plus a PARTIAL final line, no trailing newline
sp = [f"{a}={writeraw(f'D1F_{a}.out', fp if a == 'B0' else chr(10).join(synth(a, 2000)) + chr(10))}" for a in ARMS]
allok &= run("D1(3) NEG: partial final line (no newline) under -F -> FAIL (no exemption)", sp, True, item=("B0", 0))
nf = cutn(synth("B0", 2000), 400) + ["[warn] non-finite loss at step 77"]
sp = [f"{a}={writeraw(f'D1N_{a}.out', (chr(10).join(nf) + chr(10) + 'e0 step420/2000 loss=0.73') if a == 'B0' else chr(10).join(cutn(synth(a, 2000), 400)) + chr(10))}" for a in ARMS]
allok &= run("D1(4) NEG: a complete [warn] non-finite line just before a partial final line, mid-run -> FAIL", sp, True, runs=mr, final=False, item=("B0", 8))
# --- sub-check ISOLATION: each LR sub-check, alone, must catch the LR negatives it is meant to (kills 'check removed' / 'tolerance widened' mutants of a REDUNDANT sub-check)
CHK0 = (H / "check_first_run_l1.sh").read_text()
def checker(name, edits):
    s = CHK0
    for o_, n_ in edits:
        if s.count(o_) != 1:   # a sub-check this isolation needs to REMOVE is itself missing/changed in the shipped checker: report, do not crash
            print(f"## WRONG isolation builder '{name}': sub-check text not found exactly once in check_first_run_l1.sh ({o_[:60]!r}) -> that sub-check was removed or altered"); global allok; allok = False; return None
        s = s.replace(o_, n_)
    pth = FX / f"chk_{name}.sh"; pth.write_text(s); return pth
CF = ("bad = [(g, lr) for", "bad = [] and [(g, lr) for"); TB = ("        if g > TOTAL: continue\n", "        continue\n")
DC = ('if SMOKE == 0: out(arm, 1, "FAIL" if (late and max(late) > 0.8e-4)', 'if False: out(arm, 1, "FAIL" if (late and max(late) > 0.8e-4)')
Z0 = ("if [g for g, (_, lr, _) in steps.items() if lr == 0.0]:", "if False:")
ISO = {"closed-form": checker("cf", [TB, DC, Z0]), "table": checker("tb", [CF, DC, Z0]), "decay": checker("dc", [CF, TB, Z0]), "lr==0": checker("z0", [CF, TB, DC])}
LRN = {"const 1e-4 after warmup": ("B0", lambda g, v: "1.00e-04" if g > 500 else f"{v:.2e}"), "lr 0 from 1600": ("S3", lambda g, v: "0.00e+00" if g >= 1600 else f"{v:.2e}"),
       "lr x2": ("S2", lambda g, v: f"{2 * v:.2e}"), "lr x1.05": ("S2", lambda g, v: f"{1.05 * v:.2e}"), "lr x0.97": ("S2", lambda g, v: f"{0.97 * v:.2e}"),
       "warmup missing": ("B0b", lambda g, v: "1.00e-04" if g <= 500 else f"{v:.2e}"), "warmup 100": ("S4", lambda g, v: f"{1e-4 * min(1.0, g / 100) if g <= 500 else v:.2e}"),
       "warmup 450": ("S4", lambda g, v: f"{1e-4 * min(1.0, g / 450) if g <= 500 else v:.2e}"), "4000-cosine in K=2000": ("S5", lambda g, v: f"{1e-4 * scale(g, 4000):.2e}"),
       "lr 0 at last row only": ("S3", lambda g, v: "0.00e+00" if g == 2000 else f"{v:.2e}")}
MUST = {"closed-form": set(LRN), "table": set(LRN) - {"warmup 450"}, "decay": {"const 1e-4 after warmup"}, "lr==0": {"lr 0 from 1600", "lr 0 at last row only"}}
for sub, chkp in ISO.items():
    if chkp is None: continue
    for lab in sorted(MUST[sub]):
        arm, f = LRN[lab]; pth = write("ISO.out", relr(synth(arm, 2000), f))
        allok &= run(f"NEG[iso-sub-check] the {sub} check ALONE catches: {lab}", specs(2000, {arm: pth}), True, chk=chkp, item=(arm, 1), cfgd=H / "configs/k2000")
print("\nNEGATIVE-CONTROL SUITE:", "ALL AS EXPECTED" if allok else "UNEXPECTED RESULT"); sys.exit(0 if allok else 1)

"""Fixtures + NEGATIVE CONTROLS for check_optimizer_lr.py. Fixtures are SLIM checkpoints {cfg, step, optimizer.param_groups} built from the REAL c3_n46 step12000 checkpoint's param_groups
(structure/keys real, 188/64/2 params) with lr set from the closed form, cfg = the committed arm yaml (a real run's ckpt['cfg'] equalled its resolved yaml key for key: verified on the c3_n46 checkpoint, see output).
Run (venv python):  ~/.venvs/ramen/bin/python make_optfixtures_and_negctl_l1.py [real_step12000.pt]"""
import copy, math, subprocess, sys
from pathlib import Path
import torch, yaml
H = Path(__file__).resolve().parent; FX = H / "stopcheck_fixtures/opt"; FX.mkdir(parents=True, exist_ok=True)
REAL = sys.argv[1] if len(sys.argv) > 1 else "/mnt/DATA/DegradedDepth/checkpoints_v3/eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt"
real = torch.load(REAL, map_location="cpu", weights_only=False)
base = yaml.safe_load(open(H.parents[1] / "confirm/stage1/base/effective_ramen_n46.yaml"))
print("real c3_n46 step12000 ckpt: ckpt['cfg'] == resolved config yaml:", real["cfg"] == base, "| param groups:", [len(g["params"]) for g in real["optimizer"]["param_groups"]], [g["initial_lr"] for g in real["optimizer"]["param_groups"]])
assert real["cfg"] == base
def scale(g, K, W=500):
    s = g - 1; return (s + 1) / W if s < W else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - W) / (K - W))))
def smoke_cfg(arm, K, N):
    """the arm yaml after the sbatch's OWN sed: the 3 -e lines are EXTRACTED from train_template_l1.sbatch and executed by bash (not re-typed here)."""
    tpl = (H / "train_template_l1.sbatch").read_text().splitlines()
    i = next(j for j, l in enumerate(tpl) if l.lstrip().startswith("sed -e \"s/^  max_steps"))
    cmd = "\n".join(tpl[i:i + 3]); assert cmd.rstrip().endswith('"$CFG_REL" > "$RUN_CFG"'), cmd
    out = FX / f"smoke_{arm}_{K}_{N}.yaml"
    r = subprocess.run(["bash", "-c", f'ARM={arm}; K={K}; N={N}; CFG_REL={H}/configs/k{K}/arm_{arm}.yaml; RUN_CFG={out}\n{cmd}'], capture_output=True, text=True); assert r.returncode == 0, r.stderr
    return yaml.safe_load(open(out))
def fixture(arm, K, step, name, mut=None, cfg=None):
    cfg = cfg if cfg is not None else yaml.safe_load(open(H / f"configs/k{K}/arm_{arm}.yaml")); pg = copy.deepcopy(real["optimizer"]["param_groups"]); head = 1e-5 if arm == "S5" else 1e-4
    for g, il in zip(pg, (1e-4, head, 1e-3)): g["initial_lr"] = il; g["lr"] = il * scale(step, K)
    d = {"cfg": cfg, "step": step, "optimizer": {"param_groups": pg, "state": {}}}
    if mut: mut(d)
    p = FX / name; torch.save(d, p); return p
def run(label, p, arm, K, want_fail, extra=(), need=None, probe=None):
    """need: a substring that must appear on a `FAIL ...` line (so the NEG cannot pass because some OTHER check failed); probe: the probe script to run (default the shipped one)."""
    r = subprocess.run([sys.executable, str(probe or H / "check_optimizer_lr.py"), "--ckpt", str(p), "--arm", arm, "-K", str(K), *extra], capture_output=True, text=True)
    ok = (r.returncode == 1) == want_fail
    if need: ok = ok and any(l.startswith("FAIL") and need in l for l in r.stdout.splitlines())
    print(f"## {label}: rc={r.returncode} expected {'FAIL' if want_fail else 'no FAIL'} -> {'OK' if ok else 'WRONG'}")
    for l in [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("FAIL") or l.startswith("RESULT") or l.startswith("WAIT")][:3]: print("   ", l[:230])
    return ok
allok = True
ARMS = ["B0", "B0b", "S1", "S2", "S3", "S4", "S5", "S6"]
for K in (2000, 4000):
    for a in ARMS:
        for st in (120, 500, 1000, K):
            allok &= run(f"POSITIVE {a} K={K} step {st}", fixture(a, K, st, f"{a}_k{K}_s{st}.pt"), a, K, False, ("--expect-step", str(st)))
        allok &= run(f"POSITIVE -F final {a} K={K} stepK", fixture(a, K, K, f"step{K}.pt"), a, K, False, ("-F",)) if False else True
# ---- second review: sub-check ISOLATION for REDUNDANT checks (each alone must catch its own fault) + the smoke substitution-count check on its own ----
PROBE0 = (H / "check_optimizer_lr.py").read_text()
SC = 'len(pg) == 3 and all(abs(g["lr"] - g["initial_lr"] * sc) <= 1e-6 * max(g["initial_lr"] * sc, 1e-30) + 1e-18 for g in pg)'
INI = 'len(pg) == 3 and all(abs(g.get("initial_lr", -1) / w - 1) < 1e-9 for g, w in zip(pg, want_init))'
RAT = 'len(pg) == 3 and pg[0]["lr"] > 0 and abs(pg[1]["lr"] / pg[0]["lr"] - head / 1e-4) < 1e-6 and abs(pg[2]["lr"] / pg[0]["lr"] - 10) < 1e-6, ""'
def variant(name, removed):
    s = PROBE0
    for o_, n_ in removed:
        if s.count(o_) != 1: print(f"## WRONG isolation builder '{name}': check text not found exactly once in check_optimizer_lr.py ({o_[:50]!r})"); global allok; allok = False; return None
        s = s.replace(o_, n_)
    pth = FX / f"probe_{name}.py"; pth.write_text(s.replace("HERE = Path(__file__).resolve().parent", f"HERE = Path({str(H)!r})")); return pth
V_INI = variant("init_only", [(SC, "True"), (RAT, 'True, ""')]); V_RAT = variant("ratio_only", [(SC, "True"), (INI, "True")]); V_RNG = variant("range_only", [(SC, "True"), (INI, "True"), (RAT, 'True, ""')])
NOOP = lambda d: None
if V_INI:
    f = lambda d: d["optimizer"]["param_groups"][1].update(initial_lr=1e-4, lr=1e-4 * scale(1000, 2000))
    allok &= run("NEG[iso] initial_lr check ALONE: S5 head group at 1e-4 (lr consistent with it)", fixture("S5", 2000, 1000, "iso1.pt", f), "S5", 2000, True, need="initial_lr per group", probe=V_INI)
    f = lambda d: d["optimizer"]["param_groups"][2].update(initial_lr=1e-4, lr=1e-4 * scale(500, 2000))
    allok &= run("NEG[iso] initial_lr check ALONE: gate group not 10x", fixture("S2", 2000, 500, "iso2.pt", f), "S2", 2000, True, need="initial_lr per group", probe=V_INI)
if V_RAT:
    f = lambda d: d["optimizer"]["param_groups"][1].update(initial_lr=1e-4, lr=1e-4 * scale(1000, 2000))
    allok &= run("NEG[iso] ratio check ALONE: S5 head group lr ratio 1 instead of 0.1", fixture("S5", 2000, 1000, "iso3.pt", f), "S5", 2000, True, need="group lr ratios", probe=V_RAT)
    f = lambda d: d["optimizer"]["param_groups"][1].update(initial_lr=1e-5, lr=1e-5 * scale(1000, 2000))
    allok &= run("NEG[iso] ratio check ALONE: B0 head group at 0.1x", fixture("B0", 2000, 1000, "iso4.pt", f), "B0", 2000, True, need="group lr ratios", probe=V_RAT)
if V_RNG:
    allok &= run("NEG[iso] step-range check ALONE: step 2100 > K (lr consistent = 0)", fixture("S3", 2000, 2100, "iso5.pt"), "S3", 2000, True, need="step is positive and <= K", probe=V_RNG)
    allok &= run("NEG[iso] step-range check ALONE: step 0", fixture("B0", 2000, 0, "iso6.pt"), "B0", 2000, True, need="step is positive and <= K", probe=V_RNG)
# smoke substitution COUNT on its own: an arm yaml whose max_steps line is formatted so the sbatch's sed pattern does not match; the ckpt cfg is that same yaml unsubstituted -> the cfg compare PASSES, only the count check can FAIL
cdx = FX / "cfg_nomatch"; (cdx).mkdir(exist_ok=True); yt = (H / "configs/k2000/arm_B0.yaml").read_text(); assert "  max_steps: 2000\n" in yt
(cdx / "arm_B0.yaml").write_text(yt.replace("  max_steps: 2000\n", "  max_steps:  2000\n")); c_nm = yaml.safe_load((cdx / "arm_B0.yaml").read_text())
allok &= run("NEG SMOKE[iso] substitution count: the sbatch sed would not match max_steps (cfg compare alone would PASS)", fixture("B0", 2000, 120, "iso7.pt", cfg=c_nm), "B0", 2000, True, ("--smoke", "120", "--expect-step", "120", "--cfg-dir", str(cdx)), need="smoke substitutions each apply exactly once")
def neg(label, arm, K, step, mut, extra=()):
    global allok; allok &= run("NEG " + label, fixture(arm, K, step, "neg.pt", mut), arm, K, True, extra)
def set_lr(f): return lambda d: [g.__setitem__("lr", f(g)) for g in d["optimizer"]["param_groups"]]
# ---- B1 (second review): the smoke checkpoint carries the SMOKE config (sbatch sed), --smoke N must accept exactly that and nothing else ----
sm0 = smoke_cfg("B0", 2000, 120); sm_arm = yaml.safe_load(open(H / "configs/k2000/arm_B0.yaml"))
assert sm0["training"]["max_steps"] == 120 and sm0["training"]["checkpoint_dir"] == "runs/l1_smoke_B0_k2000" and sm0["training"]["checkpoint_at_steps"] == [120]
for K in (2000, 4000):
    for a in ARMS:
        allok &= run(f"POSITIVE SMOKE {a} K={K} (cfg = the sbatch sed of the arm yaml, step 120)", fixture(a, K, 120, f"smoke_{a}_{K}.pt", cfg=smoke_cfg(a, K, 120)), a, K, False, ("--smoke", "120", "--expect-step", "120", "-F"))
allok &= run("FORCED-FAIL (the second review's B1): the same correct smoke checkpoint WITHOUT --smoke is rejected (full-run yaml compared)", fixture("S5", 2000, 120, "smoke_nos.pt", cfg=smoke_cfg("S5", 2000, 120)), "S5", 2000, True, ("--expect-step", "120"))
def smneg(label, arm, K, mutcfg, extra=("--smoke", "120", "--expect-step", "120"), step=120):
    global allok
    c = smoke_cfg(arm, K, 120); mutcfg(c); allok &= run("NEG SMOKE " + label, fixture(arm, K, step, "smneg.pt", cfg=c), arm, K, True, extra)
smneg("one UNDECLARED key changed (data.seed 47)", "S5", 2000, lambda c: c["data"].__setitem__("seed", 47))
smneg("one undeclared key changed (training.head_lr_mult)", "S5", 2000, lambda c: c["training"].__setitem__("head_lr_mult", 1.0))
smneg("an undeclared extra key added", "S1", 2000, lambda c: c["training"].__setitem__("zzz", 1))
smneg("a declared key not substituted: max_steps still 2000", "S5", 2000, lambda c: c["training"].__setitem__("max_steps", 2000))
smneg("a declared key not substituted: checkpoint_dir is the full-run dir", "S5", 2000, lambda c: c["training"].__setitem__("checkpoint_dir", "runs/l1_S5_k2000"))
smneg("a declared key not substituted: checkpoint_at_steps still [2000]", "S5", 2000, lambda c: c["training"].__setitem__("checkpoint_at_steps", [2000]))
smneg("smoke dir of ANOTHER arm", "S5", 2000, lambda c: c["training"].__setitem__("checkpoint_dir", "runs/l1_smoke_S1_k2000"))
smneg("smoke of 60 steps declared as --smoke 120", "S5", 2000, lambda c: c["training"].__setitem__("max_steps", 60), ("--smoke", "120", "--expect-step", "120"))
smneg("resume_from set in a smoke config", "S1", 2000, lambda c: c["training"].__setitem__("resume_from", "/x.pt"))
allok &= run("NEG SMOKE: --smoke 120 on a FULL-run checkpoint (cfg = arm yaml)", fixture("S5", 2000, 120, "full_as_smoke.pt"), "S5", 2000, True, ("--smoke", "120", "--expect-step", "120"))
neg("S5 head group NOT scaled (initial_lr 1e-4)", "S5", 2000, 1000, lambda d: d["optimizer"]["param_groups"][1].update(initial_lr=1e-4, lr=1e-4 * scale(1000, 2000)))
neg("B0 head group wrongly at 0.1x", "B0", 2000, 1000, lambda d: d["optimizer"]["param_groups"][1].update(initial_lr=1e-5, lr=1e-5 * scale(1000, 2000)))
neg("S5 head lr not decayed (stale at initial)", "S5", 2000, 1500, lambda d: d["optimizer"]["param_groups"][1].update(lr=1e-5))
neg("gate group not 10x (initial 1e-4)", "S2", 2000, 500, lambda d: d["optimizer"]["param_groups"][2].update(initial_lr=1e-4, lr=1e-4 * scale(500, 2000)))
neg("gate group lr frozen at 0", "S3", 2000, 500, lambda d: d["optimizer"]["param_groups"][2].update(lr=0.0))
neg("no decay: lr == initial at step 1500", "B0", 2000, 1500, set_lr(lambda g: g["initial_lr"]))
neg("12000-step cosine saved lr at step 1000", "B0b", 2000, 1000, set_lr(lambda g: g["initial_lr"] * 0.5 * (1 + math.cos(math.pi * (1000 - 500) / 11500))))
neg("lr x2", "S1", 2000, 1000, set_lr(lambda g: g["lr"] * 2))
neg("only 2 param groups", "S4", 2000, 500, lambda d: d["optimizer"]["param_groups"].pop())
neg("wrong group sizes", "S4", 2000, 500, lambda d: d["optimizer"]["param_groups"][0].__setitem__("params", d["optimizer"]["param_groups"][0]["params"][:-1]))
neg("executed cfg differs: S6 cfg lacks the far keys", "S6", 2000, 500, lambda d: [d["cfg"]["loss"].pop("metric_silog_far_weight"), d["cfg"]["loss"].pop("metric_silog_far_from")])
neg("executed cfg differs: S5 cfg has head_lr_mult 1.0 (optimizer says 1e-5)", "S5", 2000, 500, lambda d: d["cfg"]["training"].__setitem__("head_lr_mult", 1.0))
neg("executed cfg differs: B0b cfg has seed 46", "B0b", 2000, 500, lambda d: d["cfg"]["data"].__setitem__("seed", 46))
neg("executed cfg differs: K=4000 yaml in a K=2000 check", "B0", 2000, 500, lambda d: d["cfg"]["training"].__setitem__("max_steps", 4000))
neg("executed cfg has a resume_from", "S2", 2000, 500, lambda d: d["cfg"]["training"].__setitem__("resume_from", "/x.pt"))
neg("step beyond K", "S3", 2000, 2100, None)
neg("expected step mismatch (--expect-step 1000, ckpt at 500)", "B0", 2000, 500, None, ("--expect-step", "1000"))
neg("final snapshot but not at K (-F on step2000.pt saved at step 1500)", "B0", 2000, 1500, None, ("-F",)) if False else None
# final-mode: file named stepK.pt but saved at step 1500
import shutil
p = fixture("B0", 2000, 1500, "step2000.pt"); allok &= run("NEG -F: step2000.pt whose saved step is 1500", p, "B0", 2000, True, ("-F",))
p = fixture("B0", 2000, 2000, "step2000.pt"); allok &= run("POSITIVE -F: step2000.pt saved at 2000", p, "B0", 2000, False, ("-F",))
# partial file: WAIT (rc 0) without -F, FAIL with -F
(FX / "partial.pt").write_bytes((FX / "step2000.pt").read_bytes()[:200])
allok &= run("POSITIVE-WAIT: truncated file => WAIT, rc 0", FX / "partial.pt", "B0", 2000, False)
allok &= run("NEG -F: truncated file => FAIL", FX / "partial.pt", "B0", 2000, True, ("-F",))
print("\nOPTIMIZER-PROBE NEGATIVE-CONTROL SUITE:", "ALL AS EXPECTED" if allok else "UNEXPECTED RESULT"); sys.exit(0 if allok else 1)

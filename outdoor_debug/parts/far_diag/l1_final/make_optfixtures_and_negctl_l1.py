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
def fixture(arm, K, step, name, mut=None):
    cfg = yaml.safe_load(open(H / f"configs/k{K}/arm_{arm}.yaml")); pg = copy.deepcopy(real["optimizer"]["param_groups"]); head = 1e-5 if arm == "S5" else 1e-4
    for g, il in zip(pg, (1e-4, head, 1e-3)): g["initial_lr"] = il; g["lr"] = il * scale(step, K)
    d = {"cfg": cfg, "step": step, "optimizer": {"param_groups": pg, "state": {}}}
    if mut: mut(d)
    p = FX / name; torch.save(d, p); return p
def run(label, p, arm, K, want_fail, extra=()):
    r = subprocess.run([sys.executable, str(H / "check_optimizer_lr.py"), "--ckpt", str(p), "--arm", arm, "-K", str(K), *extra], capture_output=True, text=True)
    ok = (r.returncode == 1) == want_fail
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
def neg(label, arm, K, step, mut, extra=()):
    global allok; allok &= run("NEG " + label, fixture(arm, K, step, "neg.pt", mut), arm, K, True, extra)
def set_lr(f): return lambda d: [g.__setitem__("lr", f(g)) for g in d["optimizer"]["param_groups"]]
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

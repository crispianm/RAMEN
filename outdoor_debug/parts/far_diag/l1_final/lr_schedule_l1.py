"""L1 numeric LR schedule from the REAL train.make_lr_scale applied to each committed arm yaml, vs an independent closed form; prints the table that goes into L1_FINAL.md and the stop check.
printed `lr=` at step G (group 0) = initial_lr * scale(G-1), because the trainer sets g['lr'] BEFORE `gstep += 1`.  Run from the repo root: ~/.venvs/ramen/bin/python outdoor_debug/parts/far_diag/l1_final/lr_schedule_l1.py"""
import math, sys
from pathlib import Path
import yaml
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO)); HERE = Path(__file__).resolve().parent
import train
def closed(s, K, W=500):   # independent: linear warmup (s+1)/W, then half-cosine to 0 at K
    return (s + 1) / W if s < W else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - W) / (K - W))))
ok = True
for K in (2000, 4000):
    scales = {}
    for a in ("B0", "B0b", "S1", "S2", "S3", "S4", "S5", "S6"):
        t = yaml.safe_load(open(HERE / f"configs/k{K}/arm_{a}.yaml"))["training"]
        f = train.make_lr_scale(t, int(t["warmup_steps"]), int(t["max_steps"])); scales[a] = [f(s) for s in range(K)]
        assert t["lr_decay_steps"] == K == t["max_steps"] and t["warmup_steps"] == 500 and "lr_restart_step" not in t
    same = all(scales[a] == scales["B0"] for a in scales); ok &= same
    err = max(abs(scales["B0"][s] - closed(s, K)) for s in range(K)); ok &= err < 1e-12
    print(f"## K={K}: 8 arms identical schedule: {same}; max |real - closed form| over all {K} steps = {err:.2e}")
    print("printed step G | lr group0 (1e-4*scale(G-1)) | S5 head group (1e-5*scale) | gate group (1e-3*scale) | note")
    pts = [1, 20, 40, 100, 250, 480, 500, 501, 520, 600, 1000, K // 2, K - 520, K - 100, K - 20, K - 1, K]
    for G in sorted(set(p for p in pts if 1 <= p <= K)):
        s = scales["B0"][G - 1]; note = "warmup" if G <= 500 else ("final step" if G == K else "cosine")
        print(f"{G:6d} | {1e-4 * s:.3e} | {1e-5 * s:.3e} | {1e-3 * s:.3e} | {note}")
    print()
print("RESULT:", "OK" if ok else "MISMATCH"); sys.exit(0 if ok else 1)

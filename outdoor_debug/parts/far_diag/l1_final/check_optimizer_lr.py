#!/usr/bin/env python3
"""L1 checkpoint-side LR/feature probe (the stdout `lr=` shows GROUP 0 only, so the head group of S5 and the gate group are checked HERE, on what the trainer actually saved).
usage (cluster venv python; torch + pyyaml; the checkpoint is our own file, loaded with weights_only=False):
  check_optimizer_lr.py --ckpt runs/l1_<ARM>_k<K>/{last.pt|step<K>.pt|step<S>.pt} --arm ARM -K 2000|4000 [--cfg-dir DIR] [--expect-step N] [-F]
Checks (each FAIL => exit 1; an unreadable/partial file is WAIT (exit 0) unless -F):
  1 the config the TRAINER EXECUTED (ckpt['cfg']) == the committed arm yaml, key for key
  2 optimizer: exactly 3 param groups with (188, 64, 2) params; initial_lr = (1e-4, 1e-4 | 1e-5 for S5, 1e-3) read from the saved state
  3 saved lr of EVERY group == initial_lr * closed-form scale(step-1) (warmup 500 then cosine to 0 at K): the schedule really multiplied all groups (incl. the S5 head group and the gate group)
  4 step: the checkpoint's own step (== --expect-step when given; with -F for a final stepK.pt it must be K)
Needs only ckpt['cfg'], ckpt['step'], ckpt['optimizer']['param_groups'] (the optimizer state itself is not inspected)."""
import argparse, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
def scale(g, K, W=500):
    s = g - 1
    return (s + 1) / W if s < W else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - W) / (K - W))))
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--arm", required=True); ap.add_argument("-K", type=int, required=True, choices=(2000, 4000))
    ap.add_argument("--cfg-dir"); ap.add_argument("--expect-step", type=int); ap.add_argument("-F", action="store_true"); a = ap.parse_args(argv)
    import torch, yaml
    res = []
    def chk(n, ok, d=""): res.append((n, bool(ok), d))
    try:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg, step, pg = ck["cfg"], int(ck["step"]), ck["optimizer"]["param_groups"]
    except Exception as e:
        print(("FAIL " if a.F else "WAIT ") + f"checkpoint unreadable/partial ({type(e).__name__}: {str(e)[:120]}): re-run later"); return 1 if a.F else 0
    cfgp = Path(a.cfg_dir or HERE / f"configs/k{a.K}") / f"arm_{a.arm}.yaml"; want = yaml.safe_load(open(cfgp))
    chk("executed config (ckpt['cfg']) == committed arm yaml", cfg == want, "differs: " + str(sorted(k for k in set(cfg) | set(want) if cfg.get(k) != want.get(k)))[:200])
    chk("optimizer has exactly 3 param groups with (188, 64, 2) params", [len(g["params"]) for g in pg] == [188, 64, 2], str([len(g["params"]) for g in pg]))
    head = 1e-5 if a.arm == "S5" else 1e-4; want_init = [1e-4, head, 1e-3]
    chk(f"initial_lr per group == {want_init} (S5: head group x0.1; gate group 1e-3 = 10x)", len(pg) == 3 and all(abs(g.get("initial_lr", -1) / w - 1) < 1e-9 for g, w in zip(pg, want_init)), str([g.get("initial_lr") for g in pg]))
    sc = scale(step, a.K)
    chk(f"every group's saved lr == initial_lr * scale({step - 1}) = *{sc:.6e} (warmup 500, cosine to 0 at {a.K})", len(pg) == 3 and all(abs(g["lr"] - g["initial_lr"] * sc) <= 1e-6 * max(g["initial_lr"] * sc, 1e-30) + 1e-18 for g in pg),
        str([(g["lr"], g["initial_lr"] * sc) for g in pg]))
    chk("group lr ratios 1 : head : 10 (S5 head 0.1)", len(pg) == 3 and pg[0]["lr"] > 0 and abs(pg[1]["lr"] / pg[0]["lr"] - head / 1e-4) < 1e-6 and abs(pg[2]["lr"] / pg[0]["lr"] - 10) < 1e-6, "")
    if a.expect_step is not None: chk(f"step == {a.expect_step}", step == a.expect_step, f"got {step}")
    if a.F and Path(a.ckpt).name == f"step{a.K}.pt": chk(f"final snapshot step == {a.K}", step == a.K, f"got {step}")
    chk("step is positive and <= K", 0 < step <= a.K, f"got {step}")
    for n, ok, d in res: print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   [{d}]"))
    nf = sum(1 for _, ok, _ in res if not ok); print(f"RESULT: {'ALL PASS' if nf == 0 else str(nf) + ' FAIL'} ({len(res)} checks) step={step} arm={a.arm} K={a.K}"); return 0 if nf == 0 else 1
if __name__ == "__main__":
    sys.exit(main())

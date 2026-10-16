"""Partial-cache exactness (no scoring, synthetic 54-frame clip, no timing). Window 1 = frames 0..31; window 2 as bench_joint.vda_windows:
cur2 = frames[22:54] with cur2[:10] = cur1[KEYFRAMES]. Question: are R (stage1/2), consumed input and encoder tokens at the carried slots
BITWISE equal to window-1's at KEYFRAMES[j]?  C3: restorer slots 4..7 (audit) ; encoder = C3 consumed input slots 4..7 ; Metric-VDA-S: encoder all 10 slots.
  test_cache_exact.py --model c3|mvda --prec fp32|fp16 --variant base|e1e2"""
import argparse, json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import run_variant as RV
import patches
KEY = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--prec"); ap.add_argument("--variant"); ap.add_argument("--h", type=int, default=518)
a = ap.parse_args()
import subprocess
if any(l and int(l.rsplit(",", 1)[1]) >= 1024 for l in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True).strip().splitlines()):
    raise SystemExit("GPU busy")
flags = RV.VARIANTS[a.variant]
fwd, log, net = RV.build(a.model, flags)
frames = RV.make_input(a.h, a.h, 54 + 0).cuda()
cur1 = frames[0:32].clone(); cur2 = frames[22:54].clone(); cur2[:10] = cur1[KEY]
res = {"model": a.model, "prec": a.prec, "variant": a.variant}
cap = {}
enc = (net.vda.core if a.model == "c3" else net).pretrained
orig = enc.get_intermediate_layers
def wrap(*args, **kw):
    out = orig(*args, **kw); cap["tok"] = [(f.detach().clone(), c.detach().clone()) for f, c in out]; return out
enc.get_intermediate_layers = wrap
def run(cur):
    cap.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=a.prec == "fp16"):
        if a.model == "c3":
            rgb = cur.permute(1, 0, 2, 3)[None]
            R = net.restore_batch(rgb)
            # stage-1 only, for the stage split
            R1 = net.restore_clip(cur, None)
            C = net.consume(rgb, R)
            net._depth_and_feats(C)
            return dict(R1=R1.float(), R=R[0].permute(1, 0, 2, 3).float(), C=C[0].permute(1, 0, 2, 3).float(), tok=cap["tok"])
        fwd(cur)
        return dict(C=cur.float(), tok=cap["tok"])
o1, o2 = run(cur1), run(cur2)
def eq(x, y): return bool(torch.equal(x, y)), float((x - y).abs().max())
slots = {"R1": list(range(3, 9)), "R": list(range(4, 8)), "C": list(range(4, 8)), "tok": list(range(4, 8))} if a.model == "c3" else {"C": list(range(10)), "tok": list(range(10))}
if a.model == "c3":
    slots["R1_all10"] = list(range(10)); slots["R_all10"] = list(range(10)); slots["tok_all10"] = list(range(10))
for k, sl in slots.items():
    base = k.replace("_all10", "")
    ok, mx = True, 0.0
    per = {}
    for j in sl:
        i1 = KEY[j]
        if base == "tok":
            pairs = [(x2[0][j], x1[0][i1]) for x1, x2 in zip(o1["tok"], o2["tok"])] + [(x2[1][j], x1[1][i1]) for x1, x2 in zip(o1["tok"], o2["tok"])]
        else:
            pairs = [(o2[base][j], o1[base][i1])]
        e = [eq(p, q) for p, q in pairs]
        per[j] = all(b for b, _ in e); mx = max(mx, max(m for _, m in e))
    res[k] = {"slots": sl, "bitwise_per_slot": per, "all_bitwise": all(per.values()), "max_abs": mx}
print(json.dumps(res))
json.dump(res, open(f"{HERE}/out/cache_{a.model}_{a.variant}_{a.prec}.json", "w"), indent=1)

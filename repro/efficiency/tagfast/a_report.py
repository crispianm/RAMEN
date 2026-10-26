import os
import sys, json, numpy as np, torch
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("RAMEN_SCORER_WORKTREE", "."))
import subprocess; print("scorer HEAD", subprocess.check_output(["git","-C",os.environ.get("RAMEN_SCORER_WORKTREE", "."),"rev-parse","HEAD"],text=True).strip())
from evaluation.depth_probe import depth_agf, depth_agf_torch
from repro.tag_fast import agf_v2, agf_v3, Agf3Stream
rng = np.random.default_rng(20261013)
def gen(kind, T, hw):
    H, W = hw; t = np.arange(T, dtype=np.float32)[:, None, None]; one = np.ones((T, H, W), np.float32)
    if kind == "constant": return 7.0*one
    if kind == "step_up": return (3.0 + 4.0*(t >= T//2))*one
    if kind == "step_down": return (7.0 - 4.0*(t >= T//2))*one
    if kind == "ramp": return (1.0 + 20.0*t/max(T-1, 1))*one
    if kind == "noise": return (7.0*(1+0.03*rng.standard_normal((T, H, W)))).astype(np.float32)
    if kind == "near0": return (1e-6*rng.random((T, H, W)) + 1e-7*rng.standard_normal((T, H, W))).astype(np.float32).clip(0)
    if kind == "near0_mix": a = (rng.random((T, H, W))*3e-6).astype(np.float32); a[:, :H//2] = 2.0*(1+0.1*rng.standard_normal((T, H//2, W))); return a.astype(np.float32)
    if kind == "far655": return (655.0*(1+0.02*rng.standard_normal((T, H, W)))).astype(np.float32)
    if kind == "zeros": a = (5.0*(1+0.05*rng.standard_normal((T, H, W)))).astype(np.float32); a[:, ::7, ::5] = 0; a[T//2:, 3:9, 3:9] = 0; return a
    if kind == "wide": return np.exp(rng.uniform(np.log(0.1), np.log(655), (1, H, W))).astype(np.float32)*(1+0.1*rng.standard_normal((T, H, W))).astype(np.float32)
KINDS = ["constant","step_up","step_down","ramp","noise","near0","near0_mix","far655","zeros","wide"]
SET = [(0.2, 0.06), (0.5, 0.12)]
def stats(o, ref):
    r = torch.from_numpy(ref).cuda(); mx = 0.0; ma = 0.0; sub = []; fin = True
    for a in range(0, r.shape[0], 50):
        rr = r[a:a+50].double(); oo = o[a:a+50].double()
        e = (oo-rr).abs(); rel = e/rr.abs().clamp_min(1e-6)
        mx = max(mx, float(rel.max())); ma = max(ma, float(e.max())); fin &= bool(torch.isfinite(oo).all()); sub.append(rel.flatten()[::7].float().cpu())
    q = torch.cat(sub).numpy()
    return dict(max=mx, p999=float(np.quantile(q, 0.999)), med=float(np.median(q)), max_abs_m=ma, finite=fin)
rows = []; worst = {}
def run(kind, T, hw):
    x = gen(kind, T, hw); xt = torch.from_numpy(x).cuda()
    for amin, tau in SET:
        ref = depth_agf(x, amin, tau)
        # reference output is fp32 (dtype of input); also compare against it as scored
        outs = {"V1_torch_fp64": lambda: depth_agf_torch(xt, amin, tau), "V2_torch_fp32": lambda: agf_v2(xt, amin, tau), "V3_triton": lambda: agf_v3(xt, amin, tau)}
        # V3s streaming, windows of 32 with state carried across windows (one Agf3Stream for the whole stitched sequence)
        st = Agf3Stream(xt[0], amin, tau); sl = [xt[0].float()]
        for i in range(1, T): sl.append(st.step(xt[i]))
        s3 = torch.stack(sl); del sl
        v3 = agf_v3(xt, amin, tau); bit = bool(torch.equal(s3, v3)); del v3
        outs["V3s_stream_carried"] = lambda: s3
        # V3 window-by-window: chunking through carried-state stream equals the full-sequence kernel (checked above); a naive per-window reset is class A and is NOT used.
        for n, fo in outs.items():
            o = fo(); s = stats(o, ref); del o; torch.cuda.empty_cache()
            s.update(kind=kind, T=T, hw=list(hw), setting=f"{amin}/{tau}", variant=n); rows.append(s)
            w = worst.setdefault(n, 0.0); worst[n] = max(w, s["max"])
        rows[-1]["stream_bitexact_vs_V3"] = bit
for kind in KINDS:
    for T in (1, 2, 32, 33):
        run(kind, T, (518, 518))
    run(kind, 33, (374, 1242))
for kind in ("noise", "step_up", "near0_mix", "zeros"):
    run(kind, 1000, (187, 621))
# bit-exactness stream vs window over all
print("stream==V3 bit-exact in all runs:", all(r.get("stream_bitexact_vs_V3", True) for r in rows))
print("worst max rel per variant:", worst)
for n in worst:
    sel = [r for r in rows if r["variant"] == n]
    print(n, "max", max(r["max"] for r in sel), "worst p99.9", max(r["p999"] for r in sel), "worst median", max(r["med"] for r in sel), "all finite", all(r["finite"] for r in sel))
json.dump(rows, open("a_rows.json", "w"))
# NaN / inf behaviour
x = gen("noise", 12, (32, 40)); x[5, 3, 3] = np.nan; x[7, 10, 10] = np.inf; x[2, 20, 20] = -np.inf
xt = torch.from_numpy(x).cuda()
import warnings
with warnings.catch_warnings(), np.errstate(all="ignore"):
    ref = depth_agf(x, 0.2, 0.06)
for n, f in {"V1": lambda: depth_agf_torch(xt, .2, .06), "V2": lambda: agf_v2(xt, .2, .06), "V3": lambda: agf_v3(xt, .2, .06)}.items():
    o = f().cpu().numpy()
    print("NaN test", n, "nan-mask equal to ref:", np.array_equal(np.isnan(o), np.isnan(ref)), "n_nan", int(np.isnan(o).sum()), "ref n_nan", int(np.isnan(ref).sum()), "inf-mask equal", np.array_equal(np.isinf(o), np.isinf(ref)))
    d = np.isnan(o) != np.isnan(ref)
    if d.any(): print("   differing px (frame,y,x) first:", np.argwhere(d)[:3].tolist())

import sys, os, json, time
sys.dont_write_bytecode = True
WT = "/mnt/DATA/DegradedDepth/RAMEN/.claude/worktrees/paper-repro"
sys.path.insert(0, WT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, triton, triton.language as tl
import bench_joint as bj
from evaluation import depth_probe as dp
from bench import agf_f32, agf_triton, _k
CK = os.environ.get("RAMEN_CKPT", "/mnt/DATA/DegradedDepth/checkpoints_release/ramen.pt")
NCLIP = int(os.environ.get("NCLIP", 3))
PARAMS = {"agf0.2t0.06": (0.2, 0.06), "agf0.5t0.12": (0.5, 0.12)}

@triton.jit
def _kstep(d_ptr, acc_ptr, s_ptr, o_ptr, P, amin, k, beta, BLOCK: tl.constexpr):
    offs = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK); m = offs < P
    acc = tl.load(acc_ptr+offs, mask=m, other=1.0); s = tl.load(s_ptr+offs, mask=m, other=0.0)
    d = tl.load(d_ptr+offs, mask=m, other=1.0).to(tl.float32)
    u = (d-acc)/tl.maximum(tl.abs(acc), 1e-6)
    s = beta*s + (1-beta)*u*u
    a = amin + (1-amin)*(1 - tl.exp(-(s*k)))
    acc = acc + a*(d-acc)
    tl.store(acc_ptr+offs, acc, mask=m); tl.store(s_ptr+offs, s, mask=m); tl.store(o_ptr+offs, acc, mask=m)

class Stream:
    def __init__(s, first, amin, tau, beta=0.5):
        s.acc = first.float().clone().contiguous(); s.s = torch.zeros_like(s.acc); s.P = s.acc.numel(); s.p = (amin, 1/(tau*tau), beta)
    def step(s, d):
        o = torch.empty_like(s.acc); _kstep[(triton.cdiv(s.P, 256),)](d.contiguous(), s.acc, s.s, o, s.P, *s.p, BLOCK=256); return o

def tri_stream(x, amin, tau):
    st = Stream(x[0], amin, tau); out = [x[0].float()]
    for t in range(1, x.shape[0]): out.append(st.step(x[t]))
    return torch.stack(out)

def tri_win(x, amin, tau, o32=False):
    # note bench.agf_triton uses module-level defaults for beta; pass amin/tau
    return agf_triton(x, False, amin, tau)

def variants(depth_np, amin, tau):
    x = torch.from_numpy(depth_np).cuda()
    h = x.half()
    out = {}
    out["np_ref"] = dp.depth_agf(depth_np, amin, tau)
    out["torch_fp64"] = dp.depth_agf_torch(x, amin, tau).cpu().numpy()
    out["loop_fp32"] = agf_f32(x, amin, tau).cpu().numpy()
    out["triton_fp32"] = agf_triton(x, False, amin, tau).cpu().numpy()
    out["triton_stream"] = tri_stream(x, amin, tau).cpu().numpy()
    out["loop_fp32_fp16in"] = agf_f32(h, amin, tau).float().cpu().numpy()   # NB output cast to half by empty_like
    out["triton_fp16in"] = agf_triton(h, False, amin, tau).float().cpu().numpy()  # fp16 out too
    # fp16 input, fp32 output (state fp32)
    out["triton_fp16in_f32out"] = agf_triton(h.float(), False, amin, tau).cpu().numpy()
    out["_bitexact_stream_vs_window"] = bool(np.array_equal(out["triton_stream"], out["triton_fp32"]))
    return out

def main():
    p = bj.build_pipeline(f"ours:{CK}", "cuda")
    rows = []
    cases = [("kitti", "llvar"), ("bonn", "llvar"), ("scannetpp", "llvar"), ("kitti", "st_llvar"), ("bonn", "st_llvar"), ("scannetpp", "st_llvar")]
    tof = None
    for ds, c in cases:
        proto = bj.DATASETS[ds]
        clips = bj.select_clips(ds, "dev", NCLIP)
        for key, frames in clips:
            clean, raw_hw = bj.load_frames(ds, frames); gt = bj.load_gt(ds, frames)
            cam = bj.camera(ds, key, frames[:proto["max_len"]]) if (ds not in bj.OPTIN_DATASETS and c not in bj.STATIC_CELLS) else None
            blur, sigma = bj.parse_cell(c)
            static = c in bj.STATIC_CELLS
            if static:
                sclean, sgt = bj.static_clip(clean, gt)
                x = bj.degrade_static(sclean, c, bj.noise_seed(ds, key, bj.seed_cell(c))).to("cuda")
            else:
                draw = bj.llvar_draw(ds, key)
                x = bj.degrade_llvar(clean, c, draw, bj.llvar_noise_seed(ds, key, c)).to("cuda")
            assert "depth_post" not in p
            depth, restored = bj.run_pipeline(p, x, sigma, blur, raw_hw)
            print(ds, c, key, depth.shape, depth.dtype, flush=True)
            torch.cuda.empty_cache()
            for tok, (amin, tau) in PARAMS.items():
                vs = variants(depth, amin, tau)
                bit = vs.pop("_bitexact_stream_vs_window")
                ref = vs["np_ref"].astype(np.float64)
                # also the "unfiltered" for context
                allv = {"raw_unfiltered": depth, **vs}
                recs = {}
                for n, dv in allv.items():
                    dv = np.ascontiguousarray(dv.astype(np.float32))
                    err = np.abs(dv.astype(np.float64) - ref)
                    rec = {"max_abs_m": float(err.max()), "mean_abs_m": float(err.mean()),
                           "max_rel": float((err/np.maximum(np.abs(ref), 1e-6)).max()), "mean_rel": float((err/np.maximum(np.abs(ref), 1e-6)).mean())}
                    if static:
                        r = bj.score_clip(dv, p["metric"], sgt, restored, x, sclean.to("cuda"), proto["max_depth"], proto["min_depth"], None, None, bj.ll_cfg(blur)["ev"], sky_m=proto.get("sky_depth_m"))
                        r.update(bj.flicker_static(dv, sgt, proto["max_depth"], proto["min_depth"]))
                    else:
                        sd, sg, sr, sx, sc, scam = bj._vda5.score_slice(proto, dv, gt, restored, x, clean.to("cuda"), cam)
                        r = bj.score_clip(sd, p["metric"], sg, sr, sx, sc, proto["max_depth"], proto["min_depth"], scam, None, draw["ev"], sky_m=proto.get("sky_depth_m"))
                    rec["metrics"] = {k: r[k] for k in ("abs_rel_aligned", "abs_rel_unaligned", "tae", "eps_d", "flicker", "abs_rel_static", "delta1_aligned") if k in r}
                    recs[n] = rec
                rows.append({"ds": ds, "cell": c, "clip": key, "tok": tok, "stream_bitexact_vs_window": bit, "variants": recs})
            json.dump(rows, open("val_rows.json", "w"))
main()

import os
import sys, time, torch, triton, triton.language as tl
sys.path.insert(0, os.environ.get("RAMEN_SCORER_WORKTREE", "."))
from evaluation.depth_probe import depth_agf_torch
AMIN, TAU, BETA = 0.2, 0.06, 0.5

@torch.no_grad()
def agf_f32(x, amin=AMIN, tau=TAU, beta=BETA):
    out = torch.empty_like(x); acc = x[0].float(); out[0] = acc; s = torch.zeros_like(acc); k = 1/(tau*tau)
    for t in range(1, x.shape[0]):
        d = x[t].float(); u = (d-acc)/acc.abs().clamp_min(1e-6)
        s = beta*s + (1-beta)*u*u
        a = amin + (1-amin)*(1-torch.exp(-s*k)); acc = a*d+(1-a)*acc; out[t] = acc
    return out

@triton.jit
def _k(x_ptr, o_ptr, N, P, amin, k, beta, RATIONAL: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid*BLOCK + tl.arange(0, BLOCK); m = offs < P
    acc = tl.load(x_ptr+offs, mask=m, other=1.0).to(tl.float32)
    tl.store(o_ptr+offs, acc, mask=m)
    s = tl.zeros([BLOCK], dtype=tl.float32)
    for t in range(1, N):
        d = tl.load(x_ptr+t*P+offs, mask=m, other=1.0).to(tl.float32)
        u = (d-acc)/tl.maximum(tl.abs(acc), 1e-6)
        s = beta*s + (1-beta)*u*u
        z = s*k
        if RATIONAL:
            g = z/(1+z)
        else:
            g = 1 - tl.exp(-z)
        a = amin + (1-amin)*g
        acc = acc + a*(d-acc)
        tl.store(o_ptr+t*P+offs, acc, mask=m)

def agf_triton(x, rational=False, amin=AMIN, tau=TAU, beta=BETA):
    x = x.contiguous(); o = torch.empty_like(x); N = x.shape[0]; P = x[0].numel()
    _k[(triton.cdiv(P, 256),)](x, o, N, P, amin, 1/(tau*tau), beta, RATIONAL=rational, BLOCK=256)
    return o

def timeit(f, x, n=20):
    for _ in range(3): f(x)
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(n): f(x)
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1)/n/x.shape[0]

if __name__ == "__main__":
    torch.manual_seed(0)
    # synthetic metric-depth-like window: smooth scene + noise-driven flicker + one real change
    N, H = 32, 518
    base = 5 + 3*torch.rand(1, H, H, device="cuda")
    x = base * (1 + 0.03*torch.randn(N, H, H, device="cuda"))
    x[20:, : H//2] *= 1.3
    ref = depth_agf_torch(x, AMIN, TAU)
    variants = {"ref fp64 loop": lambda v: depth_agf_torch(v, AMIN, TAU), "fp32 loop": agf_f32,
                "triton fp32 exact-exp": agf_triton, "triton fp32 rational": lambda v: agf_triton(v, True)}
    for n, f in variants.items():
        o = f(x); err = (o.double()-ref.double()).abs()
        print(f"{n:24s} {timeit(f, x):8.4f} ms/frame  max|d|={err.max().item():.2e} m  mean|d|={err.mean().item():.2e}  rel={(err/ref.double().abs()).max().item():.2e}")

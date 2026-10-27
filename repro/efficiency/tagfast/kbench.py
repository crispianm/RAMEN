import os
import sys, subprocess, json, numpy as np, torch
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("RAMEN_SCORER_WORKTREE", "."))
from evaluation.depth_probe import depth_agf, depth_agf_torch
from repro.tag_fast import agf_v2, agf_v3, Agf3Stream
procs = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader,nounits"], text=True).strip().splitlines()
assert not [l for l in procs if int(l.rsplit(",",1)[1]) >= 1024], procs
N,H = 32,518
g = torch.Generator(device="cuda").manual_seed(0)
base = 0.5 + 9*torch.rand(4,1,H,H,device="cuda",generator=g)
xs = [base[i]*(1+0.04*torch.randn(N,H,H,device="cuda",generator=g)) for i in range(4)]
def stream_win(x, a, t):
    st = Agf3Stream(x[0], a, t); return torch.stack([x[0]] + [st.step(x[i]) for i in range(1, N)])
res = {}
for amin, tau in [(0.2,0.06),(0.5,0.12)]:
    fns = {"cpu_numpy+D2H": lambda x: torch.from_numpy(depth_agf(x.cpu().numpy(), amin, tau)),
           "V1_torch_fp64": lambda x: depth_agf_torch(x, amin, tau),
           "V2_torch_fp32": lambda x: agf_v2(x, amin, tau),
           "V3_triton": lambda x: agf_v3(x, amin, tau),
           "V3s_stream_31steps": lambda x: stream_win(x, amin, tau)}
    for n, f in fns.items():
        for i in range(5): f(xs[i%4])
        torch.cuda.synchronize(); ms=[]; chk=0.0
        for i in range(30):
            e0,e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize(); e0.record(); o = f(xs[i%4]); e1.record(); torch.cuda.synchronize()
            ms.append(e0.elapsed_time(e1)); chk += float(o.double().sum())
        res[f"{amin}/{tau} {n}"] = dict(window_ms=float(np.median(ms)), frame_ms=float(np.median(ms))/N, checksum=chk)
        print(f"agf{amin}t{tau} {n:22s} window {np.median(ms):9.3f} ms  frame {np.median(ms)/N:8.4f} ms  chk {chk:.6e}", flush=True)
json.dump(res, open("kbench.json","w"), indent=1)

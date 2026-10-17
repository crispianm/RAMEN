"""S6 OFF-proof: the opt-in far-weighted SiLog changes NOTHING when off. Loads the PRE-S6 train.py from git (BASE, default 8e1d9ae) as a separate module and compares the real
_compute_loss_vda (loss value AND gradient, torch.equal, i.e. bit-identical) against the current train.py on many random batches, for: keys absent, far_weight 1.0, far_weight 1.0 + far_from.
CONTROLS (forced fail): the same comparison for far_weight 4.0 must report DIFFERENT (else the comparator is vacuous). Also prints the source diff stat of train.py vs BASE.
Run from the repo root:  ~/.venvs/ramen/bin/python outdoor_debug/parts/far_diag/l1_final/s6_offproof.py [BASE_SHA]"""
import importlib.util, subprocess, sys, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO))
import torch
BASE = sys.argv[1] if len(sys.argv) > 1 else "8e1d9ae"
src = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE}:train.py"], capture_output=True, text=True, check=True).stdout
tmp = Path(tempfile.mkdtemp()) / "train_base.py"; tmp.write_text(src)
spec = importlib.util.spec_from_file_location("train_base", tmp); tb = importlib.util.module_from_spec(spec); sys.modules["train_base"] = tb; spec.loader.exec_module(tb)
import train as tn
assert tb is not tn and not hasattr(tb, "silog_far_weighted") and hasattr(tn, "silog_far_weighted"), "module identity / version check"
class Z(torch.nn.Module):
    def __init__(self, **_): super().__init__()
    def forward(self, prediction, target, mask): return {"total_loss": prediction.sum() * 0.0}
for m in (tb, tn): m._get_vda_video_depth_loss = lambda repo, lc: Z()
def cfg(**l): return {"model": {"arch": "vda", "metric": True}, "loss": {"loss_mode": "vda", "sky_zero_disparity": False, "metric_silog_weight": 1.0, "metric_depth_clamp": [0.1, 1e5], **l}}
def batch(seed):
    g = torch.Generator().manual_seed(seed); B, T = 2, 3
    depth = torch.rand(B, 1, T, 16, 16, generator=g) * 400 + 0.5
    mask = (torch.rand(B, 1, T, 16, 16, generator=g) > 0.2).float()
    b = {"depth": depth, "mask": mask}
    if seed % 2: b["metric_sup"] = torch.tensor([1.0, 0.0] if seed % 4 == 1 else [1.0, 1.0]); b["silog_ceil"] = torch.tensor([655.0, float("inf")])
    return depth * torch.exp(torch.randn(depth.shape, generator=g) * 0.4), b
def run(mod, c, seed):
    p, b = batch(seed); p = p.clone().requires_grad_(True); l = mod._compute_loss_vda(p, b, c, False)
    l = l if torch.is_tensor(l) else torch.tensor(float(l))
    if l.requires_grad: l.backward(); g = p.grad.clone()
    else: g = torch.zeros_like(p)
    return l.detach(), g
def compare(c, n=60):
    same = 0; nz = 0
    for s in range(n):
        lb, gb = run(tb, c, s); ln, gn = run(tn, c, s); same += bool(torch.equal(lb, ln) and torch.equal(gb, gn)); nz += bool(gb.abs().sum() > 0)
    assert nz >= n // 2, f"only {nz}/{n} batches reach the SiLog term (vacuous comparison)"
    return same, n
ok = True
for label, c in (("keys absent", cfg()), ("far_weight 1.0", cfg(metric_silog_far_weight=1.0)), ("far_weight 1.0 + far_from 50", cfg(metric_silog_far_weight=1.0, metric_silog_far_from=50.0)),
                 ("log_terms on, keys absent", cfg(log_terms=True))):
    s, n = compare(c); print(f"OFF [{label}]: bit-identical loss+grad vs BASE {BASE} on {s}/{n} batches"); ok &= s == n
s, n = compare(cfg(metric_silog_far_weight=4.0, metric_silog_far_from=50.0))
print(f"CONTROL (forced fail) [far_weight 4.0 + far_from 50]: bit-identical on {s}/{n} batches (must be < {n}: the comparator can tell ON from OFF)"); ok &= s < n
print(subprocess.run(["git", "-C", str(REPO), "diff", "--stat", BASE, "--", "train.py"], capture_output=True, text=True).stdout.strip())
print("S6 OFF-PROOF:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

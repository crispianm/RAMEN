#!/usr/bin/env python3
"""far_diag (2): per-pixel gradient of each loss term vs GT depth, through the REAL train._compute_loss_vda (CPU, synthetic scene, effective C3/FT loss block).
Scene: T=4, 8 horizontal bands (equal pixel count) with GT depth 5,10,20,40,80,150,300,500 m (+-10% horizontal ramp, 1%/frame drift), no sky, all valid, metric_sup=1.
For each band b and each log-error e in {0.2231 (pred=0.8 GT), 0.5978 (pred=0.55 GT)}: pred = GT everywhere except band b, which is GT*exp(-e). Report mean |dL/dlog(pred)| per pixel IN band b (pixel-pooled loss as trained)
for terms: SSI only, gradient only, TGM only, VDA total, SiLog (= full - VDA). Also the response when ALL bands carry the same error (global scale): VDA is invariant, SiLog is not.
Run: CUDA_VISIBLE_DEVICES= ~/.venvs/ramen/bin/python -I outdoor_debug/parts/far_diag/grad_accounting.py > grad_accounting_output.txt
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C1_losses"))
from c1_common import *
DEPTHS = [5, 10, 20, 40, 80, 150, 300, 500]; ROWS = 8; Wd = 32; Tn = 4
H_ = ROWS * len(DEPTHS)
base = np.zeros((H_, Wd))
for i, dd in enumerate(DEPTHS):
    base[i * ROWS:(i + 1) * ROWS] = dd * (1 + 0.1 * np.linspace(-1, 1, Wd))[None, :]
dep = torch.tensor(np.stack([base * (1 + 0.01 * k) for k in range(Tn)])[None, None], dtype=torch.float32)
mask = torch.ones_like(dep); sky = torch.zeros_like(dep)
batch = make_batch(dep, mask, sky)
band = torch.tensor(np.repeat(np.arange(len(DEPTHS)), ROWS * Wd).reshape(1, 1, 1, H_, Wd)).expand_as(dep)
N = dep.numel()
def G(pred, **over):
    return loss_and_grad(pred, batch, cfg(metric_silog_weight=0.0, **over))[1]
# VideoDepthLoss only reads loss.gradient_weight (alpha) and loss.tgm_weight (train.py:833-836; ssi_weight is ignored), and total = spatial + tgm_weight*stable (linear), so:
def gr(term, pred):
    if term == "silog": return silog_only(pred, batch, cfg())[1]
    g_ssi = G(pred, tgm_weight=0.0, gradient_weight=0.0)
    g_spatial = G(pred, tgm_weight=0.0)
    g_full = G(pred)
    return {"ssi": g_ssi, "gradmatch": g_spatial - g_ssi, "tgm": g_full - g_spatial, "vda": g_full}[term]
out = {}
print("mean |dL/dlog(pred)| per pixel in the perturbed band (x1e-6), loss pixel-pooled over the whole clip; N px =", N)
for e, name in ((0.2231, "pred=0.80*GT"), (0.5978, "pred=0.55*GT")):
    print(f"\n== error in ONE band only, {name} ==")
    print("term      " + " ".join(f"{d:>9d}m" for d in DEPTHS) + "   ratio(20m/150m)  ratio(20m/500m)")
    for term in ("ssi", "gradmatch", "tgm", "vda", "silog"):
        row = []
        for b in range(len(DEPTHS)):
            pred = dep.clone(); sel = band == b
            pred[sel] = dep[sel] * math.exp(-e)
            g = gr(term, pred)
            row.append(float((g[sel].abs() * pred[sel]).mean()) * 1e6)
        out[(name, term)] = row
        print(f"{term:9s} " + " ".join(f"{v:10.2f}" for v in row) + f"   {row[2]/row[5]:8.2f}   {row[2]/row[7]:8.2f}")
print("\n== SAME error in all bands (global scale), pred=0.55*GT: mean |dL/dlog(pred)| per pixel (x1e-6) ==")
pred = dep * 0.55
for term in ("vda", "silog"):
    g = gr(term, pred)
    print(term, [round(float((g[band == b].abs() * pred[band == b]).mean()) * 1e6, 3) for b in range(len(DEPTHS))])
# SiLog pooled over the eligible pixels: far (>=150 m) band carries error pred=0.55*GT, all other eligible pixels are perfect; vary the far share of the eligible pixels (mask subsampling, fixed seed)
print("\n== SiLog (as trained: pooled over eligible px, lambda 0.5): far error 0.55*GT, rest perfect; far share f of the eligible pixels ==")
print("f_far   mean|dL/dlog p| per far px (x1e-6)   total far pull = sum over far px (x1e-3)   SiLog value")
rng = np.random.default_rng(0)
nearsel = (band <= 4).numpy(); farsel = (band >= 5).numpy()
for f in (0.014, 0.03, 0.05, 0.10, 0.25, 0.50):
    nn_, nf = int(nearsel.sum()), int(farsel.sum())
    keep_far = min(nf, int(round(f / (1 - f) * nn_)))
    keep_near = nn_ if keep_far < nf else int(round((1 - f) / f * nf))
    mk = np.zeros(dep.shape, bool)
    for sel, k in ((nearsel, keep_near), (farsel, keep_far)):
        ix = np.flatnonzero(sel.reshape(-1)); mk.reshape(-1)[rng.choice(ix, k, replace=False)] = True
    b2 = make_batch(dep, torch.tensor(mk.astype(np.float32)), sky)
    pred = dep.clone(); fm = torch.tensor(mk & farsel); pred[fm] = dep[fm] * 0.55
    Ls, gs = silog_only(pred, b2, cfg())
    print(f"{mk[farsel].sum()/mk.sum():.3f}   {float((gs[fm].abs()*pred[fm]).mean())*1e6:12.1f}   {float((gs[fm].abs()*pred[fm]).sum())*1e3:14.2f}   {Ls:.4f}")

"""Shared helpers for the C1 loss audit tests (CPU only). Calls the REAL train._compute_loss_vda,
train.silog_mask, train.sparse_ssi_loss, train.sky_pixels and the upstream VDA loss/loss.py."""
import math, os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
import numpy as np
import torch
import train

VDA_REPO = "/mnt/DATA/DegradedDepth/competition/Video-Depth-Anything"   # local copy of the upstream repo (loss/loss.py); the cluster config points at /projects/b5dh/repos/Video-Depth-Anything

# the effective loss block of C3 seed45 / RAMEN n46 / FT n46 (outdoor_debug/parts/C2_config_train/effective_*.yaml): identical in all three
EFFECTIVE_LOSS = dict(ssi_scales=4, ssi_trim=0.0, gradient_weight=0.5, tgm_weight=10.0, loss_mode="vda",
                      metric_silog_weight=1.0, metric_silog_lambda=0.5, metric_depth_clamp=[0.1, 100000.0],
                      sky_hinge_weight=0.0, sky_vda_zero_disparity=True, log_terms=False)

def cfg(**over):
    loss = dict(EFFECTIVE_LOSS); loss.update(over)
    return {"model": {"vda_repo": VDA_REPO, "metric": True}, "loss": loss}

H = W = 56
T = 4
SKY_M = 655.0

def outdoor_scene(h=H, w=W, t=T):
    """(depth (1,1,T,H,W), valid mask, sky mask, bin label (H,W)). rows 0-19 sky (655 sentinel), 20-35 buildings 150-400 m,
    36-43 mid 25-60 m, 44-55 road 4-25 m. Static scene, depth drifts 1%/frame so the TGM term is non-trivial."""
    d = np.zeros((h, w), np.float64)
    lab = np.zeros((h, w), np.int64)
    cols = np.linspace(0, 1, w)[None, :]
    for r in range(h):
        if r < 20:
            d[r] = SKY_M
        elif r < 36:
            d[r] = 150 + 250 * cols[0] * ((r - 20) / 15.0 * 0.5 + 0.5)
        elif r < 44:
            d[r] = 60 - (r - 36) * (35.0 / 7)
        else:
            d[r] = 25 - (r - 44) * (21.0 / 11)
    sky = d >= SKY_M
    lab[sky] = 5
    lab[(~sky) & (d >= 150)] = 4
    lab[(d < 150) & (d >= 80)] = 3
    lab[(d < 80) & (d >= 30)] = 2
    lab[(d < 30) & (d >= 10)] = 1
    lab[(d < 10)] = 0
    dep = np.stack([d * (1 + 0.01 * k) for k in range(t)])[None, None]
    dep = np.where(np.broadcast_to(sky, dep.shape), SKY_M, dep)
    m = np.broadcast_to(~sky, dep.shape).copy()
    return (torch.tensor(dep, dtype=torch.float32), torch.tensor(m.astype(np.float32)),
            torch.tensor(np.broadcast_to(sky, dep.shape).astype(np.float32).copy()), lab)

BIN_NAMES = ["<10m", "10-30m", "30-80m", "80-150m", "150-400m(far valid)", "sky"]

def make_batch(depth, mask, sky, metric_sup=1.0, silog_ceil=None, sparse=None):
    B = depth.shape[0]
    b = {"depth": depth, "mask": mask, "sky": sky, "metric_sup": torch.full((B,), float(metric_sup))}
    if silog_ceil is not None:
        b["silog_ceil"] = torch.full((B,), float(silog_ceil))
    if sparse is not None:
        b["sparse"] = torch.full((B,), float(sparse))
    return b

def loss_and_grad(pred, batch, c):
    p = pred.clone().requires_grad_(True)
    L = train._compute_loss_vda(p, batch, c, pred_is_disparity=False)
    L.backward()
    return float(L.detach()), p.grad.detach()

def silog_only(pred, batch, c):
    """exact SiLog contribution = L(w) - L(0) (the SiLog term is added last, linear in w)."""
    c0 = {"model": c["model"], "loss": dict(c["loss"], metric_silog_weight=0.0)}
    L1, g1 = loss_and_grad(pred, batch, c)
    L0, g0 = loss_and_grad(pred, batch, c0)
    w = c["loss"]["metric_silog_weight"]
    return (L1 - L0) / w, (g1 - g0) / w

def per_bin(g, depth, lab, pred):
    """mean |dL/dpred| per pixel (per metre) and mean |dL/dlog(pred)| = |g|*pred per pixel, for each bin; L1 sum too."""
    B, _, t, h, w = g.shape
    out = []
    labt = torch.tensor(lab)[None, None, None].expand_as(g)
    for k, n in enumerate(BIN_NAMES):
        sel = labt == k
        if not bool(sel.any()):
            out.append((n, 0, float('nan'), float('nan'), float('nan'))); continue
        ga = g[sel].abs()
        out.append((n, int(sel.sum()), float(ga.mean()), float((ga * pred[sel]).mean()), float(ga.sum())))
    return out

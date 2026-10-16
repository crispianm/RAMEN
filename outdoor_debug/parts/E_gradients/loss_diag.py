#!/usr/bin/env python3
"""PLAN Phase 11: loss / gradient diagnostics on REAL training batches (outdoor_debug, subagent E_gradients).

Real code paths used: train.load_config, train._build_multisource (dataset + weighted 'mass' sampler logic, local roots, NYU/ScanNet++ train absent),
default collate, train._compute_loss_vda (every loss component is a re-call of the REAL function with one switch changed; no loss is re-implemented),
models.build_model + the checkpoint's trainable_ema, bf16 autocast forward in model.train() as train.py does. No optimizer step; the one-step lookahead
modifies the parameters in place and restores them bit-exactly from a clone (also BN buffers).

Run from the repo root:  python -I outdoor_debug/parts/E_gradients/loss_diag.py --out <dir> [--batches 20] [--smoke]
Output: <out>/loss_diagnostics.csv (long format: ckpt,batch,group,item,component,metric,value) + batch_meta.json
"""
import argparse, copy, csv, json, os, pathlib, sys, time, bisect
import numpy as np
import torch
import yaml

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import train
from models import build_model, trainable_modules

D = "/mnt/DATA/DegradedDepth/"
T_ROOT = pathlib.Path(D + "data/TRAIN")
VDA_REPO = D + "competition/Video-Depth-Anything"
CK = {
    "FT_n46": D + "checkpoints_v3/eval_20261012_ft_n46/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n46/step12000.pt",
    "RAMEN_n46": D + "checkpoints_v3/eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt",
    "C3_s45": D + "checkpoints_v3/eval_20261009/RAMEN_train_dn01_v3_bfix_C3_B/runs/dn01_v3_bfix_C3_B/step12000.pt",
}
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--batches", type=int, default=20)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--frames", type=int, default=16)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--ckpt_blocks", action="store_true")
ap.add_argument("--models", default="released,FT_n46,RAMEN_n46,C3_s45")
ap.add_argument("--step_norm", type=float, default=0.3, help="L2 norm of the matched lookahead step (~ one Adam step at lr 1e-4 on 8.5M params)")
ap.add_argument("--sgd_lr", type=float, default=1e-2)
args = ap.parse_args()
OUT = pathlib.Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"

# ----------------------------------------------------------------------------------------------- config / models
cfg_ft = train.load_config(REPO / "configs/experiments/dn01_c3rep_20261009_ft_n46.yaml")
cfg_c3 = train.load_config(REPO / "configs/experiments/dn01_v3_bfix_C3_B.yaml")
LOSS = copy.deepcopy(cfg_ft["loss"]); LOSS["log_terms"] = False
assert LOSS == {**cfg_c3["loss"], "log_terms": False, "enh_weight": LOSS["enh_weight"]} or True
for k in ("sky_vda_zero_disparity", "metric_silog_weight", "metric_silog_lambda", "tgm_weight", "gradient_weight", "metric_depth_clamp", "sky_hinge_weight", "loss_mode"):
    assert LOSS[k] == cfg_c3["loss"][k], k           # effective loss block identical in FT and C3
assert LOSS["sky_vda_zero_disparity"] is True and LOSS["sky_hinge_weight"] == 0.0 and LOSS["metric_silog_weight"] == 1.0


def local_model_cfg(cfg):
    cfg = copy.deepcopy(cfg); m = cfg["model"]
    m["vda_repo"] = VDA_REPO
    m["vda_checkpoint_path"] = D + "RAMEN/checkpoints/metric_video_depth_anything_vits.pth"
    m["da_v2_checkpoint_path"] = D + "RAMEN/checkpoints/depth_anything_v2_vits.pth"
    m["st_temp2_init"] = None
    return cfg


def load_model(name):
    if name == "released":
        net = build_model(local_model_cfg(cfg_ft))              # plain metric VDA-S, released weights (= FT step 0)
        src = "released Metric-VDA-S (FT arch, vda_init=released, no trained weights)"
    else:
        ck = torch.load(CK[name], map_location="cpu", weights_only=False)
        assert int(ck["step"]) == 12000, ck["step"]
        net = build_model(local_model_cfg(ck["cfg"]))
        torch.nn.ModuleDict(trainable_modules(net)).load_state_dict(ck.get("trainable_ema", ck.get("trainable")), strict=True)
        src = f"{CK[name]} step {ck['step']} trainable_ema"
        del ck
    net = net.to(dev)
    if args.ckpt_blocks:
        # runtime patch (no repo edit): activation checkpointing of the DINOv2 blocks so the RAMEN graph (gradient flows through the frozen backbone into the restorer) fits 24 GB.
        # Recomputation is numerically the same forward; only memory changes.
        from torch.utils.checkpoint import checkpoint
        nblk = 0
        for mod in net.modules():
            if hasattr(mod, "attn") and hasattr(mod, "mlp") and hasattr(mod, "norm1"):
                f = mod.forward
                mod.forward = (lambda *a, _f=f, **k: checkpoint(_f, *a, use_reentrant=False, **k))
                nblk += 1
        if hasattr(net, "restore_two_stage"):        # RAMEN / C3 (SharedTemporalNet): the full-resolution restorer U-Nets dominate the activation memory -> checkpoint per clip
            net.restore_two_stage = (lambda clip, _f=net.restore_two_stage: checkpoint(_f, clip, use_reentrant=False))
        print("checkpointed blocks:", nblk, "restorer:", hasattr(net, "restore_two_stage"), flush=True)
    net.train()                                                # train.py: core.train() (backbone stays eval via its override)
    return net, src


def group_of(n):
    if "motion_modules" in n: return "motion"
    if "output_conv" in n: return "output_conv"
    if any(s in n for s in ("refinenet1.", "refinenet2.", "layer1_rn", "layer2_rn")): return "last_dpt"
    if any(s in n for s in ("scratch.", "projects.", "resize_layers.")): return "dpt_other"
    return "other"


# ----------------------------------------------------------------------------------------------- data (real dataset classes)
def build_loader():
    d = copy.deepcopy(cfg_ft["data"]); d["seed"] = 45
    roots = {"tartanair": T_ROOT / "tartanair", "vkitti": T_ROOT / "vkitti", "irs": T_ROOT / "irs", "spring": T_ROOT / "spring", "kitti_raw": T_ROOT / "kitti_raw"}
    srcs, names, dropped = [], [], []
    for s in d["sources"]:
        s = dict(s); t = s["type"]
        if t in ("nyu_train", "scannetpp_train"):
            dropped.append(t); continue                               # train data not on this host (scannetpp_train_full has no depth-train split wired by the config; NYU absent)
        if t == "pointodyssey":
            nm = "pointodyssey_" + ("train" if s["root"].endswith("/train") else "val")
            s["root"] = str(T_ROOT / "point_odyssey" / ("train" if nm.endswith("train") else "val"))
            kl = json.load(open(REPO / s["po_keep_list"])); kl["kept"] = [k for k in kl["kept"] if (pathlib.Path(s["root"]) / k).is_dir()]
            p = OUT / f"po_keep_local_{nm}.json"; json.dump(kl, open(p, "w")); s["po_keep_list"] = str(p); s.pop("po_keep_sha256", None)
        else:
            nm = t; s["root"] = str(roots[t])
            if "gt_root" in s: s["gt_root"] = str(T_ROOT / "kitti_raw_gt")
        srcs.append(s); names.append(nm)
    d["sources"] = srcs; d["num_frames"] = args.frames; d["num_workers"] = 2
    cfg = {"data": d, "training": {"batch_size": args.bs}}
    loader, _ = train._build_multisource(cfg, "train", 1, 0)
    ds = loader.dataset
    offs = ds.cumulative_sizes
    weights = list(loader.sampler.weights)
    train._set_severity_max(ds, 1.0)                                  # step 12000: severity cap 1.0 (severity_max_cap, warmup 500)
    train._set_dataset_attr(ds, "clean_prob", float(d["clean_prob"])); train._set_dataset_attr(ds, "seeded_noise", bool(d["seeded_noise"]))
    train._set_dataset_attr(ds, "blur_only_prob", float(d["blur_only_prob"]))
    train._set_dataset_attr(ds, "static_prob", float(d["static_prob"]))
    train._set_dataset_attr(ds, "static_noise_mult", tuple(float(v) for v in d["static_noise_mult"]))
    train._set_dataset_attr(ds, "static_ev_jitter", float(d["static_ev_jitter"]))
    train._set_dataset_attr(ds, "lowlight", d["lowlight"])

    class Wrap(torch.utils.data.Dataset):
        def __len__(self): return len(ds)
        def __getitem__(self, i):
            it = ds[i]; it["src"] = torch.tensor(bisect.bisect_right(offs, i)); return it
    gen = torch.Generator().manual_seed(45)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=gen)
    # CPU workers 2, seed 45; same sampler type / weights ('mass') as train.py; batch of bs clips of `frames` frames
    dl = torch.utils.data.DataLoader(Wrap(), batch_size=args.bs, sampler=sampler, num_workers=2, drop_last=True)
    mass = {n: float(s["sampling_mass"]) for n, s in zip(names, srcs)}
    return dl, names, dropped, mass


# ----------------------------------------------------------------------------------------------- loss components (REAL train._compute_loss_vda)
def lossv(pred, batch, **over):
    cfg = {"model": {"vda_repo": VDA_REPO, "metric": True}, "loss": {**LOSS, **over}}
    return train._compute_loss_vda(pred, batch, cfg, pred_is_disparity=False)


VARIANTS = {                      # name -> overrides
    "full": {},
    "vda": dict(metric_silog_weight=0.0),
    "vda_notgm": dict(metric_silog_weight=0.0, tgm_weight=0.0),
    "ssi": dict(metric_silog_weight=0.0, tgm_weight=0.0, gradient_weight=0.0),
    "vda_skyoff": dict(metric_silog_weight=0.0, sky_vda_zero_disparity=False),
}
# derived gradient = linear combination of variant gradients (the terms add linearly: total = spatial + stable_scale*stable + w*silog)
DERIVED = {
    "full": {"full": 1},
    "vda": {"vda": 1},
    "silog": {"full": 1, "vda": -1},
    "ssi": {"ssi": 1},
    "grad": {"vda_notgm": 1, "ssi": -1},
    "tgm": {"vda": 1, "vda_notgm": -1},
    "sky_term": {"vda": 1, "vda_skyoff": -1},          # the sky's whole VDA contribution: sky pixels at target disparity 0 + the knock-on through the per-frame normalisers
}
COMPS = list(DERIVED)

CLASSES = ["near_lt10", "mid_10_80", "far_80_655", "far_ge655_valid", "sky", "invalid"]


def class_masks(batch):
    depth = batch["depth"].to(dev); m = (batch["mask"].to(dev) * (depth > 0)) > 0.5
    sky = (batch["sky"].to(dev) > 0.5) & ~m
    c = {"near_lt10": m & (depth < 10), "mid_10_80": m & (depth >= 10) & (depth < 80), "far_80_655": m & (depth >= 80) & (depth < 655),
         "far_ge655_valid": m & (depth >= 655), "sky": sky, "invalid": ~m & ~sky}
    return c, m, sky, depth


rows = []


def emit(ck, b, group, item, comp, metric, val):
    rows.append((ck, b, group, item, comp, metric, float(val)))


def stats_param(vjps, names):
    """per group: l2 norm, zero fraction (exactly 0), non-finite fraction, n elements"""
    acc = {}
    for n, g in zip(names, vjps):
        gr = group_of(n)
        for grp in (gr, "all"):
            a = acc.setdefault(grp, [0.0, 0, 0, 0]);
            if g is None:
                a[1] += 0; a[3] += 0
                continue
            gf = g.detach().float()
            fin = torch.isfinite(gf)
            a[0] += float((gf[fin] ** 2).sum()); a[1] += int((gf == 0).sum()); a[2] += int((~fin).sum()); a[3] += gf.numel()
    return acc


def flat(vjps, names, grp):
    return torch.cat([g.detach().float().flatten() for n, g in zip(names, vjps) if g is not None and (grp == "all" or group_of(n) == grp)])


def cos(a, b):
    na, nb = a.norm(), b.norm()
    return float((a @ b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def run_model(name, net, batches_iter_cache):
    params = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    pnames = [n for n, _ in params]; plist = [p for _, p in params]
    gcount = {}
    for n in pnames: gcount[group_of(n)] = gcount.get(group_of(n), 0) + 1
    print(name, "trainable tensors", len(plist), "numel", sum(p.numel() for p in plist), "groups", gcount, flush=True)
    return pnames, plist


def process(ck, bi, net, pnames, plist, batch, src_names):
    t0 = time.time()
    rgb = batch.get("degraded_frames", batch.get("frames")).to(dev)
    bufs = {k: v.clone() for k, v in net.state_dict().items() if not v.requires_grad} if False else None
    sd0 = {k: v.detach().clone() for k, v in net.state_dict().items()}       # exact restore point (params + buffers)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = net(rgb)
    pred_f = pred.detach().float()
    cls, m, sky, depth = class_masks(batch)
    B = pred.shape[0]
    srcs = batch["src"].tolist()
    # ---- per-variant gradient wrt the predicted depth map (chain rule: the loss depends on the params only through pred)
    gv = {}
    lv = {}
    for v, over in VARIANTS.items():
        pp = pred.detach().clone().requires_grad_(True)
        L = lossv(pp, batch, **over)
        gv[v] = torch.autograd.grad(L, pp)[0].float()
        lv[v] = float(L.detach())
    gp = {c: sum(coef * gv[v] for v, coef in lin.items()) for c, lin in DERIVED.items()}
    for c in COMPS:
        emit(ck, bi, "loss_value", "batch", c, "value", sum(coef * lv[v] for v, coef in DERIVED[c].items()))
    # pixel counts
    npx = m.numel()
    for k, msk in cls.items():
        emit(ck, bi, "pixels", k, "all", "frac_of_pixels", float(msk.float().mean()))
    # ---- (1) pred-map gradient by pixel class: L1 mass per pixel, share of total, per-log variant, sign stats
    plog = pred_f.clamp(0.1, 1e5)
    for c in COMPS:
        g = gp[c].abs(); gl = g * plog                                # per metre / per log-depth
        tot, totl = float(g.sum()), float(gl.sum())
        emit(ck, bi, "pred_grad", "all_pixels", c, "l1_total", tot)
        emit(ck, bi, "pred_grad", "all_pixels", c, "nonfinite_frac", float((~torch.isfinite(gp[c])).float().mean()))
        for k, msk in cls.items():
            n = int(msk.sum())
            if n == 0: continue
            s, sl = float(g[msk].sum()), float(gl[msk].sum())
            emit(ck, bi, "pred_grad", k, c, "l1_per_px_per_m", s / n)
            emit(ck, bi, "pred_grad", k, c, "l1_per_px_per_log", sl / n)
            emit(ck, bi, "pred_grad", k, c, "share_l1_per_m", s / tot if tot > 0 else float("nan"))
            emit(ck, bi, "pred_grad", k, c, "share_l1_per_log", sl / totl if totl > 0 else float("nan"))
            emit(ck, bi, "pred_grad", k, c, "exact_zero_frac", float((gp[c][msk] == 0).float().mean()))
    # sign agreement / cosine of VDA vs SiLog on the pred map per class (SiLog is 0 on sky/invalid by construction)
    for k, msk in cls.items():
        if int(msk.sum()) == 0: continue
        a, s_ = gp["vda"][msk], gp["silog"][msk]
        emit(ck, bi, "pred_cos", k, "vda_vs_silog", "cosine", cos(a, s_))
        nz = (a != 0) & (s_ != 0)
        emit(ck, bi, "pred_cos", k, "vda_vs_silog", "frac_pixels_both_nonzero", float(nz.float().mean()))
        if nz.any():
            emit(ck, bi, "pred_cos", k, "vda_vs_silog", "sign_agree_frac_among_both_nonzero", float((torch.sign(a[nz]) == torch.sign(s_[nz])).float().mean()))
            emit(ck, bi, "pred_cos", k, "vda_vs_silog", "silog_to_vda_l1_ratio", float(s_.abs().sum() / a.abs().sum().clamp_min(1e-30)))
    # median pred / gt & sky depth
    for k in ("near_lt10", "mid_10_80", "far_80_655", "far_ge655_valid"):
        if int(cls[k].sum()):
            emit(ck, bi, "pred_stat", k, "pred", "median_pred_over_gt", (pred_f[cls[k]] / depth[cls[k]]).median())
            emit(ck, bi, "pred_stat", k, "pred", "median_pred_m", pred_f[cls[k]].median())
    if int(cls["sky"].sum()):
        emit(ck, bi, "pred_stat", "sky", "pred", "median_pred_m", pred_f[cls["sky"]].median())
        emit(ck, bi, "pred_stat", "sky", "pred", "frac_pred_lt_80m", float((pred_f[cls["sky"]] < 80).float().mean()))
    # ---- (2) parameter-space gradient per component (vjp of the component's pred-gradient through the model)
    def vjp(gmap):
        outs = torch.autograd.grad(pred, plist, grad_outputs=gmap.to(pred.dtype), retain_graph=True, allow_unused=True)
        return outs
    V = {}
    for c in COMPS:
        V[c] = vjp(gp[c])
        for grp, (l2sq, nz, nf, n) in stats_param(V[c], pnames).items():
            emit(ck, bi, "param_grad", grp, c, "l2_norm", l2sq ** 0.5)
            emit(ck, bi, "param_grad", grp, c, "exact_zero_frac", nz / max(n, 1))
            emit(ck, bi, "param_grad", grp, c, "nonfinite_frac", nf / max(n, 1))
    # class-restricted VDA / SiLog / sky-term / full gradients in parameter space
    R = {}
    for k in ("near_lt10", "mid_10_80", "far_80_655", "far_ge655_valid", "sky"):
        if int(cls[k].sum()) == 0: continue
        for c in ("full", "vda", "silog", "sky_term"):
            R[(c, k)] = vjp(gp[c] * cls[k].to(gp[c].dtype))
        for grp in ("output_conv", "last_dpt", "all"):
            for c in ("full", "vda", "silog", "sky_term"):
                emit(ck, bi, "param_grad_by_class_l2", f"{k}|{grp}", c, "l2_norm", flat(R[(c, k)], pnames, grp).norm())
    # ---- (3) cosines (parameter space)
    for grp in ("output_conv", "last_dpt", "all"):
        fl = {c: flat(V[c], pnames, grp) for c in COMPS}
        emit(ck, bi, "param_cos", grp, "vda_vs_silog", "cosine", cos(fl["vda"], fl["silog"]))
        emit(ck, bi, "param_cos", grp, "vda_vs_silog", "silog_to_vda_norm_ratio", fl["silog"].norm() / fl["vda"].norm().clamp_min(1e-30))
        emit(ck, bi, "param_cos", grp, "sky_term_vs_silog", "cosine", cos(fl["sky_term"], fl["silog"]))
        emit(ck, bi, "param_cos", grp, "sky_term_vs_vda", "cosine", cos(fl["sky_term"], fl["vda"]))
        emit(ck, bi, "param_cos", grp, "sky_term_vs_vda", "norm_ratio", fl["sky_term"].norm() / fl["vda"].norm().clamp_min(1e-30))
        emit(ck, bi, "param_cos", grp, "sky_term_vs_full", "norm_ratio", fl["sky_term"].norm() / fl["full"].norm().clamp_min(1e-30))
        for k in ("near_lt10", "mid_10_80", "far_80_655", "far_ge655_valid"):
            if ("full", k) not in R: continue
            vk, sk_, fk = flat(R[("vda", k)], pnames, grp), flat(R[("silog", k)], pnames, grp), flat(R[("full", k)], pnames, grp)
            emit(ck, bi, "param_cos", f"{grp}|{k}", "vda_vs_silog_on_class", "cosine", cos(vk, sk_))
            emit(ck, bi, "param_cos", f"{grp}|{k}", "sky_term_vs_class_full", "cosine", cos(fl["sky_term"], fk))
            emit(ck, bi, "param_cos", f"{grp}|{k}", "sky_term_vs_class_vda", "cosine", cos(fl["sky_term"], vk))
            emit(ck, bi, "param_cos", f"{grp}|{k}", "sky_term_vs_class_silog", "cosine", cos(fl["sky_term"], sk_))
        if ("vda", "sky") in R:                  # gradient of the VDA loss restricted to SKY pixels vs the geometry gradients
            sv = flat(R[("vda", "sky")], pnames, grp)
            emit(ck, bi, "param_cos", grp, "vda_on_sky_pixels_vs_silog_total", "cosine", cos(sv, fl["silog"]))
            emit(ck, bi, "param_cos", grp, "vda_on_sky_pixels_vs_vda_total", "cosine", cos(sv, fl["vda"]))
            for k in ("near_lt10", "far_80_655"):
                if ("full", k) in R:
                    emit(ck, bi, "param_cos", f"{grp}|{k}", "vda_on_sky_pixels_vs_class_full", "cosine", cos(sv, flat(R[("full", k)], pnames, grp)))
                    emit(ck, bi, "param_cos", f"{grp}|{k}", "vda_on_sky_pixels_vs_class_silog", "cosine", cos(sv, flat(R[("silog", k)], pnames, grp)))
    # ---- (4) per-source contribution: exact per-clip decomposition of the batch gradient (pred-gradient is separable per clip)
    sparse = batch.get("sparse", torch.zeros(B)).float().tolist()
    for c in ("full", "vda", "silog"):
        per = []
        for i in range(B):
            gi = torch.zeros_like(gp[c]); gi[i] = gp[c][i]
            per.append(vjp(gi))
        for grp in ("output_conv", "last_dpt", "all"):
            fl_i = [flat(p, pnames, grp) for p in per]
            tot_v = flat(V[c], pnames, grp)
            norms = [float(x.norm()) for x in fl_i]; sn = sum(norms) or 1.0
            tn = float(tot_v.norm()) or 1.0
            for i in range(B):
                sname = src_names[srcs[i]]
                emit(ck, bi, "source", f"{sname}|{grp}", c, "clip_grad_l2", norms[i])
                emit(ck, bi, "source", f"{sname}|{grp}", c, "norm_share_of_sum_of_clip_norms", norms[i] / sn)
                emit(ck, bi, "source", f"{sname}|{grp}", c, "projection_share_on_total", float((fl_i[i] @ tot_v) / tn ** 2))
    for i in range(B):
        sname = src_names[srcs[i]]
        v_i = m[i]; s_i = sky[i]
        emit(ck, bi, "source_pixels", sname, "all", "n_clips", 1)
        emit(ck, bi, "source_pixels", sname, "all", "valid_px", int(v_i.sum()))
        emit(ck, bi, "source_pixels", sname, "all", "sky_px", int(s_i.sum()))
        emit(ck, bi, "source_pixels", sname, "all", "total_px", v_i.numel())
        sup = float(batch["metric_sup"][i]) > 0.5
        ceil = float(batch["silog_ceil"][i]) if "silog_ceil" in batch else float("inf")
        emit(ck, bi, "source_pixels", sname, "all", "silog_px", int((v_i & (depth[i] < ceil)).sum()) if sup else 0)
        emit(ck, bi, "source_pixels", sname, "all", "sparse_flag", sparse[i])
        # L1 pred-gradient mass of this clip
        for c in ("full", "vda", "silog"):
            emit(ck, bi, "source_predgrad_l1", sname, c, "l1", float(gp[c][i].abs().sum()))
    # ---- (5) one-step lookahead: sky-only direction vs others, on a restore-able copy
    def measure(tag):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            p2 = net(rgb)
        p2f = p2.detach().float()
        with torch.no_grad():
            Lf = lossv(p2f, batch); Lv = lossv(p2f, batch, metric_silog_weight=0.0)
        silog = float(Lf - Lv)
        res = {"silog": silog, "vda_total": float(Lv)}
        for k in ("near_lt10", "far_80_655"):
            if int(cls[k].sum()):
                res[f"median_pred_over_gt_{k}"] = float((p2f[cls[k]] / depth[cls[k]]).median())
        if int(cls["sky"].sum()):
            res["median_pred_sky_m"] = float(p2f[cls["sky"]].median())
        return res

    base = measure("base")
    for kk, vv in base.items(): emit(ck, bi, "lookahead", "baseline", "none", kk, vv)

    def restore():
        net.load_state_dict(sd0, strict=True)

    steps = []
    for c in ("sky_term", "full", "vda", "silog"):
        vec = [None if g is None else g.detach().float() for g in V[c]]
        gn = float(torch.sqrt(sum((g ** 2).sum() for g in vec if g is not None)))
        steps.append((f"sgd_lr{args.sgd_lr:g}|{c}", c, args.sgd_lr, vec, gn))
        if gn > 0: steps.append((f"norm{args.step_norm:g}|{c}", c, args.step_norm / gn, vec, gn))
    for tag, c, lr, vec, gn in steps:
        with torch.no_grad():
            for p, g in zip(plist, vec):
                if g is not None: p.sub_(lr * g.to(p.dtype))
        r = measure(tag)
        restore()
        emit(ck, bi, "lookahead", tag, c, "step_l2", lr * gn)
        for kk, vv in r.items():
            emit(ck, bi, "lookahead", tag, c, kk, vv)
            if kk in base:
                emit(ck, bi, "lookahead", tag, c, "delta_" + kk, vv - base[kk])
    # state restored; make sure
    for k, v in net.state_dict().items():
        assert torch.equal(v, sd0[k]), k
    print(f"  [{ck}] batch {bi} done {time.time()-t0:.1f}s  src={[src_names[s] for s in srcs]}  loss full={lv['full']:.4f} vda={lv['vda']:.4f} skyterm={lv['vda']-lv['vda_skyoff']:.5f}", flush=True)
    del V, R, gv, gp, pred
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------- main
def main():
    torch.manual_seed(45); np.random.seed(45)          # data seed 45: worker base seeds (degradation RNG) derive from the global generator
    lock = pathlib.Path("/home/wg19671/.claude/jobs/a0d391cb/tmp/GPU_LOCK")
    dl, src_names, dropped, mass = build_loader()
    print("sources", src_names, "dropped (train data absent on this host)", dropped, flush=True)
    nb = 1 if args.smoke else args.batches
    names = args.models.split(",")
    nets = {}
    for nm in names:
        net, srcdesc = load_model(nm); nets[nm] = (net, srcdesc)
        print("loaded", nm, srcdesc, flush=True)
    pn = {nm: run_model(nm, nets[nm][0], None) for nm in names}
    meta = {"sources": src_names, "dropped": dropped, "sampling_mass_local": mass, "batches": nb, "bs": args.bs, "frames": args.frames, "models": {n: nets[n][1] for n in names},
            "seed": 45, "workers": 2, "sgd_lr": args.sgd_lr, "step_norm": args.step_norm}
    json.dump(meta, open(OUT / "batch_meta.json", "w"), indent=1)
    it = iter(dl)
    for bi in range(nb):
        batch = next(it)
        for nm in names:
            net = nets[nm][0]
            net.train()
            process(nm, bi, net, pn[nm][0], pn[nm][1], batch, src_names)
        with open(OUT / "loss_diagnostics.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["ckpt", "batch", "group", "item", "component", "metric", "value"]); w.writerows(rows)
    print("wrote", OUT / "loss_diagnostics.csv", len(rows), "rows")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase-1 baseline (outdoor_debug PLAN): DEV-only inference + metrics for 4 models. Inference path = bench_joint.run_pipeline
(same windowing / resolution / resize as the paper scorer); no repo file is modified. Metrics computed on GPU per clip, pooled by sums.
Usage: python -I run_baseline.py <repo> <outdir_small> <outdir_big> [--limit-clips N]"""
import sys, json, time, hashlib
from pathlib import Path
import numpy as np, torch, cv2

repo, out_small, out_big = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
LIMIT = int(sys.argv[sys.argv.index("--limit-clips") + 1]) if "--limit-clips" in sys.argv else None
sys.path.insert(0, str(repo))
import bench_joint as bj
from evaluation.sky_metrics import kitti_sky_masks

D = "/mnt/DATA/DegradedDepth/"
MODELS = {
    "ramen_s45": f"ours:{D}checkpoints_v3/eval_20261009/RAMEN_train_dn01_v3_bfix_C3_B/runs/dn01_v3_bfix_C3_B/step12000.pt",
    "ramen_n46": f"ours:{D}checkpoints_v3/eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt",
    "ft_n46": f"metric_vda_s_ft:{D}checkpoints_v3/eval_20261012_ft_n46/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n46/step12000.pt",
    "mvda_s_stock": "metric_vda_s",
}
DATASETS = ["kitti", "vkitti", "bonn", "scannetpp", "nyuv2", "sintel"]
GROUP = {"kitti": "outdoor", "vkitti": "outdoor", "bonn": "indoor", "scannetpp": "indoor", "nyuv2": "indoor", "sintel": "sintel_mixed_synthetic"}
CY = {"kitti": 0.46, "vkitti": 0.50}          # horizon row (fraction of GT height): KITTI principal point cy~172/374, VKITTI2 cy~187/375
BAND = 8                                       # horizon band = rows cy*H +- 8 px
RANGES = {"all": (0.1, 1e9), "cap": None, "0-10": (0.1, 10.0), "10-30": (10.0, 30.0), "30-80": (30.0, 80.0), ">80": (80.0, 1e9)}
VIS = {("kitti", "2011_09_26_drive_0005_sync"), ("kitti", "2011_09_26_drive_0023_sync"), ("vkitti", "Scene01_overcast"),
       ("vkitti", "Scene18_sunset"), ("bonn", "rgbd_bonn_synchronous"), ("scannetpp", "05993dba6f")}
dev = "cuda"
HBINS, HLO, HHI = 2000, -3.0, 3.5              # log10 histogram of predictions (m): 1e-3 .. ~3162
SAMP_PER_CLIP = 30000

pipes = {m: bj.build_pipeline(s, dev) for m, s in MODELS.items()}
for m, p in pipes.items():
    assert p["metric"], m


def new_acc():
    return dict(n=0.0, rel=0.0, se=0.0, d1=0.0, sd=0.0, sd2=0.0)


acc = {}        # (model, ds, range, mode) -> sums
frame_macro = {}  # (model, ds, mode) -> [sum of per-frame abs_rel/rmse/d1 (scorer style), n_frames]
hist = {}       # (model, ds, region) -> hist counts ; extras
extra = {}      # (model, ds) -> dict of scalars
samples = {}    # (model, ds) -> lists
sky_acc = {}    # (model, ds) -> dict
band_acc = {}
vis = {}
t0 = time.time()


def add(model, ds, rng, mode, p, g, m):
    """p, g, m flat/any-shape tensors on gpu (float32/bool). accumulate abs_rel, se, delta1, log-diff sums over mask m."""
    n = int(m.sum())
    if n == 0:
        return
    pm, gm = p[m].double(), g[m].double()
    d = torch.log(pm.clamp_min(1e-3)) - torch.log(gm)
    a = acc.setdefault((model, ds, rng, mode), new_acc())
    a["n"] += n
    a["rel"] += float(((pm - gm).abs() / gm).sum())
    a["se"] += float(((pm - gm) ** 2).sum())
    a["d1"] += float((torch.maximum(pm / gm.clamp_min(1e-6), gm / pm.clamp_min(1e-6)) < 1.25).sum())
    a["sd"] += float(d.sum())
    a["sd2"] += float((d * d).sum())


def frame_stats(p, g, m):
    """scorer-style per-frame means (VDA metric.py): returns list of (absrel, rmse, d1) for frames with any valid."""
    out = []
    for t in range(p.shape[0]):
        mt = m[t]
        if not mt.any():
            continue
        pm, gm = p[t][mt].double(), g[t][mt].double()
        out.append((float(((pm - gm).abs() / gm).mean()), float(((pm - gm) ** 2).mean().sqrt()),
                    float((torch.maximum(pm / gm, gm / pm) < 1.25).double().mean())))
    return out


def hist_add(key, vals):
    h = torch.histc(torch.log10(vals.clamp_min(10 ** HLO)).float(), bins=HBINS, min=HLO, max=HHI).cpu().numpy()
    e = hist.setdefault(key, dict(h=np.zeros(HBINS), n=0, zero=0, max=0.0, min=1e30))
    e["h"] += h
    e["n"] += int(vals.numel())
    e["zero"] += int((vals <= 1e-6).sum())
    e["max"] = max(e["max"], float(vals.max()))
    e["min"] = min(e["min"], float(vals.min()))


for ds in DATASETS:
    proto = bj.DATASETS[ds]
    clips = bj.select_clips(ds, "dev", LIMIT)
    print(f"== {ds}: {len(clips)} DEV clips", flush=True)
    for key, frames in clips:
        if ds == 'nyuv2':
            frames = frames[:160]                                           # 500-frame clips OOM the 24 GB GPU (window stacking); first 160 frames (5 windows)
        clean, raw_hw = bj.load_frames(ds, frames)
        gt_np = bj.load_gt(ds, frames)
        N, H, W = gt_np.shape
        gt = torch.from_numpy(gt_np).to(dev).float()
        if ds == "kitti":
            sky = torch.from_numpy(kitti_sky_masks(bj.PATHS["kitti_root"], frames[:proto["max_len"]], proto["crop"], gt_np)).to(dev)
        elif ds == "vkitti":
            sky = gt >= proto["sky_depth_m"]
        else:
            sky = torch.zeros_like(gt, dtype=torch.bool)
        base = (gt > 0.1) & ~sky
        capm = base & (gt < proto["max_depth"])
        rmasks = {"all": base, "cap": capm}
        rmasks["0-10"] = base & (gt <= 10.0)
        rmasks["10-30"] = base & (gt > 10) & (gt <= 30)
        rmasks["30-80"] = base & (gt > 30) & (gt <= 80)
        rmasks[">80"] = base & (gt > 80)
        g_ = torch.Generator().manual_seed(int(hashlib.md5(f"{ds}{key}".encode()).hexdigest()[:8], 16))
        bidx = torch.nonzero(base.flatten()).squeeze(1).cpu()
        sidx = bidx[torch.randperm(len(bidx), generator=g_)[:SAMP_PER_CLIP]] if len(bidx) else bidx
        sidx = sidx.to(dev)
        preds_vis = {}
        for model, p in pipes.items():
            x = clean.to(dev)
            with torch.autocast('cuda', dtype=torch.float16):                 # fp32 OOMs the restorer on KITTI (24 GB 4090); fp16 autocast like scripts/dn01_latency.py
                depth, _ = bj.run_pipeline(p, x, 0.0, 0.0, raw_hw)           # (N,H0,W0) metres (metric), as the scorer
            del x
            if depth.shape[-2:] != (H, W):
                depth = np.stack([cv2.resize(d, (W, H)) for d in depth])      # scorer's get_infer resize
            pred = torch.from_numpy(depth).to(dev).float()
            # ----- alignment params fit on the scorer's valid mask (0.1<gt<max_depth) -----
            pc, gc = pred[capm].double().clamp_min(1e-3), gt[capm].double()
            s_scale = float(torch.exp(torch.median(torch.log(gc) - torch.log(pc))))
            disp, gd = 1.0 / pc, 1.0 / gc
            Aa = torch.tensor([[float((disp * disp).sum()), float(disp.sum())], [float(disp.sum()), float(len(disp))]], dtype=torch.float64)
            s2, b2 = torch.linalg.solve(Aa, torch.tensor([float((disp * gd).sum()), float(gd.sum())], dtype=torch.float64)).tolist()
            p_sc = pred * s_scale
            p_ss = 1.0 / (s2 * (1.0 / pred.clamp_min(1e-3)) + b2).clamp_min(1e-3)
            p_ss = p_ss.clamp(1e-3, 1000.0)
            variants = {"raw": pred, "aligned_scale": p_sc, "aligned_scaleshift": p_ss}
            for mode, pv in variants.items():
                for r, m in rmasks.items():
                    add(model, ds, r, mode, pv, gt, m)
                fs = frame_stats(pv.clamp(1e-3, proto["max_depth"]) if mode != "raw" else pv.clamp(1e-3, proto["max_depth"]), gt, capm)
                fm = frame_macro.setdefault((model, ds, mode), [0.0, 0.0, 0.0, 0])
                for a_, r_, d_ in fs:
                    fm[0] += a_; fm[1] += r_; fm[2] += d_; fm[3] += 1
            extra.setdefault((model, ds), dict(fit_scale=[], fit_s2=[], fit_b2=[]))
            extra[(model, ds)]["fit_scale"].append(s_scale); extra[(model, ds)]["fit_s2"].append(s2); extra[(model, ds)]["fit_b2"].append(b2)
            # ----- output range / saturation -----
            hist_add((model, ds, "all"), pred.flatten())
            if int(base.sum()):
                hist_add((model, ds, "gtvalid"), pred[base])
            if ds in ("kitti", "vkitti"):
                hist_add((model, ds, "far>80"), pred[rmasks[">80"]]) if int(rmasks[">80"].sum()) else None
            # ----- samples for pred-vs-GT plot -----
            sm = samples.setdefault((model, ds), dict(gt=[], pred=[], pred_sc=[]))
            sm["gt"].append(gt.flatten()[sidx].cpu().numpy()); sm["pred"].append(pred.flatten()[sidx].cpu().numpy()); sm["pred_sc"].append(p_sc.flatten()[sidx].cpu().numpy())
            # ----- sky -----
            if ds in ("kitti", "vkitti"):
                sa = sky_acc.setdefault((model, ds), dict(px=[], nearer=[], frames=0, sky_px=0, ge80=0, lstd=[]))
                scene = base & (gt < 80)
                for t in range(N):
                    st = sky[t]
                    if int(st.sum()) < 50 or not bool(scene[t].any()):
                        continue
                    vs = pred[t][st]
                    p95 = torch.quantile(gt[t][scene[t]].double().flatten()[:2_000_000], 0.95).float()
                    sa["nearer"].append(float((vs < p95).float().mean())); sa["frames"] += 1; sa["sky_px"] += int(vs.numel()); sa["ge80"] += int((vs >= 80).sum())
                    sa["lstd"].append(float(torch.log(vs.clamp_min(1e-3)).std()))
                    sa["px"].append(vs[torch.randperm(vs.numel(), device=dev)[:3000]].cpu().numpy())
                # horizon band
                ba = band_acc.setdefault((model, ds), dict(pred_nonsky=[], pred_valid=[], gt_valid=[], ratio=[]))
                r0 = int(CY[ds] * H)
                bm = torch.zeros_like(sky); bm[:, r0 - BAND:r0 + BAND + 1] = True
                nm = bm & ~sky
                vm = bm & base & (gt < 80)
                def sub(v, k=20000):
                    v = v.flatten()
                    return v[torch.randperm(v.numel(), device=dev)[:k]].cpu().numpy() if v.numel() else np.zeros(0)
                ba["pred_nonsky"].append(sub(pred[nm])); ba["pred_valid"].append(sub(pred[vm])); ba["gt_valid"].append(sub(gt[vm]))
                ba["ratio"].append(sub(pred[vm] / gt[vm]))
            if (ds, key) in VIS:
                preds_vis[model] = pred[N // 2].cpu().numpy()
            del pred, p_sc, p_ss
        if (ds, key) in VIS:
            mid = N // 2
            vis[f"{ds}|{key}"] = dict(rgb=clean[mid].permute(1, 2, 0).numpy(), gt=gt[mid].cpu().numpy(), sky=sky[mid].cpu().numpy(), **preds_vis)
        torch.cuda.empty_cache()
        print(f"  {ds} {key} done ({time.time() - t0:.0f}s)", flush=True)

# ----------------------------------------------------------------------------- write outputs
rows = []


def fin(a):
    n = a["n"]
    mu = a["sd"] / n
    return dict(abs_rel=a["rel"] / n, rmse=(a["se"] / n) ** 0.5, delta1=a["d1"] / n, silog=max(a["sd2"] / n - mu * mu, 0) ** 0.5), n


def groups(ds):
    return [(ds, ds), (GROUP[ds], ds)]


gagg = {}
for (model, ds, r, mode), a in acc.items():
    for gname in (ds, GROUP[ds]):
        t = gagg.setdefault((model, gname, r, mode), new_acc())
        for k in t:
            t[k] += a[k]
for (model, g, r, mode), a in gagg.items():
    m, n = fin(a)
    for k, v in m.items():
        rows.append((model, g, GROUP.get(g, "dataset") if g in DATASETS else "group", r, k, mode, v, int(n)))
# scorer-style per-frame macro (cap mask: 0.1<gt<max_depth, preds clamped to [1e-3,max_depth])
for (model, ds, mode), (sa, sr, sd_, nf) in frame_macro.items():
    if nf:
        for k, v in (("abs_rel", sa / nf), ("rmse", sr / nf), ("delta1", sd_ / nf)):
            rows.append((model, ds, GROUP[ds], "cap_perframe_macro", k, mode, v, nf))
# saturation / output range
sat = {}
for (model, ds, region), e in hist.items():
    edges = np.linspace(HLO, HHI, HBINS + 1)
    c = np.cumsum(e["h"])
    tot = e["n"]
    mx, mn = e["max"], e["min"]
    near_max = float(e["h"][edges[:-1] >= np.log10(max(mx * 0.99, 10 ** HLO))].sum() / tot) if mx > 10 ** HLO else float("nan")   # bin-resolved (0.25%/bin): bins whose lower edge >= 0.99 max
    near_min = float(e["zero"] / tot)                                                                                      # exactly 0 (ReLU floor)
    def pct(q):
        i = int(np.searchsorted(c, q * c[-1])); return float(10 ** (edges[min(i, HBINS - 1)]))
    sat[f"{model}|{ds}|{region}"] = dict(n=tot, frac_eq0=near_min, frac_lt_0p1m=float(c[np.searchsorted(edges, -1.0) - 1] / tot),
                                        frac_within1pct_of_observed_max=near_max, max=mx, min=mn, p01=pct(0.001), p50=pct(0.5), p99=pct(0.99), p999=pct(0.999))
    for k in ("frac_eq0", "frac_lt_0p1m", "frac_within1pct_of_observed_max", "max", "p50", "p99", "p999"):
        rows.append((model, ds, GROUP[ds], f"pred_{region}", f"out_{k}", "raw", sat[f"{model}|{ds}|{region}"][k], tot))
# sky + horizon
skyrep = {}
for (model, ds), sa in sky_acc.items():
    px = np.concatenate(sa["px"]) if sa["px"] else np.zeros(1)
    d = dict(sky_px=sa["sky_px"], frames=sa["frames"], median_pred_sky_m=float(np.median(px)), p05=float(np.percentile(px, 5)), p95=float(np.percentile(px, 95)),
             frac_ge80=sa["ge80"] / max(sa["sky_px"], 1), frac_nearer_than_scene_gt_p95=float(np.mean(sa["nearer"])), mean_logstd_within_frame=float(np.mean(sa["lstd"])))
    skyrep[f"{model}|{ds}"] = d
    for k in ("median_pred_sky_m", "p05", "p95", "frac_ge80", "frac_nearer_than_scene_gt_p95", "mean_logstd_within_frame"):
        rows.append((model, ds, "outdoor", "sky", f"sky_{k}", "raw", d[k], sa["sky_px"]))
for (model, ds), ba in band_acc.items():
    cat = lambda k: np.concatenate(ba[k]) if ba[k] else np.zeros(1)
    d = dict(median_pred_nonsky=float(np.median(cat("pred_nonsky"))), median_pred_gtvalid=float(np.median(cat("pred_valid"))), median_gt_gtvalid=float(np.median(cat("gt_valid"))),
             median_ratio_pred_over_gt=float(np.median(cat("ratio"))), p90_pred_nonsky=float(np.percentile(cat("pred_nonsky"), 90)))
    skyrep[f"{model}|{ds}|horizon"] = d
    for k, v in d.items():
        rows.append((model, ds, "outdoor", "horizon_band", f"band_{k}", "raw", v, len(cat("pred_nonsky"))))

import csv
out_small.mkdir(parents=True, exist_ok=True); out_big.mkdir(parents=True, exist_ok=True)
with open(out_small / "baseline_metrics.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "dataset", "group", "depth_range", "metric", "raw_or_aligned", "value", "n_pixels"])
    for r in rows:
        w.writerow(r)
json.dump(dict(saturation=sat, sky_horizon=skyrep, mapping={m: s for m, s in MODELS.items()}, hist_bins=dict(n=HBINS, log10_lo=HLO, log10_hi=HHI),
               scale_fits={f"{m}|{d}": {k: float(np.median(v)) for k, v in e.items()} for (m, d), e in extra.items()}), open(out_small / "baseline_extra.json", "w"), indent=1)
np.savez_compressed(out_big / "samples.npz", **{f"{m}|{d}|{k}": np.concatenate(v) for (m, d), s in samples.items() for k, v in s.items()})
np.savez_compressed(out_big / "vis_frames.npz", **{f"{k}|{kk}": vv for k, v in vis.items() for kk, vv in v.items()})
np.savez_compressed(out_big / "hists.npz", **{f"{m}|{d}|{r}": e["h"] for (m, d, r), e in hist.items()})
print("DONE", time.time() - t0, flush=True)

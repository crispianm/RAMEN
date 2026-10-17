#!/usr/bin/env python3
"""T_trajectory eval: 7 checkpoints (released, ft_n46 / c3_n46 x step 6000/8000/12000), DEV only, fp16 autocast, bench_joint pipeline as the scorer.
usage: python -I run_traj_eval.py <repo> <out_small> <out_big> [--datasets kitti,vkitti,horse,scannetpp] [--limit N]"""
import sys, json, time, csv, hashlib
from pathlib import Path
import numpy as np, torch, cv2
repo, out_small, out_big = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
DS = sys.argv[sys.argv.index("--datasets") + 1].split(",") if "--datasets" in sys.argv else ["kitti", "vkitti", "horse", "scannetpp"]
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
sys.path.insert(0, str(repo))
import bench_joint as bj
from evaluation.sky_metrics import kitti_sky_masks
D = "/mnt/DATA/DegradedDepth/"; C = D + "checkpoints_v3/"; O = C + "outdoor_debug_20261016/"
SPEC = {("released", 0): "metric_vda_s"}
FT12 = C + "eval_20261012_ft_n46/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n46/step12000.pt"
C312 = C + "eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt"
for s in (6000, 8000, 12000):
    SPEC[("ft_n46", s)] = "metric_vda_s_ft:" + (FT12 if s == 12000 else O + f"ft_n46/step{s}.pt")
    SPEC[("c3_n46", s)] = "ours:" + (C312 if s == 12000 else O + f"c3_n46/step{s}.pt")
dev = "cuda"
pipes = {k: bj.build_pipeline(v, dev) for k, v in SPEC.items()}
for k, p in pipes.items(): assert p["metric"], k
out_small.mkdir(parents=True, exist_ok=True); out_big.mkdir(parents=True, exist_ok=True)
rows = []   # model, step, dataset, quantity, value, n
def R(k, ds, q, v, n): rows.append((k[0], k[1], ds, q, float(v), int(n)))
def flush():
    with open(out_small / "trajectory_eval_part.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "step", "dataset", "quantity", "value", "n"]); w.writerows(rows)
t0 = time.time()
def fwd(p, x, hw):
    with torch.autocast("cuda", dtype=torch.float16):
        d, _ = bj.run_pipeline(p, x.to(dev), 0.0, 0.0, hw)
    return d
def resz(d, H, W): return np.stack([cv2.resize(f, (W, H)) for f in d]) if d.shape[-2:] != (H, W) else d
panels = {}
rng = np.random.default_rng(0)

if "kitti" in DS:
    proto = bj.DATASETS["kitti"]
    sky_px = {k: [] for k in SPEC}; sky_fmed = {k: [] for k in SPEC}; cnt = {k: dict(n=0, g80=0, g200=0) for k in SPEC}
    acc = {k: dict(n=0, rel=0.0, relsc=0.0) for k in SPEC}
    clips = bj.select_clips("kitti", "dev", LIMIT)
    for key, frames in clips:
        clean, raw_hw = bj.load_frames("kitti", frames); gt_np = bj.load_gt("kitti", frames); N, H, W = gt_np.shape
        gt = torch.from_numpy(gt_np).to(dev).float()
        sky_np = kitti_sky_masks(bj.PATHS["kitti_root"], frames[:proto["max_len"]], proto["crop"], gt_np); sky = torch.from_numpy(sky_np).to(dev)
        base = (gt > 0.1) & ~sky; capm = base & (gt < proto["max_depth"])
        cells = [("b0s0", clean)]
        if key == "2011_09_26_drive_0023_sync":
            blur, sigma = bj.parse_cell("llb3"); seed = bj.noise_seed("kitti", key, bj.seed_cell(blur) if isinstance(blur, bj.LowLight) else sigma)
            cells.append(("llb3", bj.degrade(clean, blur, sigma, seed)))
        for cell, x in cells:
            for k, p in pipes.items():
                d = torch.from_numpy(resz(fwd(p, x, raw_hw), H, W)).to(dev).float()
                if cell == "b0s0":
                    for t in range(N):
                        st = sky[t]
                        if int(st.sum()) >= 50:
                            v = d[t][st]; sky_px[k].append(v.cpu().numpy()); sky_fmed[k].append(float(v.median()))
                            cnt[k]["n"] += v.numel(); cnt[k]["g80"] += int((v > 80).sum()); cnt[k]["g200"] += int((v > 200).sum())
                    pc, gc = d[capm].double().clamp_min(1e-3), gt[capm].double()
                    sc = float(torch.exp(torch.median(torch.log(gc) - torch.log(pc))))
                    pm, gm = d[base].double(), gt[base].double()
                    acc[k]["n"] += gm.numel(); acc[k]["rel"] += float(((pm - gm).abs() / gm).sum()); acc[k]["relsc"] += float(((pm * sc - gm).abs() / gm).sum())
                if key == "2011_09_26_drive_0023_sync":
                    v = d[109][sky[109]].cpu().numpy()
                    nm = f"f109_{cell}"
                    for q, val in (("sky_median_m", np.median(v)), ("sky_p10_m", np.percentile(v, 10)), ("sky_p90_m", np.percentile(v, 90)), ("sky_frac_gt80", (v > 80).mean()), ("sky_frac_gt200", (v > 200).mean())):
                        R(k, "kitti_0023_" + nm, q, val, v.size)
                    panels[f"kitti0023_f109_{cell}|{k[0]}|{k[1]}"] = d[109].cpu().numpy()
                    if k == ("released", 0): panels[f"kitti0023_f109_{cell}|rgb"] = x[109].permute(1, 2, 0).numpy(); panels["kitti0023_f109_sky"] = sky_np[109]
        print("kitti", key, f"{time.time()-t0:.0f}s", flush=True)
    for k in SPEC:
        a = np.concatenate(sky_px[k]); fm = np.array(sky_fmed[k]); n = a.size
        R(k, "kitti_dev", "sky_median_pooled_m", np.median(a), n); R(k, "kitti_dev", "sky_p10_pooled_m", np.percentile(a, 10), n); R(k, "kitti_dev", "sky_p90_pooled_m", np.percentile(a, 90), n)
        R(k, "kitti_dev", "sky_median_of_frame_medians_m", np.median(fm), len(fm)); R(k, "kitti_dev", "sky_frac_gt80", cnt[k]["g80"] / cnt[k]["n"], n); R(k, "kitti_dev", "sky_frac_gt200", cnt[k]["g200"] / cnt[k]["n"], n)
        R(k, "kitti_dev", "absrel_raw", acc[k]["rel"] / acc[k]["n"], acc[k]["n"]); R(k, "kitti_dev", "absrel_aligned_scale", acc[k]["relsc"] / acc[k]["n"], acc[k]["n"])
    flush()

if "vkitti" in DS:
    proto = bj.DATASETS["vkitti"]
    rngs = {"0-10": (0, 10), "10-30": (10, 30), "30-80": (30, 80), ">80": (80, 1e9)}
    skyv = {k: [] for k in SPEC}; ratios = {k: {"near<30": [], "far80-655": []} for k in SPEC}
    acc = {k: {r: dict(n=0, rel=0.0) for r in rngs} for k in SPEC}
    for key, frames in bj.select_clips("vkitti", "dev", LIMIT):
        clean, raw_hw = bj.load_frames("vkitti", frames); gt_np = bj.load_gt("vkitti", frames); N, H, W = gt_np.shape
        gt = torch.from_numpy(gt_np).to(dev).float(); sky = gt >= proto["sky_depth_m"]; base = (gt > 0.1) & ~sky
        g_ = torch.Generator().manual_seed(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
        for k, p in pipes.items():
            d = torch.from_numpy(resz(fwd(p, clean, raw_hw), H, W)).to(dev).float()
            if int(sky.sum()):
                v = d[sky].flatten(); skyv[k].append(v[torch.randperm(v.numel(), generator=g_)[:200000].to(dev)].cpu().numpy())
            for r, (lo, hi) in rngs.items():
                m = base & (gt > lo) & (gt <= hi)
                if int(m.sum()):
                    a = acc[k][r]; a["n"] += int(m.sum()); a["rel"] += float(((d[m].double() - gt[m].double()).abs() / gt[m].double()).sum())
            for nm, m in (("near<30", base & (gt < 30)), ("far80-655", (gt >= 80) & (gt < proto["sky_depth_m"]))):
                if int(m.sum()):
                    v = (d[m] / gt[m]); ratios[k][nm].append(v[torch.randperm(v.numel(), generator=g_)[:100000].to(dev)].cpu().numpy())
            del d
        print("vkitti", key, f"{time.time()-t0:.0f}s", flush=True)
    for k in SPEC:
        s = np.concatenate(skyv[k]); R(k, "vkitti_dev", "sky_median_m", np.median(s), s.size); R(k, "vkitti_dev", "sky_frac_gt200", (s > 200).mean(), s.size)
        for nm in ratios[k]:
            r_ = np.concatenate(ratios[k][nm]); R(k, "vkitti_dev", f"pred_over_gt_median_{nm}", np.median(r_), r_.size)
        for r in rngs: R(k, "vkitti_dev", f"absrel_raw_{r}", acc[k][r]["rel"] / max(acc[k][r]["n"], 1), acc[k][r]["n"])
    flush()

if "horse" in DS:
    paths = sorted(Path("/mnt/DATA/Horse/test_resized").glob("*.png")); bj._vda_path()
    from video_depth_anything.util.transform import Resize
    resize = Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC)
    rgb = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]
    clean = torch.from_numpy(np.stack([np.clip(resize({"image": r.astype(np.float32) / 255})["image"], 0, 1) for r in rgb])).permute(0, 3, 1, 2).contiguous(); hw = rgb[0].shape[:2]
    H = hw[0]; bands = {"sky_top15": slice(0, int(.15 * H)), "treeline_38_46": slice(int(.38 * H), int(.46 * H)), "ground_bottom10": slice(int(.9 * H), H)}
    panels["horse_rgb25"] = rgb[25]
    for k, p in pipes.items():
        d = fwd(p, clean, hw)
        for b, sl in bands.items():
            R(k, "horse", f"{b}_median_m", np.median(d[:, sl]), d[:, sl].size); R(k, "horse", f"{b}_f25_median_m", np.median(d[25, sl]), d[25, sl].size)
        R(k, "horse", "whole_clip_max_m", d.max(), d.size)
        panels[f"horse_f25|{k[0]}|{k[1]}"] = d[25].copy()
    flush()

if "scannetpp" in DS:
    acc = {k: dict(n=0, rel=0.0) for k in SPEC}
    for key, frames in bj.select_clips("scannetpp", "dev", LIMIT):
        clean, raw_hw = bj.load_frames("scannetpp", frames); gt_np = bj.load_gt("scannetpp", frames); N, H, W = gt_np.shape
        gt = torch.from_numpy(gt_np).to(dev).float(); base = gt > 0.1
        for k, p in pipes.items():
            d = torch.from_numpy(resz(fwd(p, clean, raw_hw), H, W)).to(dev).float()
            acc[k]["n"] += int(base.sum()); acc[k]["rel"] += float(((d[base].double() - gt[base].double()).abs() / gt[base].double()).sum())
        print("scannetpp", key, f"{time.time()-t0:.0f}s", flush=True)
    for k in SPEC: R(k, "scannetpp_dev", "absrel_raw", acc[k]["rel"] / acc[k]["n"], acc[k]["n"])
    flush()
np.savez_compressed(out_big / "panels.npz", **panels)
flush(); print("DONE", time.time() - t0)

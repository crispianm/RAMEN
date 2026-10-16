#!/usr/bin/env python3
"""Lead-requested cases: (A) KITTI drive_0023 frame 109 llb3 sky + all KITTI DEV frames' sky median; (B) horse real clip (51 frames, 518 short side, cubic, no degradation).
usage: python -I run_cases.py <repo> <small_out> <big_out>"""
import sys, json, time
from pathlib import Path
import numpy as np, torch, cv2
repo, out_small, out_big = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(repo))
import bench_joint as bj
from evaluation.sky_metrics import kitti_sky_masks
D = "/mnt/DATA/DegradedDepth/"
MODELS = {
    "mvda_s_stock": "metric_vda_s",
    "ramen_s45": f"ours:{D}checkpoints_v3/eval_20261009/RAMEN_train_dn01_v3_bfix_C3_B/runs/dn01_v3_bfix_C3_B/step12000.pt",
    "ramen_n46": f"ours:{D}checkpoints_v3/eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt",
    "ft_n46": f"metric_vda_s_ft:{D}checkpoints_v3/eval_20261012_ft_n46/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n46/step12000.pt",
}
dev = "cuda"
pipes = {m: bj.build_pipeline(s, dev) for m, s in MODELS.items()}
out_small.mkdir(parents=True, exist_ok=True); out_big.mkdir(parents=True, exist_ok=True)
rows, res = [], {}


def run(p, x, hw, amp):
    if amp:
        with torch.autocast("cuda", dtype=torch.float16):
            d, _ = bj.run_pipeline(p, x.to(dev), 0.0, 0.0, hw)
    else:
        d, _ = bj.run_pipeline(p, x.to(dev), 0.0, 0.0, hw)
    return d


def q(v):
    return dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)), gt80=float((v > 80).mean()), gt200=float((v > 200).mean()))


# ----------------------------------------------------------------------------- (A) KITTI sky
ds = "kitti"; proto = bj.DATASETS[ds]
perframe = {}   # (cell, model) -> list of (clip, frame, sky_med, sky_px)
vis = {}
for key, frames in bj.select_clips(ds, "dev"):
    clean, raw_hw = bj.load_frames(ds, frames)
    gt = bj.load_gt(ds, frames)
    sky = kitti_sky_masks(bj.PATHS["kitti_root"], frames[:proto["max_len"]], proto["crop"], gt)
    for cell in ("b0s0", "llb3"):
        blur, sigma = bj.parse_cell(cell)
        seed = bj.noise_seed(ds, key, bj.seed_cell(blur) if isinstance(blur, bj.LowLight) else sigma)
        x = bj.degrade(clean, blur, sigma, seed)
        for m, p in pipes.items():
            d = run(p, x, raw_hw, True)
            H, W = gt.shape[1:]
            d = np.stack([cv2.resize(f, (W, H)) for f in d]) if d.shape[-2:] != (H, W) else d
            for t in range(len(d)):
                if sky[t].sum() >= 50:
                    perframe.setdefault((cell, m), []).append((key, t, float(np.median(d[t][sky[t]])), int(sky[t].sum())))
            if key == "2011_09_26_drive_0023_sync":
                v = d[109][sky[109]]
                res[f"A|{cell}|{m}"] = dict(q(v), n_sky_px=int(v.size), frame_max=float(d[109].max()))
                vis[f"{cell}|{m}"] = d[109].copy()
                vis[f"{cell}|rgb"] = x[109].permute(1, 2, 0).numpy(); vis["sky"] = sky[109]; vis["gt"] = gt[109]
    print("kitti", key, flush=True)
for (cell, m), v in perframe.items():
    a = np.array([r[2] for r in v])
    res[f"A_allframes|{cell}|{m}"] = dict(frames=len(a), median_of_frame_medians=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
                                         min=float(a.min()), max=float(a.max()), frac_frames_med_gt80=float((a > 80).mean()), frac_frames_med_gt200=float((a > 200).mean()))
json.dump({f"{c}|{m}": v for (c, m), v in perframe.items()}, open(out_small / "kitti_sky_perframe.json", "w"))
np.savez_compressed(out_big / "case_kitti0023_f109.npz", **vis)

# ----------------------------------------------------------------------------- (B) horse
paths = sorted(Path("/mnt/DATA/Horse/test_resized").glob("*.png"))
bj._vda_path()
from video_depth_anything.util.transform import Resize
resize = Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC)
rgb = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]
clean = torch.from_numpy(np.stack([np.clip(resize({"image": r.astype(np.float32) / 255})["image"], 0, 1) for r in rgb])).permute(0, 3, 1, 2).contiguous()
hw = rgb[0].shape[:2]
dump = np.load(D + "qualitative/ifig7_real_20261015/fig_teaser_many_real/real__horse/teaser_alldepth_real_horse.npz")
hd = {}
for m, p in pipes.items():
    for amp in (False, True):
        try:
            hd[(m, amp)] = run(p, clean, hw, amp)
        except torch.OutOfMemoryError:
            print("OOM", m, amp); torch.cuda.empty_cache()
H = hw[0]
sky_rows = slice(0, int(0.15 * H))
for (m, amp), d in hd.items():
    tag = "fp16" if amp else "fp32"
    res[f"B|{tag}|{m}"] = dict(sky_band_top15=q(d[:, sky_rows].ravel()), sky_band_frame25_median=float(np.median(d[25, sky_rows])), whole_frame_max=float(d.max()),
                               sky_med_per_frame_min=float(np.median(d[:, sky_rows], axis=(1, 2)).min()), sky_med_per_frame_max=float(np.median(d[:, sky_rows], axis=(1, 2)).max()),
                               sky_med_ratio_to_frame_max=float(np.mean(np.median(d[:, sky_rows], axis=(1, 2)) / d.reshape(len(d), -1).max(1))))
    for k in ("c3_raw", "ft"):
        if (m, k) in (("ramen_n46", "c3_raw"), ("ft_n46", "ft")):
            res[f"B|{tag}|{m}"]["max_abs_diff_vs_dump_m"] = float(np.abs(d - dump[k]).max())
            res[f"B|{tag}|{m}"]["dump_sky_median"] = float(np.median(dump[k][:, sky_rows]))
np.savez_compressed(out_big / "case_horse.npz", rgb25=rgb[25], **{f"{m}_{'fp16' if a else 'fp32'}": d for (m, a), d in hd.items()})
json.dump(res, open(out_small / "cases.json", "w"), indent=1)
print(json.dumps(res, indent=1)[:6000])

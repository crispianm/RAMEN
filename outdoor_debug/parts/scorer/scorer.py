#!/usr/bin/env python3
"""Outdoor-debug screen by-range scorer (parametrised successor of D_baseline/run_baseline.py + T_trajectory/run_traj_eval.py).

Usage:  python -I scorer.py --ckpt <RAMEN resumable .pt> --out <result.json> [--datasets kitti,nyuv2,scannetpp,vkitti,horse]
            [--device cuda] [--precision fp16|fp32] [--nyuv2-frames 160] [--limit-clips N] [--repo <repo root>] [--no-legacy-sample]
Reads DEV clips only (bench_joint.select_clips(ds, "dev")). Needs bench_joint.py with sha256 == BJ_SHA (asserted) and per-dataset clip-key shas == R3 pins (asserted).
The guards (unaligned AbsRel, tae_gtvalid, ...) are produced by UNCHANGED bench_joint functions (build_pipeline/run_pipeline/score_clip/camera/_vda5.score_slice), exactly as cmd_run
calls them for cell b0s0 with --temporal minus the tOF net. Only the by-range / pred:gt / sky statistics are computed here (pure functions at the top, unit-tested).
See README.md for every definition. The module imports without bench_joint / torch GPU (pure functions only need numpy)."""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

BJ_SHA = "5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef"
R3_KEY_SHA = {"kitti": "e187a28b87d43f3b3e6f687a0acc944f216fa754b4512fc373d616c5bece0a3a",
              "nyuv2": "49adc06914d2bfcdec0d602a37099f0289e498780302df39f512a010c6e58797",
              "scannetpp": "010a310b828dedeaac77700463d243644d83b35a8146d75fa88da79862bb37c6",
              "vkitti": "2f57ab4872f2ff8e82e86c30f89fa21de5db3351d457df7dc9bb4e7389f15073"}
R3_COMBINED = "aca96bbd894855cc99b8d97dbdf6e957d1c1f4efb0d49c77ff6108d8dccd7de6"
R3_N = {"kitti": 7, "nyuv2": 4, "scannetpp": 31, "vkitti": 10}
FAR_LO, SENTINEL_M = 80.0, 655.0            # VKITTI far bin: 80 <= gt < 655 (sentinel 655.35 m = 65535 cm and the 655 m sky label are excluded GT-side)
RANGES = {"0-10": (0.0, 10.0), "10-30": (10.0, 30.0), "30-80": (30.0, 80.0), ">80": (80.0, np.inf)}   # lo < gt <= hi (run_baseline/run_traj_eval)
GUARD_KEYS = ("abs_rel_unaligned", "rmse_unaligned", "delta1_unaligned", "abs_rel_aligned", "delta1_aligned", "tae", "tae_gtvalid",
              "sky_med", "sky_ge80", "sky_nearer_than_scene_p95", "far40_80_ratio_med")


# ----------------------------------------------------------------------------- pure functions (numpy; unit-tested)
def far_mask(gt: np.ndarray, lo: float = FAR_LO, sentinel: float = SENTINEL_M) -> np.ndarray:
    """VKITTI primary bin: lo <= gt < sentinel, finite. The sentinel (655.35 m / 65535 px, sky) and anything above is excluded here, on the GT, for every arm."""
    return np.isfinite(gt) & (gt >= lo) & (gt < sentinel)


def ratio_stats(ratios: np.ndarray) -> dict:
    """pred:gt statistic of a pooled pixel set. pooled_median = median over ALL pixels of (pred/gt) ('median of ratios', as D_baseline/T_trajectory). Also ratio_of_medians, mean."""
    if ratios.size == 0:
        return dict(n=0, pooled_median=float("nan"), mean=float("nan"), abs_dev=float("nan"))
    m = float(np.median(ratios))
    return dict(n=int(ratios.size), pooled_median=m, mean=float(np.mean(ratios)), abs_dev=abs(1.0 - m))


def ratio_of_medians(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.median(pred) / np.median(gt)) if pred.size else float("nan")


def range_masks(gt: np.ndarray, sky: np.ndarray | None = None, min_depth: float = 0.1) -> tuple:
    base = np.isfinite(gt) & (gt > min_depth) & (~sky if sky is not None else True)
    return {k: base & (gt > max(lo, 0.0)) & (gt <= hi) for k, (lo, hi) in RANGES.items()}, base


def absrel_sums(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> tuple:
    """(n, sum |p-g|/g) in float64 over mask (pixel-pooled AbsRel is sum/n). Pred is NOT clamped (as run_baseline 'raw')."""
    if not mask.any():
        return 0, 0.0
    p, g = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    return int(p.size), float((np.abs(p - g) / g).sum())


def perframe_absrel_unaligned(pred: np.ndarray, gt: np.ndarray, max_depth: float, min_depth: float) -> float:
    """bench_joint's abs_rel_unaligned arithmetic: pred clamped to [1e-3,max_depth], valid = min_depth<gt<max_depth, per-frame mean over valid px, mean over frames with any valid.
    (The scorer itself takes this number from bj.score_clip; this mirror exists for the unit test of the arithmetic.)"""
    vals = []
    for p, g in zip(pred, gt):
        v = (g > min_depth) & (g < max_depth)
        if v.any():
            pc = np.clip(p, 1e-3, max_depth)
            vals.append(float((np.abs(pc[v] - g[v]) / g[v]).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def sky_levels(frames_sky_px: list) -> dict:
    """frames_sky_px: list of 1-D arrays (predicted depth on sky px, one per frame with >=50 sky px). pooled median, median of per-frame medians, frame-balanced (<=3000 px/frame, seeded)."""
    if not frames_sky_px:
        return dict(frames=0, pooled_median=float("nan"), median_of_frame_medians=float("nan"), framebalanced_median=float("nan"))
    rng = np.random.default_rng(0)
    bal = np.concatenate([v if v.size <= 3000 else rng.choice(v, 3000, replace=False) for v in frames_sky_px])
    allv = np.concatenate(frames_sky_px)
    return dict(frames=len(frames_sky_px), n_px=int(allv.size), pooled_median=float(np.median(allv)),
                median_of_frame_medians=float(np.median([np.median(v) for v in frames_sky_px])), framebalanced_median=float(np.median(bal)),
                frac_gt80=float((allv > 80).mean()), frac_gt200=float((allv > 200).mean()))


def keys_sha(keys: list) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def combined_sha(by_ds: dict) -> str:
    return hashlib.sha256("\n".join(f"{ds}:{k}" for ds in ("kitti", "nyuv2", "scannetpp", "vkitti") for k in by_ds[ds]).encode()).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- main scoring
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="kitti,nyuv2,scannetpp,vkitti,horse")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"], help="fp16 autocast = how the earlier 4090 numbers were made; fp32 = bench_joint cmd_run")
    ap.add_argument("--nyuv2-frames", type=int, default=160, help="first N frames of each NYUv2 clip (160 = D_baseline; same for ALL arms; 0 = all 500)")
    ap.add_argument("--limit-clips", type=int, default=None, help="CPU unit tests only; the pins are NOT asserted when set")
    ap.add_argument("--max-frames", type=int, default=None, help="CPU smoke only: truncate every clip (pins not asserted)")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--horse-dir", default="/mnt/DATA/Horse/test_resized")
    ap.add_argument("--spec-prefix", default="ours:", help="bench_joint spec prefix; ours: for C3 / RAMEN resumable checkpoints")
    ap.add_argument("--no-legacy-sample", action="store_true")
    a = ap.parse_args(argv)
    import torch, cv2
    sys.path.insert(0, str(Path(a.repo).resolve()))
    import bench_joint as bj
    bj_sha = sha256_file(bj.__file__)
    assert bj_sha == BJ_SHA, f"bench_joint.py sha {bj_sha} != pinned {BJ_SHA}"
    pins_on = a.limit_clips is None and a.max_frames is None
    dsets = a.datasets.split(",")
    out_path = Path(a.out)
    assert not out_path.exists(), f"{out_path} exists (never overwrite)"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    wkey = "trainable_ema" if "trainable_ema" in ck else None
    assert wkey, "checkpoint has no trainable_ema (bench_joint.load_ours would silently use 'trainable'); refused"
    ck_meta = dict(path=str(a.ckpt), sha256=sha256_file(a.ckpt), weights_key=wkey, step=ck.get("step"), arch=(ck.get("cfg", {}).get("model", {}) or {}).get("arch"))
    del ck
    dev = a.device
    ac = dict(device_type="cuda", dtype=torch.float16, enabled=(a.precision == "fp16" and dev.startswith("cuda")))

    # ---- clip lists + pins (S2)
    clips = {ds: bj.select_clips(ds, "dev", a.limit_clips) for ds in ("kitti", "nyuv2", "scannetpp", "vkitti") if ds in dsets}
    clip_pins = {}
    for ds, cl in clips.items():
        ks = [k for k, _ in cl]
        clip_pins[ds] = dict(n=len(ks), keys_sha256=keys_sha(ks), pinned=R3_KEY_SHA[ds], keys=ks)
        if pins_on:
            assert clip_pins[ds]["keys_sha256"] == R3_KEY_SHA[ds] and len(ks) == R3_N[ds], f"{ds} clip-list sha {clip_pins[ds]['keys_sha256']} != R3 pin {R3_KEY_SHA[ds]}"
    comb = None
    if all(d in clips for d in ("kitti", "nyuv2", "scannetpp", "vkitti")):
        comb = combined_sha({d: [k for k, _ in clips[d]] for d in clips})
        if pins_on:
            assert comb == R3_COMBINED, f"combined clip sha {comb} != {R3_COMBINED}"

    pipe = bj.build_pipeline(a.spec_prefix + a.ckpt, dev)
    assert pipe["metric"], "scorer needs a metric model"
    t0 = time.time()

    def fwd(x, hw):
        with torch.autocast(**ac):
            d, r = bj.run_pipeline(pipe, x.to(dev), 0.0, 0.0, hw)
        return d, r

    def resize(d, H, W):
        return np.stack([cv2.resize(f, (W, H)) for f in d]) if d.shape[-2:] != (H, W) else d

    res = dict(guards={}, primary={}, by_range={}, secondary={}, pooled_absrel_raw={})
    legacy = not a.no_legacy_sample

    for ds in ("kitti", "nyuv2", "scannetpp", "vkitti"):
        if ds not in clips:
            continue
        proto = bj.DATASETS[ds]
        per_video, acc_rng = {}, {r: [0, 0.0] for r in RANGES}
        acc_pool = [0, 0.0]
        sky_frames = []; sky_all = []
        far_ratio, far_pred, far_gt, far_clip_med, near_ratio = [], [], [], [], []
        leg_sky, leg_near, leg_far = [], [], []
        for key, frames in clips[ds]:
            nmax = proto["max_len"]
            if ds == "nyuv2" and a.nyuv2_frames:
                nmax = min(nmax, a.nyuv2_frames)
            if a.max_frames:
                nmax = min(nmax, a.max_frames)
            frames = frames[:nmax]
            clean, raw_hw = bj.load_frames(ds, frames)
            gt = bj.load_gt(ds, frames)
            cam = None
            if ds in ("kitti", "scannetpp"):   # datasets with camera poses for TAE; nyuv2/vkitti have none (bj.camera raises for them), so no TAE there by construction
                try:
                    cam = bj.camera(ds, key, frames)
                except Exception as e:   # a missing camera/TAE input is a FAILED score, never a silent omission (R+D decision, Auditor S1)
                    raise RuntimeError(f"{ds}/{key}: camera failed ({e!r}); tae cannot be computed -> score failed") from e
            skym = None
            if proto.get("sky_mask") == "kitti_colour":
                from evaluation.sky_metrics import kitti_sky_masks
                skym = kitti_sky_masks(bj.PATHS[f"{ds}_root"], frames, proto["crop"], gt)
            blur, sigma = bj.parse_cell("b0s0")
            x = bj.degrade(clean, blur, sigma, bj.noise_seed(ds, key, sigma)).to(dev)     # identity for b0s0, kept for call parity with cmd_run
            depth, restored = fwd(x, raw_hw)
            sd, sg, sr, sx, sc, scam = bj._vda5.score_slice(proto, depth, gt, restored, x, clean.to(dev), cam)
            per_video[key] = bj.score_clip(sd, True, sg, sr, sx, sc, proto["max_depth"], proto["min_depth"], scam, None, None,
                                           sky_m=proto.get("sky_depth_m"), sky_mask=skym)
            # ---- own by-range statistics on the full-protocol (unsliced) prediction at GT size, float32 like run_baseline
            N, H, W = gt.shape
            pred = resize(depth, H, W).astype(np.float32)
            gtf = gt.astype(np.float32)
            sky = (gtf >= proto["sky_depth_m"]) if ds == "vkitti" else (skym if skym is not None else None)
            rm, base = range_masks(gtf, sky)
            for r, m in rm.items():
                n_, s_ = absrel_sums(pred, gtf, m); acc_rng[r][0] += n_; acc_rng[r][1] += s_
            n_, s_ = absrel_sums(pred, gtf, base); acc_pool[0] += n_; acc_pool[1] += s_
            if ds in ("kitti", "vkitti"):
                for t in range(N):
                    if sky is not None and int(sky[t].sum()) >= 50:
                        v = pred[t][sky[t]]; sky_frames.append(v.copy())
            if ds == "vkitti":
                fm = far_mask(gtf)
                r_ = pred[fm] / gtf[fm]
                far_ratio.append(r_); far_pred.append(pred[fm]); far_gt.append(gtf[fm])
                far_clip_med.append(float(np.median(r_)) if r_.size else float("nan"))
                nm_ = base & (gtf < 30); near_ratio.append(pred[nm_] / gtf[nm_])
                if legacy:                       # replicate run_traj_eval's seeded subsample (md5(key) generator; draws in the order sky, near<30, far)
                    g_ = torch.Generator().manual_seed(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
                    if int(sky.sum()):
                        v = pred[sky]; leg_sky.append(v[torch.randperm(v.size, generator=g_)[:200000].numpy()])
                    for nm, m, L in (("near", nm_, leg_near), ("far", fm, leg_far)):
                        if int(m.sum()):
                            v = pred[m] / gtf[m]; L.append(v[torch.randperm(v.size, generator=g_)[:100000].numpy()])
                sky_all.append(pred[sky][::1])
            print(f"{ds} {key} frames={N} {time.time() - t0:.0f}s", flush=True)
            del depth, pred, restored, x
        # ---- aggregate
        keys = [k for k in GUARD_KEYS if all(k in v for v in per_video.values())]
        res["guards"][ds] = dict(video_macro={k: float(np.nanmean([v[k] for v in per_video.values()])) for k in keys}, n_clips=len(per_video),
                                 per_video={k: {kk: v[kk] for kk in keys} for k, v in per_video.items()},
                                 frames_per_clip_cap=("all" if not (ds == "nyuv2" and a.nyuv2_frames) else a.nyuv2_frames))
        res["pooled_absrel_raw"][ds] = dict(n=acc_pool[0], value=acc_pool[1] / max(acc_pool[0], 1), mask="gt>0.1 & ~sky (no 80 m cap), pred unclamped, pixel-pooled")
        if ds in ("kitti", "vkitti"):
            res["by_range"][ds] = {r: dict(n=n_, absrel_raw=s_ / max(n_, 1)) for r, (n_, s_) in acc_rng.items()}
            res["secondary"][f"{ds}_sky"] = sky_levels(sky_frames)
        if ds == "vkitti":
            R = np.concatenate(far_ratio) if far_ratio else np.zeros(0, np.float32)
            P, G = (np.concatenate(far_pred), np.concatenate(far_gt)) if far_pred else (np.zeros(0), np.zeros(0))
            st = ratio_stats(R)
            st.update(ratio_of_medians=ratio_of_medians(P, G), per_clip_median_mean=float(np.nanmean(far_clip_med)) if far_clip_med else float("nan"),
                      bin_absrel=float((np.abs(P.astype(np.float64) - G) / G).mean()) if P.size else float("nan"),
                      statistic="pooled median over all px of pred/gt in 80<=gt<655 (median of ratios)",
                      mask=f"{FAR_LO}<=gt<{SENTINEL_M}, finite; sentinel 655.35 excluded on GT")
            if legacy and leg_far:
                lf = np.concatenate(leg_far); st["legacy_subsample_median"] = float(np.median(lf)); st["legacy_subsample_n"] = int(lf.size)
            res["primary"]["vkitti_far_80_655"] = st
            nr = np.concatenate(near_ratio) if near_ratio else np.zeros(0)
            res["secondary"]["vkitti_near_lt30_pred_over_gt"] = ratio_stats(nr)
            if legacy and leg_near:
                res["secondary"]["vkitti_near_lt30_pred_over_gt"]["legacy_subsample_median"] = float(np.median(np.concatenate(leg_near)))
            if legacy and leg_sky:
                res["secondary"]["vkitti_sky"]["legacy_subsample_median"] = float(np.median(np.concatenate(leg_sky)))
            sa = np.concatenate(sky_all) if sky_all else np.zeros(0)
            res["secondary"]["vkitti_sky"]["exact_pooled_median_all_px"] = float(np.median(sa)) if sa.size else float("nan")

    if "horse" in dsets:
        bj._vda_path()
        from video_depth_anything.util.transform import Resize
        paths = sorted(Path(a.horse_dir).glob("*.png"))
        if a.max_frames:
            paths = paths[:a.max_frames]
        rs = Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC)
        rgb = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]
        cl = torch.from_numpy(np.stack([np.clip(rs({"image": r.astype(np.float32) / 255})["image"], 0, 1) for r in rgb])).permute(0, 3, 1, 2).contiguous()
        d, _ = fwd(cl, rgb[0].shape[:2])
        H = rgb[0].shape[0]
        bands = {"sky_top15": slice(0, int(.15 * H)), "treeline_38_46": slice(int(.38 * H), int(.46 * H)), "ground_bottom10": slice(int(.9 * H), H)}
        hs = {f"{b}_median_m": float(np.median(d[:, sl])) for b, sl in bands.items()}
        if len(d) > 25:
            hs.update({f"{b}_f25_median_m": float(np.median(d[25, sl])) for b, sl in bands.items()})
        hs.update(whole_clip_max_m=float(d.max()), n_frames=len(d), frames_dir=str(a.horse_dir))
        res["secondary"]["horse"] = hs
    res["secondary"]["tartanair_far"] = None
    res["secondary"]["tartanair_far_note"] = "NOT IMPLEMENTED: no earlier TartanAir eval-clip definition exists (only the train-batch statistic in T_trajectory/grad_traj.py); see README open question"

    meta = dict(scorer="outdoor_debug/parts/scorer/scorer.py", scorer_sha256=sha256_file(__file__), bench_joint_sha256=bj_sha, bench_joint_pinned=BJ_SHA,
                ramen_commit=bj.git_state(bj.REPO), ckpt=ck_meta, clip_lists=clip_pins, combined_clip_sha256=comb, combined_pinned=R3_COMBINED, pins_asserted=pins_on,
                datasets=dsets, precision=a.precision, device=dev, nyuv2_frames=a.nyuv2_frames, limit_clips=a.limit_clips, max_frames=a.max_frames, spec=a.spec_prefix + "<ckpt>",
                host=__import__("socket").gethostname(), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", torch=torch.__version__,
                cell="b0s0 (clean), split dev", seconds=time.time() - t0)
    res = dict(meta=meta, **res)
    if "vkitti_far_80_655" in res["primary"]:
        res["primary"]["success_rule_quantity_abs_1_minus_pred_gt"] = res["primary"]["vkitti_far_80_655"]["abs_dev"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", out_path)


if __name__ == "__main__":
    main()

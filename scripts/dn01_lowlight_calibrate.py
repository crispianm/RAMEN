"""DN-01 final-v2 `lowlight` calibration on paired BVI-RLV (real low light vs normal light, same scene): the exposure (EV)
range and the noise-amplitude scale kappa for data.elvis_noise.degrade_clip_lowlight. At bench_joint's model resolution
(short side 518), where training and eval apply the degradation. The 3 clips of the real low-light figure are held out.
  per clip (scene x light level 10/20), first --frames frames:
    EV_real = log2(mean(low) / mean(normal));  sigma_real = bench_joint.estimate_sigma(low)  (Donoho MAD, /255)
    sigma_syn(kappa) = median over --profiles seeded bank profiles of estimate_sigma(lowlight(normal, EV_real, kappa))
  kappa: two fixed-point steps kappa <- kappa * median(sigma_real / sigma_syn(kappa)) from 1 (clipping at 0 makes the
  map mildly non-linear); reported with the calibrated median / p10 / p90 of sigma_syn / sigma_real.
  python scripts/dn01_lowlight_calibrate.py --out results/lowlight_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_joint as bj  # noqa: E402
from data.elvis_noise import degrade_clip_lowlight, load_bank  # noqa: E402

ROOT = Path("/mnt/DATA/DegradedDepth/data/test/BVI-RLV/BVI-Lowlight-videos")
HELD_OUT = {"S09_kitchen", "S18_books1", "S21_mario1"}     # the real low-light figure's clips


def load(folder: Path, n: int) -> np.ndarray:
    bj._vda_path()
    from video_depth_anything.util.transform import Resize
    rs = Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC)
    imgs = [rs({"image": cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0})["image"]
            for f in sorted(folder.glob("*.png"))[:n]]
    return np.clip(np.stack(imgs), 0, 1).transpose(0, 3, 1, 2).astype(np.float32)   # (T,3,H,W)


def sigma(x: np.ndarray) -> float:
    return bj.estimate_sigma(torch.from_numpy(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--profiles", type=int, default=16)
    a = ap.parse_args()
    bank = load_bank()
    prof_rng = np.random.RandomState(0)
    profiles = [bank[prof_rng.randint(bank.shape[0])].mean(0) for _ in range(a.profiles)]
    clips = []
    for scene in sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name not in HELD_OUT):
        for lvl in ("10", "20"):
            lo, no = ROOT / scene / f"low_light_{lvl}", ROOT / scene / f"normal_light_{lvl}"
            if not (lo.is_dir() and no.is_dir()):
                continue
            low, normal = load(lo, a.frames), load(no, a.frames)
            clips.append({"scene": scene, "level": lvl, "normal": normal,
                          "ev": float(np.log2(max(low.mean(), 1e-6) / max(normal.mean(), 1e-6))),
                          "sigma_real": sigma(low), "sigma_normal": sigma(normal)})
            print(scene, lvl, f"EV {clips[-1]['ev']:+.2f}  sigma_real {clips[-1]['sigma_real']:.2f}", flush=True)

    def syn(c, kappa):
        return float(np.median([sigma(degrade_clip_lowlight(c["normal"], c["ev"], phi, kappa, 0.0,
                                                            np.random.RandomState(1000 + i)))
                                for i, phi in enumerate(profiles)]))
    kappa, steps = 1.0, []
    for _ in range(2):
        ratios = [c["sigma_real"] / syn(c, kappa) for c in clips]
        kappa *= float(np.median(ratios))
        steps.append(kappa)
    check = np.array([syn(c, kappa) / c["sigma_real"] for c in clips])
    ev = np.array([c["ev"] for c in clips])
    res = {"source": str(ROOT), "held_out": sorted(HELD_OUT), "frames": a.frames, "profiles": a.profiles,
           "short_side": 518, "num_clips": len(clips), "kappa": kappa, "kappa_steps": steps,
           "sigma_syn_over_real_at_kappa": {"median": float(np.median(check)), "p10": float(np.percentile(check, 10)),
                                            "p90": float(np.percentile(check, 90))},
           "sigma_uncalibrated_over_real_median": float(np.median([syn(c, 1.0) / c["sigma_real"] for c in clips])),
           "ev": {"min": float(ev.min()), "p10": float(np.percentile(ev, 10)), "median": float(np.median(ev)),
                  "p90": float(np.percentile(ev, 90)), "max": float(ev.max()),
                  "by_level": {lv: float(np.median([c["ev"] for c in clips if c["level"] == lv])) for lv in ("10", "20")}},
           "ramen_commit": bj.git_state(bj.REPO),
           "per_clip": [{k: v for k, v in c.items() if k != "normal"} for c in clips]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))


if __name__ == "__main__":
    main()

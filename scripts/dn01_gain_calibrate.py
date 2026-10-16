"""DN-01 v2 blind display gain (Auditor K43(b)): fit bench_joint's GAIN_STAT / GAIN_REF on TRAINING clips only, before
any v2 job. Clips come through the real loader (train.build_dataloader on the v2 C-LL config, local VKITTI/IRS roots, the
dataset attributes train.py sets, severity cap 1.0 -- as the hunt's scripts/probe_loader.py). Truth per clip: the stops
of darkening, -EV (= -log2 degrade_params[2]) for a darkened lowlight clip, 0 otherwise. For each candidate statistic
(bench_joint.gain_stat: per-frame luma mean or percentile, median over frames) GAIN_REF is the grid value minimising the
mean |log2 g - (-EV)| on the FIT clips, g = clip(GAIN_REF / stat, 1, GAIN_MAX); the statistic with the lowest fit error
wins. The CHECK clips (disjoint draws) report the chosen rule's error. No eval clip is read.
  python scripts/dn01_gain_calibrate.py --config configs/experiments/dn01_v2_ll_ctrl.yaml --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_joint as bj  # noqa: E402
import train  # noqa: E402

LOCAL = {"vkitti": "/mnt/DATA/DegradedDepth/data/TRAIN/vkitti", "irs": "/mnt/DATA/DegradedDepth/data/TRAIN/irs"}
STATS = ["mean", "q50", "q90", "q95", "q99", "q99.5"]
GRID = 2.0 ** np.linspace(-4.0, 1.0, 1001)


def err(stat: np.ndarray, stops: np.ndarray, ref: float) -> np.ndarray:
    return np.abs(np.log2(np.clip(ref / np.maximum(stat, 1e-4), 1.0, bj.GAIN_MAX)) - stops)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit", type=int, default=400)
    ap.add_argument("--check", type=int, default=200)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--only-source", default=None, help="keep one data source (a per-source error report)")
    a = ap.parse_args()
    cfg = train.load_config(Path(a.config))
    if a.only_source:
        cfg["data"]["sources"] = [s for s in cfg["data"]["sources"] if s["type"] == a.only_source]
    for s in cfg["data"]["sources"]:
        s["root"] = LOCAL[s["type"]]
    cfg["data"].update(num_frames=a.num_frames, num_workers=12)
    cfg["training"]["batch_size"] = 4
    torch.manual_seed(1234)
    loader, _ = train.build_dataloader(cfg, "train")
    d = cfg["data"]
    for k, v in [("clean_prob", float(d["clean_prob"])), ("seeded_noise", bool(d["seeded_noise"])),
                 ("blur_only_prob", float(d["blur_only_prob"])), ("lowlight", d.get("lowlight"))]:
        train._set_dataset_attr(loader.dataset, k, v)
    train._set_severity_max(loader.dataset, 1.0)
    rows = []
    for batch in loader:
        for x, p in zip(batch["degraded_frames"], batch["degrade_params"]):
            p = p.numpy().astype(np.float64)
            stops = float(-np.log2(p[2])) if p[0] > 0 and p[2] < 1.0 else 0.0
            rows.append({"stops": stops, "params": p.tolist(),
                         **{s: bj.gain_stat(x.permute(1, 0, 2, 3), s) for s in STATS}})
        if len(rows) >= a.fit + a.check:
            break
    fit, chk = rows[:a.fit], rows[a.fit:a.fit + a.check]
    stops_f, stops_c = np.array([r["stops"] for r in fit]), np.array([r["stops"] for r in chk])
    res = {}
    for s in STATS:
        v = np.array([r[s] for r in fit])
        e = np.array([err(v, stops_f, g).mean() for g in GRID])
        res[s] = {"ref": float(GRID[e.argmin()]), "fit_mae_stops": float(e.min())}
    best = min(res, key=lambda s: res[s]["fit_mae_stops"])
    ref = res[best]["ref"]
    ec = err(np.array([r[best] for r in chk]), stops_c, ref)
    dark = stops_c > 0
    summary = {"chosen": {"GAIN_STAT": best, "GAIN_REF": ref, "GAIN_MAX": bj.GAIN_MAX},
               "candidates": res,
               "check": {"n": len(chk), "mae_stops": float(ec.mean()), "median_stops": float(np.median(ec)),
                         "mae_stops_darkened": float(ec[dark].mean()) if dark.any() else None,
                         "mae_stops_normal": float(ec[~dark].mean()) if (~dark).any() else None,
                         "frac_normal_brightened_over_half_stop": float((ec[~dark] > 0.5).mean())
                         if (~dark).any() else None, "n_darkened": int(dark.sum())},
               "config": a.config, "num_frames": a.num_frames, "ramen_commit": bj.git_state(bj.REPO),
               "uncommitted_diff_sha256": hashlib.sha256(subprocess.run(
                   ["git", "-C", str(bj.REPO), "diff", "HEAD"], capture_output=True).stdout).hexdigest(),
               "script_sha256": bj.sha256_file(__file__),
               "rows": rows}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gain_calibration.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()

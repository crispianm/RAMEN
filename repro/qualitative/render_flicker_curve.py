#!/usr/bin/env python3
"""fig:flicker_curve (pre-declared definition, Auditor R1 ruling 9, 2026-10-14). DEV only, descriptive, labelled "development clips".

For every static DEV clip of CLIP_SELECTION.json (cell st_llvar, 64 frames: one repeated still, fresh noise per frame) and every model of fig:qual
(FT n46 = seed 46, RAMEN n46 raw, RAMEN n46 + TAG; the released Metric-VDA-S curve is retired, commit Ifig):
    c_t = mean over GT-valid pixels of |D_t - D_{t-1}| / Dbar,   Dbar = temporal-mean depth per pixel over frames 7..63,   t = 8..63
(depth as the scorer sees it: resized to the GT size and clamped to [min_depth, max_depth]; this is the per-frame decomposition of bench_joint.flicker_static,
whose scored value is the mean of c_t over t = 8..63). The curve is the mean over clips of c_t at each t (+ s.e.m.). Writes flicker_curve.json (per-clip series, curves, definition) and flicker_curve.png.
The pure functions (c_series, curve_over_clips) import only numpy; the inference path imports the pinned scorer clone lazily (GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T_FIRST, T_LAST, DBAR_FROM = 8, 63, 7
DEFINITION = "c_t = mean_{GT-valid px} |D_t - D_{t-1}| / mean_{t'=7..63} D_{t'},  t = 8..63; static DEV clips of CLIP_SELECTION.json; development clips; descriptive"


def c_series(depth: np.ndarray, gt: np.ndarray, max_depth: float, min_depth: float) -> np.ndarray:
    """depth (N,H0,W0) metric metres, gt (N,H,W) of a STATIC clip (N >= 64). Returns c_t for t = 8..63 (length 56). GT-valid = min_depth < GT[0] < max_depth (the scorer's flicker mask)."""
    H, W = gt.shape[-2:]
    if depth.shape[-2:] != (H, W):
        import cv2
        depth = np.stack([cv2.resize(d, (W, H)) for d in depth])
    d = np.clip(depth.astype(np.float64), min_depth, max_depth)
    if d.shape[0] <= T_LAST:
        raise ValueError(f"need frames up to index {T_LAST}, got {d.shape[0]}")
    g0 = gt[0].astype(np.float64)
    valid = (g0 > min_depth) & (g0 < max_depth)
    dbar = d[DBAR_FROM:T_LAST + 1].mean(0)
    return np.array([float((np.abs(d[t] - d[t - 1]) / dbar)[valid].mean()) for t in range(T_FIRST, T_LAST + 1)])


def curve_over_clips(series: list[np.ndarray]) -> dict:
    m = np.stack(series)
    n = m.shape[0]
    return {"t": list(range(T_FIRST, T_LAST + 1)), "mean": m.mean(0).tolist(), "sem": (m.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(m.shape[1])).tolist(), "n_clips": n}


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def apply_host_paths(cfg, host, eval_clone):
    """Path selection only (no value is computed here): --host gh200 swaps each checkpoint to its Isambard path (checkpoint sha256 pins stay as in the config and are asserted by the caller);
    --eval-clone overrides the scorer clone (the caller still asserts HEAD == eval_commit and a clean tree)."""
    if eval_clone:
        cfg["eval_clone"] = str(eval_clone)
    if host == "gh200":
        for c in [cfg["c3"]] + ([cfg["ft"][k] for k in cfg["ft"]] if "ft" in cfg else []):
            if c.get("ckpt_gh200"):
                c["ckpt"] = c["ckpt_gh200"]
            else:
                c["ckpt"] = None          # no Isambard path declared: the checkpoint assert fails loudly
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "figures_config.json"))
    ap.add_argument("--selection", default=str(HERE / "CLIP_SELECTION.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-clips", type=int, default=None, help="smoke: first N selection entries")
    ap.add_argument("--host", choices=["local", "gh200"], default="local", help="checkpoint path set (config ckpt_gh200 on gh200); default local = the config paths")
    ap.add_argument("--eval-clone", default=None, help="override the scorer clone path (still asserted at eval_commit and clean)")
    a = ap.parse_args(argv)
    cfg, selj = json.load(open(a.config)), json.load(open(a.selection))
    apply_host_paths(cfg, a.host, a.eval_clone)
    sys.path.insert(0, cfg["eval_clone"])
    import torch
    import bench_joint as bj
    assert subprocess.check_output(["git", "-C", cfg["eval_clone"], "rev-parse", "HEAD"], text=True).strip() == cfg["eval_commit"], "eval clone not at the scorer commit"
    assert not subprocess.check_output(["git", "-C", cfg["eval_clone"], "status", "--porcelain"], text=True).strip(), "eval clone not clean"
    ft = cfg["ft"][selj["FT_CHOICE"]]
    for nm, c in (("ramen", cfg["c3"]), ("ft", ft)):
        assert sha(c["ckpt"]) == c["sha256"], f"{nm} checkpoint sha mismatch"
    specs = {"ft": f"{bj.FT_VDA}{ft['ckpt']}", "ramen": f"ours:{cfg['c3']['ckpt']}"}      # Ifig: released curve retired
    names = {"ft": "Metric-VDA-S FT (seed 46)", "ramen": "RAMEN (raw)", "ramen_tag": "RAMEN + TAG"}
    clips, seen = [], set()
    for s in selj["selected"]:                                             # median clips + failure case; each (dataset, clip) once
        if (s["dataset"], s["clip"]) not in seen:
            seen.add((s["dataset"], s["clip"]))
            clips.append((s["dataset"], s["clip"]))
    if a.max_clips:
        clips = clips[:a.max_clips]
    cell = "st_llvar"
    inputs = {}
    for ds, key in clips:                                                   # DEV only: bench_joint.load_clips(ds, 'dev'); allow_test is never passed
        frames = dict(bj.load_clips(ds, "dev"))[key]
        used = frames[:bj.DATASETS[ds]["max_len"]]
        mid = [used[len(used) // 2]]
        clean, hw = bj.load_frames(ds, mid)
        gt = bj.load_gt(ds, mid)
        clean, gt = bj.static_clip(clean, gt)
        seed = bj.noise_seed(ds, key, bj.seed_cell(cell))
        x = bj.degrade_static(clean, cell, seed)
        inputs[(ds, key)] = (x, gt, hw, seed)
    per = {k: {} for k in names}
    for mk, spec in specs.items():
        p = bj.build_pipeline(spec, a.device)
        for (ds, key), (x, gt, hw, seed) in inputs.items():
            proto = bj.DATASETS[ds]
            blur, sigma = bj.parse_cell(cell)
            depth, _ = bj.run_pipeline(p, x.to(a.device), sigma, blur, hw)
            variants = {mk: depth}
            if mk == "ramen":
                variants["ramen_tag"] = bj.apply_variant(cfg["tag"], depth, p["last_guides"])
            for vk, dv in variants.items():
                per[vk][f"{ds}/{key}"] = c_series(dv, gt, proto["max_depth"], proto["min_depth"]).tolist()
        del p
        if a.device == "cuda":
            torch.cuda.empty_cache()
    curves = {k: curve_over_clips([np.array(v) for v in per[k].values()]) for k in names}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"item": "fig:flicker_curve", "definition": DEFINITION, "label": "development clips; descriptive", "models": names, "specs": specs, "cell": cell, "tag": cfg["tag"],
           "clips": [f"{d}/{k}" for d, k in clips], "per_clip_series": per, "curves": curves, "selection_sha256": sha(a.selection), "figures_config_sha256": sha(a.config), "host_paths": a.host, "eval_clone": cfg["eval_clone"],
           "ckpts": {"ramen": {"path": cfg["c3"]["ckpt"], "sha256": cfg["c3"]["sha256"]}, "ft": {"path": ft["ckpt"], "sha256": ft["sha256"]}}, "script_sha256": sha(__file__),
           "split": "dev", "torch": torch.__version__, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "flicker_curve.json").write_text(json.dumps(doc, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for k, nm in names.items():
        c = curves[k]
        ax.plot(c["t"], c["mean"], label=nm, lw=1.6)
        ax.fill_between(c["t"], np.array(c["mean"]) - np.array(c["sem"]), np.array(c["mean"]) + np.array(c["sem"]), alpha=0.15)
    ax.set_xlabel("frame t")
    ax.set_ylabel("per-frame relative depth change $c_t$")
    ax.set_title("static scenes, development clips (descriptive)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "flicker_curve.png", dpi=200)
    fig.savefig(out / "flicker_curve.pdf")                                  # paper-ready vector PDF (same artist data as the PNG)
    print("rendered flicker_curve", flush=True)


if __name__ == "__main__":
    main()

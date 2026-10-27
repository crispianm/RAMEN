#!/usr/bin/env python3
"""STATUS-PAGE COVER render (AUD-COVER-20261012). NOT a paper figure. Number-free panels (names only).

Layout (paper-teaser layout, paper_cvpr2027/figures/make_figures.py::fig_teaser):
  row 1: Degraded input | Released Metric-VDA-S depth | C3 + TAG depth | Change: released | Change: C3 + TAG
  row 2: (blank)        | y-t slice: released        | y-t slice: C3 + TAG | (blank) | (blank)
Clip rgbd_bonn_balloon_tracking (DEV), static low-light blur cell st_llb3 (one still repeated, fresh noise per frame), RAMEN (ramen.pt, 334623ec) with TAG-S (agf0.2t0.06).

  python3 -B render_cover.py --out /mnt/DATA/DegradedDepth/qualitative/13_status_cover                       # real run (4090), after the paper renders
  python3 -B render_cover.py --smoke --device cpu --max-frames 10 --out <scratch>                             # CPU plumbing smoke (NOT a figure)

Everything model-side goes through the pinned benchmark clone (bench_joint @368b5c53), exactly as render_c3_figures.py (reused read-only: helpers imported from the cleared figures package,
its sha256 pinned in cover_config.json; the nested run_models is re-stated here because it is local to that script's main()).
The y-t slice rule is FIXED before any render (cover_config.json 'slice_rule'): column round(0.671875 * panel_width) (= x0 430 of 640, the paper's fig_teaser choice), all rows, frames 1..N-1, of the relative change maps."""
import argparse, hashlib, json, os, socket, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
CFG = json.load(open(HERE / "cover_config.json"))
CFG["figures_package"] = str(HERE)   # VENDORED: import path only (was the session_evidence package dir)
sys.path.insert(0, CFG["figures_package"])
import render_c3_figures as rf  # noqa: E402   (read-only reuse of the cleared figures package's helpers)


def sha(p):
    return rf.sha(p)


def apply_host_paths(cfg, host, eval_clone):
    """Path selection only (no value is computed here): --host gh200 swaps each checkpoint to its Isambard path (checkpoint sha256 pins stay as in the config and are asserted by the caller);
    --eval-clone overrides the scorer clone (the caller still asserts HEAD == eval_commit and a clean tree)."""
    if eval_clone:
        cfg["eval_clone"] = str(eval_clone)
    if host == "gh200":
        for c in [cfg["c3"]] + []:
            if c.get("ckpt_gh200"):
                c["ckpt"] = c["ckpt_gh200"]
            else:
                c["ckpt"] = None          # no Isambard path declared: the checkpoint assert fails loudly
    return cfg


def slice_column(width, cfg=CFG):
    """Pre-declared y-t slice column for a panel of the given pixel width."""
    return int(round(cfg["slice_rule"]["x0_fraction_of_width"] * width))


def change_maps(d):
    """|D_t - D_{t-1}| / D_t for t=1..N-1 (the figures package's teaser formula)."""
    return np.abs(d[1:] - d[:-1]) / np.clip(d[1:], 1e-3, None)


def yt_slice(chg_panels, x):
    """y-t slice of a list of RGB change panels [(H,W,3)...] (t=1..N-1) at column x -> (H, T, 3)."""
    return np.stack([p[:, x] for p in chg_panels], axis=1)


def compose(names, row1, row2, header_fn=rf.header, cv2=None):
    """Stack header + row1 + optional row2; every panel is resized to the common (h, w) of row1[0]."""
    import cv2 as _cv2
    cv2 = cv2 or _cv2
    ph, w = row1[0].shape[:2]
    r1 = np.concatenate([cv2.resize(q, (w, ph)) for q in row1], 1)
    r2 = np.concatenate([cv2.resize(q, (w, ph), interpolation=cv2.INTER_NEAREST) if q is not None else np.full((ph, w, 3), 255, np.uint8) for q in row2], 1)
    return np.concatenate([header_fn(names, w), r1, r2], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="plumbing smoke: first --max-frames frames, output labelled SMOKE_, not a figure")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--panel-h", type=int, default=240)
    ap.add_argument("--host", choices=["local", "gh200"], default="local", help="checkpoint path set (config ckpt_gh200 on gh200); default local = the config paths")
    ap.add_argument("--eval-clone", default=None, help="override the scorer clone path (still asserted at eval_commit and clean)")
    a = ap.parse_args()
    cfg = CFG
    apply_host_paths(cfg, a.host, a.eval_clone)
    assert sha(Path(cfg["figures_package"]) / "render_c3_figures.py").startswith(cfg["figures_render_script_sha256_prefix"]), "figures package render script changed since this package was built"
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, cfg["eval_clone"])
    import torch, cv2
    import bench_joint as bj
    clone_head = subprocess.check_output(["git", "-C", cfg["eval_clone"], "rev-parse", "HEAD"], text=True).strip()
    porcelain = subprocess.check_output(["git", "-C", cfg["eval_clone"], "status", "--porcelain"], text=True).strip()
    assert clone_head == cfg["eval_commit"] and not porcelain, f"eval clone not at {cfg['eval_commit']} or not clean"
    c3 = cfg["c3"]
    assert Path(c3["ckpt"]).exists() and sha(c3["ckpt"]) == c3["sha256"], "RAMEN checkpoint missing or sha mismatch"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = {"released": cfg["released"]["spec"], "c3": f"ours:{c3['ckpt']}"}
    dev = a.device
    script_sha = sha(__file__)

    def run_models(x, sigma, blur, raw_hw, names):
        res = {}
        for nm in names:
            p = bj.build_pipeline(specs[nm], dev)
            depth, _ = bj.run_pipeline(p, x.to(dev), sigma, blur, raw_hw)
            res[nm] = depth
            if nm == "c3":
                res["c3_tag"] = bj.apply_variant(cfg["tag"], depth, p["last_guides"])
            res.setdefault("_weights", {})[nm] = {k: v for k, v in p["weights"].items()}
            del p
            if dev == "cuda":
                torch.cuda.empty_cache()
        return res

    ds, key, cell = cfg["dataset"], cfg["clip"], cfg["cell"]
    proto = bj.DATASETS[ds]
    frames = dict(bj.load_clips(ds, cfg["split"]))[key]             # DEV only (KeyError if the clip is not DEV); allow_test is never passed
    clean, raw_hw = bj.load_frames(ds, frames)
    gt = bj.load_gt(ds, frames)
    if a.max_frames:
        clean, gt = clean[:a.max_frames], gt[:a.max_frames]
    tag = "SMOKE_" if a.smoke else ""
    base = f"{tag}cover_{ds}_{key}_{cell}"
    t0 = time.time()
    sclean, sgt = bj.static_clip(clean, gt) if not a.max_frames else bj.static_clip(clean, gt, n=a.max_frames)
    seed = bj.noise_seed(ds, key, bj.seed_cell(cell))
    x = bj.degrade_static(sclean, cell, seed)
    blur, sigma = bj.parse_cell(cell)
    r = run_models(x, sigma, blur, raw_hw, ["released", "c3"])
    dr, dc = r["released"].astype(np.float64), r["c3_tag"].astype(np.float64)
    mid = sgt.shape[0] // 2
    valid = (sgt[mid] > 1e-3) & (sgt[mid] < proto["max_depth"])
    gmid = sgt[mid][valid]
    lo, hi = float(np.percentile(gmid, 2)), float(np.percentile(gmid, 98))
    cr, cc = change_maps(dr), change_maps(dc)
    N = x.shape[0]
    fi = max(1, N // 2)
    ph = a.panel_h
    names = ["Degraded input", "Released Metric-VDA-S", "C3 + TAG", "Change: released", "Change: C3 + TAG"]
    names2 = ["", "y-t slice: released", "y-t slice: C3 + TAG", "", ""]
    # panel sizes follow the input aspect
    inp_t = lambda t: rf.resize_h(rf.u8(x[t]), ph)
    w = inp_t(fi).shape[1]
    rs = lambda im: cv2.resize(im, (w, ph), interpolation=cv2.INTER_AREA)
    chg_rel_panels = [rf.colour(cr[t - 1] / 0.03, "inferno") for t in range(1, N)]
    chg_ours_panels = [rf.colour(cc[t - 1] / 0.03, "inferno") for t in range(1, N)]
    xs = slice_column(chg_rel_panels[0].shape[1])
    sl_rel = cv2.resize(yt_slice(chg_rel_panels, xs), (w, ph), interpolation=cv2.INTER_NEAREST)
    sl_ours = cv2.resize(yt_slice(chg_ours_panels, xs), (w, ph), interpolation=cv2.INTER_NEAREST)
    def marks(im, t):
        """vertical green marker at the current frame in a y-t slice panel"""
        m = im.copy()
        m[:, min(w - 1, int((t - 1) / max(1, N - 2) * (w - 1)))] = (47, 176, 90)
        return m

    def comp(t):
        inp = inp_t(t).copy()
        cx = int(round(xs / chg_rel_panels[0].shape[1] * w))
        inp[:, max(0, cx - 1):cx + 1] = (47, 176, 90)                 # green line = the y-t slice column
        row1 = [inp, rs(rf.depth_rgb(dr[t], lo, hi)), rs(rf.depth_rgb(dc[t], lo, hi)), rs(chg_rel_panels[t - 1]), rs(chg_ours_panels[t - 1])]
        row2 = [None, marks(sl_rel, t), marks(sl_ours, t), None, None]
        img = compose(names, row1, row2, cv2=cv2)
        # second header strip for row 2 labels (names only, number-free)
        bar = rf.header(names2, w)
        return np.concatenate([img[:ph + 26], bar, img[ph + 26:]], 0)

    still = comp(fi)
    cv2.imwrite(str(out / f"{base}_still.png"), still[..., ::-1])
    rf.save_gif([comp(t) for t in range(1, N)], out / f"{base}.gif")
    side = {"panels": names + names2[1:3], "depth_range_m": [lo, hi], "depth_range_rule": "GT 2nd-98th percentile of the middle frame's valid GT, log, turbo, identical in both depth panels",
            "change_range": "0-3 % of depth (relative |D_t - D_{t-1}| / D_t), inferno", "yt_slice_rule": cfg["slice_rule"], "yt_slice_column_px": xs, "yt_slice_panel_width_px": int(chg_rel_panels[0].shape[1]),
            "tag": cfg["tag"], "cell": cell, "seed": seed, "frames": int(N), "still_frame_index": fi, "fps": 10, "models": {k: specs[k] for k in ("released", "c3")}}
    prov = {"PROVENANCE": "status-page cover render, AUD-COVER-20261012", "label": cfg["label"], "scope": cfg["scope"], "smoke": bool(a.smoke),
            "item": {"dataset": ds, "clip": key, "cell": cell, "split": cfg["split"]},
            "noise_draw_note": "noise draw = scored-row stream, differs from the previous cover (V2 n45 step4000, seed 0)", "script": "render_cover.py", "script_sha256": script_sha, "config_sha256": sha(HERE / "cover_config.json"),
            "figures_render_script_sha256": sha(Path(cfg["figures_package"]) / "render_c3_figures.py"),
            "ckpts": {"c3": {"path": c3["ckpt"], "sha256": c3["sha256"]}},
            "released_weights": {k: {"path": v, "sha256": sha(v)} for k, v in r.get("_weights", {}).get("released", {}).items()},
            "frames_used": [frames[clean.shape[0] // 2]["image"]], "eval_clone_head": clone_head, "eval_clone_porcelain_empty": not porcelain,
            "torch": torch.__version__, "device": dev, "gpu": torch.cuda.get_device_name() if dev == "cuda" and torch.cuda.is_available() else "cpu", "host": socket.gethostname(),
            "split": "dev (bench_joint.load_clips('dev'); allow_test never passed)", "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numbers_on_images": False, "sidecar": side}
    json.dump(prov, open(out / f"{base}.json", "w"), indent=1, default=str)
    print("rendered", base, flush=True)


if __name__ == "__main__":
    main()

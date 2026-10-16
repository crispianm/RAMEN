#!/usr/bin/env python3
"""Ifig8: LONG-VIDEO qualitative data (VDA Fig. 5 analogue): ONE benchmark DEV clip of N (500) frames, the DYNAMIC low-light cell ``llvar`` (the clip's own severity draw + noise stream of the scored
rows, bench_joint.llvar_draw / llvar_noise_seed / degrade_llvar), all models through the scored inference path (bench_joint.build_pipeline + run_pipeline = VDA's 32-frame / stride-22 key-frame windows),
TAG (cfg tag, agf0.2t0.06) on RAMEN's raw depth. Writes ONE deterministic npz per clip + a json sidecar. No figure, no number on any image: the figure itself is rendered from this npz by the
redesign template (qualitative/qual_redesign_20261015/scripts/render_P_long.py).

  python3 -B render_long_video.py --clip bonn:rgbd_bonn_placing_obstructing_box --start 30 --n 500 --manifest long_video/bonn_video_long500.json --out <dir>/bonn__rgbd_bonn_placing_obstructing_box
  python3 -B render_long_video.py --clip scannet_rgbfix:scene0009_00 --n 500 --manifest <scannet_video_rgbfix_500.json> ...
  python3 -B render_long_video.py --smoke --device cpu --n 12 --models ft,c3_raw,c3_tag --clip nyuv2:study_0002 --out <scratch>

npz keys (template contract, qualitative/qual_redesign_20261015/README.md): ft, c3_raw, c3_tag (N,H,W) float32 metres at the clip's native frame size; gt, gt_valid (H,W) = the GT of frame N//2 (the
template reads ONE GT frame) ; rgb (N,H,W,3) uint8 = the DEGRADED input; restored (N,H,W,3) uint8 = RAMEN's restored frames (== rgb when the model has no restorer; see sidecar restored_is_input);
dav2_ft, daac_ft when requested. EXTRA keys (ignored by the template): clean (N,H,W,3) uint8, gt_all (N,H,W) float32 (metres, -1 = invalid, every frame), gt_valid_all, gt_frame_index, raw_hw.
Frame images are resized from the model resolution (518 short side) to the native size with cv2 INTER_AREA so that rgb and depth share one pixel grid (the template indexes both with the same x).
DEV ONLY: the clip must be in bench_joint.load_clips(ds, 'dev'); a TEST clip is refused (there is no TEST option)."""
import argparse, hashlib, json, os, socket, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import render_c3_figures as RC   # helpers only (sha, write_depth_npz, apply_host_paths); its main() is not called

ALLOWED = ("ft", "c3_raw", "c3_tag", "dav2_ft", "daac_ft")
REQUIRED = ("ft", "c3_raw", "c3_tag")
PER_FRAME = {"dav2_ft": "dav2_s_ft:", "daac_ft": "daac_ft:"}


def is_placeholder(x):
    return x is None or str(x).upper().startswith("PLACEHOLDER") or "PLACEHOLDER" in str(x).upper()


def resolve_models(models, pins, host):
    """models: list of ALLOWED keys; pins: {dav2_ft|daac_ft: {ckpt, ckpt_gh200, sha256}}. Refuses (assert) any requested per-frame model whose checkpoint path or sha256 is still a PLACEHOLDER."""
    assert models and all(m in ALLOWED for m in models), f"--models must be a comma list of {ALLOWED}"
    assert all(m in models for m in REQUIRED), f"the template needs {REQUIRED}; got {models}"
    out = {}
    for m in models:
        if m in PER_FRAME:
            pin = (pins or {}).get(m)
            assert pin, f"model {m} requested but no pin given (params.model_pins.{m})"
            ck = pin.get("ckpt_gh200") if host == "gh200" else pin.get("ckpt")
            assert not is_placeholder(ck), f"model {m}: checkpoint path is an unresolved PLACEHOLDER ({ck!r}): the TRAIN checkpoint does not exist / was not pinned yet; refusing"
            assert not is_placeholder(pin.get("sha256")), f"model {m}: sha256 pin is an unresolved PLACEHOLDER ({pin.get('sha256')!r}); refusing"
            assert Path(ck).exists(), f"model {m}: checkpoint missing: {ck}"
            assert RC.sha(ck) == pin["sha256"], f"model {m}: checkpoint sha256 mismatch"
            out[m] = {"ckpt": str(ck), "sha256": pin["sha256"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "figures_config.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", required=True, help="dataset:key, a DEV clip of the benchmark")
    ap.add_argument("--start", type=int, default=0, help="first frame of the clip entry used (Bonn 30)")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--manifest", default=None, help="long manifest json (VDA format); default = the benchmark manifest of the dataset (NYUv2 is already 500 frames). The DEV membership is always checked on the BENCHMARK manifest")
    ap.add_argument("--manifest-sha256", default=None, help="expected sha256 of --manifest (asserted)")
    ap.add_argument("--root", default=None, help="override the dataset frame root (GH200 Bonn: /projects/b5dh/data/test/benchmarks/bonn, the tree that holds the full scenes)")
    ap.add_argument("--models", default="ft,c3_raw,c3_tag", help="comma list of " + ",".join(ALLOWED))
    ap.add_argument("--model-pins", default=None, help="json {dav2_ft|daac_ft: {ckpt, ckpt_gh200, sha256}}; a PLACEHOLDER value for a requested model aborts")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", choices=["local", "gh200"], default="local")
    ap.add_argument("--eval-clone", default=None)
    ap.add_argument("--smoke", action="store_true", help="plumbing smoke: output labelled SMOKE, not a figure")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    pins = json.loads(a.model_pins) if a.model_pins else {}
    mres = resolve_models(models, pins, a.host)       # refuses loudly BEFORE any heavy import / load
    ds, key = a.clip.split(":", 1)
    cfg = json.load(open(a.config))
    RC.apply_host_paths(cfg, a.host, a.eval_clone)
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, cfg["eval_clone"])
    import torch, cv2
    import bench_joint as bj
    clone_head = subprocess.check_output(["git", "-C", cfg["eval_clone"], "rev-parse", "HEAD"], text=True).strip()
    porcelain = subprocess.check_output(["git", "-C", cfg["eval_clone"], "status", "--porcelain"], text=True).strip()
    assert clone_head == cfg["eval_commit"] and not porcelain, f"eval clone not at {cfg['eval_commit']} or not clean"
    assert ds in bj.DATASETS and not ds.startswith("kitti"), f"unsupported dataset {ds}"
    dev_keys = [k for k, _ in bj.load_clips(ds, "dev")]          # DEV membership on the benchmark's own manifest + split rule; TEST is refused by load_clips itself
    assert key in dev_keys, f"{ds}:{key} is not a DEV clip (DEV only; this renderer has no TEST option)"
    ft = cfg["ft"]["B"]
    for nm, c in (("c3", cfg["c3"]), ("ft", ft)):
        assert c["ckpt"] and Path(c["ckpt"]).exists(), f"{nm} checkpoint missing: {c['ckpt']}"
        assert RC.sha(c["ckpt"]) == c["sha256"], f"{nm} checkpoint sha mismatch"
    specs = {"ft": f"{bj.FT_VDA}{ft['ckpt']}", "c3": f"ours:{cfg['c3']['ckpt']}"}
    for m, pre in PER_FRAME.items():
        if m in mres:
            specs[m] = f"{pre}{mres[m]['ckpt']}"
    if a.root:
        bj.PATHS[f"{ds}_root"] = a.root
    if a.manifest:
        assert not a.manifest_sha256 or RC.sha(a.manifest) == a.manifest_sha256, f"manifest sha256 mismatch: {a.manifest}"
        man = json.load(open(a.manifest))[bj.DATASETS[ds].get("manifest_key", ds)]
        frames_all = {k: v for e in man for k, v in e.items()}[key]
    else:
        frames_all = dict(bj.load_clips(ds, "dev"))[key]
    frames = frames_all[a.start:a.start + a.n]
    assert len(frames) == a.n or a.smoke, f"{ds}:{key}: {len(frames)} frames in [{a.start}, {a.start + a.n}), wanted {a.n}"
    bj.DATASETS[ds]["max_len"] = len(frames)                      # load_frames / load_gt truncate to max_len (the 90/110 benchmark cap): lift it for this clip only
    t0 = time.time()
    clean, raw_hw = bj.load_frames(ds, frames)
    gt = bj.load_gt(ds, frames)
    assert clean.shape[0] == gt.shape[0] == len(frames)
    cell = "llvar"
    drawn = bj.llvar_draw(ds, key)
    seed = bj.llvar_noise_seed(ds, key, cell)
    pin_sha = bj.llvar_pin(ds)
    x = bj.degrade_llvar(clean, cell, drawn, seed)                 # the scored rows' degradation (cmd_run l.1394-1396), on the real frames of this window
    blur, sigma = bj.parse_cell(cell)
    dev = a.device
    N = x.shape[0]
    res, restored, weights = {}, None, {}
    for nm in [m for m in ("ft", "c3", "dav2_ft", "daac_ft") if (m == "c3" and ("c3_raw" in models or "c3_tag" in models)) or m in models]:
        p = bj.build_pipeline(specs[nm], dev)
        depth, rest = bj.run_pipeline(p, x.to(dev), sigma, blur, raw_hw)
        weights[nm] = {k: v for k, v in p["weights"].items()}
        if nm == "c3":
            res["c3_raw"] = depth.astype(np.float32)
            res["c3_tag"] = bj.apply_variant(cfg["tag"], depth, p["last_guides"]).astype(np.float32)
            restored = rest.detach().float().cpu() if hasattr(rest, "detach") else None
        else:
            res[nm] = depth.astype(np.float32)
        del p, depth, rest
        if dev == "cuda":
            torch.cuda.empty_cache()
    H, W = raw_hw

    def frames_u8(t):                                              # (N,3,h,w) float [0,1] -> (N,H,W,3) uint8 at the native size (INTER_AREA), the u8() conversion the other figures use
        o = np.empty((t.shape[0], H, W, 3), np.uint8)
        for i in range(t.shape[0]):
            im = (t[i].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            o[i] = im if im.shape[:2] == (H, W) else cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        return o
    rgb = frames_u8(x)
    rest_u8 = frames_u8(restored) if restored is not None else rgb
    restored_is_input = bool(restored is None or np.array_equal(rgb, rest_u8))
    gt_shape = gt.shape[1:]
    assert tuple(gt_shape) == (H, W) and all(v.shape == (N, H, W) for v in res.values()), f"shape mismatch gt {gt_shape} raw_hw {(H, W)} depth {[v.shape for v in res.values()]}"
    mx = bj.DATASETS[ds]["max_depth"]
    gt_all = gt.astype(np.float32)
    valid_all = (gt > 1e-3) & (gt < mx)
    fi = N // 2
    arrs = {k: res[k] for k in (*REQUIRED, "dav2_ft", "daac_ft") if k in res}
    arrs.update({"gt": gt_all[fi], "gt_valid": valid_all[fi], "rgb": rgb, "restored": rest_u8, "clean": frames_u8(clean), "gt_all": gt_all, "gt_valid_all": valid_all,
                 "gt_frame_index": np.int64(fi), "raw_hw": np.array(raw_hw, np.int64)})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "SMOKE_" if a.smoke else ""
    stem = f"{tag}long_{ds}_{key}"
    rec = RC.write_depth_npz(out / f"{stem}.npz", arrs)
    side = {"PROVENANCE": "Ifig8 long-video clip data (render_long_video.py)", "smoke": bool(a.smoke), "dataset": ds, "clip": key, "split": "dev (membership checked on the benchmark manifest via bench_joint.load_clips('dev'))",
            "frames": int(N), "frame_range_in_manifest_entry": [a.start, a.start + N], "first_image": frames[0]["image"], "last_image": frames[-1]["image"], "manifest": a.manifest or "benchmark manifest of the dataset",
            "manifest_sha256": RC.sha(a.manifest) if a.manifest else None, "root": bj.PATHS[f"{ds}_root"], "cell": cell, "llvar_draw": drawn, "noise_seed": seed, "llvar_draws_sha256": pin_sha,
            "window_rule": "bench_joint.vda_windows: 32-frame windows, stride 22, 10 key-frames (VDA infer_video_depth), the same for every model; per-frame nets (window 1) in batches of 8",
            "tag": cfg["tag"], "tag_applied_to": "RAMEN n46 raw depth over the whole clip (c3_tag); ft and c3_raw are raw", "models": models, "specs": specs,
            "ckpts": {"c3": {"path": cfg["c3"]["ckpt"], "sha256": cfg["c3"]["sha256"]}, "ft": {"path": ft["ckpt"], "sha256": ft["sha256"]}, **{m: mres[m] for m in mres}},
            "base_weights": {nm: {k: {"path": v, "sha256": RC.sha(v)} for k, v in w_.items()} for nm, w_ in weights.items()},
            "restored_is_input": restored_is_input, "restored_note": "restored = RAMEN n46 restored frames (each frame from the first window that sees it as real input); FT / per-frame models have no restorer",
            "gt_note": "gt, gt_valid = frame gt_frame_index (N//2), sensor GT in metres, valid = 1e-3 < GT < max_depth; gt_all / gt_valid_all = every frame",
            "npz": rec, "dtype_shapes": rec["shapes"], "raw_hw": [H, W], "model_resolution_short_side": bj.DATASETS[ds]["short_side"], "numbers_on_images": False,
            "eval_clone": cfg["eval_clone"], "eval_clone_head": clone_head, "script_sha256": RC.sha(__file__), "figures_config_sha256": RC.sha(a.config), "torch": torch.__version__, "device": dev,
            "gpu": torch.cuda.get_device_name() if dev == "cuda" and torch.cuda.is_available() else "cpu", "host": socket.gethostname(), "host_paths": a.host,
            "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(side, open(out / f"{stem}.json", "w"), indent=1, default=str)
    print("wrote", stem, {k: list(np.asarray(v).shape) for k, v in arrs.items()}, flush=True)


if __name__ == "__main__":
    main()

"""ONE leg in ONE process (Amendment 1 (i): one leg per process, own output directory, no shared mutable state). Runs the DEV scoring loop of ``bench_joint.cmd_run``
(same pinned functions, same order, metrics from the pinned ``score_clip``) with the leg's toggles applied by runtime monkeypatch (``patches.apply``); the degraded input
comes from the verified cache (``--source cache``) or on the fly (``--source fly``, the control). Never measures or writes timing. Failed clips are retried then FAIL the
leg (resumable per clip); nothing is dropped.

Output dir layout:  leg.json  toggles.json  clips/<ds>/<clip>.json  per_clip.csv  macro.json  depth_shas.json  [depth/<ds>/<cell>/<clip>.npy with --save-depth]  DONE.json | FAILED.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.efficiency import common as C  # noqa: E402
from repro.efficiency.gpu_gate import gate as _gate  # noqa: E402

RETRIES = 2
HEAVY = False       # set in _run: manual-attention legs (no SDPA toggle) are memory-heavy


def numerics(torch) -> dict:
    return {"cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def find_patches_dir(explicit: str | None) -> Path:
    for c in [explicit, os.environ.get("EFF_PATCHES_DIR"), str(Path(__file__).resolve().parent), str(C.REPO_ROOT / "repro" / "eff")]:
        if c and (Path(c) / "patches.py").exists():
            return Path(c)
    raise SystemExit("patches.py not found (pass --patches-dir or set EFF_PATCHES_DIR; the committed copy lives under repro/eff/ per P1)")


def build(bj, patches, leg: dict, device: str):
    """Pinned pipeline with the leg's toggles. S0 (no flags) takes the untouched pinned builders."""
    import torch
    flags = leg["flags"]
    if leg["model"] == "ramen":
        if flags:
            orig = bj.load_ours
            bj.load_ours = lambda ckpt, dev: (lambda n: (patches.apply(n, flags, True), n)[1])(orig(ckpt, dev))
        p = bj.build_pipeline(f"ours:{C.RAMEN_CKPT}", device)
    else:
        if flags:
            def vda_released(dev, metric):          # line-for-line bench_joint._vda_released (checked by the Auditor in latency_copy.py) + patches.apply
                bj._vda_path()
                from video_depth_anything.video_depth import VideoDepthAnything
                key = "metric_vda_s" if metric else "vda_s"
                net = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384], metric=metric)
                net.load_state_dict(torch.load(bj.PATHS[key], map_location="cpu", weights_only=False), strict=True)
                net = net.to(dev).eval()
                patches.apply(net, flags, False)
                return dict(fwd=lambda x: (net(bj._norm(x)[None])[0], None), metric=metric, window=32, weights={key: bj.PATHS[key]})
            bj._vda_released = vda_released
        p = bj.build_pipeline("metric_vda_s", device)
    if leg["precision"] == "fp16":
        f0 = p["fwd"]

        def fwd16(x):
            with torch.autocast("cuda", dtype=torch.float16):
                d, r = f0(x)
            return (d.float() if torch.is_tensor(d) else d), (r.float() if torch.is_tensor(r) else r)
        p["fwd"] = fwd16
    return p


def toggle_record(patches, leg: dict, N: int, worker_id: int, torch) -> dict:
    from evaluation import depth_probe as dp
    rec = dict(patches.toggle_record(leg["flags"], leg["precision"])) if patches is not None else {}
    rec.update({"precision": leg["precision"], "venv": sys.prefix, "torch": torch.__version__, "agf_backend": dp.AGF_BACKEND,
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), **numerics(torch)})
    rec.setdefault("attn_backend", "manual+motion_manual"); rec.setdefault("bn_folded", False); rec.setdefault("channels_last", False)
    rec.setdefault("compiled", False); rec.setdefault("refine_hook", True); rec.setdefault("input_staging", "fp32")
    missing = [k for k in C.TOGGLE_FIELDS if k not in rec]
    assert not missing, missing
    rec.update({"leg": leg["id"], "stack": leg["stack"], "toggles": leg["toggles"], "flags": leg["flags"], "N": N, "worker_id": worker_id})
    return rec


def load_cached(cache: Path, manifest: dict, ds: str, cell: str, key: str):
    import torch
    p = C.cache_path(cache, ds, cell, key)
    arr = np.load(p)
    if C.array_sha256(arr) != manifest[(ds, cell, key)]:
        raise RuntimeError(f"cache integrity: {p} sha differs from the manifest")
    return torch.from_numpy(arr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=C.DEFAULT_CACHE)
    ap.add_argument("--source", choices=["cache", "fly"], default="cache")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, required=True)
    ap.add_argument("--patches-dir", default=None)
    ap.add_argument("--save-depth", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=list(C.DEV_DATASETS))
    ap.add_argument("--limit-clips", type=int, default=None, help="smoke only (the counts check then reports a limited run)")
    a = ap.parse_args(argv)
    leg = json.loads(a.leg_json.read_text())
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    C.write_json(out / "leg.json", leg)
    manifest = {}
    if a.source == "cache":
        vr = a.cache_dir / "verify_report.json"
        if not vr.exists() or not json.loads(vr.read_text())["summary"]["PASS"]:
            raise SystemExit("cache route requires a PASSing verify_report.json (bitwise gate); not available -> route dropped")
        manifest = {(e["dataset"], e["cell"], e["key"]): e["sha256_array"] for e in json.loads((a.cache_dir / "manifest.json").read_text())["entries"]}
    try:
        return _run(a, leg, out, manifest)
    except BaseException as e:                              # noqa: BLE001  the leg FAILS (the launcher re-runs it); partial clips are kept for resume
        C.write_json(out / "FAILED.json", {"leg": leg["id"], "error": repr(e), "trace": traceback.format_exc()})
        raise


def _run(a, leg: dict, out: Path, manifest: dict) -> int:
    bj = C.import_scorer()
    import torch
    device = "cuda"
    global HEAVY
    HEAVY = "sdpa" not in leg["flags"]
    patches = None
    if leg["flags"]:
        sys.path.insert(0, str(find_patches_dir(a.patches_dir)))
        import patches  # noqa: PLC0415
    n0 = numerics(torch)
    if n0 != C.PINNED_NUMERICS:
        raise RuntimeError(f"numerics flags differ from the pinned PyTorch defaults at start: {n0}")
    if leg["model"] == "ramen" and C.sha256_file(C.RAMEN_CKPT) != C.RAMEN_CKPT_SHA:
        raise RuntimeError("RAMEN checkpoint sha mismatch")
    p = build(bj, patches, leg, device)
    if leg["agf_backend"] == "fast_triton_v3":
        sys.path.append(str(C.REPO_ROOT))
        from repro import tag_fast_patch
        tag_fast_patch.patch()
    rec = toggle_record(patches, leg, a.n_workers, a.worker_id, torch)
    C.write_json(out / "toggles.json", rec)
    from evaluation.temporal_metrics import TOF
    from evaluation.depth_probe import MemoTOF
    variants = leg["tag_variants"]
    tofnet = MemoTOF(TOF(device)) if len(variants) > 1 else TOF(device)
    from repro.efficiency.cache_builder import gen_input
    done = 0
    for ds in a.datasets:
        proto = bj.DATASETS[ds]
        clips = bj.select_clips(ds, "dev", None)
        if a.limit_clips is None and len(clips) != C.DEV_DATASETS[ds]:
            raise RuntimeError(f"{ds}: {len(clips)} DEV clips, expected {C.DEV_DATASETS[ds]}")
        for key, frames in clips[:a.limit_clips]:
            cf = out / "clips" / ds / (C.safe_key(key) + ".json")
            if cf.exists():
                done += 1
                continue
            for attempt in range(RETRIES + 1):
                try:
                    # fix 2 (eff_b): scheduling-only GPU-memory gate around the whole clip (frames/GT/forward/scoring); inert without EFF_GATE_DIR.
                    # exclusive for memory-heavy clips (manual attention, or KITTI), shared otherwise; no data/numerics change.
                    with _gate(HEAVY or ds == "kitti"):
                        res = score_one_clip(bj, p, tofnet, proto, ds, key, frames, variants, a, manifest, device, out)
                        torch.cuda.empty_cache()             # hand the pool back so concurrent workers do not hold each other's memory
                    break
                except Exception as e:               # noqa: BLE001  OOM etc.: logged, retried, then the leg fails -- never dropped
                    with open(out / "failures.jsonl", "a") as f:
                        f.write(json.dumps({"dataset": ds, "key": key, "attempt": attempt, "error": repr(e)}) + "\n")
                    torch.cuda.empty_cache()
                    if attempt == RETRIES:
                        raise
            C.write_json(cf, res)
            done += 1
            print(f"{leg['id']} {ds} {key} done", flush=True)
    n1 = numerics(torch)
    if n1 != C.PINNED_NUMERICS:
        raise RuntimeError(f"numerics flags changed during the leg: {n1}")
    aggregate(out, leg, rec, a)
    C.write_json(out / "DONE.json", {"leg": leg["id"], "clips_done": done, "limited": a.limit_clips is not None, "source": a.source})
    return 0


def score_one_clip(bj, p, tofnet, proto, ds, key, frames, variants, a, manifest, device, out) -> dict:
    """Both cells of one clip, mirroring cmd_run (non-static and static branches)."""
    import torch
    from evaluation import vda5_sets as _vda5
    from repro.efficiency.cache_builder import gen_input
    clean, raw_hw = bj.load_frames(ds, frames)
    gt = bj.load_gt(ds, frames)
    cam = bj.camera(ds, key, frames[:bj.DATASETS[ds]["max_len"]]) if ds not in bj.OPTIN_DATASETS else None
    skym = None
    if proto.get("sky_mask") == "kitti_colour":
        from evaluation.sky_metrics import kitti_sky_masks
        skym = kitti_sky_masks(bj.PATHS[f"{ds}_root"], frames[:proto["max_len"]], proto["crop"], gt)
    res = {"dataset": ds, "key": key, "cells": {}}
    for c in C.CELLS:
        blur, sigma = bj.parse_cell(c)
        static = c in bj.STATIC_CELLS or c in bj.STATIC_LADDER_CELLS
        xcpu = load_cached(a.cache_dir, manifest, ds, c, key) if a.source == "cache" else gen_input(bj, ds, key, frames, c, clean, raw_hw)[0]
        x = xcpu.to(device)
        if static:
            sclean, sgt = bj.static_clip(clean, gt)
        depth, restored = bj.run_pipeline(p, x, sigma, blur, raw_hw)
        if hasattr(tofnet, "new_clip"):
            tofnet.new_clip()
        cell_res = {"depth_sha256": C.array_sha256(depth), "depth_finite_mask_sha256": C.array_sha256(np.isfinite(depth)), "depth_all_finite": bool(np.isfinite(depth).all()),
                    "input_sha256": C.array_sha256(xcpu), "variants": {}}
        if a.save_depth:
            dp_ = Path(out) / "depth" / ds / c / (C.safe_key(key) + ".npy")
            dp_.parent.mkdir(parents=True, exist_ok=True)
            np.save(dp_, depth)
        draw = bj.llvar_draw(ds, key) if c in bj.LLVAR_VIDEO_CELLS else None
        for v in variants:
            dv = bj.apply_variant(v, depth, p["last_guides"])
            if static:
                r = bj.score_clip(dv, p["metric"], sgt, restored, x, sclean.to(device), proto["max_depth"], proto["min_depth"], None, tofnet, bj.ll_cfg(blur)["ev"],
                                  sky_m=proto.get("sky_depth_m"))
                r.update(bj.flicker_static(dv, sgt, proto["max_depth"], proto["min_depth"]) if p["metric"] else {"flicker": float("nan"), "abs_rel_static": float("nan")})
            else:
                sd, sg, sr, sx, sc, scam = _vda5.score_slice(proto, dv, gt, restored, x, clean.to(device), cam)
                r = bj.score_clip(sd, p["metric"], sg, sr, sx, sc, proto["max_depth"], proto["min_depth"], scam, tofnet,
                                  (draw["ev"] if draw is not None else bj.ll_cfg(blur)["ev"]) if isinstance(blur, bj.LowLight) else None,
                                  sky_m=proto.get("sky_depth_m"), sky_mask=skym)
            cell_res["variants"][v] = {k: (float(val) if np.isscalar(val) else val) for k, val in r.items() if k != "frames"}
        res["cells"][c] = cell_res
        del x
    return res


def aggregate(out: Path, leg: dict, rec: dict, a) -> None:
    """Per-clip CSV, macro JSON (video_macro = nanmean as cmd_run), depth sha table, SHA256SUMS. No timing field anywhere."""
    rows, macro, shas = [], {}, {}
    for cf in sorted((out / "clips").rglob("*.json")):
        r = json.loads(cf.read_text())
        for c, cr in r["cells"].items():
            shas[f"{r['dataset']}|{c}|{r['key']}"] = {k: cr[k] for k in ("depth_sha256", "depth_finite_mask_sha256", "depth_all_finite", "input_sha256")}
            for v, m in cr["variants"].items():
                rows.append({"leg": leg["id"], "dataset": r["dataset"], "cell": c, "variant": v, "clip": r["key"], **m})
    keys = sorted({k for r in rows for k in r if k not in ("leg", "dataset", "cell", "variant", "clip")})
    with open(out / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["leg", "dataset", "cell", "variant", "clip", *keys])
        w.writeheader()
        w.writerows(rows)
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["cell"], r["variant"]), []).append(r)
    for (ds, c, v), rs in sorted(groups.items()):
        macro[f"{ds}|{c}|{v}"] = {"num_clips": len(rs), "video_macro": {k: float(np.nanmean([x[k] for x in rs if k in x])) for k in keys},
                                  "nonfinite_clip_counts": {k: int(sum(not np.isfinite(x[k]) for x in rs if k in x)) for k in keys}}
    C.write_json(out / "macro.json", {"leg": leg["id"], "toggles": rec, "limited": a.limit_clips is not None, "macro": macro})
    C.write_json(out / "depth_shas.json", shas)
    C.write_sha256sums(out)


if __name__ == "__main__":
    raise SystemExit(main())

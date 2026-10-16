"""AUD-FASTTAG-B-20261013 driver (fast-TAG (b), real-depth equivalence). New file only; scorer copy ~/.cache/ramen_scorer/368b5c53 is imported, never edited.
Per clip x cell: ONE model pass (S0 fp32 RAMEN seed45 step12000, cached verified degraded input) -> raw stitched depth saved (.npy + sha256) -> reloaded from disk
(sha re-checked) -> every arm is applied to THOSE saved bytes and scored by the scorer's own score_clip.
Arms: none | for setting in (0.2/0.06, 0.5/0.12): REF numpy depth_agf (cpu) | V1 depth_agf_torch (gpu fp64) | V3 tag_fast_patch (fast_triton_v3)."""
from __future__ import annotations
import argparse, csv, json, sys, traceback
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from repro.efficiency import common as C  # noqa: E402

SETTINGS = {"s02": (0.2, 0.06), "s05": (0.5, 0.12)}
BACKENDS = ["cpu", "gpu", "fast_triton_v3"]   # REF, V1, V3
ARM_NAME = {"cpu": "REF", "gpu": "V1", "fast_triton_v3": "V3"}


def rel_stats(y, ref):
    fy, fr = np.isfinite(y), np.isfinite(ref)
    both = fy & fr
    d = np.abs(y[both].astype(np.float64) - ref[both].astype(np.float64)) / np.maximum(np.abs(ref[both].astype(np.float64)), 1e-6)
    out = {"n_px": int(y.size), "nan_y": int(np.isnan(y).sum()), "inf_y": int(np.isinf(y).sum()), "nan_ref": int(np.isnan(ref).sum()), "inf_ref": int(np.isinf(ref).sum()),
           "finite_mask_changes": int((fy != fr).sum()), "byte_identical": bool(y.tobytes() == ref.tobytes())}
    if d.size:
        out.update(max_rel=float(d.max()), p999_rel=float(np.partition(d, int(0.999 * (d.size - 1)))[int(0.999 * (d.size - 1))]), median_rel=float(np.median(d)), n_exact=int((d == 0).sum()))
    else:
        out.update(max_rel=float("nan"), p999_rel=float("nan"), median_rel=float("nan"), n_exact=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=C.DEFAULT_CACHE)
    ap.add_argument("--limit-clips", type=int, default=None)
    ap.add_argument("--datasets", nargs="+", default=list(C.DEV_DATASETS))
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    vr = json.loads((a.cache / "verify_report.json").read_text())
    assert vr["summary"]["PASS"], "cache not verified"
    manifest = {(e["dataset"], e["cell"], e["key"]): e["sha256_array"] for e in json.loads((a.cache / "manifest.json").read_text())["entries"]}
    bj = C.import_scorer()
    import torch
    from evaluation import depth_probe as dp
    from evaluation import vda5_sets as _vda5
    from evaluation.temporal_metrics import TOF
    from repro import tag_fast_patch as tfp
    from repro.efficiency.leg_worker import numerics, build, load_cached
    n0 = numerics(torch)
    assert n0 == C.PINNED_NUMERICS, n0
    assert C.sha256_file(C.RAMEN_CKPT) == C.RAMEN_CKPT_SHA
    device = "cuda"
    leg = {"model": "ramen", "flags": [], "precision": "fp32"}
    p = build(bj, None, leg, device)
    tofnet = TOF(device)
    pin_backend = dp.AGF_BACKEND
    assert pin_backend == "cpu"
    ref_depth_agf = tfp.original("depth_agf")
    assert ref_depth_agf is dp.depth_agf

    def set_backend(b):
        if b == "fast_triton_v3":
            tfp.patch()
        else:
            tfp.unpatch()
            dp.AGF_BACKEND = b   # 'cpu' numpy reference | 'gpu' depth_agf_torch fp64
        assert dp.AGF_BACKEND == b
    rows, pix = [], []
    done_cells = out / "cells"
    for ds in a.datasets:
        proto = bj.DATASETS[ds]
        clips = bj.select_clips(ds, "dev", None)
        if a.limit_clips is None:
            assert len(clips) == C.DEV_DATASETS[ds], (ds, len(clips))
        for key, frames in clips[:a.limit_clips]:
            clean, raw_hw = bj.load_frames(ds, frames)
            gt = bj.load_gt(ds, frames)
            cam = bj.camera(ds, key, frames[:bj.DATASETS[ds]["max_len"]]) if ds not in bj.OPTIN_DATASETS else None
            skym = None
            if proto.get("sky_mask") == "kitti_colour":
                from evaluation.sky_metrics import kitti_sky_masks
                skym = kitti_sky_masks(bj.PATHS[f"{ds}_root"], frames[:proto["max_len"]], proto["crop"], gt)
            for c in C.CELLS:
                cf = done_cells / ds / (C.safe_key(key) + "__" + c + ".json")
                if cf.exists():
                    r = json.loads(cf.read_text()); rows += r["rows"]; pix += r["pix"]; continue
                blur, sigma = bj.parse_cell(c)
                static = c in bj.STATIC_CELLS or c in bj.STATIC_LADDER_CELLS
                set_backend("cpu")
                torch.cuda.empty_cache()          # attempt-2 fix: V1 fp64 / Triton buffers left the allocator fragmented -> OOM in the model pass; memory management only
                xcpu = load_cached(a.cache, manifest, ds, c, key)
                x = xcpu.to(device)
                if static:
                    sclean, sgt = bj.static_clip(clean, gt)
                depth, restored = bj.run_pipeline(p, x, sigma, blur, raw_hw)
                dpath = out / "depth" / ds / c / (C.safe_key(key) + ".npy")
                dpath.parent.mkdir(parents=True, exist_ok=True)
                np.save(dpath, depth)
                dsha = C.array_sha256(depth)
                raw = np.load(dpath)
                assert C.array_sha256(raw) == dsha and raw.tobytes() == depth.tobytes()
                del depth
                draw = bj.llvar_draw(ds, key) if c in bj.LLVAR_VIDEO_CELLS else None
                crow, cpix = [], []

                def score(dv):
                    if static:
                        r = bj.score_clip(dv, p["metric"], sgt, restored, x, sclean.to(device), proto["max_depth"], proto["min_depth"], None, tofnet, bj.ll_cfg(blur)["ev"], sky_m=proto.get("sky_depth_m"))
                        r.update(bj.flicker_static(dv, sgt, proto["max_depth"], proto["min_depth"]) if p["metric"] else {"flicker": float("nan"), "abs_rel_static": float("nan")})
                    else:
                        sd, sg, sr, sx, sc, scam = _vda5.score_slice(proto, dv, gt, restored, x, clean.to(device), cam)
                        r = bj.score_clip(sd, p["metric"], sg, sr, sx, sc, proto["max_depth"], proto["min_depth"], scam, tofnet,
                                          (draw["ev"] if draw is not None else bj.ll_cfg(blur)["ev"]) if isinstance(blur, bj.LowLight) else None, sky_m=proto.get("sky_depth_m"), sky_mask=skym)
                    return {k: (float(v) if np.isscalar(v) else v) for k, v in r.items() if k != "frames"}
                base = dict(dataset=ds, cell=c, clip=key, raw_depth_sha256=dsha, input_sha256=C.array_sha256(xcpu), raw_shape=str(list(raw.shape)), raw_dtype=str(raw.dtype))
                crow.append({**base, "arm": "none", "setting": "none", "backend": "cpu", "agf_backend": "cpu", "out_sha256": dsha, **score(raw)})
                for sname, (amin, tau) in SETTINGS.items():
                    tok = f"agf{amin}t{tau}"
                    refy = None
                    for b in BACKENDS:
                        set_backend(b)
                        y = bj.apply_variant(tok, raw, p["last_guides"])
                        assert isinstance(y, np.ndarray) and y.shape == raw.shape
                        if b == "cpu":
                            refy = y
                        st = rel_stats(y, refy)
                        crow.append({**base, "arm": ARM_NAME[b], "setting": sname, "backend": b, "agf_backend": dp.AGF_BACKEND, "out_sha256": C.array_sha256(y), **score(y)})
                        cpix.append({**base, "arm": ARM_NAME[b], "setting": sname, "agf_backend": dp.AGF_BACKEND, "out_dtype": str(y.dtype), **st})
                        if b != "cpu":
                            del y
                    del refy
                set_backend("cpu")
                assert C.array_sha256(np.load(dpath)) == dsha        # saved input never altered
                C.write_json(cf, {"rows": crow, "pix": cpix})
                rows += crow; pix += cpix
                print(ds, key, c, "done", flush=True)
                del x
                torch.cuda.empty_cache()
    keys = sorted({k for r in rows for k in r if k not in ("dataset", "cell", "clip", "arm", "setting", "backend", "agf_backend", "out_sha256", "raw_depth_sha256", "input_sha256", "raw_shape", "raw_dtype")})
    lead = ["dataset", "cell", "clip", "arm", "setting", "backend", "agf_backend", "raw_depth_sha256", "input_sha256", "out_sha256", "raw_shape", "raw_dtype"]
    with open(out / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lead + keys); w.writeheader(); w.writerows(rows)
    pk = ["dataset", "cell", "clip", "arm", "setting", "agf_backend", "raw_depth_sha256", "out_dtype", "n_px", "max_rel", "p999_rel", "median_rel", "n_exact", "byte_identical", "nan_y", "inf_y", "nan_ref", "inf_ref", "finite_mask_changes"]
    with open(out / "per_pixel_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pk, extrasaction="ignore"); w.writeheader(); w.writerows(pix)
    macro = {}
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["cell"], r["arm"], r["setting"]), []).append(r)
    for (ds, c, arm, s), rs in sorted(groups.items()):
        macro[f"{ds}|{c}|{arm}|{s}"] = {"num_clips": len(rs), "video_macro": {k: float(np.nanmean([x[k] for x in rs if k in x])) for k in keys}}
    tog = {"arm_backends": {ARM_NAME[b]: b for b in BACKENDS}, "agf_backend_per_arm": {"none": "cpu", "REF": "cpu", "V1": "gpu", "V3": tfp.BACKEND_NAME}, "tag_fast_patch_sha256": C.sha256_file(REPO / "repro" / "tag_fast_patch.py"),
           "tag_fast_sha256": C.sha256_file(REPO / "repro" / "tag_fast.py"), "venv": sys.prefix, "torch": torch.__version__, "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "precision": "fp32", **numerics(torch),
           "scorer_copy": str(C.SCORER_CLONE), "scorer_commit": C.SCORER_COMMIT, "ramen_ckpt": str(C.RAMEN_CKPT), "ramen_ckpt_sha256": C.RAMEN_CKPT_SHA, "cache": str(a.cache), "cache_verify_summary": vr["summary"],
           "limited": a.limit_clips is not None, "settings": {k: list(v) for k, v in SETTINGS.items()}}
    C.write_json(out / "macro.json", {"toggles": tog, "macro": macro})
    C.write_json(out / "DONE.json", {"rows": len(rows)})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException:
        traceback.print_exc()
        raise

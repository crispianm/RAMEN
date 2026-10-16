"""Input cache for efficiency (b) (Amendment 1 / Add.29 (ii)). A NEW file outside the pinned clone: it IMPORTS the scorer's own loaders, seeds and degradations
(bench_joint.load_frames / load_gt / static_clip / degrade_static / llvar_draw / degrade_llvar / llvar_noise_seed / noise_seed / degrade) and calls them exactly as
``cmd_run`` does. Nothing is copied or re-implemented.

Cache layout (declared directory, DEV only):  <cache>/<dataset>/<cell>/<clip>.npy   the degraded input clip (N,3,h,w) in the dtype the pinned function returns
(no cast), <cache>/manifest.json (sha256 per clip x cell: array sha over dtype+shape+bytes, plus the .npy file sha256).
Optional clean-flow cache (GPU, RAFT; used only if ``verify-flow`` shows it bitwise equal to recomputation on every clip):  <cache>/flow/<dataset>/<clip>.npy.

  build   --cache DIR [--workers K]      generate + store every DEV clip x cell (CPU only)
  verify  --cache DIR [--workers K]      regenerate on the fly for EVERY DEV clip x cell and compare sha256 with the cached array and the manifest (CPU only, no model, no metrics)
  build-flow / verify-flow               clean RAFT flow (needs the GPU; NOT part of the allowed-now CPU verify)
Any mismatch / missing / extra clip x cell / DEV count mismatch is a FAIL (exit 1) and the cache route is dropped for (b)."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.efficiency import common as C  # noqa: E402


def gen_input(bj, ds: str, key: str, frames: list, cell: str, clean=None, raw_hw=None):
    """The degraded input of one clip x cell on CPU, byte-for-byte the code path of ``bench_joint.cmd_run`` (before ``.to(device)``).
    Returns (x, clean, raw_hw); pass ``clean`` back in to avoid re-reading the frames for the second cell."""
    if clean is None:
        clean, raw_hw = bj.load_frames(ds, frames)
    if cell in bj.STATIC_CELLS or cell in bj.STATIC_LADDER_CELLS:
        gt = bj.load_gt(ds, frames)
        sclean, _ = bj.static_clip(clean, gt)
        return bj.degrade_static(sclean, cell, bj.noise_seed(ds, key, bj.seed_cell(cell))), clean, raw_hw
    blur, sigma = bj.parse_cell(cell)
    if cell in bj.LLVAR_VIDEO_CELLS:
        draw = bj.llvar_draw(ds, key)
        return bj.degrade_llvar(clean, cell, draw, bj.llvar_noise_seed(ds, key, cell)), clean, raw_hw
    seed = bj.noise_seed(ds, key, bj.seed_cell(blur) if isinstance(blur, bj.LowLight) else sigma)
    return bj.degrade(clean, blur, sigma, seed), clean, raw_hw


def _init_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""            # CPU only
    import torch
    torch.set_num_threads(1)
    C.import_scorer()


def _task(args):
    """One clip: both cells. mode 'build' writes the .npy + sidecar, 'verify' regenerates and compares. Returns a list of per-(clip x cell) result dicts."""
    mode, cache, ds, key, cells = args
    import bench_joint as bj
    frames = _CLIPS[ds][key]
    clean = raw_hw = None
    res = []
    for cell in cells:
        p = C.cache_path(cache, ds, cell, key)
        side = p.with_suffix(".json")
        x, clean, raw_hw = gen_input(bj, ds, key, frames, cell, clean, raw_hw)
        xn = x.numpy()
        sha_fly = C.array_sha256(xn)
        rec = {"dataset": ds, "cell": cell, "key": key, "file": str(p.relative_to(cache)), "dtype": xn.dtype.str, "shape": list(xn.shape), "sha256_array": sha_fly}
        if mode == "build":
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".tmp")
            with open(tmp, "wb") as f:
                np.save(f, xn)
            os.replace(tmp, p)
            rec["sha256_file"] = C.sha256_file(p)
            C.write_json(side, rec)
            rec["status"] = "built"
        else:
            if not p.exists() or not side.exists():
                rec["status"] = "MISSING"
            else:
                cached = np.load(p)
                sha_cached = C.array_sha256(cached)
                man = json.loads(side.read_text())
                ok = (sha_cached == sha_fly == man["sha256_array"] and C.sha256_file(p) == man["sha256_file"]
                      and cached.dtype == xn.dtype and cached.shape == xn.shape)
                rec.update({"sha256_cached": sha_cached, "sha256_file_ok": C.sha256_file(p) == man["sha256_file"], "status": "MATCH" if ok else "MISMATCH"})
        res.append(rec)
        del x, xn
    return res


_CLIPS: dict = {}


def dev_clips(bj, datasets=None) -> dict:
    out = {}
    for ds in (datasets or C.DEV_DATASETS):
        out[ds] = dict(bj.select_clips(ds, "dev", None))
    return out


def _run(mode: str, cache: Path, workers: int, datasets, limit: int | None = None) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    bj = C.import_scorer()
    global _CLIPS
    _CLIPS = dev_clips(bj, datasets)
    counts = {ds: len(c) for ds, c in _CLIPS.items()}
    count_ok = all(counts[ds] == C.DEV_DATASETS.get(ds, counts[ds]) for ds in counts)
    tasks = [(mode, str(cache), ds, key, C.CELLS) for ds, clips in _CLIPS.items() for key in list(clips)[:limit]]
    results: list = []
    ctx = mp.get_context("fork")                             # fork: workers inherit the imported scorer and _CLIPS (read-only)
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        for r in pool.imap_unordered(_task, tasks):
            results.extend(r)
            print(f"[{mode}] {len(results)}/{len(tasks) * len(C.CELLS)} {r[0]['dataset']} {r[0]['key']}", flush=True)
    results.sort(key=lambda r: (r["dataset"], r["cell"], r["key"]))
    return {"mode": mode, "scorer_commit": C.SCORER_COMMIT, "dev_clip_counts": counts, "expected_counts": C.DEV_DATASETS, "count_ok": count_ok,
            "cells": list(C.CELLS), "entries": results}


def summarise(rep: dict) -> dict:
    ent = rep["entries"]
    by = {}
    for e in ent:
        by[e["status"]] = by.get(e["status"], 0) + 1
    expected_units = sum(rep["dev_clip_counts"].values()) * len(rep["cells"])
    rep["summary"] = {"units": len(ent), "expected_units": expected_units, "by_status": by, "count_ok": rep["count_ok"]}
    if rep["mode"] == "verify":
        rep["summary"]["PASS"] = bool(rep["count_ok"] and len(ent) == expected_units and by.get("MATCH", 0) == len(ent))
    else:
        rep["summary"]["PASS"] = bool(rep["count_ok"] and len(ent) == expected_units)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["build", "verify", "build-flow", "verify-flow"])
    ap.add_argument("--cache", type=Path, default=C.DEFAULT_CACHE)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--datasets", nargs="+", default=list(C.DEV_DATASETS))
    ap.add_argument("--limit-clips", type=int, default=None, help="testing only: first K clips per dataset (the full-set counts then FAIL the count check on purpose)")
    a = ap.parse_args(argv)
    if a.mode.endswith("flow"):
        from repro.efficiency.flow_cache import main_flow
        return main_flow(a)
    rep = summarise(_run(a.mode, a.cache, a.workers, a.datasets, a.limit_clips))
    if a.limit_clips is not None:
        rep["summary"]["PASS"] = False
        rep["summary"]["note"] = "limited run; not a gate result"
    name = "manifest.json" if a.mode == "build" else "verify_report.json"
    if a.limit_clips is not None:
        name = "LIMITED_" + name
    C.write_json(a.cache / name, rep)
    print(json.dumps(rep["summary"], indent=1))
    return 0 if rep["summary"]["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""DN-01 K34 held-out multi-dataset eval (user 2026-09-25: outdoor-first; indoor a bonus). bench_joint's model specs,
pipeline, degradation and scorer (bench_joint.build_pipeline / run_pipeline / degrade / score_clip: aligned + unaligned
metric depth, PSNR/SSIM, eps) on datasets read with the FROZEN benchmark.py's own iterators (imported read-only; clips,
GT, caps and masks are benchmark.py's). Frames are resized to model resolution with bench_joint's VDA Resize (short side
per set); GT stays native, and depth is resized to it, as in bench_joint. The pre-declared spec is in
docs/DN01_JOB_LEDGER.md (K34).
  sets / cells:
    nuscenes_night  real (raw REAL low-light frames, no synthetic degradation; no clean reference -> no PSNR)
    kittic          clean, <corruption>:<severity> (KITTI-C's own corrupted frames; clean split = PSNR reference)
    nyu, scannetpp  bench_joint cells b<sigma_b>s<sigma> (bench_joint.degrade on the clean frames)
  python scripts/dn01_heldout_eval.py list --out results/heldout_clips.json          (no model: clip lists only)
  python scripts/dn01_heldout_eval.py run --model <bench_joint spec> --dataset kittic --cells dark:3 --out DIR
  --temporal adds VDA's TAE on nuscenes_night (the committed CAM_FRONT poses, provenance/dn01_nuscenes_poses/); the
  other sets are single images (kittic, nyu) or have no committed poses (scannetpp), so they get none.
K46 WS1 (user 2026-09-28): nuScenes-DAY from the committed DEV/TEST split BY LOG (provenance/dn01_nuscenes_day_rain/,
split sha256 adabae02), read from Joint R+D's committed manifest (nuscenes_day_v1.json; npz projections external,
same builder chain as night), iterated exactly as the frozen benchmark's _iter_nuscenes_split (scene order, frames by
timestamp_us, its own LiDAR rasteriser):
    nuscenes_day    b3s25, llb3, ll (synthetic degradation of real day frames; DEGRADED cells only, K46)
  (nuScenes-rain is NOT supported: user decision 2026-09-28.) Only the DEV half is scored by default. The TEST half needs ``--split test --allow-test`` (reserved for a locked
  model under a newly pre-declared rule); without the flag the loader refuses before any frame is read.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import socket
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_joint as bj  # noqa: E402

BENCHMARK = (Path("/projects/b5dh/benchmark/benchmark.py") if Path("/projects/b5dh").is_dir()   # Isambard
             else Path("/mnt/DATA/DegradedDepth/benchmark/benchmark.py"))
BENCHMARK_SHA256 = "4627cfd906be380d467ec21941ad9dd0a1bc0a4b40337d0e48f8d6650fd2195c"   # the frozen file, both hosts
SETS = {   # cap = benchmark.py LOADERS; short side = bench_joint's (KITTI 378, else 518)
    "nuscenes_night": dict(max_depth=80.0, min_depth=0.1, short_side=518, cells=["real"]),
    "kittic": dict(max_depth=80.0, min_depth=0.1, short_side=378,
                   cells=["clean", "dark:3", "gaussian_noise:3", "defocus_blur:3"]),
    "nyu": dict(max_depth=10.0, min_depth=0.1, short_side=518, cells=["b0s0", "b3s25"]),
    "scannetpp": dict(max_depth=10.0, min_depth=0.1, short_side=518, cells=["b0s0", "b3s25"], limit=10),
}   # scannetpp limit: the first 10 benchmark-order scenes (K34 spec: indoor bonus guard, decode cost)
    # K46 WS1: DEV split only unless --allow-test; degraded cells only (no b0s0 / clean cell, K46)
SETS["nuscenes_day"] = dict(max_depth=80.0, min_depth=0.1, short_side=518, cells=["b3s25", "llb3", "ll"],
                            manifest="nuscenes_day_v1", split="dev")
NUSCENES_SPLIT_SETS = ("nuscenes_day",)             # nuScenes-rain: not supported (user decision 2026-09-28)
# Final-v2 TEST (K35(c); Manager 2026-09-28): the committed nuScenes-night TEST set (84 scenes, 3385 keyframes; manifest
# sha256 0937374a, provenance/dn01_nuscenes_night_test/). TEST-ONLY: refused unless --split test --allow-test, before
# any frame, manifest or model is loaded. Scored ONCE, on the final model, by R+D 2's chain after AUDIT OK + GO.
SETS["nuscenes_night_test"] = dict(max_depth=80.0, min_depth=0.1, short_side=518, cells=["real"], split="test")
TEST_ONLY_SETS = ("nuscenes_night_test",)
_PROV = Path(__file__).resolve().parents[1] / "provenance"
POSES = {"nuscenes_night": _PROV / "dn01_nuscenes_poses/nuscenes_night_v1_poses.json",
         # Joint R+D's day poses (scripts/dn01_nuscenes_poses.py on the v1 manifest); --temporal needs them
         "nuscenes_day": _PROV / "dn01_nuscenes_poses/nuscenes_day_v1_poses.json",
         "nuscenes_night_test": _PROV / "dn01_nuscenes_poses/nuscenes_night_test_v1_poses.json"}
NIGHT_TEST_SRC = _PROV / "dn01_nuscenes_night_test"     # nuscenes_night_test_v1.json + SHA256SUMS + PROJECTIONS.sha256
NIGHT_TEST_MANIFEST_SHA256 = "0937374a4457bb778a1c872f486954b460755a911050df305177a309676792e6"
POSES_SUMS = _PROV / "dn01_nuscenes_poses/SHA256SUMS"
DAY_RAIN_SPLIT = _PROV / "dn01_nuscenes_day_rain/day_rain_split_v1.json"          # sha256 adabae02 (committed)
DAY_RAIN_SPLIT_SHA256 = "adabae02"
# Joint R+D's manifests commit (92c0005): the manifests + per-set projection sha lists are committed there; the npz
# projections are external (host dirs below). Every file read is checked against those committed hashes.
NUSCENES_MANIFEST_SRC = _PROV / "dn01_nuscenes_day_rain"          # nuscenes_day_v1.json + SHA256SUMS
PROJECTION_SUMS = str(_PROV / "dn01_nuscenes_day_rain/PROJECTIONS_{set}.sha256")
NUSCENES_MANIFEST_DIR = (Path("/projects/b5dh/data/nuscenes_night_manifest") if Path("/projects/b5dh").is_dir()
                         else Path("/mnt/DATA/DegradedDepth/RAMEN/results/nuscenes_manifest/out"))   # projections


class TestSplitRefused(PermissionError):
    """The nuScenes-day TEST half was requested without the explicit --allow-test flag."""


def check_split(ds: str, split: str, allow_test: bool) -> None:
    """The K46 guard: day TEST is never read unless ``allow_test`` (the CLI's --allow-test)."""
    if ds in NUSCENES_SPLIT_SETS and split not in ("dev", "test"):
        raise ValueError(f"{ds}: split must be dev or test, got {split!r}")
    if ds in NUSCENES_SPLIT_SETS and split == "test" and not allow_test:
        raise TestSplitRefused(f"{ds} TEST is held out (K46): pass --allow-test only for a locked model under a "
                               "newly pre-declared rule")
    if ds in TEST_ONLY_SETS and not allow_test:
        raise TestSplitRefused(f"{ds} is the final-v2 TEST set (K35(c)): scored ONCE, on the final model, with "
                               "--split test --allow-test after the Auditor's OK and the Manager's GO")
    if ds in TEST_ONLY_SETS and split != "test":
        raise ValueError(f"{ds}: split must be test, got {split!r}")


def night_test_manifest() -> tuple[dict, Path]:
    """The committed nuScenes-night TEST manifest (sha-checked: 0937374a) and its projection dir."""
    path = NIGHT_TEST_SRC / "nuscenes_night_test_v1.json"
    if bj.sha256_file(path) != NIGHT_TEST_MANIFEST_SHA256:
        raise RuntimeError(f"{path}: sha256 is not the committed TEST manifest {NIGHT_TEST_MANIFEST_SHA256[:8]}")
    return json.loads(path.read_text()), NUSCENES_MANIFEST_DIR / "nuscenes_night_test_v1_projections"


def iter_night_test(B, allow_test: bool = False, limit: int | None = None, verify: bool = True):
    """The committed night TEST manifest, iterated as benchmark._iter_nuscenes_split (selection order; frames by
    timestamp_us; the frozen rasteriser), every projection checked against PROJECTIONS.sha256 (``verify``)."""
    check_split("nuscenes_night_test", "test", allow_test)
    man, proj_dir = night_test_manifest()
    sums = ({ln.split()[1]: ln.split()[0] for ln in (NIGHT_TEST_SRC / "PROJECTIONS.sha256").read_text().splitlines()
             if ln.strip()} if verify else None)
    scenes = man["selection"]["night"][:limit] if limit else man["selection"]["night"]
    yield from _iter_manifest(B, man, proj_dir, sums, "night", scenes)


def day_manifest(ds: str) -> tuple[dict, Path]:
    """Joint R+D's v1 manifest for ``ds`` and its projection dir. Refuses a manifest whose DEV/TEST scene lists are
    not the committed split's (day_rain_split_v1.json, sha256 adabae02)."""
    name = SETS[ds]["manifest"]
    path = manifest_path(ds)
    sums = {ln.split()[1]: ln.split()[0] for ln in (NUSCENES_MANIFEST_SRC / "SHA256SUMS").read_text().splitlines()
            if ln.strip()}
    if bj.sha256_file(path) != sums.get(path.name):
        raise RuntimeError(f"{path}: sha256 is not the committed one in {NUSCENES_MANIFEST_SRC / 'SHA256SUMS'}")
    man = json.loads(path.read_text())
    split = json.loads(DAY_RAIN_SPLIT.read_text())
    if not bj.sha256_file(DAY_RAIN_SPLIT).startswith(DAY_RAIN_SPLIT_SHA256):
        raise RuntimeError(f"{DAY_RAIN_SPLIT} is not the committed split {DAY_RAIN_SPLIT_SHA256}")
    want = split["selection"][ds.removeprefix("nuscenes_")]
    if {k: sorted(v) for k, v in man["selection"].items()} != {k: sorted(v) for k, v in want.items()}:
        raise RuntimeError(f"{path}: selection differs from the committed split {DAY_RAIN_SPLIT.name}")
    return man, NUSCENES_MANIFEST_DIR / f"{name}_projections"


def manifest_path(ds: str) -> Path:
    return NUSCENES_MANIFEST_SRC / f"{SETS[ds]['manifest']}.json"


def projection_sums(ds: str) -> dict[str, str]:
    """{'<manifest>_projections/<token>.npz': sha256} from the committed PROJECTIONS_<set>.sha256."""
    f = Path(PROJECTION_SUMS.format(set=ds.removeprefix("nuscenes_")))
    return {ln.split()[1]: ln.split()[0] for ln in f.read_text().splitlines() if ln.strip()}


def iter_day(B, ds: str, split: str, allow_test: bool = False, limit: int | None = None, verify: bool = True):
    """benchmark._iter_nuscenes_split, verbatim in order and GT, on the day v1 manifest's ``split`` half:
    (scene, pos, frame (3,H,W) [0,1] native, gt (H,W) m, mask). Each projection is checked against the committed
    PROJECTIONS_<set>.sha256 (``verify``)."""
    check_split(ds, split, allow_test)
    man, proj_dir = day_manifest(ds)
    sums = projection_sums(ds) if verify else None
    scenes = man["selection"][split][:limit] if limit else man["selection"][split]
    yield from _iter_manifest(B, man, proj_dir, sums, split, scenes)


def _iter_manifest(B, man: dict, proj_dir: Path, sums: dict | None, frame_split: str, scenes: list[str]):
    """The frozen iterator's body on one manifest: frames whose ``split`` is ``frame_split`` in ``scenes`` (in order),
    per scene by timestamp_us; each projection sha-checked against ``sums`` (None: unchecked)."""
    by_scene: dict[str, list[dict]] = {}
    keep = set(scenes)
    for fr in man["frames"]:
        if fr["split"] == frame_split and fr["scene"] in keep:
            by_scene.setdefault(fr["scene"], []).append(fr)
    for scene in scenes:
        for pos, rec in enumerate(sorted(by_scene.get(scene, []), key=lambda r: r["timestamp_us"])):
            bgr = cv2.imread(str(B.NUSCENES_ROOT / rec["rgb"]))
            if bgr is None:
                raise ValueError(f"iter_day: failed to read {B.NUSCENES_ROOT / rec['rgb']}")
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            npz_path = proj_dir / Path(rec["projection"]).name
            if sums is not None and bj.sha256_file(npz_path) != sums[rec["projection"]]:
                raise RuntimeError(f"{npz_path}: sha256 differs from the committed PROJECTIONS list")
            npz = np.load(npz_path)
            depth_map, mask = B._nuscenes_rasterise(npz["xy"], npz["depth_m"], rec["height"], rec["width"])
            yield scene, pos, frame.transpose(2, 0, 1), depth_map, mask


def nuscenes_cameras(B, split: str = "night", ds: str = "nuscenes_night") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{scene: (K (N,3,3), c2w (N,4,4))} in the frozen iterator's frame order (benchmark._iter_nuscenes_split: the
    manifest's frames of ``split``, per scene, sorted by timestamp_us). K is native (GT is native: no crop).
    ``ds`` nuscenes_day reads its v1 manifest (``split`` dev / test) and poses instead."""
    manifest = (B._nuscenes_load_manifest()[0] if ds == "nuscenes_night" else
                night_test_manifest()[0] if ds == "nuscenes_night_test" else day_manifest(ds)[0])
    if not POSES[ds].is_file():
        raise FileNotFoundError(f"--temporal on {ds} needs the committed poses {POSES[ds]}")
    if ds == "nuscenes_night_test":                   # the TEST poses are sha-checked against the committed list
        want = {ln.split()[1]: ln.split()[0] for ln in POSES_SUMS.read_text().splitlines() if ln.strip()}
        if bj.sha256_file(POSES[ds]) != want.get(POSES[ds].name):
            raise RuntimeError(f"{POSES[ds]}: sha256 is not the committed one in {POSES_SUMS}")
    poses = json.loads(POSES[ds].read_text())
    by_scene: dict[str, list[dict]] = {}
    for fr in manifest["frames"]:
        if fr["split"] == split:
            by_scene.setdefault(fr["scene"], []).append(fr)
    out = {}
    for scene, recs in by_scene.items():
        toks = [r["camera_token"] for r in sorted(recs, key=lambda r: r["timestamp_us"])]
        out[scene] = (np.array([poses[t]["K"] for t in toks]), np.array([poses[t]["c2w"] for t in toks]))
    return out


def benchmark():
    """The frozen benchmark.py, loaded read-only for its iterators. It puts the MAIN RAMEN checkout on sys.path[0] at
    import (for its own model wrappers, unused here), which would shadow this checkout's models/ and vendor/; sys.path
    is restored, and its top-level imports are stdlib / cv2 / numpy / torch only."""
    got = bj.sha256_file(str(BENCHMARK))
    if got != BENCHMARK_SHA256:
        raise RuntimeError(f"{BENCHMARK} sha256 {got} is not the frozen benchmark.py {BENCHMARK_SHA256}")
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location("benchmark_frozen", BENCHMARK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def kitti_test_frame_keys(manifest: Path | None = None) -> set[tuple[str, int]]:
    """(drive dir name, frame index) of EVERY frame of the bench KITTI TEST clips (the odd manifest clips, as
    bench_joint.load_clips('kitti', 'test') slices them; read here as names only, no image or GT is opened)."""
    with open(manifest or bj.PATHS["kitti_manifest"]) as f:
        clips_ = [(k, v) for entry in json.load(f)["kitti"] for k, v in entry.items()]
    return {(k, int(Path(fr["image"]).stem)) for k, v in clips_[1::2] for fr in v}


def kittic_keep_indices(frame_list: list[tuple[str, str]], test_keys: set[tuple[str, int]] | None = None) -> set[int]:
    """Indices into KITTI-C's (drive, frame_id) list that are NOT frames of a bench KITTI TEST clip. KITTI-C's list is
    data/test_files.txt (Eigen test, 697 rows); 51 of them are frames of the 6 TEST clips (drives 0002: 21, 0016: 9,
    0013: 18, 0036: 3), so scoring our models on KITTI-C would have shown them TEST imagery (clean and corrupted)."""
    tk = kitti_test_frame_keys() if test_keys is None else test_keys
    return {i for i, (drive, fid) in enumerate(frame_list) if (Path(drive).name, int(fid)) not in tk}


def clips(B, ds: str, corruption: str | None = None, limit: int | None = None, short_side: int | None = None,
          split: str = "dev", allow_test: bool = False):
    """Yield (clip_id, frames, gt (N,H,W) metres native) per video / image, in benchmark order. frames: (N,3,h,w)
    tensor at model resolution when ``short_side`` is given (resized frame by frame: ScanNet++ clips are 300 native
    frames), else (N,) native (H,W) sizes. KITTI-C: one image per clip; ``corruption`` = 'corr:sev' reads that split.
    nuscenes_day: the ``split`` half (dev; test only with ``allow_test``)."""
    if ds in NUSCENES_SPLIT_SETS:
        check_split(ds, split, allow_test)
        it = iter_day(B, ds, split, allow_test, limit)
    elif ds in TEST_ONLY_SETS:
        check_split(ds, split, allow_test)
        it = iter_night_test(B, allow_test, limit)
    elif ds == "kittic":
        # ultrareview: KITTI-C is built from the Eigen test list, whose frames overlap the bench KITTI TEST clips (51
        # frames). Those frames are dropped (clip ids stay the ORIGINAL list indices, so clean / corrupted pairing and
        # ``cmd_list`` drive lookup are unchanged); ``limit`` counts KEPT frames, so it is applied here, not by the iterator.
        keep = kittic_keep_indices(B._kittic_frame_list())
        it = (B.iter_kittic(False) if corruption in (None, "clean") else
              B.iter_kittic_corruption(corruption.split(":")[0], int(corruption.split(":")[1])))
        it = itertools.islice((r for r in it if int(r[0]) in keep), limit)
    else:
        it = {"nuscenes_night": B.iter_nuscenes_night, "nyu": B.iter_nyu, "scannetpp": B.iter_scannetpp}[ds](
            False, limit)
    cur, fr, gt = None, [], []
    def pack(fr):
        return torch.cat(fr) if short_side else np.array(fr)
    for vid, _pos, frame, g, _mask in it:
        if cur is not None and vid != cur:
            yield cur, pack(fr), np.stack(gt)
            fr, gt = [], []
        cur = vid
        f = np.asarray(frame, np.float32)
        fr.append(to_model(f[None], short_side) if short_side else f.shape[-2:])
        gt.append(np.asarray(g, np.float32))
    if cur is not None:
        yield cur, pack(fr), np.stack(gt)


def to_model(frames: np.ndarray, short_side: int) -> torch.Tensor:
    """(N,3,H,W) native -> (N,3,h,w) at bench_joint's model resolution (VDA Resize, as bench_joint.load_frames)."""
    bj._vda_path()
    from video_depth_anything.util.transform import Resize
    rs = Resize(width=short_side, height=short_side, resize_target=False, keep_aspect_ratio=True,
                ensure_multiple_of=14, resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC)
    out = [rs({"image": f.transpose(1, 2, 0)})["image"] for f in frames]
    return torch.from_numpy(np.clip(np.stack(out), 0, 1)).permute(0, 3, 1, 2).contiguous()


def cmd_list(a) -> None:
    for ds in a.dataset:
        check_split(ds, a.split or SETS[ds].get("split", "all"), a.allow_test)
    B = benchmark()
    out = {"benchmark_py_sha256": bj.sha256_file(str(BENCHMARK)), "ramen_commit": bj.git_state(bj.REPO), "sets": {}}
    for ds in a.dataset:
        sp = a.split or SETS[ds].get("split", "all")
        rows = [(cid, len(f), [int(v) for v in f[0]], float((g > 1e-3).mean()), float(g[g > 1e-3].max()))
                for cid, f, g in clips(B, ds, limit=a.limit_clips, split=sp, allow_test=a.allow_test)]
        drives = [d for d, _ in B._kittic_frame_list()] if ds == "kittic" else None
        out["sets"][ds] = {**{k: v for k, v in SETS[ds].items()}, "split": sp, "num_clips": len(rows),
                           "num_frames": sum(r[1] for r in rows),
                           "clips": [{"id": r[0], "frames": r[1], "hw": r[2], "gt_valid_frac": round(r[3], 4),
                                      "gt_max_m": round(r[4], 2), **({"drive": drives[int(r[0])]} if drives else {})}
                                     for r in rows]}
        print(ds, len(rows), "clips", out["sets"][ds]["num_frames"], "frames", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


@torch.no_grad()
def score(p: dict, x: torch.Tensor, clean: torch.Tensor, gt: np.ndarray, proto: dict, cell: str,
          sigma: float = 0.0, blur: float = 0.0, cam: tuple | None = None, vs: list | None = None,
          dump: dict | None = None) -> dict:
    """``sigma`` / ``blur``: the synthetic cell's TRUE values (oracle restorers use them, blind ones ignore them); 0 on
    real / KITTI-C cells, where they are unknown -- use blind restorers there (fastdvdnet_blind, pocketdvdnet).
    ``cam`` (K, c2w) adds VDA's TAE (bench_joint.score_clip). A low-light cell (``blur`` a bench_joint.LowLight) adds
    bench_joint's v2 native-exposure fields (``ev``), exactly as bench_joint.cmd_run."""
    depth, restored = bj.run_pipeline(p, x, sigma, blur, tuple(gt.shape[-2:]))
    ev = bj.LL_CELLS[blur]["ev"] if isinstance(blur, bj.LowLight) else None
    if dump is not None:                      # probe package: the raw depth + guides of this clip, for the ghosting look
        dump.update(depth=depth, guides=p["last_guides"], gt=gt)

    def one(dv):
        r = bj.score_clip(dv, p["metric"], gt, restored, x, clean, proto["max_depth"], proto["min_depth"], cam,
                          ev=ev)
        if cell == "real":        # no clean reference: restoration metrics are undefined, not "infinite"
            for k in ("psnr", "ssim", "psnr_input", "ssim_input", "eps_r", "eps_r_input", "joint_p95"):
                r[k] = None
        return r
    if vs is None:
        return one(depth)
    return {v: {**one(bj.apply_variant(v, depth, p["last_guides"])), **bj.variant_extras(v)} for v in vs}   # probe package: {variant: result}, ONE model pass (+ F3 DVP record)


@torch.no_grad()
def score_static(p: dict, frames: torch.Tensor, gt: np.ndarray, proto: dict, ds: str, cid: str, cell: str,
                 device: str, vs: list | None = None) -> dict:
    """K49 static cell (bench_joint.STATIC_CELLS), by bench_joint's own functions, as its cmd_run static branch: the
    clip's middle frame x STATIC_LEN (bench_joint.static_clip), an independent degradation per frame
    (bench_joint.degrade_static, seeded per (set, clip, cell)), then score_clip (v2 native fields; no TAE: the camera
    is still) + bench_joint.flicker_static (unaligned metric flicker + abs_rel_static; NaN for relative models)."""
    blur, sigma = bj.parse_cell(cell)
    sclean, sgt = bj.static_clip(frames, gt)
    x = bj.degrade_static(sclean, cell, bj.noise_seed(ds, cid, cell)).to(device)
    depth, restored = bj.run_pipeline(p, x, sigma, blur, tuple(gt.shape[-2:]))

    def one(dv):
        r = bj.score_clip(dv, p["metric"], sgt, restored, x, sclean.to(device), proto["max_depth"], proto["min_depth"],
                          None, None, bj.LL_CELLS[blur]["ev"])
        r.update(bj.flicker_static(dv, sgt, proto["max_depth"], proto["min_depth"]) if p["metric"]
                 else {"flicker": float("nan"), "abs_rel_static": float("nan")})
        return r
    if vs is None:
        return one(depth)
    return {v: {**one(bj.apply_variant(v, depth, p["last_guides"])), **bj.variant_extras(v)} for v in vs}     # + F3 DVP record ({} for other variants)


def synthetic(cell: str) -> bool:
    """A bench_joint synthetic cell (b<sigma_b>s<sigma> or a low-light cell), degraded here from the clean frames."""
    return cell[0] == "b" or cell in bj.LL_CELLS


def cmd_run(a) -> None:
    for ds in a.dataset:                          # the K46 TEST guard, before any model or frame is loaded
        check_split(ds, a.split or SETS[ds].get("split", "all"), a.allow_test)
    if any(ds in TEST_ONLY_SETS or (a.split or SETS[ds].get("split", "all")) != "dev" for ds in a.dataset):
        bj.log_test_access(a)                     # Auditor WP18 B4: one append-only line per non-dev run, before any model or frame is loaded
    B = benchmark()
    p = bj.build_pipeline(a.model, a.device)
    vs = bj._grid_variants(a)                     # probe package: [None] (unchanged path) or the --grid variants, ONE model pass
    spec = lambda v: a.model if v in (None, "none") else f"{a.model}+{v}"        # noqa: E731
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in a.dataset:
        split = a.split or SETS[ds].get("split", "all")
        cams = (nuscenes_cameras(B, split if ds in NUSCENES_SPLIT_SETS else "night", ds)
                if a.temporal and ds in POSES else {})     # night TEST: its manifest's frames are split "night"
        proto = SETS[ds]
        for cell in a.cells or proto["cells"]:
            dsts = {v: out_dir / f"{bj._slug(spec(v))}__{ds}__{cell.replace(':', 's')}__{split}.json" for v in vs}
            if all(d.exists() for d in dsts.values()):
                continue
            per = {v: {} for v in vs}
            ndump = 0
            ss, lim = proto["short_side"], a.limit_clips or proto.get("limit")
            src = clips(B, ds, cell if ds == "kittic" else None, lim, ss, split, a.allow_test)
            ref = clips(B, ds, None, lim, ss) if ds == "kittic" and cell != "clean" else None
            for cid, frames, gt in src:
                if cell in bj.STATIC_CELLS:                     # K49 static cell, exactly as bench_joint.cmd_run
                    r = score_static(p, frames, gt, proto, ds, cid, cell, a.device, None if vs == [None] else vs)
                    for v in vs:
                        per[v][cid] = r if vs == [None] else r[v]
                    continue
                x = frames.to(a.device)
                blur, sigma = bj.parse_cell(cell) if synthetic(cell) else (0.0, 0.0)
                if ref is not None:                              # KITTI-C: PSNR vs the same image's clean split
                    rid, rframes, _ = next(ref)
                    assert rid == cid
                    clean = rframes.to(a.device)
                elif synthetic(cell):                            # LL: seeded per cell name, as bench_joint.cmd_run
                    clean = x
                    seed = bj.noise_seed(ds, cid, blur if isinstance(blur, bj.LowLight) else sigma)
                    x = bj.degrade(x.cpu(), blur, sigma, seed).to(a.device)
                else:
                    clean = x
                if cams:
                    assert len(cams[cid][0]) == len(gt), (cid, len(cams[cid][0]), len(gt))
                dump = {} if (getattr(a, "dump_clips", 0) and ndump < a.dump_clips and cell == "real") else None
                r = score(p, x, clean, gt, proto, cell, sigma, blur, cams.get(cid), None if vs == [None] else vs, dump)
                for v in vs:
                    per[v][cid] = r if vs == [None] else r[v]
                if dump is not None:                      # ghosting look: {variant: depth} + the guide luma + GT, first frames only
                    from evaluation.depth_probe import dump_look
                    dump_look(out_dir / "look" / f"{bj._slug(a.model)}__{ds}__{cid}.npz",
                              {v: bj.apply_variant(v, dump["depth"], dump["guides"]) for v in (vs if vs != [None] else ["none"])},
                              dump["guides"]["r"], dump["gt"])
                    ndump += 1
            for var in vs:
                dst, per_v = dsts[var], per[var]
                if vs != [None] and dst.exists():
                    continue
                keys = [k for k in next(iter(per_v.values())) if k != "frames"]
                art = {"bench": "dn01_heldout_eval (bench_joint scorer)", "model": spec(var), "dataset": ds, "cell": cell,
                   "agf_backend": getattr(sys.modules.get("evaluation.depth_probe"), "AGF_BACKEND", "cpu"),   # protocol v3 WP24 (cpu = the numpy reference; gpu = bit-identical device implementation)
                   "split": split, "limit_clips": lim, "short_side": proto["short_side"], **bj.ckpt_info(a.model),
                   **({"static_len": bj.STATIC_LEN, "static_t0": bj.STATIC_T0} if cell in bj.STATIC_CELLS else {}),
                   **({"manifest": {"path": str(manifest_path(ds)).removeprefix(f"{bj.REPO}/"),
                                    "sha256": bj.sha256_file(manifest_path(ds))},
                       "split_file": {"path": str(DAY_RAIN_SPLIT).removeprefix(f"{bj.REPO}/"),
                                      "sha256": bj.sha256_file(DAY_RAIN_SPLIT)},
                       "projections_dir": str(NUSCENES_MANIFEST_DIR), "projections_verified": True,
                       "poses": ({"path": str(POSES[ds]).removeprefix(f"{bj.REPO}/"),
                                  "sha256": bj.sha256_file(POSES[ds])} if cams else None)}
                      if ds in NUSCENES_SPLIT_SETS else {}),
                   **({"manifest": {"path": "provenance/dn01_nuscenes_night_test/nuscenes_night_test_v1.json",
                                    "sha256": NIGHT_TEST_MANIFEST_SHA256},
                       "projections_dir": str(NUSCENES_MANIFEST_DIR), "projections_verified": True,
                       "allow_test": True} if ds in TEST_ONLY_SETS else {}),
                   "max_depth": proto["max_depth"], "metric_model": p["metric"], "num_clips": len(per_v),
                   **({"temporal_metrics": {"tae": "vendor/vda_eval/eval_tae.py (VDA @4f5ae23, sha256 8e481e94)",
                                            "poses": {"path": str(POSES[ds].relative_to(bj.REPO)),
                                                      "sha256": bj.sha256_file(str(POSES[ds]))}}} if cams else {}),
                   "weights": {k: {"path": v, "sha256": bj.sha256_file(v)} for k, v in p["weights"].items()},
                   "benchmark_py_sha256": bj.sha256_file(str(BENCHMARK)), "ramen_commit": bj.git_state(bj.REPO),
                   "host": socket.gethostname(), "torch": torch.__version__,
                   "video_macro": {k: (None if any(v[k] is None for v in per_v.values()) else
                                       float(np.nanmean([v[k] for v in per_v.values()]))) for k in keys},
                   "per_video": per_v}
                dropped = bj.nan_dropped(per_v, p["metric"])
                if dropped:                   # bug-sweep (eval-harness): the macro below skips non-finite clips; record + shout (see bench_joint.nan_dropped)
                    art["nan_clips_dropped_from_macro"] = dropped
                    print(f"WARNING {dst.name}: clips with a non-finite core metric were dropped from video_macro: {dropped}", flush=True)
                dst.write_text(json.dumps(art, indent=1))
                m = art["video_macro"]
                print(f"wrote {dst.name}: aligned {m['abs_rel_aligned']:.4f} unaligned {m.get('abs_rel_unaligned')}",
                      flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k34 = [d for d in SETS if d not in NUSCENES_SPLIT_SETS + TEST_ONLY_SETS]   # the default stays the K34 list; TEST
    # is never in a default: it must be named (--dataset nuscenes_night_test) AND unlocked (--allow-test)
    for name in ("list", "run"):
        s = sub.add_parser(name)
        s.add_argument("--dataset", nargs="+", default=k34, choices=list(SETS))
        s.add_argument("--limit-clips", type=int, default=None)
        s.add_argument("--out", required=True)
        s.add_argument("--split", default=None, choices=["dev", "test"],
                       help="nuscenes_day half (default dev); other sets are always 'all'")
        s.add_argument("--allow-test", action="store_true",
                       help="REQUIRED to read the day TEST half (K46: locked model + a new pre-declared rule only)")
    s = sub.choices["run"]
    s.add_argument("--model", required=True)
    s.add_argument("--cells", nargs="+", default=None)
    s.add_argument("--temporal", action="store_true", help="add TAE where committed poses exist (nuScenes sets)")
    s.add_argument("--device", default="cuda")
    s.add_argument("--grid", nargs="+", default=None,
                   help="probe package: score these depth post-filter variants (group names or literal variants) from ONE model pass; "
                        "one result file per variant, named as a single-token run")
    s.add_argument("--dump-clips", type=int, default=0,
                   help="probe package: save the first N real-cell clips' variant depths + guide (out/look/*.npz) for the ghosting look")
    a = ap.parse_args()
    if a.split and any(d not in NUSCENES_SPLIT_SETS + TEST_ONLY_SETS for d in a.dataset):
        ap.error("--split applies to nuscenes_day and nuscenes_night_test only")
    (cmd_list if a.cmd == "list" else cmd_run)(a)


if __name__ == "__main__":
    main()

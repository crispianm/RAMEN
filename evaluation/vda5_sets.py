"""Protocol v3 WP8: the VDA-five benchmark sets that bench_joint does not have yet (MPI-Sintel, NYUv2 VIDEO, ScanNet++ as the ScanNet* column), as a standalone module so it does not collide with the WP7 edits inside
bench_joint. Nothing here edits bench_joint or the frozen benchmark.py: ``register(bench_joint)`` adds the dataset protocols and host paths to ``bench_joint.DATASETS`` / ``PATHS`` once WP7 has landed (it is idempotent
and never overwrites an existing entry); the clip manifests are the VDA-format JSONs (``{dataset: [{video: [{image, gt_depth, factor, [K, pose]}]}]}``) that bench_joint.load_clips already reads.

PROTOCOLS (VDA benchmark/eval/eval.py + dataset_extract, scout_bench.md section 1):
  sintel    TRAINING CLEAN pass (VDA's extractor reads ``training/clean``, not final), 23 scenes, first 100 frames, cap 70 m, 436x1024, GT = round(m * 65535/650) uint16 (factor 65535/650).
  nyuv2     the 8 raw-video scenes (basement_0001a bookstore_0001a cafe_0001a classroom_0001a kitchen_0003 office_0004 playroom_0002 study_0002), first 500 frames each, cap 10 m. VDA's extractor crops the RGB
            [45:471, 41:601]; OUR copy (WP3, nyu_convert.py) is pre-cropped for BOTH rgb and depth: (426, 560) each, so the protocol crop is the FULL frame (0:426, 0:560) = NO SECOND CROP (Auditor condition).
  scannet   ScanNet v2 (the real ScanNet column; mirror tenacioustommy/scannet, WP4): VDA's sorted()[:100] scenes, first 510 frames, 90-frame clips (frames 0..269 step 3), cap 10 m.
            Colour is VDA's pre-cropped 464x618 (8:-8, 11:-11; the 500-frame twin scannet_video_500.json shares the "scannet" key, so it is a manifest swap, not a second dataset; R/B swapped exactly as VDA's extractor, ``scannet_video_rgbfix.json`` is the channel-correct twin, NOT used for parity); depth is the uncropped 480x640
            uint16 mm png, so the protocol crop (8,472,11,629) on the GT gives 464x618 = the colour shape (VDA eval: a=8 b=-8 c=11 d=-11). 14/100 scenes have mirror-dropped frames; scene0013_01 has 289 frames.
  scannetpp ScanNet++ iPhone (the ScanNet* column when ScanNet v2 access does not land; labelled as such, never compared with VDA-paper ScanNet numbers): the 62 eval scenes (nvs_sem_val + nvs_test_iphone), 90 frames at
            10 Hz (stride 6), cap 10 m, exported by scripts/dn01_v3_scannetpp_eval_export.py to 480x640 RGB + nearest-upsampled 480x640 LiDAR depth (factor 1000), per-frame K and camera-to-world pose for TAE.
``min_depth`` follows bench_joint's existing sets (0.1 m; VDA's own valid mask is gt > 1e-3, which differs only below 10 cm).
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np

DATASETS = {
    "sintel": dict(max_depth=70.0, min_depth=0.1, max_len=100, crop=(0, 436, 0, 1024), short_side=518, frame_shape=(436, 1024), n_scenes=23, rgb_dir="clean"),
    # B1 (Auditor 10-04): the stored Sintel frames are R/B-SWAPPED (sintel_extract.py cv2.imwrite of a PIL-RGB array; VDA's own extractor has the same bug, so the literal ``sintel`` is VDA's parity protocol). ``sintel_rgbfix`` is the Table column:
    # the SAME files and manifest ("sintel" key), channels swapped back at load time (``swap_rb``), so it equals training/clean pixel-exactly; its llvar draws are pinned in their own file.
    "sintel_rgbfix": dict(max_depth=70.0, min_depth=0.1, max_len=100, crop=(0, 436, 0, 1024), short_side=518, frame_shape=(436, 1024), n_scenes=23, rgb_dir="clean", manifest_key="sintel", swap_rb=True),
    "nyuv2": dict(max_depth=10.0, min_depth=0.1, max_len=500, crop=(0, 426, 0, 560), short_side=518, frame_shape=(426, 560), n_scenes=8, rgb_dir="rgb", pre_cropped=True),
    "scannet": dict(max_depth=10.0, min_depth=0.1, max_len=90, crop=(8, 472, 11, 629), short_side=518, frame_shape=(464, 618), gt_shape=(480, 640), n_scenes=100, rgb_dir="color", manifest_key="scannet"),
    "scannet_rgbfix": dict(max_depth=10.0, min_depth=0.1, max_len=90, crop=(8, 472, 11, 629), short_side=518, frame_shape=(464, 618), gt_shape=(480, 640), n_scenes=100, rgb_dir="color_rgbfix", manifest_key="scannet"),
    # ScanNet TAE (the Table's TAE cell; protocol_v3 PROTOCOL_TABLE): VDA's scannet_video_tae.json (192 frames/scene, per-frame K + pose), first 20 sorted scenes, metric on frames [10:180] (eval_tae.sh --start_idx 10 --end_idx 180
    # --eval_scenes_num 20), the model sees all 192. ``scannet_tae`` = literal colour (color_origin 480x640, VDA's R/B swap: parity); ``scannet_tae_rgbfix`` = the SAME 480x640 color_origin frames and manifest with the channels swapped back at load (``swap_rb``): the Table cell, so literal vs rgbfix differ in colour ONLY.
    # B2: ``hard_crop`` = VDA eval_tae.sh --hard_crop: the 480x640 PREDICTION is CROPPED [8:-8, 11:-11] (protocol ``crop``) to the 464x618 GT, not resized (see ``score_slice``).
    "scannet_tae": dict(max_depth=10.0, min_depth=0.1, max_len=192, crop=(8, 472, 11, 629), short_side=518, frame_shape=(480, 640), gt_shape=(480, 640), n_scenes=100, rgb_dir="color_origin", manifest_key="scannet", score_slice=(10, 180), eval_scenes=20, hard_crop=True),
    "scannet_tae_rgbfix": dict(max_depth=10.0, min_depth=0.1, max_len=192, crop=(8, 472, 11, 629), short_side=518, frame_shape=(480, 640), gt_shape=(480, 640), n_scenes=100, rgb_dir="color_origin", manifest_key="scannet", score_slice=(10, 180), eval_scenes=20, hard_crop=True, swap_rb=True),
    "scannetpp": dict(max_depth=10.0, min_depth=0.1, max_len=90, crop=(0, 480, 0, 640), short_side=518, frame_shape=(480, 640), n_scenes=62, rgb_dir="rgb"),
}
NYU_SCENES = ("basement_0001a", "bookstore_0001a", "cafe_0001a", "classroom_0001a", "kitchen_0003", "office_0004", "playroom_0002", "study_0002")
_C = "/mnt/DATA/DegradedDepth/data/TEST"
_I = "/projects/b5dh/data/test/benchmarks"
PATHS_BY_HOST = {
    "crispwork": {"scannet_root": f"{_C}/scannet", "scannet_manifest": f"{_C}/scannet/scannet_video.json",
                  "scannet_rgbfix_root": f"{_C}/scannet", "scannet_rgbfix_manifest": f"{_C}/scannet/scannet_video_rgbfix.json",
                  "scannet_tae_root": f"{_C}/scannet", "scannet_tae_manifest": f"{_C}/scannet/scannet_video_tae.json",
                  "scannet_tae_rgbfix_root": f"{_C}/scannet", "scannet_tae_rgbfix_manifest": f"{_C}/scannet/scannet_video_tae.json",
                  "sintel_root": f"{_C}/sintel/video", "sintel_manifest": f"{_C}/sintel/sintel_video.json", "sintel_camdata": f"{_C}/sintel/training/camdata_left",
                  "sintel_rgbfix_root": f"{_C}/sintel/video", "sintel_rgbfix_manifest": f"{_C}/sintel/sintel_video.json",
                  "nyuv2_root": f"{_C}/nyuv2_video", "nyuv2_manifest": f"{_C}/nyuv2_video/nyuv2_video_500.json",
                  "scannetpp_root": f"{_C}/scannetpp_eval", "scannetpp_manifest": f"{_C}/scannetpp_eval/scannetpp_video.json"},
    "isambard": {"scannet_root": f"{_I}/scannet", "scannet_manifest": f"{_I}/scannet/scannet_video.json",
                 "scannet_rgbfix_root": f"{_I}/scannet", "scannet_rgbfix_manifest": f"{_I}/scannet/scannet_video_rgbfix.json",
                  "scannet_tae_root": f"{_I}/scannet", "scannet_tae_manifest": f"{_I}/scannet/scannet_video_tae.json",
                  "scannet_tae_rgbfix_root": f"{_I}/scannet", "scannet_tae_rgbfix_manifest": f"{_I}/scannet/scannet_video_tae.json",
                 "sintel_root": f"{_I}/sintel/video", "sintel_manifest": f"{_I}/sintel/sintel_video.json", "sintel_camdata": f"{_I}/sintel/training/camdata_left",
                 "sintel_rgbfix_root": f"{_I}/sintel/video", "sintel_rgbfix_manifest": f"{_I}/sintel/sintel_video.json",
                 "nyuv2_root": f"{_I}/nyuv2_video", "nyuv2_manifest": f"{_I}/nyuv2_video/nyuv2_video_500.json",
                 "scannetpp_root": f"{_I}/scannetpp_eval", "scannetpp_manifest": f"{_I}/scannetpp_eval/scannetpp_video.json"},
}


def host_paths() -> dict:
    return PATHS_BY_HOST["isambard" if Path("/projects/b5dh").is_dir() else "crispwork"]


def register(bj) -> list:
    """Add the protocols + host paths to a loaded ``bench_joint`` module (call after WP7). Never overwrites an existing entry; returns the dataset names added."""
    added = []
    for name, proto in DATASETS.items():
        if name not in bj.DATASETS:
            bj.DATASETS[name] = {k: v for k, v in proto.items() if k in ("max_depth", "min_depth", "max_len", "crop", "short_side", "manifest_key", "score_slice", "eval_scenes", "hard_crop", "swap_rb")}
            added.append(name)
    for k, v in host_paths().items():
        bj.PATHS.setdefault(k, v)
    return added


# ------------------------------------------------------------------------------------------------- manifest checks
def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text())


def check_set(name: str, root, manifest, check_images: bool = True, sample_every: int = 1) -> dict:
    """Facts + violations of one set against its protocol: scene count, frames per scene (<= max_len), the GT factor, and (``check_images``) that EVERY sampled rgb and gt has exactly ``frame_shape``
    (read from the image headers only). Returns {n_scenes, n_frames, shapes, bad_shapes, ...}; the tests / parity script assert on it."""
    from PIL import Image
    p = DATASETS[name]
    data = load_manifest(manifest)[p.get("manifest_key", name)]
    scenes = {k: v for e in data for k, v in e.items()}
    root = Path(root)
    bad, shapes, n = [], set(), 0
    for sc, frames in sorted(scenes.items()):
        for i, f in enumerate(frames):
            n += 1
            if not check_images or i % sample_every:
                continue
            with Image.open(root / f["image"]) as im, Image.open(root / f["gt_depth"]) as gt:
                rs, gs = im.size[::-1], gt.size[::-1]
            shapes.add((rs, gs))
            if rs != p["frame_shape"] or gs != p.get("gt_shape", p["frame_shape"]):
                bad.append((sc, i, rs, gs))
    return {"n_scenes": len(scenes), "n_frames": n, "frames_per_scene": {k: len(v) for k, v in sorted(scenes.items())}, "factors": sorted({f["factor"] for v in scenes.values() for f in v}),
            "shapes": sorted(shapes), "bad_shapes": bad, "max_len": p["max_len"], "frame_shape": p["frame_shape"]}


# ------------------------------------------------------------------------------------------------- Sintel poses (TAE)
TAG_FLOAT = 202021.25


def sintel_cam_read(path) -> tuple[np.ndarray, np.ndarray]:
    """Sintel ``.cam``: float32 tag 202021.25, 9 float64 (3x3 intrinsics M), 12 float64 (3x4 extrinsics N = [R | t], WORLD -> CAMERA). Returns (M (3,3), N (3,4))."""
    with open(path, "rb") as f:
        tag = struct.unpack("<f", f.read(4))[0]
        assert tag == TAG_FLOAT, f"{path}: bad .cam tag {tag}"
        M = np.frombuffer(f.read(72), "<f8").reshape(3, 3)
        N = np.frombuffer(f.read(96), "<f8").reshape(3, 4)
    return M.copy(), N.copy()


def sintel_camera(camdata_root, key: str, frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(K (N,3,3), camera-to-world (N,4,4)) for ``tae_clip`` of one Sintel scene's manifest frames (frame_XXXX.png -> camdata_left/<key>/frame_XXXX.cam)."""
    Ks, c2w = [], []
    for f in frames:
        M, N = sintel_cam_read(Path(camdata_root) / key / (Path(f["image"]).stem + ".cam"))
        w2c = np.vstack([N, [0.0, 0.0, 0.0, 1.0]])
        Ks.append(M)
        c2w.append(np.linalg.inv(w2c))
    return np.stack(Ks), np.stack(c2w)


def manifest_camera(frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(K (N,3,3), camera-to-world (N,4,4)) from the per-frame ``K`` / ``pose`` fields of a VDA-format manifest (the ScanNet++ export writes them like VDA's scannet TAE json)."""
    return np.array([f["K"] for f in frames], np.float64), np.array([f["pose"] for f in frames], np.float64)


def score_slice(proto: dict, depth, gt, restored, degraded, clean, cam):
    """Apply a dataset's ``hard_crop`` and ``score_slice`` = (lo, hi) to everything ``score_clip`` scores (frames [lo:hi]; the model ran on all of them). Identity without the keys. ``cam`` = (K (3,3) | (N,3,3), c2w (N,4,4)) or None.
    ``hard_crop`` (B2, VDA eval_tae.sh --hard_crop, eval_tae.py get_infer): the PREDICTION (N,H,W) is cropped by the protocol ``crop`` (a, b, c, d) = pred[:, a:b, c:d] so it has the GT's shape and score_clip's resize is a no-op (never a stretch)."""
    if proto.get("hard_crop"):
        a, b, c, d = proto["crop"]
        assert tuple(depth.shape[-2:]) == (a + b, c + d), f"hard_crop needs the uncropped {(a + b, c + d)} prediction (symmetric crop), got {tuple(depth.shape)}"
        depth = depth[..., a:b, c:d]
    sl = proto.get("score_slice")
    if sl is None:
        return depth, gt, restored, degraded, clean, cam
    lo, hi = sl
    if cam is not None:
        K, c2w = cam
        cam = (K[lo:hi] if np.ndim(K) == 3 else K, c2w[lo:hi])
    return depth[lo:hi], gt[lo:hi], restored[lo:hi], degraded[lo:hi], clean[lo:hi], cam


def camera(ds: str, key: str, frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """bench_joint.camera's fallback for the WP8 sets: Sintel from camdata_left, ScanNet / ScanNet++ from the manifest's per-frame K + pose (ScanNet's K is 4x4: its top-left 3x3 is used)."""
    if ds.startswith("sintel"):
        return sintel_camera(host_paths()["sintel_camdata"], key, frames)
    if ds.startswith("scannet"):
        K, c2w = manifest_camera(frames)
        return K[:, :3, :3], c2w
    raise ValueError(f"no camera poses for {ds}")


# ------------------------------------------------------------------------------------------------- ScanNet 500-frame parity variants (WP8; parity script only)
# B3 (Auditor 10-04): the parity driver must certify the REGISTERED path, so it wraps ONLY the names bench_joint does not register: the 500-frame twins of the ScanNet column. ``scannet`` / ``scannet_rgbfix`` / ``scannet_tae`` /
# ``scannet_tae_rgbfix`` come from ``register()`` unmodified (their manifests, eval_scenes, camera, hard_crop and score_slice are bench_joint's own).
#   scannet_500[_rgbfix] the 500-frame twins (frames 0..499 stride 1): the VDA PAPER's Table 1 ScanNet AbsRel/delta1 are on up-to-500-frame videos (VDA-S 0.110/0.876, VDA-L 0.089/0.926); the 90-frame set is paper Table 7 (VDA-L only: 0.075/0.954)
#                        Their clips are read from the "scannet" manifest key of scannet_video_500.json / scannet_video_rgbfix_500.json.
SCANNET_ALIASES = ("scannet_500", "scannet_500_rgbfix")


def register_scannet_variants(bj, workdir=None) -> list:
    """Register the two 500-frame ScanNet aliases in a loaded bench_joint and wrap its ``load_clips`` / ``camera`` / ``score_clip`` for THOSE NAMES ONLY (parity-run time only; bench_joint is not edited). ``workdir`` is unused (kept for callers)."""
    paths = host_paths()
    root = Path(paths["scannet_root"])
    man = {"scannet_500": root / "scannet_video_500.json", "scannet_500_rgbfix": root / "scannet_video_rgbfix_500.json"}
    for name in SCANNET_ALIASES:
        proto = dict(DATASETS["scannet"], max_len=500)
        bj.DATASETS.setdefault(name, {k: v for k, v in proto.items() if k in ("max_depth", "min_depth", "max_len", "crop", "short_side")})
        bj.PATHS[f"{name}_root"] = paths["scannet_root"]
        bj.PATHS[f"{name}_manifest"] = str(man[name])
    orig_load, orig_cam, orig_score = bj.load_clips, bj.camera, bj.score_clip

    def load_clips(dataset, split, limit=None, allow_test=False):
        if dataset not in SCANNET_ALIASES:
            return orig_load(dataset, split, limit, allow_test)
        if split != "dev" and not allow_test:
            raise SystemExit(f"load_clips: split {split!r} reads {dataset} TEST clips; refused without allow_test")
        clips = [(k, v) for e in json.loads(Path(bj.PATHS[f"{dataset}_manifest"]).read_text())["scannet"] for k, v in e.items()]
        if split != "all":
            clips = clips[{"dev": 0, "test": 1}[split]::2]
        if os.environ.get("SCANNET_SHARD"):                       # "i/n": every n-th scene from i (parity throughput only; each shard writes its own --out dir and the report merges the per_video dicts)
            i, n = map(int, os.environ["SCANNET_SHARD"].split("/"))
            clips = clips[i::n]
        return clips[:limit]

    def camera(ds, key, frames):
        if ds not in SCANNET_ALIASES:
            return orig_cam(ds, key, frames)
        K, c2w = manifest_camera(frames)
        return K[0, :3, :3], c2w

    def score_clip(depth, metric, gt_np, restored, degraded, clean, max_depth, min_depth, cam=None, *a, **k):
        if depth.shape[0] > 200:                                  # a 500-frame clip: the whole-clip float64 tensors do not fit a shared 24 GB card, so score on the CPU (same maths, only slower; no dataset-specific change)
            restored, degraded, clean = restored.cpu(), degraded.cpu(), clean.cpu()
        return orig_score(depth, metric, gt_np, restored, degraded, clean, max_depth, min_depth, cam, *a, **k)

    bj.load_clips, bj.camera, bj.score_clip = load_clips, camera, score_clip
    return list(SCANNET_ALIASES)

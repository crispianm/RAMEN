#!/usr/bin/env python3
"""DN-01 joint benchmark: degraded video in, depth + restored frames out, one file.

Scores any pipeline ``[denoiser+]depth_model`` on the same degraded clips, on three axes:
  depth        VDA's eval protocol (per-clip least-squares scale+shift in disparity, AbsRel /
               delta1 / RMSE, VDA crop and valid mask); metric models also score unaligned.
  restoration  PSNR / SSIM of the restored frames vs the clean frames at model resolution.
               A pipeline with no restorer is scored on its input (so ``vda_s`` = the input row).
  temporal     eps_r (restored-frame error flicker), eps_d (aligned depth error flicker on
               VDA's TGM mask Omega) and joint_p95 = p95 over Omega of max(eps_d, eps_r):
               the quantity the DN-01 joint temporal loss L3 targets.

Pipelines (``--model``): ``vda_s``, ``metric_vda_s``, ``dav2_s``, ``daac``, ``ours:<train.py ckpt>``,
optionally prefixed by restorers: ``fastdvdnet+vda_s`` is THE BASELINE TO BEAT. Restorers AFTER the depth
model run on its restored output only (``ours:<ckpt>+wiener_oracle``: DN-01 T0, depth unchanged).
Held-out blur (DN-01 R1 robustness, off by default): cells ``d<R>s<sigma>`` use a pillbox PSF of radius R
(``d6`` matches ``b3`` by second moment); ``wiener_oracle`` then uses the pillbox.
Static cells (DN-01 K49, off by default; existing cells unchanged): ``st_ll`` / ``st_llb3`` / ``st_llvar`` score the
clip's MIDDLE frame repeated 64x with an independent degradation per frame (st_llvar: + per-frame noise level and gain
jitter) and add ``flicker`` / ``abs_rel_static`` (:func:`flicker_static`, unaligned metric depth) to the usual fields.
Adding a competitor = one entry in DEPTH_MODELS or DENOISERS.

Protocol (fixed; bump BENCH_VERSION if any of it changes):
- Data: VDA's KITTI (13 clips) and Bonn (26 clips) manifests, <=110 frames per clip.
  ``--split dev`` = even-indexed clips, ``test`` = odd, ``all`` = both. Iterate on dev and
  score test once; tuning on test is tuning on the paper's numbers.
- Model resolution: VDA's Resize (keep aspect, multiple of 14, cubic), short side 378 on
  KITTI (native height, docs/DN01_METRIC_EVAL_PROTOCOL.md) and 518 on Bonn; clipped to [0,1].
- Degradation, per cell ``b<blur>s<sigma>``: cv2 Gaussian defocus blur (sigma px at model
  resolution), then AWGN (sigma/255) clamped to [0,1]. The WHOLE clip is degraded once with
  one seed per (dataset, clip, sigma), so every pipeline sees the identical noisy frames and
  a frame shared by two windows carries one noise realisation. (val01_eval_vda.py redraws
  noise per window inside forward, so its rows are not bit-comparable to these.)
- Video depth models run VDA's own ``infer_video_depth`` windowing (32 frames, overlap 10,
  keyframes, scale/shift stitching for relative models), reimplemented around a generic
  forward so a joint model's restored frames come out of the same windows; tested equal to
  the official function. Image models run per frame. fp32 throughout.
- Restorers (``+``-chained, left to right; e.g. ``fastdvdnet+restormer+vda_s``):
  ``fastdvdnet`` = released clipped-noise weights given the true sigma (oracle noise map;
  skipped when sigma is 0); ``fastdvdnet_blind`` = same weights, sigma estimated per clip
  (median-MAD, finest Haar diagonal band), always run; ``restormer`` = Restormer single-image
  defocus deblurring, kernel-blind, skipped when blur is 0 (the dn01_cascade baseline's rule).
  DOMAIN MISMATCH: Restormer's released defocus weights are trained on real dual-pixel defocus
  (DPDD), not this benchmark's synthetic Gaussian blur, so it under-deblurs here (KITTI b3s25:
  +0.14 dB over FastDVDnet alone at 6.5x the cost). ``wiener_oracle`` is the matched upper bound
  for a deblur stage: non-blind Wiener deconvolution with the cell's TRUE Gaussian PSF (fixed
  K=0.01, picked on dev frames; Richardson-Lucy was within 0.1 dB), skipped when blur is 0.
- Not the DN-01 gate protocol: ``b3s25`` here != ``b3s25`` in jobs/dn01_score_final.sh on
  KITTI (model resolution 378 vs 280, so blur px and noise act at a different scale; one noise
  draw per clip vs per window). F1-F4 reference rows are not bench_joint numbers.

"Beating the baseline" at a (dataset, cell) = lower aligned AbsRel AND higher PSNR AND lower
joint_p95 than ``fastdvdnet+vda_s``. ``table`` prints those three flags per row.

  python bench_joint.py run --model fastdvdnet+vda_s --out results/bench_joint
  python bench_joint.py run --model ours:runs/x/step20000.pt --dataset kitti --cells b3s25 --out ...
  python bench_joint.py latency --model fastdvdnet+restormer+vda_s --out ...   # 4090, idle GPU only;
                                                    # per-stage + total ms/frame, 518^2, 32 frames, fp16
  python bench_joint.py table --out results/bench_joint
  python bench_joint.py plot --out results/bench_joint   # latency vs AbsRel / PSNR at b3s25 (the figure)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

BENCH_VERSION = 3        # v3 (K96, 2026-10-03): moving-video llvar / llvarf cells, TEST access log, no released-only rule; new result files carry the ``__v3`` suffix (v1 DEV files stay readable)
REPO = Path(__file__).resolve().parent
BASELINE = "fastdvdnet+vda_s"
CELLS = ["b0s0", "b0s25", "b1.5s0", "b1.5s25", "b3s0", "b3s25"]   # blur px x AWGN /255
HEADLINE_CELL = "b3s25"

_HOSTS = {
    "crispwork": dict(
        vda_repo="/mnt/DATA/DegradedDepth/competition/Video-Depth-Anything",
        dav2_repo="/mnt/DATA/DegradedDepth/competition/Depth-Anything-V2",
        fastdvdnet_repo="/mnt/DATA/DegradedDepth/competition/fastdvdnet",
        restormer_repo="/mnt/DATA/DegradedDepth/competition/Restormer",
        vda_s="/mnt/DATA/DegradedDepth/RAMEN/checkpoints/video_depth_anything_vits.pth",
        metric_vda_s="/mnt/DATA/DegradedDepth/RAMEN/checkpoints/metric_video_depth_anything_vits.pth",
        dav2_s="/mnt/DATA/DegradedDepth/RAMEN/checkpoints/depth_anything_v2_vits.pth",
        daac_repo="/mnt/DATA/DegradedDepth/competition/DepthAnythingAC",      # = benchmark.py's, both hosts
        daac="/mnt/DATA/DegradedDepth/RAMEN_aaai/baselines_ckpt/depth_anything_ac/checkpoints/depth_anything_AC_vits.pth",
        # DN-01 KITTI B/R fix: the default KITTI root is the TRUE-RGB root (scripts/dn01_kitti_rgbfix.py; images under <clip>/rgb_fixed/, so manifest_sha256 tells fixed rows from swapped ones). The OLD swapped
        # root is kept ONLY as ``kitti_root_swapped_legacy`` (the fix's source and the swap-repro's reference); no scorer reads it.
        kitti_root="/mnt/DATA/DegradedDepth/data/TEST/kitti_rgbfix",
        kitti_manifest="/mnt/DATA/DegradedDepth/data/TEST/kitti_rgbfix/kitti_video.json",
        kitti_root_swapped_legacy="/mnt/DATA/DegradedDepth/RAMEN/results/vda_clean_repro/benchmark_data/kitti",
        bonn_root="/mnt/DATA/DegradedDepth/data/TEST/bonn",
        bonn_manifest="/mnt/DATA/DegradedDepth/RAMEN/results/vda_clean_repro/benchmark_data/bonn/bonn_video.json",
        kitti_raw_zips="/mnt/DATA/DegradedDepth/data/TEST/kitti_raw_val_drives",    # OXTS for TAE poses
        kitti_calib="/mnt/DATA/DegradedDepth/data/TEST/kitti_c/kitti_calib",
        # DN-01 F3 (inference-only baselines; fetched on Isambard by provenance/dn01_f3/submit_f3.sh STAGE=fetch; absent on crispwork unless fetched there)
        metric_vda_l="/mnt/DATA/DegradedDepth/RAMEN/checkpoints/metric_video_depth_anything_vitl.pth",
        vda_l="/mnt/DATA/DegradedDepth/RAMEN/checkpoints/video_depth_anything_vitl.pth",     # v3 WP29: released relative VDA-L, sha256 43df27c6...e6fa
        zerodce_repo="/mnt/DATA/DegradedDepth/competition/Zero-DCE_extension",
        dvp_repo="/mnt/DATA/DegradedDepth/competition/pytorch-deep-video-prior",
    ),
    "isambard": dict(
        vda_repo="/projects/b5dh/repos/Video-Depth-Anything",
        dav2_repo="/projects/b5dh/benchmark/competition/Depth-Anything-V2",
        fastdvdnet_repo="/projects/b5dh/benchmark/competition/fastdvdnet",
        restormer_repo="/projects/b5dh/benchmark/competition/Restormer",
        vda_s="/projects/b5dh/benchmark/weights/video_depth_anything_vits.pth",
        metric_vda_s="/projects/b5dh/benchmark/weights/metric_video_depth_anything_vits.pth",
        dav2_s="/projects/b5dh/benchmark/weights/depth_anything_v2_vits.pth",
        daac_repo="/projects/b5dh/benchmark/competition/DepthAnythingAC",
        daac="/projects/b5dh/benchmark/weights/depth_anything_AC_vits.pth",
        kitti_root="/projects/b5dh/data/test/benchmarks/kitti_rgbfix",                  # DN-01 KITTI B/R fix (built by STAGE=prep of provenance/dn01_kittifix/submit_kittifix.sh)
        kitti_manifest="/projects/b5dh/data/test/benchmarks/kitti_rgbfix/kitti_video.json",
        kitti_root_swapped_legacy="/projects/b5dh/data/test/benchmarks/kitti",
        bonn_root="/projects/b5dh/dn01_eval_data/bonn",
        bonn_manifest="/projects/b5dh/dn01_eval_data/bonn_video.json",
        kitti_raw_zips="/projects/b5dh/dn01_eval_data/kitti_raw_val_drives",
        kitti_calib="/projects/b5dh/dn01_eval_data/kitti_calib",
        metric_vda_l="/projects/b5dh/benchmark/weights/metric_video_depth_anything_vitl.pth",
        vda_l="/projects/b5dh/benchmark/weights/video_depth_anything_vitl.pth",              # v3 WP29: must be copied here (sha256 43df27c6...e6fa) before the VDA-L jobs run
        zerodce_repo="/projects/b5dh/benchmark/competition/Zero-DCE_extension",
        dvp_repo="/projects/b5dh/benchmark/competition/pytorch-deep-video-prior",
    ),
}
# Sky-fix guard dataset (OPT-IN: never in a default dataset list): the exact VKITTI sky label (655 m) on clips of the NON-training weather variants; manifest provenance/dn01_sky/vkitti_sky_v1.json (scripts/dn01_vkitti_sky_manifest.py).
_HOSTS["crispwork"].update(vkitti_root="/mnt/DATA/DegradedDepth/data/TRAIN/vkitti", vkitti_manifest=str(Path(__file__).resolve().parent / "provenance/dn01_sky/vkitti_sky_v1.json"))
_HOSTS["isambard"].update(vkitti_root="/projects/b5dh/data/train/vkitti", vkitti_manifest=str(Path(__file__).resolve().parent / "provenance/dn01_sky/vkitti_sky_v1.json"))
PATHS = _HOSTS["isambard" if Path("/projects/b5dh").is_dir() else "crispwork"]

# VDA's benchmark/eval/eval.py table (crop a:b, c:d on GT) + this benchmark's model short side.
DATASETS = {
    "kitti": dict(max_depth=80.0, min_depth=0.1, max_len=110, crop=(0, 374, 0, 1242), short_side=378, sky_mask="kitti_colour"),
    "bonn": dict(max_depth=10.0, min_depth=0.1, max_len=110, crop=(0, 480, 0, 640), short_side=518),
    "vkitti": dict(max_depth=80.0, min_depth=0.1, max_len=64, crop=(0, 375, 0, 1242), short_side=378, sky_depth_m=655.0),   # OPT-IN (OPTIN_DATASETS): the sky-fix guard
}
OPTIN_DATASETS = ("vkitti",)                                   # never in a default --dataset list; must be named
DEFAULT_DATASETS = [d for d in DATASETS if d not in OPTIN_DATASETS]
# v3 (Auditor rec a): a dataset ALIAS is a clip filter on a real dataset, named ``<alias>`` in the result file / ``dataset`` field and in the table, but scored with the base dataset's
# loaders, caps, camera poses and (draw / noise) seeds, so its clips are bit-identical to the same clips of the base. ``kitti_minus_0047`` = kitti without drive 2011_10_03_drive_0047_sync
# (the one KITTI eval drive that VKITTI Scene20 replicates), applied AFTER the dev / test split. Never in a default list.
DATASET_ALIASES = {"kitti_minus_0047": ("kitti", ("2011_10_03_drive_0047_sync",))}
# v3 WP8: the VDA-five sets bench_joint lacked (sintel, nyuv2, scannet [literal VDA frames], scannet_rgbfix [Table 1's colour], scannetpp) register here, AFTER DEFAULT_DATASETS (so a default run is still kitti + bonn);
# their protocols, host paths and camera poses live in evaluation/vda5_sets.py (scannet* entries read the manifest key "scannet": DATASETS[ds]["manifest_key"]).
from evaluation import vda5_sets as _vda5  # noqa: E402
_vda5.register(sys.modules[__name__])

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(repo: str | Path) -> str:
    try:
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "-uno"], text=True).strip()
        return head + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _vda_path() -> None:
    if PATHS["vda_repo"] not in sys.path:
        sys.path.insert(0, PATHS["vda_repo"])


def _load_file(name: str, path: Path):
    """Import one file by path (FastDVDnet's ``models.py``/``utils.py`` shadow RAMEN's)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- data

def load_clips(dataset: str, split: str, limit: int | None = None, allow_test: bool = False) -> list[tuple[str, list[dict]]]:
    """VDA manifest ``{dataset: [{video_key: [{image, gt_depth, factor}, ...]}]}``, in file order.
    Ultrareview: ``test`` / ``all`` (the TEST clips) are refused unless ``allow_test`` is passed; the only caller that passes it is ``cmd_run``, after ``check_split`` (--allow-test and the
    no-ours-on-non-DEV rule). The opt-in sky-guard clips are not dev / test halves (every split returns all of them), so they are exempt."""
    if split != "dev" and dataset not in OPTIN_DATASETS and not allow_test:
        raise SystemExit(f"bench_joint.load_clips: split {split!r} reads {dataset} TEST clips; refused without allow_test (only cmd_run, after check_split, may pass it)")
    with open(PATHS[f"{dataset}_manifest"]) as f:
        clips = [(k, v) for entry in json.load(f)[DATASETS[dataset].get("manifest_key", dataset)] for k, v in entry.items()]
    if DATASETS[dataset].get("eval_scenes"):                    # v3 WP8: ScanNet TAE scores VDA's FIRST 20 sorted scenes (eval_scenes_num 20), taken before the dev / test halves
        clips = clips[:DATASETS[dataset]["eval_scenes"]]
    if split != "all" and dataset not in OPTIN_DATASETS:        # the opt-in sky-guard clips are not dev / test halves: all of them are used under 'dev'
        clips = clips[{"dev": 0, "test": 1}[split]::2]
    return clips[:limit]


def select_clips(ds_name: str, split: str, limit: int | None = None) -> list[tuple[str, list[dict]]]:
    """The clips ``cmd_run`` scores for ``ds_name`` (a dataset or a :data:`DATASET_ALIASES` name): the base dataset's split, then the alias's drop filter, then ``limit``."""
    ds, drop = DATASET_ALIASES.get(ds_name, (ds_name, ()))
    clips = load_clips(ds, split, None if drop else limit, **({} if split == "dev" else {"allow_test": True}))
    return [(k, f) for k, f in clips if not any(t in k for t in drop)][:limit] if drop else clips


def kitti_swap_testonly() -> bool:
    from evaluation.sky_metrics import kitti_swap_testonly as f
    return f()


def load_frames(dataset: str, frames: list[dict]) -> tuple[torch.Tensor, tuple[int, int]]:
    """Clean clip ``(N,3,h,w)`` in [0,1] at model resolution, plus the raw frame size."""
    _vda_path()
    from video_depth_anything.util.transform import Resize
    s = DATASETS[dataset]["short_side"]
    resize = Resize(width=s, height=s, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                    resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC)
    imgs, raw_hw = [], None
    for f in frames[:DATASETS[dataset]["max_len"]]:
        img = cv2.cvtColor(cv2.imread(str(Path(PATHS[f"{dataset}_root"]) / f["image"])), cv2.COLOR_BGR2RGB)
        if dataset == "kitti" and kitti_swap_testonly():
            img = np.ascontiguousarray(img[..., ::-1])    # TEST-ONLY: re-introduce the old B/R swap on the fixed frames (G-FIX-3 repro / the calibration's swapped arm); recorded in the result json
        if DATASETS[dataset].get("swap_rb"):
            img = np.ascontiguousarray(img[..., ::-1])    # v3 B1/B2: the stored frames are R/B-swapped (VDA's extractors); the *_rgbfix twins swap them back here (same files as the literal set: parity differs in colour only)
        raw_hw = img.shape[:2]
        imgs.append(resize({"image": img.astype(np.float32) / 255.0})["image"])
    return torch.from_numpy(np.clip(np.stack(imgs), 0, 1)).permute(0, 3, 1, 2).contiguous(), raw_hw


def load_gt(dataset: str, frames: list[dict]) -> np.ndarray:
    """VDA eval's ``get_gt`` + crop: depth in metres, 0 -> -1 (invalid)."""
    a, b, c, d = DATASETS[dataset]["crop"]
    out = []
    for f in frames[:DATASETS[dataset]["max_len"]]:
        p = str(Path(PATHS[f"{dataset}_root"]) / f["gt_depth"])
        g = (np.load(p) if p.endswith(".npy") else cv2.imread(p, -1)).astype(np.float64) / f["factor"]
        g[g == 0] = -1
        out.append(g[a:b, c:d])
    return np.stack(out)


class Disk(float):
    """The blur of a ``d<R>s<sigma>`` cell: a pillbox (disk) PSF of radius R px, the HELD-OUT blur family
    (DN-01 R1 robustness cells, docs/DN01_FIX_PROPOSALS.md; not in CELLS, so off by default). A plain float
    blur is the Gaussian sigma. R = 6 matches b3 by per-axis second moment (disk variance R^2/4 = 3^2)."""


def disk_psf(radius: float) -> np.ndarray:
    """Normalised pillbox: the pixels whose centre lies within ``radius`` of the kernel centre."""
    n = int(np.ceil(radius))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    k = (xx ** 2 + yy ** 2 <= radius ** 2).astype(np.float32)
    return k / k.sum()


class LowLight(str):
    """A low-light cell (DN-01 final-v2, user 2026-09-26): data.elvis_noise.degrade_clip_lowlight with the BVI-RLV
    calibration (scripts/dn01_lowlight_calibrate.py -> provenance/dn01_lowlight_calibration/: kappa 0.186, EV median
    -1.68) and ONE fixed canonical noise profile (the ELVIS bank's per-parameter median), seeded per clip. Blind cells:
    sigma/blur-oracle restorers get no true values (use fastdvdnet_blind / pocketdvdnet)."""


LL_KAPPA, LL_EV_MEDIAN = 0.186, -1.68
LL_CELLS = {"ll": dict(ev=LL_EV_MEDIAN, blur=0.0),           # calibrated median dark + real noise
            "llb3": dict(ev=LL_EV_MEDIAN, blur=3.0),         # + Gaussian blur sigma 3 px (PRIMARY with b3s25)
            "llhard": dict(ev=LL_EV_MEDIAN - 1.0, blur=3.0)}  # one stop darker (DESCRIPTIVE)


# DN-01 NOISE STRESS (additive; the cells above are untouched): a NOISE LADDER at the llb3 exposure (EV -1.68) and blur
# (3 px). ``llb3x<k>`` multiplies the calibrated low-light noise amplitude (kappa 0.186) by k: kappa = LL_KAPPA * k, k in
# {1, 2, 4, 8, 16} (kappa 0.186 .. 2.98). ``llb3kappa1`` is ONE ABSOLUTE rung: kappa = 1.0 = the ELVIS bank's median
# profile at full strength (what an un-calibrated ELVIS training mix applies), i.e. x5.376 (between x4 and x8). Exposure,
# blur, canonical profile and the per-clip SEED (the base cell's name, so a rung is the same draw scaled, nested across
# rungs) are those of llb3: ``llb3x1`` reproduces ``llb3`` bit for bit. ``st_<rung>`` = the static variant of st_llb3.
LADDER_K = (1, 2, 4, 8, 16)
LADDER_CELLS = {**{f"llb3x{k}": dict(base="llb3", kappa=LL_KAPPA * k) for k in LADDER_K},
                "llb3kappa1": dict(base="llb3", kappa=1.0)}
STATIC_LADDER_CELLS = {f"st_{n}": n for n in LADDER_CELLS}


def ll_cfg(cell: str) -> dict:
    """The low-light cell's ``ev`` / ``blur`` / ``kappa`` (a :data:`LL_CELLS` cell: the calibrated kappa; a ladder rung: its own)."""
    if cell in LADDER_CELLS:
        return {**LL_CELLS[LADDER_CELLS[cell]["base"]], "kappa": LADDER_CELLS[cell]["kappa"]}
    return {**LL_CELLS[cell], "kappa": LL_KAPPA}


def seed_cell(cell: str) -> str:
    """The cell name a noise seed is derived from: a ladder rung uses its base cell's name (llb3 / st_llb3), every other cell its own."""
    if cell in LADDER_CELLS:
        return LADDER_CELLS[cell]["base"]
    if cell in STATIC_LADDER_CELLS:
        return "st_" + LADDER_CELLS[STATIC_LADDER_CELLS[cell]]["base"]
    return cell


def _lowlight(clean: torch.Tensor, cell: str, seed: int) -> torch.Tensor:
    from data.elvis_noise import degrade_clip_lowlight, load_bank
    c = ll_cfg(cell)
    phi = np.median(load_bank().reshape(-1, 9), 0)
    return torch.from_numpy(degrade_clip_lowlight(clean.numpy(), c["ev"], phi, c["kappa"], c["blur"],
                                                  np.random.RandomState(seed % 2 ** 32)))


# DN-01 K49 (user 2026-09-28, Auditor (i) / (k)): STATIC cells for metric-depth temporal consistency. One still frame
# of an EVAL clip (its MIDDLE frame) repeated STATIC_LEN (64) times with its GT; every frame an INDEPENDENT degradation draw,
# so any depth change between frames is noise-induced flicker. st_ll / st_llb3 = the ll / llb3 degradation (one rng:
# independent per-frame noise); st_llvar = llb3 plus a per-frame noise level (kappa x U[0.5, 1.5]) and global gain
# (EV + U[-0.15, +0.15] stops) -- the auto-gain case. Bias ((i)): WS2's static-clip training regime mirrors these cells.
STATIC_CELLS = {"st_ll": "ll", "st_llb3": "llb3", "st_llvar": "llb3"}
# AMENDMENT 2026-09-28 (Manager-approved, PRE-RESULT: no WS2 arm exists; the Auditor has the final word): STATIC_LEN
# 32 -> 64 so a static clip spans 3 VDA windows (stitching included), as F_noise's 64 frames. Disclosed: a 1-clip smoke of
# released metric_vda_s ONLY (1 K + 1 B dev clip, 32 frames, before this freeze; not an artifact) preceded the change.
STATIC_LEN, STATIC_T0 = 64, 8
LLVAR_KAPPA_MULT, LLVAR_EV_JITTER = (0.5, 1.5), 0.15


def static_clip(clean: torch.Tensor, gt: np.ndarray, n: int = STATIC_LEN, idx: int | None = None):
    """``clean`` (N,3,h,w), ``gt`` (N,H,W) -> frame ``idx`` (default: the middle, N // 2) repeated ``n`` times:
    ((n,3,h,w), (n,H,W))."""
    i = clean.shape[0] // 2 if idx is None else idx
    return clean[i:i + 1].expand(n, -1, -1, -1).contiguous(), np.repeat(gt[i:i + 1], n, 0)


def degrade_static(clean: torch.Tensor, cell: str, seed: int) -> torch.Tensor:
    """A static clip ``(n,3,h,w)`` -> its degraded frames for ``cell`` in :data:`STATIC_CELLS` (seeded, CPU)."""
    if cell in STATIC_LADDER_CELLS:                 # noise-stress rung: llb3 with a scaled kappa, one rng (as st_llb3)
        return _lowlight(clean, STATIC_LADDER_CELLS[cell], seed)
    base = STATIC_CELLS[cell]
    if cell != "st_llvar":
        return _lowlight(clean, base, seed)
    from data.elvis_noise import degrade_clip_lowlight, load_bank
    c = LL_CELLS[base]
    phi = np.median(load_bank().reshape(-1, 9), 0)
    rng = np.random.RandomState(seed % 2 ** 32)
    out = []
    for t in range(clean.shape[0]):
        m, j = rng.uniform(*LLVAR_KAPPA_MULT), rng.uniform(-LLVAR_EV_JITTER, LLVAR_EV_JITTER)
        out.append(degrade_clip_lowlight(clean[t:t + 1].numpy(), c["ev"] + j, phi, LL_KAPPA * m, c["blur"], rng))
    return torch.from_numpy(np.concatenate(out))


# DN-01 v3 (K96, protocol table section 1): MOVING-VIDEO llvar cells over real clips. ``llvar`` (HEADLINE) draws ONE severity per clip: EV ~ U[-3, -0.5] stops and
# kappa = LL_KAPPA (0.186, the BVI-RLV calibration of llb3, EV -1.68) x one of {0.5, 1, 2} (uniform), at llb3's fixed 3 px Gaussian blur, the ELVIS bank's median noise
# profile, the same ISP / noise model as llb3 (data.elvis_noise.degrade_clip_lowlight). ``llvarf`` (SUPPLEMENT only) keeps that per-clip draw as its centre and adds the
# st_llvar per-frame jitter (kappa x U[0.5, 1.5], EV + U[-0.15, +0.15]) on every frame. The DRAW depends only on (dataset, clip key, cell family "llvar"): never on the
# model, the run order, --limit-clips or the split; llvarf shares llvar's draw (nested, like the ladder rungs) with its own noise stream. Pinned by provenance/dn01_v3/llvar_draws_<dataset>.json.
LLVAR_EV_RANGE, LLVAR_KAPPA_MULTS, LLVAR_BLUR = (-3.0, -0.5), (0.5, 1.0, 2.0), LL_CELLS["llb3"]["blur"]
LLVAR_VIDEO_CELLS = ("llvar", "llvarf")
LLVARF_KAPPA_MULT, LLVARF_EV_JITTER = LLVAR_KAPPA_MULT, LLVAR_EV_JITTER      # = st_llvar's per-frame jitter
LLVAR_DRAW_DIR = REPO / "provenance" / "dn01_v3"


def llvar_draw(dataset: str, key: str) -> dict:
    """The per-clip severity of ``llvar`` / ``llvarf``: ``{ev, kappa_mult, kappa, blur}``, a pure function of (dataset, key)."""
    rng = np.random.RandomState(noise_seed(dataset, key, "llvar|draw") % 2 ** 32)
    ev, i = float(rng.uniform(*LLVAR_EV_RANGE)), int(rng.randint(len(LLVAR_KAPPA_MULTS)))
    return {"ev": ev, "kappa_mult": LLVAR_KAPPA_MULTS[i], "kappa": LL_KAPPA * LLVAR_KAPPA_MULTS[i], "blur": LLVAR_BLUR}


def degrade_llvar(clean: torch.Tensor, cell: str, draw: dict, seed: int) -> torch.Tensor:
    """``clean`` (N,3,h,w) -> the ``llvar`` / ``llvarf`` degraded clip for the clip's ``draw`` (seeded, CPU). ``seed`` = :func:`llvar_noise_seed`."""
    from data.elvis_noise import degrade_clip_lowlight, load_bank
    phi = np.median(load_bank().reshape(-1, 9), 0)
    rng = np.random.RandomState(seed % 2 ** 32)
    if cell == "llvar":
        return torch.from_numpy(degrade_clip_lowlight(clean.numpy(), draw["ev"], phi, draw["kappa"], draw["blur"], rng))
    out = []
    for t in range(clean.shape[0]):
        m, j = rng.uniform(*LLVARF_KAPPA_MULT), rng.uniform(-LLVARF_EV_JITTER, LLVARF_EV_JITTER)
        out.append(degrade_clip_lowlight(clean[t:t + 1].numpy(), draw["ev"] + j, phi, draw["kappa"] * m, draw["blur"], rng))
    return torch.from_numpy(np.concatenate(out))


def llvar_noise_seed(dataset: str, key: str, cell: str) -> int:
    return noise_seed(dataset, key, f"{cell}|noise")


def llvar_draw_table(dataset: str) -> dict:
    """Every clip key of ``dataset``'s manifest (keys only, no frame is read) -> its draw."""
    with open(PATHS[f"{dataset}_manifest"]) as f:
        keys = [k for entry in json.load(f)[DATASETS[dataset].get("manifest_key", dataset)] for k in entry]
    return {"cell_spec": {"ev_range": list(LLVAR_EV_RANGE), "kappa_mults": list(LLVAR_KAPPA_MULTS), "kappa_base": LL_KAPPA, "blur_px": LLVAR_BLUR,
                          "llvarf_kappa_mult": list(LLVARF_KAPPA_MULT), "llvarf_ev_jitter": LLVARF_EV_JITTER,
                          "isp_noise": "data.elvis_noise.degrade_clip_lowlight, ELVIS bank median profile, as llb3 (ll_cfg); exposure, blur, noise per draw",
                          "seed": "noise_seed(dataset, key, 'llvar|draw'); noise stream noise_seed(dataset, key, '<cell>|noise')"},
            "dataset": dataset, "draws": {k: llvar_draw(dataset, k) for k in keys}}


def llvar_draws_file(dataset: str) -> Path:
    return LLVAR_DRAW_DIR / f"llvar_draws_{dataset}.json"


def llvar_pin(dataset: str) -> str | None:
    """sha256 of the committed draw JSON of ``dataset`` after checking that every draw it holds equals the one this code computes (a drifted RNG / constant is refused).
    None when no JSON is committed for the dataset (a dataset added later)."""
    p = llvar_draws_file(dataset)
    if not p.exists():
        return None
    held = json.loads(p.read_text())["draws"]
    bad = [k for k, d in held.items() if llvar_draw(dataset, k) != d]
    if bad:
        raise SystemExit(f"bench_joint: llvar draws of {dataset} differ from {p.name} for {bad[:3]}...: refusing")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def flicker_static(depth: np.ndarray, gt_np: np.ndarray, max_depth: float, min_depth: float,
                   t0: int = STATIC_T0) -> dict:
    """K49 (k), on a static clip, UNALIGNED metric depth. ``depth`` (N,H0,W0) metres (resized to the GT size as
    score_clip), clamped to [min_depth, max_depth]; valid = the (constant) GT in (min_depth, max_depth). Over frames
    t >= t0 (the first t0 frames are warm-up):
      flicker        = mean over valid pixels of [mean_t |D_t - D_{t-1}|] / [mean_t D_t]   (pairs t = t0..N-1)
      abs_rel_static = mean over valid pixels and t of |D_t - GT| / GT                         (the degeneracy guard)"""
    H, W = gt_np.shape[-2:]
    if depth.shape[-2:] != (H, W):
        depth = np.stack([cv2.resize(d, (W, H)) for d in depth])
    d = np.clip(depth.astype(np.float64), min_depth, max_depth)
    g = gt_np[0].astype(np.float64)
    valid = (g > min_depth) & (g < max_depth)
    step = np.abs(np.diff(d[t0 - 1:], axis=0)).mean(0) / d[t0 - 1:].mean(0)
    return {"flicker": float(step[valid].mean()),
            "abs_rel_static": float((np.abs(d[t0:] - g) / np.where(valid, g, 1.0))[:, valid].mean())}


# DN-01 v2 (user 2026-09-26, Auditor K43; provenance/dn01_lowlight_defect/): an LL cell is scored at NATIVE exposure
# against data.elvis_noise.lowlight_reference(clean, ev) (the ``*_native`` fields, primary); a blind per-clip global
# DISPLAY gain computed from the INPUT only -- the same function for every pipeline -- brightens the restored frames,
# which are then scored against the normally exposed clean frame (the ``*_display`` fields, secondary).
LUMA = (0.299, 0.587, 0.114)
# Fitted on 400 training-mix clips only (scripts/dn01_gain_calibrate.py -> provenance/dn01_gain_calibration/): the
# 99.5th luma percentile beat the mean and q50-q99; held-out training clips' error 0.22 stops (mean), 0.02 (median).
GAIN_STAT, GAIN_REF, GAIN_MAX = "q99.5", 0.8796490759224356, 16.0


def gain_stat(x: torch.Tensor, stat: str) -> float:
    """A clip's input brightness: the median over frames of each frame's luma ``stat`` ('mean' or 'q<percent>')."""
    luma = (x.detach().double().cpu() * torch.tensor(LUMA, dtype=torch.float64).view(1, 3, 1, 1)).sum(1)
    luma = luma.flatten(1).numpy()
    per = luma.mean(1) if stat == "mean" else np.quantile(luma, float(stat[1:]) / 100.0, axis=1)
    return float(np.median(per))


def blind_gain(x: torch.Tensor) -> float:
    """The per-clip global display gain g = clip(GAIN_REF / stat(input), 1, GAIN_MAX): input only, never brightens
    less than 1x, identical for every pipeline (the restored frames never enter it)."""
    return float(np.clip(GAIN_REF / max(gain_stat(x, GAIN_STAT), 1e-4), 1.0, GAIN_MAX))


def parse_cell(cell: str) -> tuple[float, float]:
    """``b<sigma_b>s<sigma>`` (Gaussian blur), ``d<R>s<sigma>`` (disk blur, returned as :class:`Disk`) or a
    :data:`LL_CELLS` name (returned as :class:`LowLight`, sigma 0)."""
    if cell in LL_CELLS or cell in LADDER_CELLS or cell in LLVAR_VIDEO_CELLS:    # llvar / llvarf: the per-clip draw is made in cmd_run (the cell name only marks "low light, blind")
        return LowLight(cell), 0.0
    if cell in STATIC_LADDER_CELLS:
        return LowLight(STATIC_LADDER_CELLS[cell]), 0.0
    if cell in STATIC_CELLS:                   # K49 static cell: its base LL cell (the frames: degrade_static)
        return LowLight(STATIC_CELLS[cell]), 0.0
    blur, sigma = cell[1:].split("s")
    return (Disk(blur) if cell[0] == "d" else float(blur)), float(sigma)


def noise_seed(dataset: str, video_key: str, sigma: float) -> int:
    """Same derivation as scripts/val01_eval_vda.py::noise_seed."""
    return int(hashlib.sha256(f"{dataset}|{video_key}|awgn|{sigma}".encode()).hexdigest()[:8], 16)


def degrade(clean: torch.Tensor, blur: float, sigma: float, seed: int) -> torch.Tensor:
    """Blur (train's cv2 defocus, data/video_clip_dataset.py::_blur_frames), then clamped AWGN.
    CPU generator, so the noise is identical on every host. A :class:`LowLight` blur is a low-light cell."""
    if isinstance(blur, LowLight):
        return _lowlight(clean, blur, seed)
    x = clean.clone()
    if isinstance(blur, Disk):
        ker = disk_psf(blur)
        for t in range(x.shape[0]):
            img = np.ascontiguousarray(x[t].permute(1, 2, 0).numpy())
            x[t] = torch.from_numpy(cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_REFLECT_101)).permute(2, 0, 1)
    elif blur > 0:
        k = int(2 * round(3 * blur) + 1)
        for t in range(x.shape[0]):
            img = np.ascontiguousarray(x[t].permute(1, 2, 0).numpy())
            x[t] = torch.from_numpy(cv2.GaussianBlur(img, (k, k), blur)).permute(2, 0, 1)
    if sigma > 0:
        gen = torch.Generator().manual_seed(seed)
        x = (x + torch.randn(x.shape, generator=gen) * (sigma / 255.0)).clamp(0, 1)
    return x


# ----------------------------------------------------------------------------- models
# depth builder(device) -> dict(fwd, metric, window, weights). fwd((T,3,h,w) [0,1]) ->
# (depth-like (T,h,w), restored (T,3,h,w) | None); metric=True means fwd returns metres.
# restorer builder(device) -> dict(fn, weights). fn((T,3,h,w) [0,1], sigma/255, blur px) -> same;
# sigma and blur are the cell's TRUE values: an oracle restorer may use them, a blind one ignores them.

def _vda_released(device: str, metric: bool) -> dict:
    _vda_path()
    from video_depth_anything.video_depth import VideoDepthAnything
    key = "metric_vda_s" if metric else "vda_s"
    net = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384], metric=metric)
    net.load_state_dict(torch.load(PATHS[key], map_location="cpu", weights_only=False), strict=True)
    net = net.to(device).eval()
    return dict(fwd=lambda x: (net(_norm(x)[None])[0], None), metric=metric, window=32,
                weights={key: PATHS[key]})


def _metric_vda_l(device: str) -> dict:
    """DN-01 F3 second network: RELEASED Metric-Video-Depth-Anything-Large (HF depth-anything/Metric-Video-Depth-Anything-Large @607fcdbd, metric_video_depth_anything_vitl.pth
    sha256 24eba253...e620, CC BY-NC 4.0), the repo's own class and run.py's vitl config, through the SAME harness as metric_vda_s (VDA's infer_video_depth windowing 32 / overlap 10 /
    keyframes, the scorer's resolution, fp32). Metric: the agf stage applies directly (tau is relative)."""
    _vda_path()
    from video_depth_anything.video_depth import VideoDepthAnything
    net = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], metric=True)
    net.load_state_dict(torch.load(PATHS["metric_vda_l"], map_location="cpu", weights_only=False), strict=True)
    net = net.to(device).eval()
    return dict(fwd=lambda x: (net(_norm(x)[None])[0], None), metric=True, window=32, weights={"metric_vda_l": PATHS["metric_vda_l"]})


def _vda_l(device: str) -> dict:
    """v3 WP29: RELEASED relative Video-Depth-Anything-Large (video_depth_anything_vitl.pth, sha256 43df27c6...e6fa), the same class / vitl config / harness as ``metric_vda_l``, relative
    output (aligned metrics only), like ``vda_s``."""
    _vda_path()
    from video_depth_anything.video_depth import VideoDepthAnything
    net = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], metric=False)
    net.load_state_dict(torch.load(PATHS["vda_l"], map_location="cpu", weights_only=False), strict=True)
    net = net.to(device).eval()
    return dict(fwd=lambda x: (net(_norm(x)[None])[0], None), metric=False, window=32, weights={"vda_l": PATHS["vda_l"]})


VDA_CFG = {"vits": dict(features=64, out_channels=[48, 96, 192, 384]), "vitl": dict(features=256, out_channels=[256, 512, 1024, 1024])}
UPSTREAM_INPUT_SIZE = 518          # VDA run.py --input_size default; infer_video_depth's fp32=False default = fp16 autocast


MOTION_PREFIX = "motion_modules."
ALLOWED_TRAINABLE = ("motion_modules", "temp2", "proj")        # depth path = motion_modules; temp2 / proj are the restorer (dropped, never touch depth when st_depth_inject is off)


def load_motion_modules(net, ck: dict) -> str:
    """DN-01 F1-UP: overlay a train.py checkpoint's trained motion modules onto an UPSTREAM ``VideoDepthAnything`` (released Metric-VDA weights already loaded). The F1 / fv2_s arms are
    SharedTemporalNet = stock Metric-VDA-S whose ONLY depth-path trainables are ``vda.core.head.motion_modules`` (temp2 / proj are the restorer and never touch depth), and ``vda.core`` IS
    the upstream class, so the overlay is a strict load of the ``motion_modules.*`` keys of ``trainable_ema`` (``trainable`` only with a printed warning). A missing / unexpected key or a
    shape mismatch raises (strict). Returns the checkpoint key used."""
    if "trainable_ema" in ck:
        which = "trainable_ema"
    elif "trainable" in ck:
        which = "trainable"
        print("WARNING load_motion_modules: checkpoint has no trainable_ema; using the raw trainable weights", file=sys.stderr)
    else:
        raise KeyError("checkpoint has neither trainable_ema nor trainable")
    m = ck.get("cfg", {}).get("model", {})
    if m.get("st_depth_inject"):                                      # restorer -> VDA motion-module inputs: the depth path needs dproj, which this overlay cannot carry
        raise ValueError("REFUSED: checkpoint cfg sets st_depth_inject (depth path depends on the restorer); a motion-module-only overlay would be a silent partial load")
    if int(m.get("vda_lora_rank", 0) or 0) > 0:                       # encoder LoRA changes the depth path (the plain lora_rank key is dormant and inherited by every arm)
        raise ValueError("REFUSED: checkpoint cfg sets vda_lora_rank > 0 (encoder LoRA); a motion-module-only overlay would be a silent partial load")
    outside = sorted({k.split(".")[0] for k in ck[which] if k.split(".")[0] not in ALLOWED_TRAINABLE})
    if outside:                                                       # e.g. lora / dproj / head: trained weights this overlay would silently drop
        raise ValueError(f"REFUSED: {which} has keys outside {sorted(ALLOWED_TRAINABLE)}: {outside}")
    sd ={k[len(MOTION_PREFIX):]: v for k, v in ck[which].items() if k.startswith(MOTION_PREFIX)}
    net.head.motion_modules.load_state_dict(sd, strict=True)         # empty sd (no motion modules in this checkpoint) raises as missing keys
    return which


def _vda_upstream_net(encoder: str, ck: dict | None = None, metric: bool = True):
    """Upstream VideoDepthAnything + the released metric weights (+ ``ck``'s motion modules when given); on CPU, not yet eval / on device. ``metric=False`` (vits only; the KITTI-swap calibration's
    relative VDA-S) loads the released RELATIVE weights; every existing caller keeps the default."""
    _vda_path()
    from video_depth_anything.video_depth import VideoDepthAnything
    key = {"vits": "metric_vda_s", "vitl": "metric_vda_l"}[encoder] if metric else {"vits": "vda_s"}[encoder]
    net = VideoDepthAnything(encoder=encoder, metric=metric, **VDA_CFG[encoder])
    net.load_state_dict(torch.load(PATHS[key], map_location="cpu", weights_only=False), strict=True)
    if ck is not None:
        enc = ck.get("cfg", {}).get("model", {}).get("vda_encoder", encoder)
        if enc != encoder:
            raise ValueError(f"checkpoint encoder {enc} != {encoder}")
        load_motion_modules(net, ck)
    return net


def _vda_upstream(device: str, encoder: str, ckpt: str | None = None, metric: bool = True) -> dict:
    """DN-01 F3 pre-ruling (c) (audit/F3_prerulings_20261001.md): the released metric VDA run at its UPSTREAM default, i.e. VDA's OWN ``infer_video_depth(frames, fps,
    input_size=518, fp32=False)`` (fp16 autocast; its Resize lower_bound / multiple of 14 / cubic; NOTE its aspect rule: a frame with long/short side > 1.78 gets input_size
    518 * 1.777 / ratio, so KITTI (ratio ~3.3) runs at ~280 short side, Bonn at 518), on the harness's degraded frames (float, no 8-bit rounding), output resized to the
    scorer's size (bilinear, align_corners). A DESCRIPTIVE check row (KITTI + Bonn llb3, none) beside the scorer-pipeline row of the same network.
    DN-01 F1-UP: with ``ckpt`` (a train.py checkpoint of the shared-temporal / motion-unfrozen arms) the SAME path runs the released weights + that checkpoint's trained motion modules
    (``load_motion_modules``; the restorer is dropped, depth path only): same weights as the harness row of that arm, a different eval path."""
    key = {"vits": "metric_vda_s", "vitl": "metric_vda_l"}[encoder] if metric else {"vits": "vda_s"}[encoder]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt else None
    net = _vda_upstream_net(encoder, ck, metric)
    net = net.to(device).eval()

    def upstream(x, out_hw):
        frames = (x.permute(0, 2, 3, 1).float().cpu().numpy() * 255.0)
        d, _ = net.infer_video_depth(frames, 30, input_size=UPSTREAM_INPUT_SIZE, device=x.device.type, fp32=False)
        d = torch.from_numpy(np.ascontiguousarray(d)).float()
        return F.interpolate(d[:, None], size=out_hw, mode="bilinear", align_corners=True)[:, 0].numpy()
    return dict(fwd=lambda x: (net(_norm(x)[None])[0], None), metric=metric, window=32, upstream=upstream, weights={key: PATHS[key], **({"ours_up": ckpt} if ckpt else {})})


def _zerodcepp(device: str) -> dict:
    """DN-01 F3 enhancement-cascade front: RELEASED Zero-DCE++ (evaluation/f3_baselines.py header), blind."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from evaluation import f3_baselines as f3
    net = f3.build_zerodce(PATHS["zerodce_repo"], device)
    return dict(fn=lambda x, sigma, blur: f3.zerodce_enhance(net, x), net=net,
                weights={"zerodcepp": str(Path(PATHS["zerodce_repo"]) / f3.ZERODCE_WEIGHTS)})


def _dav2_s(device: str) -> dict:
    if PATHS["dav2_repo"] not in sys.path:
        sys.path.insert(0, PATHS["dav2_repo"])
    from depth_anything_v2.dpt import DepthAnythingV2
    net = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    net.load_state_dict(torch.load(PATHS["dav2_s"], map_location="cpu", weights_only=False), strict=True)
    net = net.to(device).eval()
    return dict(fwd=lambda x: (net(_norm(x)), None), metric=False, window=1, weights={"dav2_s": PATHS["dav2_s"]})


DAAC_DROP = ("geo_prior_gen.weight",)   # the one training-only key: benchmark.py's strict=False load reports it as
                                        # unexpected and NO missing key (checked 2026-09-28), so dropping it and
                                        # loading strictly gives the same parameters


def _daac(device: str) -> dict:
    """DA-AC (Depth Anything AC, arXiv:2507.01634; degraded-trained single-image depth), exactly as the frozen
    benchmark.py's ``build_daac``: the repo's own ``DepthAnything_AC`` class (vits, features 64, v2 layer indices), the
    same weights file, ImageNet normalisation, ``forward()['out']`` = relu(disparity). RELATIVE (affine-invariant
    disparity, like DAv2): ``metric=False``, so only aligned depth metrics exist. Per frame (window 1), fp32 (the
    benchmark defaults to bf16). DINOv2 is built without its hub download: the checkpoint overwrites every parameter.
    The repo is APPENDED to sys.path so its ``util``/``torchhub`` packages never shadow VDA's."""
    repo = PATHS["daac_repo"]
    if repo not in sys.path:
        sys.path.append(repo)
    import torchhub.facebookresearch_dinov2_main.hubconf as hub
    from depth_anything.dpt import DepthAnything_AC
    orig = hub.dinov2_vits14
    hub.dinov2_vits14 = lambda pretrained=True, **kw: orig(pretrained=False, **kw)   # looked up at build time
    try:
        net = DepthAnything_AC({"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384],
                                "version": "v2"})
    finally:
        hub.dinov2_vits14 = orig
    sd = torch.load(PATHS["daac"], map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    net.load_state_dict({k: v for k, v in sd.items() if k not in DAAC_DROP}, strict=True)
    net = net.to(device).eval()
    return dict(fwd=lambda x: (net(_norm(x))["out"], None), metric=False, window=1, weights={"daac": PATHS["daac"]})


def load_ours(ckpt: str, device: str):
    """A train.py checkpoint, rebuilt exactly as train.py builds and saves it: build_model(cfg),
    then ``trainable_ema`` into ModuleDict(trainable_modules(model)) -- so new modules the
    architecture adds are picked up with no change here."""
    sys.path.insert(0, str(REPO))
    _vda_path()                                  # a cfg with vda_repo unset falls back to a crispwork path
    from models import build_model, trainable_modules
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "rgb_restorer" in ck or "depth_restorer" in ck:   # composed RAMEN / RAMEN-L: ours: would silently drop the display head
        raise ValueError(f"{ckpt} is a composed checkpoint (display head); score it with l15d:<path>, not ours:")
    cfg = ck["cfg"]
    if cfg.get("model", {}).get("shared_temporal"):   # trainable_ema carries all of temp2 (BN buffers incl.); the
        cfg = {**cfg, "model": {**cfg["model"], "st_temp2_init": None}}   # init path is a training-host path
    net = build_model(cfg)
    torch.nn.ModuleDict(trainable_modules(net)).load_state_dict(ck.get("trainable_ema", ck.get("trainable")),
                                                                 strict=True)
    if getattr(net, "lora", None) is not None and hasattr(net, "merge_lora"):
        net.merge_lora()                         # DN-01 L1: fold the encoder LoRA -> plain VDA-S cost
    return net.to(device).eval()


def _ours(device: str, ckpt: str) -> dict:
    net = load_ours(ckpt, device)
    if getattr(net, "depth_input", "raw") == "restored":   # DN-01 design (c): depth CONSUMES the (gated) restored frames, so R is computed ONCE per window inside the forward and is the model's restored output
        def fwd_c(x):                            # (the restorer's time is inside the model's stage: the latency command counts it once; restored = each frame's WINDOW-level R, as consumed by depth)
            rgb = x.permute(1, 0, 2, 3)[None]
            restored = net.restore_batch(rgb)
            d, _ = net._depth_and_feats(net.consume(rgb, restored))
            return d[0, 0], restored[0].permute(1, 0, 2, 3)
        return dict(fwd=fwd_c, metric=not net.predicts_disparity, window=32, weights={"ours": ckpt})
    if hasattr(net, "restore_clip"):             # DN-01 shared-temporal: depth by windows, restoration per CLIP
        def fwd_st(x):
            d, feats = net.depth_and_feats(x.permute(1, 0, 2, 3)[None])
            return d[0, 0], feats                # vda_windows keeps each frame's features from its real-input window
        return dict(fwd=fwd_st, metric=not net.predicts_disparity, window=32, weights={"ours": ckpt},
                    st_restore=lambda clip, feats: net.restore_clip(clip, feats))
    joint = bool(getattr(net, "enh_decoder", False))
    fe = getattr(net, "frontend", None)
    kw = {"frontend_done": True} if fe is not None else {}

    def fwd(x):
        rgb = x.permute(1, 0, 2, 3)[None]                              # (1,3,T,h,w)
        if not joint:
            return net(rgb, **kw)[0, 0], None
        d, r = net(rgb, return_recon=True, **kw)
        return d[0, 0], r[0].permute(1, 0, 2, 3).clamp(0, 1)
    p = dict(fwd=fwd, metric=not net.predicts_disparity, window=32, weights={"ours": ckpt})
    if fe is not None:    # DN-01 FR: the front-end denoises the WHOLE clip before VDA's windowing, as in the cascade
        p["denoise"] = lambda x, sigma, blur: net.front(x)       # front-end (+ FR+W's Wiener)
        if getattr(net, "frontend_branch", False):   # DN-01 FR2: (restored y, depth input x_enc)
            p["denoise"] = lambda x, sigma, blur: net.front_pair(x)
        p["stages"] = [("frontend", p["denoise"])]
    return p


FT_VDA = "metric_vda_s_ft:"                # protocol v3 WP21: the FT Metric-VDA-S (ours minus TAG and restorer) = a train.py arch=vda checkpoint, scored through the ``ours:`` path
PERFRAME_FT = ("dav2_s_ft:", "daac_ft:", "da3_s_ft:")   # protocol v3 WP22/23: a fine-tuned per-frame net (train.py arch dav2 / daac), spec ``dav2_s_ft:<ckpt>`` / ``daac_ft:<ckpt>`` / ``da3_s_ft:<ckpt>``


def _vda_ft(device: str, ckpt: str) -> dict:
    """``metric_vda_s_ft:<ckpt>``: the ``ours:`` path, refusing a checkpoint that carries a restorer / front-end / shared-temporal module (that is ours, not the matched baseline)."""
    m = torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]["model"]
    bad = [k for k in ("shared_temporal", "restoration_head", "input_adapter") if m.get(k) not in (None, False, "none")] + (["frontend"] if m.get("frontend", "none") != "none" else [])
    assert m.get("arch") == "vda" and m.get("metric") and not bad, f"{FT_VDA} needs a plain metric arch=vda checkpoint, got arch={m.get('arch')} metric={m.get('metric')} restorer keys {bad}"
    return _ours(device, ckpt)


def _perframe_ft(device: str, spec: str) -> dict:
    """A train.py checkpoint of the per-frame arches (``models.dav2_net.DAV2Net`` / ``models.daac_net.DAACNet``), rebuilt by ``load_ours``. Window 1 (no temporal module); metric nets are scored unaligned."""
    kind, ckpt = spec.split(":", 1)
    net = load_ours(ckpt, device)
    arch = {"dav2_s_ft": "DAV2Net", "daac_ft": "DAACNet", "da3_s_ft": "DA3Net"}[kind]
    assert type(net).__name__ == arch, f"{kind}: checkpoint is a {type(net).__name__}, not a {arch}"
    return dict(fwd=lambda x: (net(x.permute(1, 0, 2, 3)[None])[0, 0], None), metric=bool(net.metric), window=1, weights={kind: ckpt})


def _fastdvdnet(device: str, blind: bool = False) -> dict:
    """Oracle: the true sigma as noise map, skipped at sigma 0 (FastDVDnet at sigma 0 costs ~9 dB on
    clean KITTI frames). Blind: sigma estimated per clip (estimate_sigma), always run."""
    repo = Path(PATHS["fastdvdnet_repo"])
    net = _load_file("fastdvdnet_models", repo / "models.py").FastDVDnet(num_input_frames=5)
    wpath = repo / "model_clipped_noise.pth"          # README: the weights for clipped AWGN
    sd = torch.load(wpath, map_location="cpu", weights_only=False)
    net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
    net = net.to(device).eval()

    def fn(x, sigma, blur):
        if blind:
            return fastdvdnet_denoise(net, x, estimate_sigma(x))
        return fastdvdnet_denoise(net, x, sigma) if sigma > 0 else x
    return dict(fn=fn, net=net, weights={"fastdvdnet": str(wpath)})


def _pocketdvdnet(device: str, ckpt: str | None = None) -> dict:
    """PocketDVDNet (blind; models.restoration.PocketFrontEnd, the module FR trains): the v1.0 weights, or with
    ``pocketdvdnet:<train.py checkpoint>`` that checkpoint's whole clip-level front path (``VDANet.front``: the
    fine-tuned front-end, + FR+W's Wiener if present) -- the FR / FR+W matched controls."""
    if ckpt:
        net = load_ours(ckpt, device)
        return dict(fn=lambda x, sigma, blur: net.front(x), net=net,
                    weights={"pocketdvdnet": net.frontend.checkpoint, "pocketdvdnet_ft": ckpt})
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from models.restoration import PocketFrontEnd
    fe = PocketFrontEnd().to(device).eval()
    return dict(fn=lambda x, sigma, blur: fe(x), net=fe, weights={"pocketdvdnet": fe.checkpoint})


def estimate_sigma(x: torch.Tensor) -> float:
    """Blind AWGN sigma (/255) of a clip (T,3,H,W): Donoho's median-MAD, median(|HH|)/0.6745 over the
    finest orthonormal Haar diagonal band of every frame and channel."""
    h, w = x.shape[-2] // 2 * 2, x.shape[-1] // 2 * 2
    y = x[..., :h, :w].float()
    hh = (y[..., 0::2, 0::2] - y[..., 0::2, 1::2] - y[..., 1::2, 0::2] + y[..., 1::2, 1::2]) / 2
    return float(hh.abs().median() / 0.6745 * 255)


WIENER_K = 0.01   # picked on 8 dev frames/dataset after oracle FastDVDnet, K in {1e-3..1e-1}; RL (5-40 it) was
                  # within 0.1 dB mean at ~40x the FFTs, so the faster tie is the fair speed baseline


def wiener_deblur(x: torch.Tensor, blur: float, k_reg: float = WIENER_K, batch: int = 8) -> torch.Tensor:
    """Non-blind Wiener deconvolution of (T,3,H,W) with the degradation's own PSF (cv2's truncated Gaussian,
    ksize 2*round(3 sigma)+1; a :class:`Disk` blur uses its pillbox), reflect-padded like cv2's border before
    the FFT, clamped to [0,1]."""
    if isinstance(blur, LowLight):
        raise ValueError("wiener_oracle has no true PSF on a low-light cell; use a blind restorer")
    if isinstance(blur, Disk):
        psf = torch.from_numpy(disk_psf(blur))
        k = psf.shape[0]
    else:
        k = int(2 * round(3 * blur) + 1)
        g = torch.from_numpy(cv2.getGaussianKernel(k, blur)[:, 0]).float()
        psf = torch.outer(g, g)
    xp_hw = (x.shape[-2] + 2 * k, x.shape[-1] + 2 * k)
    ker = torch.zeros(xp_hw, device=x.device)
    ker[:k, :k] = psf.to(x.device)
    H = torch.fft.rfft2(ker.roll((-(k // 2), -(k // 2)), (0, 1)))
    W = H.conj() / (H.abs() ** 2 + k_reg)
    out = []
    for s in range(0, x.shape[0], batch):
        xp = F.pad(x[s:s + batch].float(), (k, k, k, k), mode="reflect")
        out.append(torch.fft.irfft2(torch.fft.rfft2(xp) * W, s=xp_hw)[..., k:-k, k:-k].clamp(0, 1).to(x.dtype))
    return torch.cat(out)


def _wiener_oracle(device: str) -> dict:
    """Matched deblur upper bound for the cascade: the TRUE sigma_b's PSF, skipped at blur 0."""
    return dict(fn=lambda x, sigma, blur: wiener_deblur(x, blur) if blur > 0 else x, weights={})


def _restormer(device: str) -> dict:
    """Restormer single-image defocus deblurring (Zamir et al., CVPR 2022, v1.0 weights), exactly as the
    dn01_cascade baseline (scripts/dn01_prefilters.py): Options/DefocusDeblur_Single_8bit_Restormer.yml
    network_g, demo.py's reflect pad to a multiple of 8, per frame, clamped. Oracle on blur presence:
    skipped at blur 0 (kernel-blind otherwise)."""
    repo = Path(PATHS["restormer_repo"])
    arch = _load_file("restormer_arch", repo / "basicsr/models/archs/restormer_arch.py")
    net = arch.Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                         heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias",
                         dual_pixel_task=False)
    wpath = repo / "Defocus_Deblurring/pretrained_models/single_image_defocus_deblurring.pth"
    net.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True)["params"])
    net = net.to(device).eval()

    def fn(x, sigma, blur):
        if blur == 0:
            return x
        h, w = x.shape[-2:]
        return torch.cat([net(F.pad(x[t:t + 1], (0, (-w) % 8, 0, (-h) % 8), "reflect"))[..., :h, :w]
                          for t in range(x.shape[0])]).clamp(0, 1)
    return dict(fn=fn, weights={"restormer_defocus": str(wpath)})


def fastdvdnet_denoise(net, x: torch.Tensor, sigma: float, batch: int = 8) -> torch.Tensor:
    """fastdvdnet.py::denoise_seq_fastdvdnet (5-frame window, reflected at the clip ends, pad
    to a multiple of 4, clamp), batched over frames. Tested equal to the original."""
    n, _, h, w = x.shape
    ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
    xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
    idx = torch.arange(n)[:, None] + torch.arange(-2, 3)[None]         # (n,5) neighbours
    idx = torch.where(idx < 0, -idx, idx)
    idx = torch.where(idx > n - 1, 2 * (n - 1) - idx, idx).clamp(0, n - 1)   # n < 3 (single images): no wrap
    out = []
    for s in range(0, n, batch):
        stack = xp[idx[s:s + batch]].flatten(1, 2)                       # (b,15,H,W)
        nmap = torch.full((stack.shape[0], 1, *stack.shape[-2:]), sigma / 255.0, device=x.device, dtype=x.dtype)
        out.append(net(stack, nmap).clamp(0, 1)[..., :h, :w])
    return torch.cat(out)


DEPTH_MODELS = {
    "vda_s": lambda dev: _vda_released(dev, metric=False),
    "metric_vda_s": lambda dev: _vda_released(dev, metric=True),
    "dav2_s": _dav2_s,
    "daac": _daac,          # DN-01 K46: external degraded-trained depth (relative; aligned metrics only)
    "metric_vda_l": _metric_vda_l,   # DN-01 F3: the second (metric) video-depth network
    "vda_l": _vda_l,                 # v3 WP29: released relative VDA-L
    "metric_vda_s_up": lambda dev: _vda_upstream(dev, "vits"),   # DN-01 F3 pre-ruling (c): the upstream-default check rows
    "metric_vda_l_up": lambda dev: _vda_upstream(dev, "vitl"),
    "vda_s_up": lambda dev: _vda_upstream(dev, "vits", None, False),   # DN-01 KITTI-swap calibration: RELEASED relative VDA-S at VDA's upstream defaults (aligned metrics only)
}
DENOISERS = {"fastdvdnet": _fastdvdnet, "fastdvdnet_blind": lambda dev: _fastdvdnet(dev, blind=True),
             "restormer": _restormer, "wiener_oracle": _wiener_oracle, "pocketdvdnet": _pocketdvdnet,
             "zerodcepp": _zerodcepp}       # DN-01 F3: Zero-DCE++ (low-light enhancer) -> depth


# DN-01 K49 (Manager 2026-09-28): post-hoc temporal DEPTH smoothing baselines, appended to a spec as post tokens
# (``metric_vda_s+ema0.3``, ``metric_vda_s+tmed5``). They act on the depth model's stitched output (metres for a metric
# model, disparity for a relative one) at GT size, before scoring; restoration is untouched. Parameters are chosen on
# bench_joint DEV only (provenance/dn01_evalx/README.md).
# DN-01 probe package (2026-09-30): the causal probe filters ``agf...`` / ``sgf...`` (evaluation/depth_probe.py) are further post tokens; the
# existing ``ema`` / ``tmed`` behaviour is unchanged.
# DN-01 F3: ``dvp`` = Deep Video Prior (evaluation/f3_baselines.py; blind, NON-causal test-time training per clip) is one more post token (group 6).
# v3 WP29 / FBASE: ``kal<q/r>`` (Kalman), ``oneeuro[<mincutoff>b<beta>]`` (1-euro), ``flowema<alpha>`` (flow-aligned EMA): causal, evaluation/fbase_filters.py (groups 7-13).
DEPTH_POST_RE = re.compile(r"(ema)(0?\.\d+|1(?:\.0)?)|(tmed)(\d+)|((?:agf|sgf|rgf)[0-9a-z.]+)|(dvp)|(kal)(\d*\.?\d+)|(oneeuro)(?:(\d*\.?\d+)b(\d*\.?\d+))?|(flowema)(0?\.\d+|1(?:\.0)?)")


def depth_ema(depth: np.ndarray, alpha: float) -> np.ndarray:
    """Causal exponential moving average over frames: y_0 = d_0, y_t = alpha d_t + (1 - alpha) y_{t-1}."""
    out = np.empty_like(depth)
    acc = depth[0].astype(np.float64)
    out[0] = acc
    for t in range(1, depth.shape[0]):
        acc = alpha * depth[t] + (1.0 - alpha) * acc
        out[t] = acc
    return out


def depth_tmed(depth: np.ndarray, width: int) -> np.ndarray:
    """Centered temporal median of odd ``width`` per pixel, truncated at the clip ends (non-causal)."""
    if width % 2 == 0 or width < 1:
        raise ValueError(f"tmed width must be odd, got {width}")
    k, n = width // 2, depth.shape[0]
    return np.stack([np.median(depth[max(0, t - k):t + k + 1], axis=0) for t in range(n)]).astype(depth.dtype)


def depth_post(token: str):
    """``ema<alpha>`` / ``tmed<width>`` / a probe token -> fn((N,H,W) numpy, guides=None) -> (N,H,W). ``guides`` = {'i': input frames, 'r': restored
    frames} (run_pipeline provides them); only the guided ``sgf`` tokens read it."""
    m = DEPTH_POST_RE.fullmatch(token)
    if m is None:
        raise ValueError(f"not a depth post-filter: {token}")
    if m.group(5):
        from evaluation import depth_probe
        return depth_probe.make_post(token)
    if m.group(1):
        a = float(m.group(2))
        if not 0.0 < a <= 1.0:
            raise ValueError(f"ema alpha must be in (0, 1]: {token}")
        return lambda d, guides=None: depth_ema(d, a)
    if m.group(6):
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from evaluation import f3_baselines
        return f3_baselines.make_dvp_post(PATHS["dvp_repo"])
    if m.group(7) or m.group(9) or m.group(12):
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from evaluation import fbase_filters as fb
        if m.group(7):
            return lambda d, guides=None: fb.depth_kalman(d, float(m.group(8)))
        if m.group(9):
            mc, be = (float(m.group(10)), float(m.group(11))) if m.group(10) else fb.EURO_DEFAULT
            return lambda d, guides=None: fb.depth_oneeuro(d, mc, be)
        a = float(m.group(13))
        return lambda d, guides=None: fb.depth_flowema(d, None if guides is None else guides.get("i"), a)
    return lambda d, guides=None: depth_tmed(d, int(m.group(4)))


def apply_variant(name: str, depth: np.ndarray, guides: dict | None) -> np.ndarray:
    """A grid variant ('none', 'ema0.7', 'agf0.3t0.06', 'sgf4e0.1r+ema0.7') applied to the raw pipeline depth, exactly as a single-token run would."""
    for tok in ([] if name == "none" else name.split("+")):
        depth = depth_post(tok)(depth, guides)
    return depth


def variant_extras(v) -> dict:
    """DN-01 F3 (pre-ruling 2): the per-clip DVP record (out-of-range outputs clamped, floored depths, pixels, wall-clock seconds) for a variant that contains the ``dvp`` token,
    read from the DVP memo right after the variant was applied; {} for every other variant (their results are unchanged)."""
    if v is None or "dvp" not in v.split("+"):
        return {}
    from evaluation import f3_baselines
    return dict(f3_baselines.MEMO.stats)


def split_spec(spec: str) -> tuple[list[str], str, list[str]]:
    """``[pre+...+]depth[+post+...]`` -> (pre, depth, post). Pre-restorers run on the depth model's input;
    post-restorers run on its restored output and leave depth untouched (DN-01 T0:
    ``ours:<ckpt>+wiener_oracle`` = oracle Wiener on the joint model's restored frames)."""
    parts = spec.split("+")
    i = next(i for i, s in enumerate(parts) if s.startswith(("ours:", "ours_up:", "l15d:", FT_VDA) + PERFRAME_FT) or s in DEPTH_MODELS)
    return parts[:i], parts[i], parts[i + 1:]


def _chain(names: list[str], device: str):
    """Restorers applied left to right -> (fn(x, sigma, blur), [(name, fn)], weights)."""
    ds = [_pocketdvdnet(device, c.split(":", 1)[1]) if c.startswith("pocketdvdnet:") else DENOISERS[c](device)
          for c in names]

    def fn(x, sigma, blur):
        for d in ds:
            x = d["fn"](x, sigma, blur)
        return x
    return fn, [(c, d["fn"]) for c, d in zip(names, ds)], {k: v for d in ds for k, v in d["weights"].items()}


def build_pipeline(spec: str, device: str) -> dict:
    """``[restorer+...+]depth[+restorer+...]``, restorers applied left to right; depth is a DEPTH_MODELS key
    or ``ours:<ckpt>``."""
    chain, dep, post = split_spec(spec)
    if dep.startswith("ours_up:"):               # DN-01 F1-UP: released upstream VDA-S + the checkpoint's motion modules, through infer_video_depth
        p = _vda_upstream(device, "vits", dep[len("ours_up:"):])
    elif dep.startswith("l15d:"):                # L15 stage 3 (new path only): DECOUPLED packed checkpoint (depth <- the original restorer, restored RGB <- rgb_restorer)
        from evaluation.l15_spec import ours_l15
        p = ours_l15(device, dep)
    elif dep.startswith(FT_VDA):
        p = _vda_ft(device, dep[len(FT_VDA):])
    elif dep.startswith(PERFRAME_FT):
        p = _perframe_ft(device, dep)
    else:
        p = _ours(device, dep[len("ours:"):]) if dep.startswith("ours:") else DEPTH_MODELS[dep](device)
    if chain:
        p["denoise"], p["stages"], w = _chain(chain, device)
        p["weights"] = {**w, **p["weights"]}
    dpost = [t for t in post if DEPTH_POST_RE.fullmatch(t)]
    post = [t for t in post if not DEPTH_POST_RE.fullmatch(t)]
    if dpost:                                    # K49: depth smoothers, applied in order
        fns = [depth_post(t) for t in dpost]

        def smooth(d, guides=None):
            for f in fns:
                d = f(d, guides)
            return d
        p["depth_post"], p["depth_post_tokens"] = smooth, dpost
    if post:
        p["post"], p["post_stages"], w = _chain(post, device)
        p["weights"] = {**p["weights"], **w}
    return p


# ----------------------------------------------------------------------------- inference

INFER_LEN, OVERLAP, INTERP_LEN = 32, 10, 8
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]


@torch.no_grad()
def vda_windows(fwd, clip: torch.Tensor, out_hw: tuple[int, int], metric: bool):
    """VDA's ``infer_video_depth`` (video_depth.py @4f5ae23) around ``fwd``: 32-frame windows
    stepping 22, the first 10 slots of each later window replaced by the previous window's
    keyframes, depth upsampled to ``out_hw`` (bilinear, align_corners) and stitched with
    per-window scale/shift (identity for metric models). Restored frames: each frame from the
    first window that sees it as a real input (window 0 all, later windows slots 10..31)."""
    _vda_path()
    from utils.util import compute_scale_and_shift, get_interpolate_frames
    n, step = clip.shape[0], INFER_LEN - OVERLAP
    pad = (step - n % step) % step + OVERLAP
    frames = torch.cat([clip, clip[-1:].expand(pad, -1, -1, -1)])
    depth_list, restored, pre = [], [], None
    for fid in range(0, n, step):
        cur = frames[fid:fid + INFER_LEN].clone()
        if pre is not None:
            cur[:OVERLAP] = pre[KEYFRAMES]
        d, r = fwd(cur)
        d = F.interpolate(d.float()[:, None], size=out_hw, mode="bilinear", align_corners=True)[:, 0]
        depth_list += list(d.cpu().numpy())
        if r is not None:
            restored.append(r if pre is None else r[OVERLAP:])
        pre = cur

    aligned, ref_align = [], []
    align_len = OVERLAP - INTERP_LEN
    kf_align = KEYFRAMES[:align_len]
    for fid in range(0, len(depth_list), INFER_LEN):          # verbatim from infer_video_depth
        if not aligned:
            aligned += depth_list[:INFER_LEN]
            ref_align += [depth_list[fid + k] for k in kf_align]
            continue
        curr_align = [depth_list[fid + i] for i in range(len(kf_align))]
        if metric:
            scale, shift = 1.0, 0.0
        else:
            scale, shift = compute_scale_and_shift(np.concatenate(curr_align), np.concatenate(ref_align),
                                                   np.concatenate(np.ones_like(ref_align) == 1))
        pre_d = aligned[-INTERP_LEN:]
        post_d = depth_list[fid + align_len:fid + OVERLAP]
        for i in range(len(post_d)):
            post_d[i] = post_d[i] * scale + shift
            post_d[i][post_d[i] < 0] = 0
        aligned[-INTERP_LEN:] = get_interpolate_frames(pre_d, post_d)
        for i in range(OVERLAP, INFER_LEN):
            nd = depth_list[fid + i] * scale + shift
            nd[nd < 0] = 0
            aligned.append(nd)
        ref_align = ref_align[:1]
        for k in kf_align[1:]:
            nd = depth_list[fid + k] * scale + shift
            nd[nd < 0] = 0
            ref_align.append(nd)
    return np.stack(aligned[:n]), (torch.cat(restored)[:n] if restored else None)


@torch.no_grad()
def run_pipeline(p: dict, x: torch.Tensor, sigma: float, blur: float, out_hw: tuple[int, int]):
    """Degraded clip ``(N,3,h,w)`` on device -> (depth-like ``(N,*out_hw)`` numpy, restored ``(N,3,h,w)``)."""
    if "denoise" in p:
        x = p["denoise"](x, sigma, blur)
    y = None
    if isinstance(x, tuple):                     # DN-01 FR2: (restored, depth input)
        y, x = x
    if p["window"] == 1:
        ds = [p["fwd"](x[s:s + 8])[0] for s in range(0, x.shape[0], 8)]
        depth = F.interpolate(torch.cat(ds).float()[:, None], size=out_hw, mode="bilinear",
                              align_corners=True)[:, 0].cpu().numpy()
        restored = x
    elif "upstream" in p:                        # DN-01 F3 pre-ruling (c): VDA's own infer_video_depth at its defaults (no restorer output)
        depth, restored = p["upstream"](x, out_hw), x
    else:
        depth, restored = vda_windows(p["fwd"], x, out_hw, p["metric"])
        if "st_restore" in p:                    # shared-temporal: clip-level triplets + per-frame VDA features
            restored = p["st_restore"](x, restored)
        restored = restored if restored is not None else x
    restored = restored if y is None else y
    p["last_guides"] = {"i": x, "r": restored}   # probe package: the guides of the LAST call (i = depth-model input, r = restored; r IS i without a restorer)
    if "depth_post" in p:                        # K49 post-hoc temporal depth smoothing (depth only)
        depth = p["depth_post"](depth, p["last_guides"])
    return depth, (p["post"](restored, sigma, blur) if "post" in p else restored)


# ----------------------------------------------------------------------------- metrics

def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-frame PSNR (dB), ``(N,3,H,W)`` in [0,1]."""
    return 10.0 * torch.log10(1.0 / (a - b).pow(2).flatten(1).mean(1).clamp_min(1e-12))


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-frame SSIM on luma (Wang et al. 2004: 11x11 Gaussian, sigma 1.5, valid window)."""
    w = torch.tensor([0.299, 0.587, 0.114], device=a.device).view(1, 3, 1, 1)
    a, b = (a * w).sum(1, keepdim=True), (b * w).sum(1, keepdim=True)
    g = torch.exp(-(torch.arange(11, device=a.device) - 5.0) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :]).view(1, 1, 11, 11)
    mu_a, mu_b = F.conv2d(a, k), F.conv2d(b, k)
    s_aa = F.conv2d(a * a, k) - mu_a ** 2
    s_bb = F.conv2d(b * b, k) - mu_b ** 2
    s_ab = F.conv2d(a * b, k) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return m.flatten(1).mean(1)


LAG_KMAX, LAG_T0 = 4, 4


def lag_metrics(disp: torch.Tensor, gt: torch.Tensor, max_depth: float, kmax: int = LAG_KMAX,
                t0: int = LAG_T0) -> dict:
    """DN-01 K49 LAG of a depth video (WS2 smoothing Pareto; definition by R+D 2, 2026-09-28). ``disp`` (N,H,W) the
    predicted disparity as score_clip forms it (1 / metric depth, or a relative model's disparity), ``gt`` (N,H,W)
    metres. For k in 0..kmax, over frames t = t0..N-1: D_t is aligned to GT_{t-k} by score_clip's per-clip
    least-squares (scale, shift) on disparity (VDA eval's fit, on GT_{t-k}'s valid mask), and AbsRel_k = _depth_errors'
    abs_rel (per-frame mean, then mean over frames) of the aligned depth vs GT_{t-k}. ``lag_k`` = argmin_k AbsRel_k
    (ties -> the smaller k; an unlagged model gives ~0, an EMA > 0); ``lag_absrel0`` = AbsRel_0 on the same frames.
    NaN when the clip has <= t0 frames."""
    n = disp.shape[0]
    if n <= t0:
        return {"lag_k": float("nan"), "lag_absrel0": float("nan")}
    rel = []
    for k in range(kmax + 1):
        p, g = disp[t0:], gt[t0 - k:n - k]
        v = (g > 1e-3) & (g < max_depth)
        gd = torch.where(v, 1.0 / (g + 1e-8), 0.0)
        pv, tv = p[v], gd[v]
        m = pv.new_tensor(float(pv.numel()))
        A = torch.stack([torch.stack([(pv * pv).sum(), pv.sum()]), torch.stack([pv.sum(), m])])
        s, b = torch.linalg.solve(A, torch.stack([(pv * tv).sum(), tv.sum()]))
        rel.append(_depth_errors((1.0 / (s * p + b).clamp_min(1e-3)).clamp(1e-3, max_depth), g, v)["abs_rel"])
    return {"lag_k": float(min(range(kmax + 1), key=lambda k: (rel[k], k))), "lag_absrel0": float(rel[0])}


def _depth_errors(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> dict:
    """VDA metric.py: per-frame mean over valid pixels, then mean over frames with any valid."""
    keep = valid.flatten(1).any(1)
    pred, gt, valid = pred[keep], gt[keep], valid[keep]
    n = valid.flatten(1).sum(1)
    rel = torch.where(valid, (pred - gt).abs() / gt, 0).flatten(1).sum(1) / n
    d1 = torch.where(valid & (torch.maximum(pred / gt, gt / pred) < 1.25), 1.0, 0.0).flatten(1).sum(1) / n
    rmse = (torch.where(valid, (pred - gt) ** 2, 0).flatten(1).sum(1) / n).sqrt()
    return {"abs_rel": float(rel.mean()), "delta1": float(d1.mean()), "rmse": float(rmse.mean())}


def camera(ds: str, key: str, frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(K (3,3) at the GT crop, camera-to-world (N,4,4)) for TAE (evaluation/temporal_metrics.py). KITTI: rectified
    camera 2 from the raw drive's OXTS + that date's calibration (the GT crop starts at (0,0), so K is unchanged);
    Bonn: the dataset page's sensor-pose convention and intrinsics."""
    from evaluation import temporal_metrics as tm
    if ds == "bonn":
        return tm.BONN_K, tm.bonn_poses(Path(PATHS["bonn_root"]) / key, [f["image"] for f in frames])
    if ds == "kitti":
        import zipfile
        date = key[:10]
        K, T_cam2_imu = tm.kitti_calib(Path(PATHS["kitti_calib"]) / date)
        with zipfile.ZipFile(Path(PATHS["kitti_raw_zips"]) / f"{key}.zip") as z:
            rows = [np.array([float(v) for v in z.read(f"{date}/{key}/oxts/data/{Path(f['image']).stem}.txt")
                              .decode().split()]) for f in frames]
        return K, tm.kitti_poses(rows, T_cam2_imu)
    from evaluation import vda5_sets
    return vda5_sets.camera(ds, key, frames)                    # v3 WP8: sintel / scannet* / scannetpp (raises for any other dataset)


def shadow_lift(y: torch.Tensor, g: torch.Tensor, r: int = 32, thr: float = 0.2) -> float:
    """Mean signed luma error (y - g, in /255) over the pixels whose (2r+1)^2 local-mean GT luma < ``thr`` (shadows):
    the low-light hunt's detector of learned shadow lifting (provenance/dn01_lowlight_defect/scripts/eval_clean.py,
    verbatim; r shrinks only on frames smaller than 2r+1, i.e. in tests). NaN when the clip has no shadow pixel.
    Images (N,3,h,w)."""
    r = min(r, y.shape[-2] - 1, y.shape[-1] - 1)
    w = torch.tensor(LUMA, device=y.device, dtype=y.dtype).view(1, 3, 1, 1)
    e, gl = ((y - g) * w).sum(1, keepdim=True), (g * w).sum(1, keepdim=True)
    lm = F.avg_pool2d(F.pad(gl, (r,) * 4, mode="reflect"), 2 * r + 1, stride=1)
    m = lm < thr
    return float(e[m].mean() * 255) if m.any() else float("nan")


def score_clip(depth: np.ndarray, metric: bool, gt_np: np.ndarray, restored: torch.Tensor,
               degraded: torch.Tensor, clean: torch.Tensor, max_depth: float, min_depth: float,
               cam: tuple | None = None, tof=None, ev: float | None = None, sky_m: float | None = None,
               sky_mask: np.ndarray | None = None) -> dict:
    """All metrics for one clip. ``depth`` (N,H0,W0) raw-frame size, metres if ``metric`` else
    disparity; ``gt_np`` (N,H,W) cropped GT; images (N,3,h,w) on one device. ``cam`` (K, c2w) adds VDA's TAE and
    ``tof`` (evaluation.temporal_metrics.TOF) adds tOF of the restored and of the input video (both lower-better).
    ``ev`` (an LL cell's exposure) adds the v2 fields: ``*_native`` vs lowlight_reference(clean, ev) (on other cells
    they equal the plain fields), ``shadow_lift`` vs that reference, and the blind display gain's ``*_display``
    (+ ``*_display_oracle`` and ``gain_err_stops`` on LL cells), both vs the normally exposed clean frame."""
    # ``sky_m`` (opt-in VKITTI: GT >= sky_m is the exact sky label) / ``sky_mask`` (KITTI: the colour rule's (N,H,W) mask) add sky_frac / sky_med / sky_ge80 / sky_nearer_than_scene_p95 (evaluation/sky_metrics.py); ``tae_gtvalid`` below = TAE without sky / no-GT pixels.
    dev = restored.device
    H, W = gt_np.shape[-2:]
    # E4 far-field guard fields (additive; bug-sweep 'sky push' diagnostic): the raw metric depth BEFORE the resize / clamp below, so depth pushed past the 80 m cap is not hidden.
    far = ({"far_frac80": float((depth >= 80.0).mean()), "median_depth": float(np.median(depth[:, ::4, ::4]))} if metric
           else {"far_frac80": float("nan"), "median_depth": float("nan")})
    if depth.shape[-2:] != (H, W):                                # VDA eval's get_infer resize
        depth = np.stack([cv2.resize(d, (W, H)) for d in depth])
    pd = torch.from_numpy(depth).to(dev, torch.float64)
    gt = torch.from_numpy(gt_np).to(dev)
    disp = 1.0 / pd.clamp_min(1e-3) if metric else pd
    disp = disp.clamp_min(1e-3)
    valid = (gt > 1e-3) & (gt < max_depth)                        # VDA eval's mask
    gd = torch.where(valid, 1.0 / (gt + 1e-8), 0.0)
    # VDA eval: one least-squares (scale, shift) on disparity per clip (normal equations, fp64).
    p, t = disp[valid], gd[valid]
    n = p.new_tensor(float(p.numel()))
    A = torch.stack([torch.stack([(p * p).sum(), p.sum()]), torch.stack([p.sum(), n])])
    s, b = torch.linalg.solve(A, torch.stack([(p * t).sum(), t.sum()]))
    aligned_disp = s * disp + b
    out = {"frames": int(pd.shape[0]), "fit_scale": float(s), "fit_shift": float(b), **far}
    out.update({f"{k}_aligned": v for k, v in _depth_errors(
        (1.0 / aligned_disp.clamp_min(1e-3)).clamp(1e-3, max_depth), gt, valid).items()})
    if metric:
        out.update({f"{k}_unaligned": v for k, v in _depth_errors(
            pd.clamp(1e-3, max_depth), gt, (gt > min_depth) & (gt < max_depth)).items()})
    out.update(lag_metrics(disp, gt, max_depth))
    if sky_m is not None or sky_mask is not None:       # sky-fix guard fields (evaluation/sky_metrics.py): exact VKITTI label (gt >= sky_m) or the KITTI colour rule's mask
        from evaluation.sky_metrics import sky_stats
        out.update(sky_stats(depth, (gt_np >= sky_m) if sky_m is not None else sky_mask, gt_np))

    out["psnr"] = float(psnr(restored, clean).mean())
    out["ssim"] = float(ssim(restored, clean).mean())
    out["psnr_input"] = float(psnr(degraded, clean).mean())
    out["ssim_input"] = float(ssim(degraded, clean).mean())

    # Temporal (train.py depth_flicker 'disparity_affine' / restore_flicker, on the eval fit):
    # E = (aligned - gt disparity) / clip median gt disparity; eps = |E_{t+1} - E_t|.
    err = (aligned_disp - gd) / gd[valid].median()
    eps_d = (err[1:] - err[:-1]).abs()
    lo = torch.where(valid, gd, torch.inf).amin((-2, -1))
    hi = torch.where(valid, gd, -torch.inf).amax((-2, -1))
    omega = valid[1:] & valid[:-1] & ((gd[1:] - gd[:-1]).abs() < ((hi - lo) * 0.05)[1:, None, None])
    eps_r = ((restored - clean)[1:] - (restored - clean)[:-1]).abs().mean(1)          # (N-1,h,w)
    eps_in = ((degraded - clean)[1:] - (degraded - clean)[:-1]).abs().mean(1)
    joint = torch.maximum(eps_d, F.interpolate(eps_r[:, None].double(), size=(H, W), mode="bilinear",
                                               align_corners=False)[:, 0])[omega]
    out.update({"eps_r": float(eps_r.mean()), "eps_r_input": float(eps_in.mean()),
                "eps_d": float(eps_d[omega].mean()) if joint.numel() else float("nan"),
                "joint_p95": float(joint.kthvalue(max(1, int(np.ceil(0.95 * joint.numel()))))[0])
                if joint.numel() else float("nan"),
                "omega_frac": float(omega.double().mean())})
    if cam is not None:
        from evaluation.temporal_metrics import tae_clip
        out["tae"] = tae_clip(depth, metric, gt_np, cam[0], cam[1], max_depth)
        out["tae_gtvalid"] = tae_clip(depth, metric, gt_np, cam[0], cam[1], max_depth, gt_valid_only=True)      # sky-fix guard (descriptive): TAE with no sky / no-GT pixels
    if tof is not None:
        out["tof"], out["tof_input"] = tof(restored, clean), tof(degraded, clean)

    # v2 (K43): native-exposure restoration (primary), shadow lift, blind display gain (secondary).
    ref = clean
    if ev is not None:
        from data.elvis_noise import lowlight_reference
        ref = torch.from_numpy(lowlight_reference(clean.cpu().numpy(), ev)).to(dev)
    for k, (a, b) in {"psnr_native": (restored, ref), "psnr_input_native": (degraded, ref)}.items():
        out[k] = float(psnr(a, b).mean())
    for k, (a, b) in {"ssim_native": (restored, ref), "ssim_input_native": (degraded, ref)}.items():
        out[k] = float(ssim(a, b).mean())
    if tof is not None:
        out["tof_native"], out["tof_input_native"] = ((tof(restored, ref), tof(degraded, ref)) if ev is not None
                                                      else (out["tof"], out["tof_input"]))
    out["shadow_lift"] = shadow_lift(restored, ref)
    g = blind_gain(degraded)
    shown = (restored * g).clamp(0.0, 1.0)
    out.update(gain=g, psnr_display=float(psnr(shown, clean).mean()), ssim_display=float(ssim(shown, clean).mean()))
    if ev is not None:
        orc = (restored * 2.0 ** -ev).clamp(0.0, 1.0)
        out.update(gain_err_stops=abs(float(np.log2(g)) + ev), psnr_display_oracle=float(psnr(orc, clean).mean()),
                   ssim_display_oracle=float(ssim(orc, clean).mean()))
    return out


# ----------------------------------------------------------------------------- commands

_HASHES: dict[str, str] = {}


def _sha(path: str) -> str:
    if path not in _HASHES:
        _HASHES[path] = sha256_file(path)
    return _HASHES[path]


def ckpt_info(spec: str) -> dict:
    """The checkpoint file(s) a spec points to (``ours:<ckpt>``, ``pocketdvdnet:<ckpt>``, ``*_ft:<ckpt>``, optionally followed by ``+post`` tokens), WITHOUT loading any model:
    ``ckpt_sha256`` / ``ckpt_basename`` of the first one (None for released / hub models), ``ckpts`` = {basename: sha256} of all (Auditor Table addendum e: joins the TEST log to the results across hosts)."""
    files = [t.split(":", 1)[1] for t in spec.split("+") if ":" in t and Path(t.split(":", 1)[1]).is_file()]
    shas = {Path(f).name: _sha(f) for f in files}
    return {"ckpt_sha256": _sha(files[0]) if files else None, "ckpt_basename": Path(files[0]).name if files else None, "ckpts": shas}


def _slug(spec: str) -> str:
    return spec.replace("ours_up:", "oursup-").replace("ours:", "ours-").replace("l15d:", "l15d-").replace("/", "_").replace(".pt", "")


CORE_FINITE = ("abs_rel_aligned", "delta1_aligned", "fit_scale", "tae")          # must be finite on every clip of every model; + the metric-only ones below
CORE_FINITE_METRIC = ("abs_rel_unaligned", "flicker")                              # NaN is legitimate for a relative model


def nan_dropped(pv: dict, metric: bool) -> dict:
    """{metric: n clips dropped by nanmean} for the core metrics that should never be non-finite (bug-sweep 2026-09-30, eval-harness: video_macro uses nanmean, so a model that emits
    NaN / degenerate depth on a hard clip is scored on the easy clips only while num_clips still counts them). Empty when every clip is finite (the current state of every pulled result)."""
    out = {}
    for k in CORE_FINITE + (CORE_FINITE_METRIC if metric else ()):
        vals = [v.get(k) for v in pv.values() if k in v]
        n = sum(1 for z in vals if z is None or not np.isfinite(z))
        if vals and n:
            out[k] = n
    return out


def _grid_variants(a) -> list:
    """[None] (the unchanged single-model path) or the expanded ``--grid`` variants (evaluation.depth_probe.GROUPS names or literal variants)."""
    if not getattr(a, "grid", None):
        return [None]
    from evaluation.depth_probe import expand_grid
    if any(DEPTH_POST_RE.fullmatch(t) for t in split_spec(a.model)[2]):
        raise SystemExit("--grid needs a model spec WITHOUT depth post tokens (the variants are the post tokens)")
    return expand_grid(a.grid)


def check_split(split: str, allow_test: bool, model: str | None = None) -> None:
    """Bug-sweep 2026-09-30 (eval-harness): bench_joint had no TEST guard (heldout's check_split only covered dn01_heldout_eval). `test` / `all` include the TEST clips (odd clips of KITTI / Bonn):
    refused without --allow-test, before any model or frame is loaded. `dev` (the default) is unaffected."""
    if split != "dev" and not allow_test:
        raise SystemExit(f"bench_joint: --split {split} reads TEST clips; refused without --allow-test (TEST is reserved for a locked model under a pre-declared rule)")
    # v3 (K96, user 2026-10-03/04: "TEST now", select on TEST): the 2026-10-01 "released only" rule that stood here (a whitelist refusing every ours: / ours_up: / checkpoint spec on a
    # non-dev split) is DROPPED. --allow-test is still required and every non-dev run is appended to TEST_access_log.jsonl (cmd_run -> log_test_access). ``model`` is kept for the call shape.


def cmd_run(a) -> None:
    check_split(a.split, getattr(a, "allow_test", False), a.model)
    if a.split != "dev":
        log_test_access(a)
    p = build_pipeline(a.model, a.device)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds_name in a.dataset:
        ds, drop = DATASET_ALIASES.get(ds_name, (ds_name, ()))         # ds = the base dataset (loaders, seeds); ds_name = the name written to the result file
        proto = DATASETS[ds]
        clips = select_clips(ds_name, a.split, a.limit_clips)          # check_split (above) already vetted a non-dev split
        # DN-01 probe package: --grid scores many depth post-filter variants from ONE model pass; each variant writes the file (and the 'model' /
        # 'depth_post' fields) a single-token run of it would. Without --grid the code path and every output are unchanged.
        vs = _grid_variants(a)
        spec = lambda v: a.model if v in (None, "none") else f"{a.model}+{v}"        # noqa: E731
        for c in a.cells:
            for v in vs:
                check_resumable(art_path(out_dir, spec(v), ds_name, c, a.split), a.limit_clips)
        todo = [c for c in a.cells if any(not art_path(out_dir, spec(v), ds_name, c, a.split).exists() for v in vs)]
        if not todo:
            continue
        per_cell = {c: {v: {} for v in vs} for c in todo}
        diag = {c: {v: {} for v in vs} for c in todo}
        tofnet = None
        if getattr(a, "temporal", False):
            from evaluation.temporal_metrics import TOF
            tofnet = TOF(a.device)
        if tofnet is not None and vs != [None]:
            from evaluation.depth_probe import MemoTOF
            tofnet = memo = MemoTOF(tofnet)                        # the flow does not depend on the depth post-filter
        else:
            memo = None
        for key, frames in clips:                                 # load each clip once, all cells
            clean, raw_hw = load_frames(ds, frames)
            gt = load_gt(ds, frames)
            cam = camera(ds, key, frames[:DATASETS[ds]["max_len"]]) if tofnet is not None and ds not in OPTIN_DATASETS else None
            skym = None
            if proto.get("sky_mask") == "kitti_colour":              # sky-fix guard: the colour rule on the RAW clean frames, once per clip (non-static cells only)
                from evaluation.sky_metrics import kitti_sky_masks
                skym = kitti_sky_masks(PATHS[f"{ds}_root"], frames[:proto["max_len"]], proto["crop"], gt)
            for c in todo:
                blur, sigma = parse_cell(c)
                if c in STATIC_CELLS or c in STATIC_LADDER_CELLS:           # K49: the clip's middle frame x STATIC_LEN, fresh noise per frame
                    sclean, sgt = static_clip(clean, gt)
                    x = degrade_static(sclean, c, noise_seed(ds, key, seed_cell(c))).to(a.device)
                    depth, restored = run_pipeline(p, x, sigma, blur, raw_hw)
                    if memo is not None:
                        memo.new_clip()
                    for v in vs:
                        dv = depth if v is None else apply_variant(v, depth, p["last_guides"])
                        r = score_clip(dv, p["metric"], sgt, restored, x, sclean.to(a.device), proto["max_depth"],
                                       proto["min_depth"], None, tofnet, ll_cfg(blur)["ev"], sky_m=proto.get("sky_depth_m"))
                        r.update(flicker_static(dv, sgt, proto["max_depth"], proto["min_depth"]) if p["metric"]
                                 else {"flicker": float("nan"), "abs_rel_static": float("nan")})   # metric models only
                        r.update(variant_extras(v))     # F3: the DVP record ({} for every other variant)
                        per_cell[c][v][key] = r
                        if getattr(a, "diag", False):
                            from evaluation.depth_probe import flicker_diag
                            diag[c][v][key] = flicker_diag(dv)
                        print(f"{spec(v)} {ds} {c} {key}: flicker {r['flicker']:.4f} abs_rel_static "
                              f"{r['abs_rel_static']:.4f}", flush=True)
                    continue
                draw = llvar_draw(ds, key) if c in LLVAR_VIDEO_CELLS else None       # v3: the clip's own severity, the same for every model / order / split
                if draw is not None:
                    x = degrade_llvar(clean, c, draw, llvar_noise_seed(ds, key, c)).to(a.device)
                else:
                    seed = noise_seed(ds, key, seed_cell(blur) if isinstance(blur, LowLight) else sigma)   # LL: per cell name
                    x = degrade(clean, blur, sigma, seed).to(a.device)
                depth, restored = run_pipeline(p, x, sigma, blur, raw_hw)
                if memo is not None:
                    memo.new_clip()
                for v in vs:
                    dv = depth if v is None else apply_variant(v, depth, p["last_guides"])
                    sd, sg, sr, sx, sc, scam = _vda5.score_slice(proto, dv, gt, restored, x, clean.to(a.device), cam)      # v3 WP8: ScanNet TAE scores frames [10:180] of the 192 the model sees (identity elsewhere)
                    per_cell[c][v][key] = score_clip(sd, p["metric"], sg, sr, sx, sc,
                                                     proto["max_depth"], proto["min_depth"], scam, tofnet,
                                                     (draw["ev"] if draw is not None else ll_cfg(blur)["ev"]) if isinstance(blur, LowLight) else None,
                                                     sky_m=proto.get("sky_depth_m"), sky_mask=skym)
                    if draw is not None:
                        per_cell[c][v][key]["llvar_draw"] = draw      # additive per_video fields (v3): the drawn EV / kappa / blur of this clip
                    per_cell[c][v][key].update(variant_extras(v))     # F3: the DVP record ({} for every other variant)
                    print(f"{spec(v)} {ds} {c} {key}: {per_cell[c][v][key]['abs_rel_aligned']:.4f} "
                          f"{per_cell[c][v][key]['psnr']:.2f}dB", flush=True)
        for c, v in [(c, v) for c in todo for v in vs]:
            path = art_path(out_dir, spec(v), ds_name, c, a.split)
            if vs != [None] and path.exists():
                continue
            pv = per_cell[c][v]
            keys = [k for k in next(iter(pv.values())) if k not in ("frames", "llvar_draw")]
            art = {"bench": "bench_joint", "bench_version": BENCH_VERSION, "model": spec(v), "dataset": ds_name, **({"source_dataset": ds, "dropped_clips": list(drop)} if drop else {}),
                   "cell": c, "noise_sigma_255": parse_cell(c)[1],
                   **({"noise_kappa": ll_cfg(parse_cell(c)[0])["kappa"], "noise_kappa_mult_vs_llb3": ll_cfg(parse_cell(c)[0])["kappa"] / LL_KAPPA}
                      if c in LADDER_CELLS or c in STATIC_LADDER_CELLS else {}),
                   **({"blur_disk_radius_px": parse_cell(c)[0]} if c[0] == "d" else
                      {"blur_sigma_px": parse_cell(c)[0]}),
                   **({"llvar_draws_sha256": llvar_pin(ds), "llvar_spec": llvar_draw_table(ds)["cell_spec"]} if c in LLVAR_VIDEO_CELLS else {}),
                   "split": a.split, "limit_clips": a.limit_clips, "short_side": proto["short_side"], **ckpt_info(a.model),
                   "metric_model": p["metric"], "num_clips": len(pv),
                   "restorers": split_spec(a.model)[0],    # fastdvdnet: oracle sigma; _blind: estimated
                   "post_restorers": [t for t in split_spec(a.model)[2] if not DEPTH_POST_RE.fullmatch(t)],
                   "depth_post": (p.get("depth_post_tokens", []) if v is None else
                                  ([] if v == "none" else v.split("+"))),   # K49 depth smoothers (depth only)
                   "agf_backend": getattr(sys.modules.get("evaluation.depth_probe"), "AGF_BACKEND", "cpu"),   # protocol v3 WP24: which agf implementation filtered the depth (cpu = the numpy reference; gpu = depth_agf_torch, bit-identical); 'cpu' also when no agf token ran
                                                            # per clip; restormer: oracle blur presence
                   "weights": {k: {"path": v, "sha256": _sha(v)} for k, v in p["weights"].items()},
                   "manifest_sha256": _sha(PATHS[f"{ds}_manifest"]), "ramen_commit": git_state(REPO),
                   **({"kitti_channel_swap_testonly": True} if ds == "kitti" and kitti_swap_testonly() else {}),   # additive: only a swap-repro / calibration row carries it
                   "vda_repo_commit": git_state(PATHS["vda_repo"]), "host": socket.gethostname(),
                   "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                   "torch": torch.__version__,
                   **({"temporal_metrics": {"tae": "vendor/vda_eval/eval_tae.py (VDA @4f5ae23, sha256 8e481e94)",
                                            "tof": "torchvision raft_large C_T_SKHT_V2 sha256 ff5fadd5, 12 updates"}}
                      if tofnet is not None else {}),
                   "video_macro": {k: float(np.nanmean([v[k] for v in pv.values()])) for k in keys},
                   "shadow_lift_clips": int(sum(np.isfinite(v["shadow_lift"]) for v in pv.values())),  # nanmean's n
                   "per_video": pv}
            if getattr(a, "diag", False) and diag[c][v]:
                art["diag"] = diag[c][v]                      # probe item 1 (static cells only); absent without --diag
            dropped = nan_dropped(pv, p["metric"])
            if dropped:                                       # bug-sweep (eval-harness): nanmean silently scores a model on its easy clips only; record + shout, never hide
                art["nan_clips_dropped_from_macro"] = dropped
                print(f"WARNING {path.name}: clips with a non-finite core metric were dropped from video_macro: {dropped}", flush=True)
            path.write_text(json.dumps(art, indent=2))
            vm = art["video_macro"]
            print(f"wrote {path}: AbsRel {vm['abs_rel_aligned']:.4f} PSNR {vm['psnr']:.2f} "
                  f"joint_p95 {vm['joint_p95']:.4f}", flush=True)


def depth_post_stages(p: dict, size: int, device: str) -> list:
    """The timed stage of a spec's depth post-filter tokens (``+agf0.2t0.06``, ema, sgf, dvp): [] without tokens, else one stage that copies a
    ``(INFER_LEN, size, size)`` depth window to the host and runs ``p['depth_post']`` on it, exactly as ``run_pipeline`` does. Before this
    ``cmd_latency`` never called it, so ``latency --model <depth>+agf...`` was filed under the agf spec with the bare model's time."""
    if "depth_post" not in p:
        return []
    probe = torch.rand(INFER_LEN, size, size, device=device)
    from evaluation import depth_probe
    on_device = device == "cuda" and depth_probe.AGF_BACKEND == "gpu" and all(t.startswith("agf") for t in p["depth_post_tokens"])   # protocol v3 WP24: the GPU agf stage runs on the device-resident window (no host copy)

    def stage(y, s, b):
        if device == "cuda":
            torch.cuda.synchronize()             # CPU work must not overlap the previous stage's queued GPU work
        guides = {"i": y, "r": y} if isinstance(y, torch.Tensor) else None
        if on_device:
            p["depth_post"](probe.float(), guides)
            return y
        p["depth_post"](probe.float().cpu().numpy(), guides)
        return y
    return [("+".join(p["depth_post_tokens"]), stage)]


@torch.no_grad()
def cmd_latency(a) -> None:
    """Per-frame ms of every stage of the pipeline and their total, on one 32-frame window at
    ``--size``^2 (518: scripts/dn01_latency.py's method -- fp16 autocast, CUDA events, median after
    warmup). Every restorer runs (the b3s25 cell: sigma 25, blur 3). Needs an otherwise idle GPU;
    latency figures come only from the 4090."""
    gpu_procs = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                         "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    busy = [ln for ln in gpu_procs if int(ln.rsplit(",", 1)[1]) >= 1024]
    if busy:                                     # desktop compositor / editor use < 1 GB
        raise SystemExit(f"GPU not idle, refusing to time: {busy}")
    p = build_pipeline(a.model, "cuda")
    depth = p["fwd"] if p["window"] == 32 else (lambda y: [p["fwd"](y[s:s + 8]) for s in range(0, INFER_LEN, 8)])
    def depth_stage(y, s, b):                    # hands a joint model's restored frames to any post-restorer
        if isinstance(y, tuple):                 # DN-01 FR2: (restored, depth input)
            y, x = y
            depth(x)
            return y
        out = depth(y)
        if "st_restore" in p:                    # shared-temporal: the restoration is part of the model's time
            return p["st_restore"](y, out[1])
        return out[1] if isinstance(out, tuple) and out[1] is not None else y
    stages = p.get("stages", []) + [(split_spec(a.model)[1], depth_stage)] + p.get("post_stages", [])
    stages += depth_post_stages(p, a.size, "cuda")        # agf/ema/sgf/dvp tokens: timed as their own stage
    x = torch.rand(INFER_LEN, 3, a.size, a.size, device="cuda")
    ms = {name: [] for name, _ in stages} | {"total": []}
    for i in range(a.warmup + a.iters):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(stages) + 1)]
        y = x
        ev[0].record()
        with torch.autocast("cuda", dtype=torch.float16):
            for k, (_, fn) in enumerate(stages):
                y = fn(y, 25.0, 3.0)
                ev[k + 1].record()
        torch.cuda.synchronize()
        if i >= a.warmup:
            for k, (name, _) in enumerate(stages):
                ms[name].append(ev[k].elapsed_time(ev[k + 1]) / INFER_LEN)
            ms["total"].append(ev[0].elapsed_time(ev[-1]) / INFER_LEN)
    art = {"bench": "bench_joint", "bench_version": BENCH_VERSION, "model": a.model, "size": [a.size, a.size],
           "frames": INFER_LEN, "precision": "fp16 autocast", "gpu": torch.cuda.get_device_name(),
           "host": socket.gethostname(), "ramen_commit": git_state(REPO), "iters": a.iters, "warmup": a.warmup,
           "gpu_idle_check": {"other_compute_procs_before": [ln.split(",")[1].strip()[:40] + ":" +
                                                             ln.rsplit(",", 1)[1].strip() + "MiB" for ln in gpu_procs],
                              "rule": "no other compute process >= 1 GB"},
           "depth_post_timed": p.get("depth_post_tokens", []), "agf_backend": getattr(sys.modules.get("evaluation.depth_probe"), "AGF_BACKEND", "cpu"),
           "depth_post_impl": "numpy on CPU (the scoring implementation); the stage time includes the device-to-host copy of the depth window"
                              if "depth_post" in p else None,
           "depth_post_probe": {"shape": [INFER_LEN, a.size, a.size], "content": "torch.rand uniform noise (not model depth)",
                                "resolution_note": f"the post-filter is timed at the latency size {a.size}^2, not at the scoring clip's native depth resolution"}
                               if "depth_post" in p else None,
           "ms_per_frame_definition": f"window time / {INFER_LEN} frames; the stitched pipeline advances {INFER_LEN - OVERLAP} new frames "
                                      f"per window, so per new streamed frame multiply by {INFER_LEN}/{INFER_LEN - OVERLAP}",
           "stage_ms_p50": {k: float(np.median(v)) for k, v in ms.items() if k != "total"},
           "per_frame_ms_p50": float(np.median(ms["total"])), "per_frame_ms_p95": float(np.percentile(ms["total"], 95))}
    out = Path(a.out) / f"latency__{_slug(a.model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(art, indent=2))
    print(json.dumps(art, indent=2))


def art_path(out_dir: Path, model_spec: str, ds: str, cell: str, split: str) -> Path:
    """A bench_joint result file. v3 (BENCH_VERSION >= 3) adds a ``__v3`` suffix, so a v1 ``...__<split>.json`` in the same directory is never mistaken for a finished v3 cell
    (resume-by-existence) and never overwritten; ``_results`` / ``table`` read both."""
    return Path(out_dir) / f"{_slug(model_spec)}__{ds}__{cell}__{split}{'__v3' if BENCH_VERSION >= 3 else ''}.json"


def log_test_access(a, path: str | Path | None = None) -> None:
    """v3 (Auditor rec c): one APPEND-ONLY line per non-dev run, written BEFORE any model or frame is loaded: the evidence for the "selected on TEST" disclosure.
    Default file ``~/.ramen/TEST_access_log.jsonl`` (outside the repo), or ``$RAMEN_TEST_ACCESS_LOG``."""
    import getpass
    p = Path(path or os.environ.get("RAMEN_TEST_ACCESS_LOG") or Path.home() / ".ramen" / "TEST_access_log.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "host": socket.gethostname(), "user": getpass.getuser(),
           "bench_version": BENCH_VERSION, "ramen_commit": git_state(REPO), "model": a.model, "datasets": list(getattr(a, "dataset", None) or []), "cells": list(getattr(a, "cells", None) or []), "split": getattr(a, "split", None),
           "allow_test": bool(getattr(a, "allow_test", False)), "limit_clips": getattr(a, "limit_clips", None), "out": str(a.out), **ckpt_info(a.model)}
    with open(p, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def check_resumable(path: str | Path, limit_clips: int | None) -> None:
    """Auditor WP18 B5: resume-by-existence must not take a file written under another ``limit_clips`` (a --limit-clips smoke) or ``bench_version`` for this run's result. An existing file that
    does not match is REFUSED (move it away or use another --out), never silently skipped or overwritten."""
    p = Path(path)
    if not p.exists():
        return
    art = json.loads(p.read_text())
    if art.get("limit_clips") != limit_clips or art.get("bench_version") != BENCH_VERSION:
        raise SystemExit(f"bench_joint: REFUSED to resume {p}: it has limit_clips={art.get('limit_clips')!r} bench_version={art.get('bench_version')!r}, "
                         f"this run has limit_clips={limit_clips!r} bench_version={BENCH_VERSION!r}")


def check_aggregable(art: dict, where: str | Path = "?") -> None:
    """Auditor WP18 B5 / part-1 condition 1b: a smoke result (``limit_clips`` set) or a llvar / llvarf cell without its ``llvar_draws_sha256`` pin never enters an aggregate."""
    if art.get("limit_clips") is not None:
        raise ValueError(f"REFUSED {where}: limit_clips={art['limit_clips']!r} (a smoke result) cannot be aggregated")
    if art.get("cell") in LLVAR_VIDEO_CELLS and art.get("llvar_draws_sha256") is None:
        raise ValueError(f"REFUSED {where}: {art.get('cell')} result has a null llvar_draws_sha256 (no committed draw file for {art.get('dataset')})")


def _results(out: str) -> list[dict]:
    rows = []
    for p in sorted(Path(out).glob("*__*__*__*.json")):
        if p.name.startswith("latency__"):
            continue
        art = json.loads(p.read_text())
        check_aggregable(art, p)
        rows.append(art)
    return rows


def _head(spec: str) -> str:
    """Head variant of an ``ours:`` spec, from its run dir name (DN-01 names it: *_h8_*, *_h16_*,
    *_half_*; untagged runs such as A1/C/E use the default width 32)."""
    run = Path(spec.split("ours:")[1]).parent.name
    return next((t for t in ("h8", "h16", "half") if f"_{t}_" in f"{run}_"), "h32")


def _latencies(out: str) -> dict[str, float]:
    return {r["model"]: r["per_frame_ms_p50"]
            for r in (json.loads(p.read_text()) for p in Path(out).glob("latency__*.json"))}


def latency_of(lat: dict[str, float], m: str) -> float | None:
    """ms/frame of ``m``. Latency depends on architecture, not weights: an ``ours:`` spec with no timing
    of its own takes that of a timed ``ours:`` run with the same head variant."""
    if m in lat:
        return lat[m]
    if m.endswith("metric_vda_s"):   # same VDA-S backbone and head architecture as vda_s (timed 6.00 vs 6.04 ms):
        return lat.get(m.removesuffix("metric_vda_s") + "vda_s")   # reuse the relative chain's timing
    if "ours:" not in m:
        return None
    pre, _, post = split_spec(m)
    if pre or post:                  # extra stages change the cost: only an exact timing counts
        return None
    return next((v for k, v in lat.items() if "ours:" in k and _head(k) == _head(m)), None)


def short_name(spec: str) -> str:
    """``ours:<clone>/runs/dn01_x/step20000.pt`` -> ``dn01_x@20000``; baselines keep their spec."""
    if "ours:" not in spec and "ours_up:" not in spec:
        return spec
    pre, dep, post = split_spec(spec)
    up = dep.startswith("ours_up:")                                  # F1-UP: the same checkpoint through VDA's upstream infer_video_depth
    ck = Path(dep[len("ours_up:" if up else "ours:"):])
    return "+".join(pre + [f"{ck.parent.name}@{ck.stem.removeprefix('step')}" + ("_up" if up else "")] + post)


def cmd_plot(a) -> None:
    """Latency vs quality at one cell: x = total ms/frame (``latency``), y = aligned AbsRel and PSNR,
    one row per dataset, one point per pipeline that has both a result and a latency. Blue = ours
    (models with ``ours:``), orange = cascades and single models; every point direct-labelled.
    Writes ``<out>/latency_vs_quality_<cell>_<split>.png`` and the plotted points as ``.csv``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lat = _latencies(a.out)
    pts = [(r["dataset"], r["model"], latency_of(lat, r["model"]), r["video_macro"]) for r in _results(a.out)
           if r["cell"] == a.cell and r["split"] == a.split and latency_of(lat, r["model"]) is not None]
    dss = sorted({p[0] for p in pts})
    if not dss:
        raise SystemExit(f"no model has both a {a.cell}/{a.split} result and a latency JSON in {a.out}")
    stem = Path(a.out) / f"latency_vs_quality_{a.cell}_{a.split}"
    with open(f"{stem}.csv", "w") as f:
        f.write("dataset,model,ms_per_frame,abs_rel_aligned,psnr\n")
        f.writelines(f"{d},{m},{ms:.3f},{v['abs_rel_aligned']:.5f},{v['psnr']:.3f}\n" for d, m, ms, v in pts)
    ink, muted, grid = "#0b0b0b", "#52514e", "#e5e4e0"
    fig, axes = plt.subplots(len(dss), 2, figsize=(9, 3.4 * len(dss)), squeeze=False, facecolor="#fcfcfb")
    for i, ds in enumerate(dss):
        for j, (key, ylab) in enumerate([("abs_rel_aligned", "depth AbsRel (aligned) ↓"),
                                         ("psnr", "restored PSNR (dB) ↑")]):
            ax = axes[i][j]
            ax.set_facecolor("#fcfcfb")
            here = [(ms, v[key], m) for d, m, ms, v in pts if d == ds]
            xr = max(p[0] for p in here) or 1.0
            yr = (max(p[1] for p in here) - min(p[1] for p in here)) or 1.0
            labels: list[list] = []                       # near-coincident points share one label
            for x, y, m in sorted(here, key=lambda p: p[1]):
                ax.scatter(x, y, s=64, color="#2a78d6" if "ours:" in m else "#eb6834",
                           edgecolors="#fcfcfb", linewidths=2, zorder=3)
                near = next((lb for lb in labels if abs(lb[0] - x) < 0.1 * xr and abs(lb[1] - y) < 0.06 * yr), None)
                if near:
                    near[2].append(short_name(m))
                else:
                    labels.append([x, y, [short_name(m)]])
            for x, y, names in labels:
                ax.annotate("\n".join(names), (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8,
                            color=ink, va="bottom")
            ax.set_xlabel("ms / frame (4090, 518², fp16)", color=muted, fontsize=9)
            ax.set_ylabel(ylab, color=muted, fontsize=9)
            ax.set_title(f"{ds} {a.cell} ({a.split})", color=ink, fontsize=10, loc="left")
            ax.grid(True, color=grid, linewidth=0.8)
            ax.set_xlim(0, max(p[2] for p in pts) * 1.6)       # room for the right-most label
            ax.margins(y=0.15)
            ax.tick_params(colors=muted, labelsize=8)
            for s in ax.spines.values():
                s.set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=8, label=t)
               for c, t in (("#2a78d6", "ours (joint model)"), ("#eb6834", "cascade / single model"))]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{stem}.png", dpi=200)
    print(f"wrote {stem}.png and .csv ({len(pts)} points)")


def cmd_table(a) -> None:
    rows = _results(a.out)
    lat = _latencies(a.out)
    by = {(r["model"], r["dataset"], r["cell"], r["split"]): r["video_macro"] for r in rows}
    cols = ["abs_rel_aligned", "delta1_aligned", "abs_rel_unaligned", "psnr", "ssim", "eps_r", "joint_p95"]
    for ds, cell, split in sorted({(r["dataset"], r["cell"], r["split"]) for r in rows}):
        base = by.get((BASELINE, ds, cell, split))
        print(f"\n### {ds} {cell} split={split}" + ("  (headline cell)" if cell == HEADLINE_CELL else ""))
        print("| model | AbsRel | d1 | AbsRel unal | PSNR | SSIM | eps_r | joint_p95 | ms/f | beats D R T |")
        print("|---|" + "---|" * 9)
        for (m, d, c, s), v in sorted(by.items()):
            if (d, c, s) != (ds, cell, split):
                continue
            f = [f"{v[k]:.4f}" if k in v and k != "psnr" else (f"{v[k]:.2f}" if k in v else "-") for k in cols]
            beats = "-" if base is None or m == BASELINE else " ".join(
                "Y" if ok else "n" for ok in (v["abs_rel_aligned"] < base["abs_rel_aligned"], v["psnr"] > base["psnr"],
                                              v["joint_p95"] < base["joint_p95"]))
            ms = latency_of(lat, m)
            star = "*" if ms and m not in lat else ""
            print(f"| {m} | " + " | ".join(f) + f" | {f'{ms:.1f}{star}' if ms else '-'} | {beats} |")
    print("\n\\* latency reused from a timed pipeline of the same architecture (metric_vda_s chains <- the vda_s "
          "chain, same backbone; ours <- a timed run with the same restoration head)")


def cmd_llvar_draws(a) -> None:
    LLVAR_DRAW_DIR.mkdir(parents=True, exist_ok=True)
    for ds in a.dataset:
        p = llvar_draws_file(ds)
        p.write_text(json.dumps(llvar_draw_table(ds), indent=1, sort_keys=True) + "\n")
        print(f"{p}: sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "latency"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True, help="[restorer+...+]depth, e.g. fastdvdnet+vda_s, ours:<ckpt>")
        s.add_argument("--out", required=True)
        if name == "run":
            s.add_argument("--dataset", nargs="+", default=DEFAULT_DATASETS, choices=list(DATASETS) + list(DATASET_ALIASES))
            s.add_argument("--cells", nargs="+", default=CELLS)
            s.add_argument("--split", default="dev", choices=["dev", "test", "all"])
            s.add_argument("--limit-clips", type=int, default=None, help="smoke use; recorded in the JSON")
            s.add_argument("--device", default="cuda")
            s.add_argument("--temporal", action="store_true", help="add TAE (GT poses) and tOF (pinned RAFT)")
            s.add_argument("--grid", nargs="+", default=None,
                           help="probe package: score these depth post-filter variants (group names ema/agf/sgf/sgfe/sgfi/soup/none or literal "
                                "variants) from ONE model pass; one result file per variant, named as a single-token run")
            s.add_argument("--diag", action="store_true", help="probe package: add the flicker diagnostic to static-cell results")
            s.add_argument("--allow-test", action="store_true", help="REQUIRED for --split test / all (TEST clips): a locked model under a pre-declared rule only")
        else:
            s.add_argument("--size", type=int, default=518)
            s.add_argument("--iters", type=int, default=30)
            s.add_argument("--warmup", type=int, default=5)
    s = sub.add_parser("llvar-draws", help="v3: write provenance/dn01_v3/llvar_draws_<dataset>.json (keys only, no frame is read) and print its sha256")
    s.add_argument("--dataset", nargs="+", default=["kitti", "bonn"], choices=list(DATASETS))
    sub.add_parser("table").add_argument("--out", required=True)
    s = sub.add_parser("plot")
    s.add_argument("--out", required=True)
    s.add_argument("--cell", default=HEADLINE_CELL)
    s.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    a = ap.parse_args()
    {"run": cmd_run, "latency": cmd_latency, "table": cmd_table, "plot": cmd_plot, "llvar-draws": cmd_llvar_draws}[a.cmd](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""VAL-01: score a VDA-S checkpoint on VDA's own eval harness -- reusing
benchmark/eval/eval.py + metric.py and video_depth_anything.video_depth.VideoDepthAnything
directly from the vendored repo (pinned commit recorded in the output artifact), not
reimplementing them. Same reuse-not-reimplement principle as models/vda_net.py.

Two-stage pipeline, matching the vendored repo's own infer.py/eval.py split exactly:
  1. run_official_inference(): the real model runs each clip, writing one .npy disparity
     array per frame (infer.py's own naming convention).
  2. run_official_eval(): the real eval_depthcrafter() aligns each clip's disparity to GT
     by one least-squares scale+shift fit (over the whole clip) and computes AbsRel/RMSE/
     delta1 -- VDA's own alignment code, untouched.

KITTI self-consistency check (Fable, 2026-08-27; named deliberately NOT "tier0"/
"benchmark_fidelity" -- docs/BENCHMARK_PROTOCOL.md's BENCH-06 freeze forbids citing VDA/
KITTI as a fidelity pass, and artifact field names propagate into later documents):
before trusting a score for any checkpoint, running the RELEASED VDA-S checkpoint through
this exact script on KITTI must reproduce BENCH-01d/05b's already-recorded published-vs-
ours figures within tolerance. This verifies THIS SCRIPT'S OWN plumbing reproduces a prior
measurement -- it is not a fresh benchmark_fidelity claim about VDA/KITTI, run
automatically whenever the checkpoint being scored is the released one.

Output: one hashed JSON artifact per (checkpoint, dataset) with checkpoint sha256,
manifest sha256, the vendored repo's commit, and (for KITTI + the released checkpoint) the
self-consistency verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# run_unaligned_eval's `from metrics import compute_depth_metrics` needs the repo root on
# sys.path -- present when this module is imported (pytest inserts the repo root), but NOT
# when this file is run directly as `python scripts/val01_eval_vda.py` (Python only adds the
# script's own directory). Found running for real on crispwork, 2026-08-27.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RELEASED_VDA_S_SHA256 = "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609"

# BENCH-01d/05b's recorded published-vs-ours KITTI figures (docs/BENCHMARK_PROTOCOL.md) --
# the as-run figure is what upstream's OWN released script yields end to end; this script
# reuses that same code, so it should reproduce the as-run figure specifically.
KITTI_SELF_CONSISTENCY_REFERENCE = {"abs_rel": 0.0931, "delta1": 0.9333}
KITTI_SELF_CONSISTENCY_ABS_REL_TOLERANCE_REL = 0.03    # 3% relative, matching BENCH-06's freeze tolerance
KITTI_SELF_CONSISTENCY_DELTA1_TOLERANCE_PTS = 0.01     # 1.0 point, matching BENCH-06's freeze tolerance

# VDA's own benchmark/eval/eval.py dataset table, verbatim (a, b, c, d = crop box on GT).
DATASET_PROTOCOL = {
    "kitti":      dict(manifest_key="kitti",   max_depth_eval=80.0, min_depth_eval=0.1, max_eval_len=110, crop=(0, 374, 0, 1242)),
    "kitti_500":  dict(manifest_key="kitti",   max_depth_eval=80.0, min_depth_eval=0.1, max_eval_len=500, crop=(0, 374, 0, 1242)),
    "sintel":     dict(manifest_key="sintel",  max_depth_eval=70.0, min_depth_eval=0.1, max_eval_len=100, crop=(0, 436, 0, 1024)),
    "nyuv2_500":  dict(manifest_key="nyuv2",   max_depth_eval=10.0, min_depth_eval=0.1, max_eval_len=500, crop=(45, 471, 41, 601)),
    "bonn":       dict(manifest_key="bonn",    max_depth_eval=10.0, min_depth_eval=0.1, max_eval_len=110, crop=(0, 480, 0, 640)),
    "bonn_500":   dict(manifest_key="bonn",    max_depth_eval=10.0, min_depth_eval=0.1, max_eval_len=500, crop=(0, 480, 0, 640)),
    "scannet":    dict(manifest_key="scannet", max_depth_eval=10.0, min_depth_eval=0.1, max_eval_len=90,  crop=(8, -8, 11, -11)),
    "scannet_500": dict(manifest_key="scannet", max_depth_eval=10.0, min_depth_eval=0.1, max_eval_len=500, crop=(8, -8, 11, -11)),
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_commit(repo: str) -> str:
    return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()


def infer_npy_path(infer_dir: str, dataset: str, image_rel: str) -> str:
    """Matches VDA's own benchmark/infer/infer.py naming exactly."""
    return (infer_dir + "/" + dataset + "/" + image_rel).replace(".jpg", ".npy").replace(".png", ".npy")


def load_manifest_clips(json_path: str, manifest_key: str) -> list[tuple[str, list[dict]]]:
    """Parses VDA's own manifest JSON format: {manifest_key: [{video_key: [{"image","gt_depth","factor"}, ...]}]}."""
    with open(json_path) as f:
        path_json = json.load(f)
    clips = []
    for entry in path_json[manifest_key]:
        for video_key, frames in entry.items():
            clips.append((video_key, frames))
    return clips


def check_kitti_self_consistency(measured_abs_rel: float, measured_delta1: float) -> dict:
    abs_rel_delta_pct = (measured_abs_rel - KITTI_SELF_CONSISTENCY_REFERENCE["abs_rel"]) / KITTI_SELF_CONSISTENCY_REFERENCE["abs_rel"] * 100
    delta1_delta_points = (measured_delta1 - KITTI_SELF_CONSISTENCY_REFERENCE["delta1"]) * 100
    abs_rel_ok = abs(abs_rel_delta_pct) <= KITTI_SELF_CONSISTENCY_ABS_REL_TOLERANCE_REL * 100
    delta1_ok = abs(delta1_delta_points) <= KITTI_SELF_CONSISTENCY_DELTA1_TOLERANCE_PTS * 100
    return {
        "reference": KITTI_SELF_CONSISTENCY_REFERENCE,
        "measured": {"abs_rel": measured_abs_rel, "delta1": measured_delta1},
        "abs_rel_delta_pct": abs_rel_delta_pct,
        "delta1_delta_points": delta1_delta_points,
        "abs_rel_result": "PASS" if abs_rel_ok else "FAIL",
        "delta1_result": "PASS" if delta1_ok else "FAIL",
        "overall": "PASS" if (abs_rel_ok and delta1_ok) else "FAIL",
        "tolerance": f"within {KITTI_SELF_CONSISTENCY_ABS_REL_TOLERANCE_REL*100:.1f}% relative AbsRel and "
                     f"{KITTI_SELF_CONSISTENCY_DELTA1_TOLERANCE_PTS*100:.1f} point delta1, against BENCH-01d/05b's "
                     "as-run figure (upstream's own released script, unmodified)",
    }


def _import_vda_modules(vda_repo: str):
    if vda_repo not in sys.path:
        sys.path.insert(0, vda_repo)
    eval_dir = str(Path(vda_repo) / "benchmark" / "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from video_depth_anything.video_depth import VideoDepthAnything
    import eval as vda_eval_module
    import metric as vda_metric_module
    return VideoDepthAnything, vda_eval_module, vda_metric_module


_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def load_model_for_inference(vda_repo: str, checkpoint_path: str, encoder: str, metric: bool,
                             ramen_checkpoint: bool = False, da_v2_checkpoint_path: str | None = None):
    """A standalone, complete checkpoint (released VDA-S, DA-V2-init baseline) is a flat
    state dict loadable with strict=True directly. A checkpoint SAVED BY THIS REPO'S OWN
    train.py (``ramen_checkpoint=True``) is a completely different shape -- found running
    this for real the first time, 2026-08-27, step2000.pt of job 6155634 -- train.py's
    ``_save()`` wraps everything in ``{cfg, epoch, step, ..., trainable, trainable_ema}``,
    and (frozen-backbone runs) never saves ``pretrained.*`` at all, only the trainable head
    under ``trainable``/``trainable_ema`` (EMA-smoothed, matching train.py's own
    ``_validate_with_ema`` convention -- the head weights actually used for eval during
    training). Reconstructs the full model by reusing ``models.vda_net.VDANet``'s own
    correct DA-V2 init (same recipe training itself used, not reimplemented differently),
    then overlays the checkpoint's ``trainable_ema`` onto ``core.head`` directly."""
    import torch

    VideoDepthAnything, _, _ = _import_vda_modules(vda_repo)
    if not ramen_checkpoint:
        model = VideoDepthAnything(**_MODEL_CONFIGS[encoder], metric=metric)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False), strict=True)
        return model

    if da_v2_checkpoint_path is None:
        raise ValueError("ramen_checkpoint=True requires da_v2_checkpoint_path (the same DA-V2 "
                         "init checkpoint the training run itself used)")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.vda_net import VDANet

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mc = ckpt.get("cfg", {}).get("model", {})
    wrapper = VDANet(repo=vda_repo, encoder=encoder, metric=metric, init="da_v2",
                     da_v2_checkpoint_path=da_v2_checkpoint_path,
                     lora_rank=int(mc.get("vda_lora_rank", 0)), lora_alpha=float(mc.get("vda_lora_alpha", 16.0)),
                     input_adapter=str(mc.get("input_adapter", "none")))
    trainable = ckpt.get("trainable_ema", ckpt.get("trainable"))
    if trainable is None:
        raise ValueError(f"{checkpoint_path}: neither 'trainable_ema' nor 'trainable' present -- "
                         "not a recognized train.py checkpoint (unfrozen-backbone runs save "
                         "'model'/'model_ema' instead, not yet supported here)")
    head_state = {k[len("head."):]: v for k, v in trainable.items() if k.startswith("head.")}
    wrapper.core.head.load_state_dict(head_state, strict=True)
    wrapper.load_lora_merged(trainable)   # DN-01 L1 encoder LoRA, folded in; no-op without one
    wrapper.load_adapter(trainable)
    from models.vda_net import refuse_frontend
    refuse_frontend(trainable, "val01_eval_vda.py")
    if wrapper.adapter is not None:
        # DN-01 PA: VDANet.forward applies the input adapter, but this path calls core directly (the StreamingOVDA
        # front-end bug, docs/FINDINGS.md:276), so wrap core.forward; registered on core so .to(device) moves it too
        wrapper.core.input_adapter = wrapper.adapter
        wrapper.core.forward = adapter_forward(wrapper.core.forward, wrapper.adapter)
    return wrapper.core


def run_official_inference(vda_repo: str, checkpoint_path: str, encoder: str,
                           clips: list[tuple[str, list[dict]]], benchmark_root: str,
                           dataset: str, infer_dir: str, device: str = "cuda",
                           metric: bool = False, ramen_checkpoint: bool = False,
                           da_v2_checkpoint_path: str | None = None, noise_sigma: float = 0.0,
                           short_side: int | None = None, blur_sigma: float = 0.0,
                           clamp_input: bool = False) -> None:
    """VDA's own benchmark/infer/infer.py inference call, unmodified, run per clip.

    ``metric`` (Opus's audit, 2026-08-27): this constructor call was previously missing
    ``metric=`` entirely -- ALWAYS building a relative-mode VideoDepthAnything regardless
    of which checkpoint was loaded, so a metric checkpoint's state dict either failed to
    load or loaded into the wrong head configuration silently. Threading this through is
    required before this script can correctly score any metric checkpoint at all.

    The saved .npy files are ALWAYS disparity-space, uniformly for both modes (verified
    directly against the vendored benchmark/eval/eval.py on crispwork, 2026-08-27: its
    ``eval_depthcrafter``/``get_infer`` treat a loaded .npy as disparity unconditionally --
    there is no metric-specific branch anywhere in that file). For ``metric=True``, whose
    native model output is depth-like (models/vda_net.py's own convention), this function
    inverts to disparity (``1/depth``) before saving -- NOT because that is VDA's own
    published metric-eval protocol (no such protocol exists in the vendored eval.py; this
    is this repo's own choice, so ``run_official_eval``'s scale+shift fit still operates on
    the disparity-space quantity it was written for) rather than silently mis-scoring a
    metric checkpoint through code built for a different quantity. See
    ``run_unaligned_eval`` for the metric-native, alignment-free counterpart that avoids
    this inversion entirely."""
    import cv2
    import numpy as np

    model = load_model_for_inference(vda_repo, checkpoint_path, encoder, metric,
                                     ramen_checkpoint=ramen_checkpoint,
                                     da_v2_checkpoint_path=da_v2_checkpoint_path)
    model = model.to(device).eval()
    wrap = noise_sigma > 0 or blur_sigma > 0 or clamp_input
    clean_forward = model.forward if wrap else None

    for video_key, frames in clips:
        if wrap:   # DN-01: (clamp, then blur, then) fixed-seed AWGN at model resolution
            fwd = clean_forward
            if noise_sigma > 0:
                fwd = add_noise_forward(fwd, noise_sigma, noise_seed(dataset, video_key, noise_sigma), device)
            fwd = add_blur_forward(fwd, blur_sigma) if blur_sigma > 0 else fwd
            model.forward = clamp_forward(fwd) if clamp_input else fwd   # outermost = applied first
        videos = []
        infer_paths = []
        for f in frames:
            img = cv2.imread(str(Path(benchmark_root) / f["image"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            videos.append(img)
            infer_paths.append(infer_npy_path(infer_dir, dataset, f["image"]))
        videos = np.stack(videos, axis=0)
        input_size = 518 if short_side is None else input_size_for_short_side(*videos.shape[1:3], short_side)
        depths, _fps = model.infer_video_depth(videos, target_fps=1, input_size=input_size,
                                                device=device, fp32=True)
        if metric:
            depths = 1.0 / np.clip(depths, a_min=1e-3, a_max=None)   # depth -> disparity, see docstring
        for p, d in zip(infer_paths, depths):
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            np.save(p, d)


def input_size_for_short_side(h: int, w: int, short_side: int) -> int:
    """input_size to pass so infer_video_depth's memory clause (video_depth.py:72-75: aspect
    > 1.78 shrinks input_size, 518 -> 280 on KITTI) lands on ``short_side`` instead. A metric
    model's scale follows input resolution, so metric runs must pin this explicitly."""
    ratio = max(h, w) / min(h, w)
    if ratio <= 1.78:
        return short_side
    n = math.ceil(short_side * ratio / 1.777)
    while round(int(n * 1.777 / ratio) / 14) * 14 < short_side:
        n += 1
    return n


def noise_seed(dataset: str, video_key: str, sigma: float) -> int:
    """DN-01: one deterministic noise stream per (dataset, video, sigma)."""
    return int(hashlib.sha256(f"{dataset}|{video_key}|awgn|{sigma}".encode()).hexdigest()[:8], 16)


def add_noise_forward(forward, sigma: float, seed: int, device: str):
    """DN-01: wrap VideoDepthAnything.forward so each window gets AWGN (sigma in /255 units)
    added in [0,1] RGB space at model resolution -- after VDA's own resize, matching how
    training degrades after the VDA resize/crop. infer_video_depth feeds ImageNet-normalized
    (B,T,3,H,W) windows, so denormalize -> noise -> clip -> renormalize."""
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    gen = torch.Generator(device=device).manual_seed(seed)

    def noisy(x):
        rgb = x * std + mean
        n = torch.randn(rgb.shape, generator=gen, device=rgb.device, dtype=rgb.dtype)
        return forward(((rgb + n * (sigma / 255.0)).clamp(0.0, 1.0) - mean) / std)
    return noisy


def adapter_forward(forward, adapter):
    """DN-01 PA: infer_video_depth feeds ImageNet-normalized (B,T,3,H,W) windows and VDANet.forward applies the input
    adapter in [0,1] RGB before normalizing, so denormalize -> adapter -> renormalize. Wrapped at load time, so the
    noise/blur/clamp wrappers sit outside it and degrade first, as in training."""
    import torch

    def adapted(x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
        return forward((adapter(x * std + mean) - mean) / std)
    return adapted


def clamp_forward(forward):
    """DN-01 H4 (2026-09-24): clamp each window to [0,1] RGB before anything else, as degrade=true training does
    to its inputs even at severity 0 (VDA's bicubic resize overshoots; the plain clean path leaves that in)."""
    import torch

    def clamped(x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
        return forward(((x * std + mean).clamp(0.0, 1.0) - mean) / std)
    return clamped


def add_blur_forward(forward, sigma_px: float):
    """DN-01 A1-blur: wrap ``forward`` (outside the noise wrapper, so blur comes first) with
    train's cv2 defocus blur (``_blur_frames``, sigma in px at model resolution). Blur is
    linear with a unit-sum kernel, so blurring the ImageNet-normalized window equals blurring
    [0,1] RGB and renormalizing -- no round trip needed."""
    import torch

    from data.video_clip_dataset import _blur_frames

    def blurred(x):
        arr = x.detach().float().cpu().numpy().reshape(-1, *x.shape[-3:]).copy()   # (B*T,3,H,W); never mutate x
        return forward(torch.from_numpy(_blur_frames(arr, sigma_px)).view(x.shape).to(x.device, x.dtype))
    return blurred


def run_official_eval(vda_repo: str, benchmark_root: str, dataset: str,
                      clips: list[tuple[str, list[dict]]], infer_dir: str) -> dict:
    """VDA's own benchmark/eval/eval.py eval_depthcrafter(), unmodified, aggregated the
    same way their own main() does (mean over clips)."""
    import numpy as np

    proto = DATASET_PROTOCOL[dataset]
    _, vda_eval_module, _ = _import_vda_modules(vda_repo)
    a, b, c, d = proto["crop"]
    args = SimpleNamespace(max_eval_len=proto["max_eval_len"], max_depth_eval=proto["max_depth_eval"],
                           min_depth_eval=proto["min_depth_eval"], a=a, b=b, c=c, d=d)
    results = []
    for video_key, frames in clips:
        infer_paths = [infer_npy_path(infer_dir, dataset, f["image"]) for f in frames]
        depth_gt_paths = [str(Path(benchmark_root) / f["gt_depth"]) for f in frames]
        factors = [f["factor"] for f in frames]
        results.append(vda_eval_module.eval_depthcrafter(infer_paths, depth_gt_paths, factors, args))
    mean = np.mean(np.array(results), axis=0)
    return {name: float(mean[i]) for i, name in enumerate(vda_eval_module.eval_metrics)}


def compute_alignment_stats(vda_repo: str, benchmark_root: str, dataset: str,
                            clips: list[tuple[str, list[dict]]], infer_dir: str) -> dict:
    """Recovers the per-clip (scale, shift) that VDA's own eval_depthcrafter fits and then
    discards (it returns only the post-alignment metrics, never the fit itself) -- Opus's
    follow-up, 2026-08-27, to turn the observed ~3x unaligned/aligned AbsRel gap on a metric
    checkpoint from an inference into a measured fact: scale ~= 1.0 means the model's raw
    absolute scale is already correct and the gap is dispersion/noise; scale far from 1.0
    means a systematic scale offset instead.

    Mirrors eval_depthcrafter's own fit exactly (same GT/prediction loading via the
    vendored get_gt/get_infer, same crop box, same valid mask, same
    ``A = [pred_disp, ones]; X = lstsq(A, gt_disp)`` -- benchmark/eval/eval.py's own lines,
    not reimplemented differently) rather than re-deriving a different alignment; this
    duplicates only the few lines needed to keep the fit itself instead of discarding it,
    since the vendored function offers no hook to recover it otherwise."""
    import numpy as np

    proto = DATASET_PROTOCOL[dataset]
    _, vda_eval_module, _ = _import_vda_modules(vda_repo)
    a, b, c, d = proto["crop"]
    args = SimpleNamespace(max_eval_len=proto["max_eval_len"], max_depth_eval=proto["max_depth_eval"],
                           min_depth_eval=proto["min_depth_eval"], a=a, b=b, c=c, d=d)
    scales, shifts = [], []
    for video_key, frames in clips:
        infer_paths = [infer_npy_path(infer_dir, dataset, f["image"]) for f in frames]
        depth_gt_paths = [str(Path(benchmark_root) / f["gt_depth"]) for f in frames]
        factors = [f["factor"] for f in frames]
        gts, infs = [], []
        for infer_path, gt_path, factor in zip(infer_paths, depth_gt_paths, factors):
            if not Path(infer_path).exists():
                continue
            depth_gt = vda_eval_module.get_gt(gt_path, factor, args)[a:b, c:d]
            infer = vda_eval_module.get_infer(infer_path, args, target_size=depth_gt.shape)
            gts.append(depth_gt)
            infs.append(infer)
        if not gts:
            continue
        gts = np.stack(gts, axis=0)[: proto["max_eval_len"]]
        infs = np.stack(infs, axis=0)[: proto["max_eval_len"]]
        valid_mask = (gts > 1e-3) & (gts < proto["max_depth_eval"])
        if not valid_mask.any():
            continue
        gt_disp_masked = 1.0 / (gts[valid_mask].reshape((-1, 1)).astype(np.float64) + 1e-8)
        pred_disp_masked = np.clip(infs, a_min=1e-3, a_max=None)[valid_mask].reshape((-1, 1)).astype(np.float64)
        ones = np.ones_like(pred_disp_masked)
        A = np.concatenate([pred_disp_masked, ones], axis=-1)
        fit = np.linalg.lstsq(A, gt_disp_masked, rcond=None)[0].flatten()
        scales.append(float(fit[0]))
        shifts.append(float(fit[1]))
    scales_arr, shifts_arr = np.array(scales), np.array(shifts)
    return {
        "num_clips": len(scales),
        "scale": {"median": float(np.median(scales_arr)), "iqr": float(np.percentile(scales_arr, 75) -
                 np.percentile(scales_arr, 25)), "min": float(scales_arr.min()), "max": float(scales_arr.max())},
        "shift": {"median": float(np.median(shifts_arr)), "iqr": float(np.percentile(shifts_arr, 75) -
                 np.percentile(shifts_arr, 25)), "min": float(shifts_arr.min()), "max": float(shifts_arr.max())},
        "interpretation": "scale ~= 1.0 -> correct absolute scale, gap is dispersion error; "
                          "scale far from 1.0 -> systematic scale offset",
    }


def run_unaligned_eval(vda_repo: str, benchmark_root: str, dataset: str,
                       clips: list[tuple[str, list[dict]]], infer_dir: str) -> dict:
    """RAMEN's own alignment-free scoring (metrics.py::compute_depth_metrics), kept
    SEPARATE from run_official_eval's VDA-native scale-shift-aligned protocol above --
    this is the number that actually tests whether a metric checkpoint's raw output
    already sits at GT's absolute scale; an aligned-only report could mask a metric model
    that is scale-wrong but shape-right.

    Reuses VDA's own vendored ``get_gt``/``get_infer`` directly (correct factor scaling,
    .npy-vs-image loading, the same crop box) rather than reimplementing GT loading --
    only the least-squares scale+shift fit and the depth<->disparity round-trip inside
    ``eval_depthcrafter`` are skipped. The saved .npy predictions are always disparity-space
    (see run_official_inference's docstring), so this inverts them back to depth
    (``1/disparity``) before comparing to GT depth directly -- exact up to float precision
    except at pixels ``eval_depthcrafter``'s own masking already excludes from every metric
    anyway.

    Definition (ultrareview): identical to bench_joint.score_clip's ``*_unaligned`` -- predictions clamped to
    [1e-3, max_depth_eval] and every GT-valid pixel counted (``invalid_pred="penalize"``). Rows produced before this
    change used ``"exclude"`` with no upper clamp (pred <= 0.1 m dropped, far pushes uncapped) and are NOT comparable
    to bench_joint's unaligned numbers or to rows produced after it."""
    import numpy as np
    import torch

    from metrics import compute_depth_metrics

    proto = DATASET_PROTOCOL[dataset]
    _, vda_eval_module, _ = _import_vda_modules(vda_repo)
    a, b, c, d = proto["crop"]
    args = SimpleNamespace(max_eval_len=proto["max_eval_len"], max_depth_eval=proto["max_depth_eval"],
                           min_depth_eval=proto["min_depth_eval"], a=a, b=b, c=c, d=d)
    per_clip = []
    for video_key, frames in clips:
        infer_paths = [infer_npy_path(infer_dir, dataset, f["image"]) for f in frames]
        depth_gt_paths = [str(Path(benchmark_root) / f["gt_depth"]) for f in frames]
        factors = [f["factor"] for f in frames]
        gts, disps = [], []
        for infer_path, gt_path, factor in zip(infer_paths, depth_gt_paths, factors):
            if not Path(infer_path).exists():
                continue
            depth_gt = vda_eval_module.get_gt(gt_path, factor, args)[a:b, c:d]
            disp = vda_eval_module.get_infer(infer_path, args, target_size=depth_gt.shape)
            gts.append(depth_gt)
            disps.append(disp)
        if not gts:
            continue
        gts = np.stack(gts, axis=0)[: proto["max_eval_len"]]
        disps = np.stack(disps, axis=0)[: proto["max_eval_len"]]
        # ultrareview: the SAME unaligned definition as bench_joint.score_clip (pred clamped to [1e-3, max_depth], every GT-valid pixel
        # counted): a zero / near-zero prediction is penalised (not silently dropped by invalid_pred="exclude") and a far push is
        # capped at the dataset cap. min_depth=1e-3 is the clamp floor; the GT-validity band is the mask's (gt > min_depth_eval).
        pred_depth = np.clip(1.0 / np.clip(disps, a_min=1e-3, a_max=None), 1e-3, proto["max_depth_eval"])
        mask = (gts > proto["min_depth_eval"]) & (gts < proto["max_depth_eval"])
        per_clip.append(compute_depth_metrics(
            torch.from_numpy(pred_depth), torch.from_numpy(gts), torch.from_numpy(mask),
            min_depth=1e-3, max_depth=proto["max_depth_eval"], invalid_pred="penalize",
        ))
    keys = per_clip[0].keys()
    return {k: float(np.mean([c[k] for c in per_clip])) for k in keys}


def score_checkpoint(vda_repo: str, checkpoint_path: str, encoder: str, benchmark_root: str,
                     manifest_path: str, dataset: str, infer_dir: str, device: str = "cuda",
                     metric: bool = False, unaligned: bool = False, alignment_stats: bool = False,
                     ramen_checkpoint: bool = False, da_v2_checkpoint_path: str | None = None,
                     noise_sigma: float = 0.0, short_side: int | None = None,
                     blur_sigma: float = 0.0, clamp_input: bool = False) -> dict:
    proto = DATASET_PROTOCOL[dataset]
    clips = load_manifest_clips(manifest_path, proto["manifest_key"])
    run_official_inference(vda_repo, checkpoint_path, encoder, clips, benchmark_root, dataset,
                           infer_dir, device, metric=metric, ramen_checkpoint=ramen_checkpoint,
                           da_v2_checkpoint_path=da_v2_checkpoint_path, noise_sigma=noise_sigma,
                           short_side=short_side, blur_sigma=blur_sigma, clamp_input=clamp_input)
    metrics = run_official_eval(vda_repo, benchmark_root, dataset, clips, infer_dir)

    artifact = {
        "task": "VAL-01",
        "dataset": dataset,
        "metric": metric,
        "vda_repo_commit": git_commit(vda_repo),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "num_clips": len(clips),
        "noise_sigma_255": noise_sigma,
        "short_side": short_side,   # None = VDA default (518 with its aspect clamp; 280 on KITTI)
        **({"blur_sigma_px": blur_sigma} if blur_sigma > 0 else {}),
        **({"clamp_input": True} if clamp_input else {}),
        "metrics": metrics,
    }
    if unaligned:
        artifact["unaligned_metrics"] = run_unaligned_eval(vda_repo, benchmark_root, dataset, clips, infer_dir)
    if alignment_stats:
        artifact["alignment_stats"] = compute_alignment_stats(vda_repo, benchmark_root, dataset, clips, infer_dir)
    if dataset == "kitti" and artifact["checkpoint_sha256"] == RELEASED_VDA_S_SHA256 and noise_sigma == 0 \
            and blur_sigma == 0:
        artifact["kitti_self_consistency_check"] = check_kitti_self_consistency(
            metrics["abs_relative_difference"], metrics["delta1_acc"])
    return artifact


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vda-repo", default="/mnt/DATA/DegradedDepth/competition/Video-Depth-Anything")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--encoder", default="vits", choices=list(_MODEL_CONFIGS))
    ap.add_argument("--dataset", required=True, choices=list(DATASET_PROTOCOL))
    ap.add_argument("--benchmark-root", required=True,
                    help="directory containing <dataset>/... referenced by the manifest's relative paths")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--infer-dir", required=True, help="scratch dir for per-frame .npy predictions")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metric", action="store_true",
                    help="score a metric-mode checkpoint -- required to correctly load a metric "
                         "state dict into the vendored model at all (previously always built a "
                         "relative-mode model regardless of the checkpoint; Opus's audit, 2026-08-27)")
    ap.add_argument("--unaligned", action="store_true",
                    help="also compute RAMEN's own alignment-free metrics.py::compute_depth_metrics "
                         "scoring, kept separate from VDA's own scale-shift-aligned protocol -- the "
                         "figure that actually tests absolute-scale correctness for a metric checkpoint")
    ap.add_argument("--alignment-stats", action="store_true",
                    help="also record the per-clip (scale, shift) VDA's own eval_depthcrafter fits and "
                         "discards -- median/IQR across clips, to distinguish 'correct absolute scale, "
                         "just dispersion error' (scale ~= 1.0) from a systematic offset (Opus's "
                         "follow-up, 2026-08-27)")
    ap.add_argument("--ramen-checkpoint", action="store_true",
                    help="the checkpoint is one of THIS repo's own train.py saves (a wrapper dict with "
                         "trainable/trainable_ema, never a flat standalone state dict) -- required to "
                         "score an in-progress training run's own step{N}.pt/last.pt at all (found "
                         "missing running this for real the first time, 2026-08-27, job 6155634's "
                         "step2000.pt). Requires --da-v2-checkpoint-path.")
    ap.add_argument("--da-v2-checkpoint-path", default=None,
                    help="required with --ramen-checkpoint: the same DA-V2 init checkpoint the training "
                         "run itself used, to reconstruct the frozen backbone that --ramen-checkpoint's "
                         "own save format never includes")
    ap.add_argument("--noise-sigma", type=float, default=0.0,
                    help="DN-01: AWGN sigma in /255 units added at model resolution (fixed seed per "
                         "video); 0 = clean, the VAL-01 protocol unchanged")
    ap.add_argument("--short-side", type=int, default=None,
                    help="model input short side, bypassing VDA's aspect clamp (518 -> 280 on KITTI). "
                         "Metric protocol: 378 on KITTI (native height). Default: VDA's own behaviour")
    ap.add_argument("--blur-sigma", type=float, default=0.0,
                    help="DN-01 A1-blur: Gaussian defocus sigma in px at model resolution, fixed per "
                         "video, applied before --noise-sigma's noise; 0 = no blur (old path unchanged)")
    ap.add_argument("--clamp-input", action="store_true",
                    help="DN-01 H4: clamp each model-resolution window to [0,1] RGB first, as degrade=true "
                         "training does even at severity 0; default off (old path unchanged)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    artifact = score_checkpoint(args.vda_repo, args.checkpoint, args.encoder, args.benchmark_root,
                                args.manifest, args.dataset, args.infer_dir, args.device,
                                metric=args.metric, unaligned=args.unaligned,
                                alignment_stats=args.alignment_stats,
                                ramen_checkpoint=args.ramen_checkpoint,
                                da_v2_checkpoint_path=args.da_v2_checkpoint_path,
                                noise_sigma=args.noise_sigma, short_side=args.short_side,
                                blur_sigma=args.blur_sigma, clamp_input=args.clamp_input)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(json.dumps(artifact, indent=2))
    if "kitti_self_consistency_check" in artifact and artifact["kitti_self_consistency_check"]["overall"] != "PASS":
        raise SystemExit(
            "KITTI self-consistency check FAILED -- this script's own plumbing does not reproduce "
            "BENCH-01d/05b's recorded figure for the released checkpoint; do not trust any "
            "other result from this script until this is resolved."
        )


if __name__ == "__main__":
    main()

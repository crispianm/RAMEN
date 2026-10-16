#!/usr/bin/env python3
"""VAL-01 metric variant: empirically determine what units the released Metric-Video-
Depth-Anything-Small checkpoint's raw head output is actually in, when run through
THIS repo's own training wrapper (models/vda_net.py::VDANet.forward()) -- not just the
vendored infer_video_depth() convenience function, since VDANet.forward() is the code
path train.py actually trains and evaluates against, and its own docstring currently
states an ASSUMPTION ("depth-like for the metric variant") rather than a measured fact.

PRE-REGISTERED BEFORE RUNNING (Fable/Opus's ruling, 2026-08-27; the whole point of
writing this before any measurement is that it cannot be quietly adjusted after seeing
the numbers):

  If the raw output is genuinely depth-like (metres):
    - correlation with GT depth: positive
    - correlation with GT inverse depth: near zero or negative
    - unaligned AbsRel vs aligned AbsRel: same neighbourhood (roughly within 2x; ~0.1 on
      KITTI is the ballpark either figure should land in if the model is any good at all)

  If the raw output is actually disparity-like (inverse depth):
    - correlation with GT depth: near zero or negative
    - correlation with GT inverse depth: positive
    - unaligned AbsRel vs aligned AbsRel: unaligned an order of magnitude worse -- an
      unaligned disparity value compared directly to a depth GT is a unit mismatch, not
      a small error

  Context only, NOT evidence of units (do not treat this as a units measurement):
  the vendored video_depth_anything/video_depth.py:135-136's own windowed-stitching loop
  inside infer_video_depth() sets ``scale, shift = 1.0, 0.0`` for metric mode's
  cross-window alignment, instead of computing one via compute_scale_and_shift the way it
  does for relative mode. This is suggestive that upstream expects metric windows to
  already share ONE common absolute scale (no cross-window drift correction needed) --
  but "windows agree with each other" does not by itself prove that shared scale is
  metres specifically; it is upstream's own design assumption, cited here as context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PRE_REGISTRATION = {
    "hypothesis_under_test": "models/vda_net.py's VDANet.forward() docstring assumption: "
                             "metric=True's raw output is depth-like (metres), not disparity",
    "if_depth_like": {
        "correlation_with_gt_depth": "positive",
        "correlation_with_gt_inverse_depth": "near zero or negative",
        "unaligned_vs_aligned_abs_rel": "same neighbourhood, roughly within 2x of each other",
    },
    "if_disparity_like": {
        "correlation_with_gt_depth": "near zero or negative",
        "correlation_with_gt_inverse_depth": "positive",
        "unaligned_vs_aligned_abs_rel": "unaligned an order of magnitude worse than aligned",
    },
    "context_only_not_a_units_measurement": (
        "video_depth_anything/video_depth.py:135-136 (vendored, commit "
        "4f5ae23172ba60fd7bc11ef671cca678842c7072): infer_video_depth()'s windowed-stitching "
        "loop sets scale,shift=1.0,0.0 for metric mode instead of computing one via "
        "compute_scale_and_shift (relative mode's own branch) -- suggestive upstream expects "
        "metric windows to already share one common absolute scale, but this proves windows "
        "AGREE with each other, not that the shared scale is metres specifically."
    ),
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pearson_corr(a, b) -> float:
    a = a.double() - a.double().mean()
    b = b.double() - b.double().mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a * b).sum() / denom)


def run_units_check(vda_repo: str, checkpoint_path: str, benchmark_root: str, manifest_path: str,
                    clip_index: int, device: str = "cuda") -> dict:
    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from val01_eval_vda import DATASET_PROTOCOL, _import_vda_modules, load_manifest_clips

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.vda_net import VDANet

    proto = DATASET_PROTOCOL["kitti"]
    clips = load_manifest_clips(manifest_path, proto["manifest_key"])
    video_key, frames = clips[clip_index]

    _, vda_eval_module, _ = _import_vda_modules(vda_repo)
    a, b, c, d = proto["crop"]

    model = VDANet(repo=vda_repo, encoder="vits", metric=True, init="released",
                   checkpoint_path=checkpoint_path).to(device).eval()

    raw_frames, gt_frames = [], []
    for f in frames:
        img = cv2.imread(str(Path(benchmark_root) / f["image"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img14 = cv2.resize(img, (518, 518))
        gt = vda_eval_module.get_gt(str(Path(benchmark_root) / f["gt_depth"]), f["factor"],
                                    argparse.Namespace())[a:b, c:d]
        with torch.no_grad():
            x = torch.from_numpy(img14).float().permute(2, 0, 1)[None, :, None] / 255.0  # (1,3,1,H,W)
            out = model(x.to(device))   # (1,1,1,518,518) -- VDANet's own native output, unmodified
        pred = out[0, 0, 0].cpu().numpy()
        pred_resized = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        raw_frames.append(pred_resized)
        gt_frames.append(gt)

    raw = torch.from_numpy(np.stack(raw_frames, axis=0))
    gt = torch.from_numpy(np.stack(gt_frames, axis=0))
    mask = (gt > proto["min_depth_eval"]) & (gt < proto["max_depth_eval"]) & torch.isfinite(raw)

    from metrics import compute_depth_metrics
    corr_depth = pearson_corr(raw[mask], gt[mask])
    corr_inv_depth = pearson_corr(raw[mask], 1.0 / gt[mask].clamp_min(1e-3))
    unaligned_single_clip = compute_depth_metrics(raw, gt, mask, min_depth=proto["min_depth_eval"],
                                                  max_depth=proto["max_depth_eval"], invalid_pred="exclude")

    return {
        "video_key": video_key,
        "num_frames_in_clip": len(frames),
        "correlation_with_gt_depth": corr_depth,
        "correlation_with_gt_inverse_depth": corr_inv_depth,
        "single_clip_unaligned_metrics": unaligned_single_clip,
        "raw_output_range": {"min": float(raw[mask].min()), "max": float(raw[mask].max()),
                             "mean": float(raw[mask].mean())},
        "gt_depth_range": {"min": float(gt[mask].min()), "max": float(gt[mask].max()),
                          "mean": float(gt[mask].mean())},
    }


def interpret(measured: dict) -> dict:
    depth_like = measured["correlation_with_gt_depth"] > 0.3 > measured["correlation_with_gt_inverse_depth"]
    disparity_like = measured["correlation_with_gt_inverse_depth"] > 0.3 > measured["correlation_with_gt_depth"]
    if depth_like and not disparity_like:
        conclusion = "depth_like"
    elif disparity_like and not depth_like:
        conclusion = "disparity_like"
    else:
        conclusion = "inconclusive"
    return {
        "matches_if_depth_like": depth_like,
        "matches_if_disparity_like": disparity_like,
        "conclusion": conclusion,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vda-repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--benchmark-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    measured = run_units_check(args.vda_repo, args.checkpoint, args.benchmark_root,
                               args.manifest, args.clip_index, args.device)
    artifact = {
        "task": "VAL-01",
        "purpose": "Determine empirically whether Metric-VDA-S's raw head output (as produced by "
                   "THIS repo's own models/vda_net.py::VDANet.forward()) is depth-like or "
                   "disparity-like -- pre-registered before running, per Fable/Opus's ruling.",
        "checkpoint_path": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "pre_registration": PRE_REGISTRATION,
        "measured": measured,
        "verdict": interpret(measured),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()

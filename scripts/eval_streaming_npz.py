#!/usr/bin/env python3
"""True sequential RAMEN evaluation from an ordered NPZ frame manifest.

The input is a manifest written by ``evaluation.alignment_free.write_frame_manifest``:
``video_id``, ``frame_index``, ``frame``, ``gt`` and optional ``mask``, one row per
frame, stored ragged-safe (frames need not share one resolution). Frames for each
video must be contiguous and chronological. The runner resets only when
``video_id`` changes and records every newest prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.alignment_free import (ALIGNMENT_MODES, ALIGNMENT_SCOPES, ContinuousStreamConsumer,
                                       DepthEvaluator, EvaluationMetadata,
                                       canonical_result_payload, canonical_sha256, finite_json,
                                       load_frame_source, sha256_file, validate_video_id_contiguity)
# RECIPE_REGISTRY and MODEL_REGISTRY both live in non-frozen modules, not inline here,
# so a per-benchmark or per-architecture builder can register itself by touching only
# non-frozen code -- see evaluation/recipe_registry.py's and evaluation/model_registry.py's
# own docstrings for why a gate re-run would never have detected anything about a
# registry entry's correctness anyway.
from evaluation.model_registry import MODEL_REGISTRY
from evaluation.recipe_registry import RECIPE_REGISTRY


def validate_chronological_manifest(video_ids, frame_indices) -> None:
    """Validate contiguity and a strictly increasing per-video order key."""
    if len(frame_indices) != len(video_ids):
        raise ValueError("frame_index has inconsistent leading length")
    validate_video_id_contiguity(video_ids)
    current, previous = None, None
    for row, (vid_raw, index_raw) in enumerate(zip(video_ids, frame_indices)):
        vid = str(vid_raw)
        index = np.asarray(index_raw)
        if index.ndim != 0 or not np.issubdtype(index.dtype, np.number) or not np.isfinite(index):
            raise ValueError(f"frame_index at row {row} must be a finite scalar number")
        index = float(index)
        if vid != current:
            current = vid
            previous = None
        if previous is not None and index <= previous:
            raise ValueError(f"video {vid!r} frame_index is not strictly increasing at row {row}")
        previous = index


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {"measured_steps": len(values),
            "p50_ms_per_frame": float(np.percentile(values, 50)) if values else None,
            "p95_ms_per_frame": float(np.percentile(values, 95)) if values else None,
            "fps_at_p50": float(1000.0 / np.percentile(values, 50)) if values else None}


def metadata(args, checkpoint_hash: str, steady: bool) -> EvaluationMetadata:
    return EvaluationMetadata(
        model_name=args.model.upper(), dataset_name=args.dataset, split=args.split,
        checkpoint_sha256=checkpoint_hash, manifest_sha256=sha256_file(args.manifest),
        output_semantics=args.output_semantics, condition=args.condition,
        streaming={
            "mode": "causal_continuous", "window": args.window,
            "warmup_scored": not steady, "state_reset": "on_video_boundary",
            "frame_order": "chronological",
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest")
    ap.add_argument("--model", default="ramen", choices=sorted(MODEL_REGISTRY),
                    help="Model-registry entry to run this manifest through")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--output-semantics", default="relative_depth",
                    choices=("relative_depth", "metric_depth"))
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--alignment-scope", default="per_video", choices=ALIGNMENT_SCOPES)
    ap.add_argument("--alignment-mode", default="affine", choices=ALIGNMENT_MODES)
    ap.add_argument("--timing-warmup", type=int, default=5,
                    help="Exclude this many initial calls from latency statistics")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = ap.parse_args()
    if args.timing_warmup < 0:
        ap.error("--timing-warmup must be non-negative")

    recipe_factory = RECIPE_REGISTRY.get(args.dataset)
    manifest = load_frame_source(
        args.manifest, regenerate_range=recipe_factory(args.condition) if recipe_factory else None)
    n = len(manifest)
    validate_chronological_manifest(manifest.video_id, manifest.frame_index)

    predict_frame, reset, digest = MODEL_REGISTRY[args.model](args.checkpoint, args.device, args.dtype, args.window)
    all_eval = DepthEvaluator(metadata(args, digest, steady=False), args.alignment_scope, args.alignment_mode)
    steady_eval = DepthEvaluator(metadata(args, digest, steady=True), args.alignment_scope, args.alignment_mode)

    groups: dict[str, list[int]] = {}
    for row, vid_raw in enumerate(manifest.video_id):
        groups.setdefault(str(vid_raw), []).append(row)

    timings: list[float] = []
    per_video_timings: dict[str, dict[str, list[float]]] = {}

    # Set by video_samples() just before each frame is yielded, so timed_predict
    # (called back-to-back with that same frame) knows which video/position/row
    # it belongs to without retaining any prediction past its own call.
    cursor = {"video_id": None, "position": -1, "row": None}

    def wrapped_reset() -> None:
        reset()
        cursor["position"] = -1

    def video_samples(vid: str, idxs: list[int]):
        for row in idxs:
            cursor["video_id"], cursor["row"], cursor["position"] = vid, row, cursor["position"] + 1
            yield manifest.frame(row), manifest.gt(row), manifest.mask(row)

    def timed_predict(frame: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(frame)).to(args.device).float()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        pred = predict_frame(tensor)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        vid, row, position = cursor["video_id"], cursor["row"], cursor["position"]
        if row >= args.timing_warmup:
            timings.append(elapsed_ms)
            phase = "cold_start" if position < args.window - 1 else "steady_state"
            per_video_timings.setdefault(vid, {"cold_start": [], "steady_state": []})[phase].append(elapsed_ms)
        if position >= args.window - 1:
            steady_eval.add_frame(vid, pred, manifest.gt(row), manifest.mask(row))
        return pred

    videos = {vid: video_samples(vid, idxs) for vid, idxs in groups.items()}
    ContinuousStreamConsumer(all_eval, timed_predict, wrapped_reset).consume(videos)

    timing_by_video = {vid: {phase: latency_summary(vals) for phase, vals in phases.items()}
                       for vid, phases in per_video_timings.items()}
    cold = [v for phases in per_video_timings.values() for v in phases["cold_start"]]
    steady = [v for phases in per_video_timings.values() for v in phases["steady_state"]]
    result = {
        "schema_version": "1.0.0",
        "all_frames": all_eval.result(),
        "steady_state": steady_eval.result(),
        "runtime": {
            "warmup_steps_excluded": min(args.timing_warmup, n),
            "measured_steps": len(timings),
            "median_ms_per_frame": float(np.median(timings)) if timings else None,
            "fps": float(1000.0 / np.median(timings)) if timings else None,
            "cold_start": latency_summary(cold),
            "steady_state": latency_summary(steady),
            "per_video": timing_by_video,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0,
        },
    }
    result["provenance_sha256"] = canonical_sha256({
        "all_frames": result["all_frames"]["provenance_sha256"],
        "steady_state": result["steady_state"]["provenance_sha256"],
    })
    result["result_sha256"] = canonical_sha256(canonical_result_payload(result))
    Path(args.out).write_text(json.dumps(finite_json(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": args.out, "runtime": result["runtime"]}, indent=2))


if __name__ == "__main__":
    main()

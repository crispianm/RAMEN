#!/usr/bin/env python3
"""Build a recipe-locked manifest for the KITTI 13-drive endpoint population
(docs/DEGRADED_EVAL_PROTOCOL.md §3/§6), for either the ``clean`` or
``degraded_s<severity>`` condition.

This population is THE SAME 13 drives `scripts/build_kitti_streaming_manifest.py`
built for `BENCH-05b` (docs/BENCHMARK_PROTOCOL.md) -- the one streaming-path
validation this project has. This builder deliberately mirrors that script's
conventions exactly rather than harmonising with Bonn's or VKITTI's, verified
by reading it directly (RAMEN Opus's review) rather than assuming either of
the other two carries over -- two of the three conventions below genuinely
differ from both.

**NO RESIZE, NATIVE RESOLUTION, RAGGED FRAMES.** Unlike Bonn (resized to
518x518) and VKITTI (square-cropped then resized to 518x518), KITTI frames
and depth stay at native per-drive resolution. Two reasons this is not
optional: (1) KITTI GT is SPARSE LiDAR projection -- resampling it (nearest
or otherwise) would shift, duplicate or destroy individual point
measurements, unlike dense synthetic (VKITTI) or dense RGB-D (Bonn) depth,
which tolerate a resize. (2) This is the exact population `BENCH-05b`
validated; changing its geometry would mean the one streaming-path
validation this project has no longer describes the population the endpoint
table actually uses. `write_recipe_manifest`/`RecipeManifest` never
materialise or stack pixels across videos, so raggedness across drives is a
non-issue for them (frames within one drive share one camera's resolution,
which is all `degrade_frame_range`'s per-video stacking needs).

**CONSEQUENCE FOR §1, recorded here rather than left a silent exception:**
§1 states degradation is applied "after resize to `data.image_size`
(518x518)". For KITTI, there is no resize -- degradation is applied at
NATIVE resolution instead. `degrade_frame_range` has no resolution
assumption baked in, so this needs no code change, only this note.

**OPEN RISK, not resolved by this builder, worth carrying into the pilot
pass (§8 step 7):** KITTI's native resolution (e.g. 1242x375) is not a
multiple of 14 or 16. VDA's own class handled this internally for
`BENCH-05b`; whether RAMEN's patch embedding accepts native KITTI dimensions
the same way is unverified -- check before assuming, not after the pilot
fails on a shape mismatch.

**MASK BAKES IN THE 80m DEPTH CAP** (`(depth > MIN_DEPTH) & (depth <
MAX_DEPTH)`, ported verbatim from `build_kitti_streaming_manifest.py`) --
UNLIKE VKITTI, whose mask deliberately keeps geometry past 80m valid and
lets the cap bite at scoring time instead. This is not a bug and not
changed here: `metrics.compute_depth_metrics` re-applies `gt < max_depth`
anyway (idempotent), and this is the exact mask convention `BENCH-05b`
validated. The three endpoint benchmarks therefore construct masks by THREE
DIFFERENT conventions (Bonn: no cap at all in gt, cap masked by construction
via 1e-3/no-cap; VKITTI: cap excluded from the mask entirely; KITTI: cap
baked into the mask) -- do not "unify" them later without re-deriving every
number that depends on the one being changed.

**Depth convention**: 16-bit PNG, raw/256.0 = metres (KITTI depth-completion
convention, matching VDA's own `eval.py`'s `factor=256.0`) -- a THIRD depth
scale (Bonn: /5000, VKITTI: /100).

**Pairing**: by matching FILENAME between a drive's `rgb/` and `depth/`
subdirectories (`depth_dir / rgb_path.name`), not timestamp association
(Bonn) or sorted positional index (VKITTI) -- a THIRD pairing convention.
Missing depth for an existing rgb file RAISES rather than silently dropping
the frame, matching `build_kitti_streaming_manifest.py` exactly: KITTI's
depth-completion GT is expected to exist for every frame that survived
VDA's own extraction, unlike Bonn's known systematic gaps.

Source data location, recorded because it is not a "canonical" TEST/TRAIN
path the way Bonn's and VKITTI's are: `KITTI_ROOT` points inside a specific
prior run's output directory
(`.../results/vda_clean_repro/benchmark_data/kitti/`), the actual extraction
VDA's own `dataset_extract_kitti.py` produced for `BENCH-01d`/`BENCH-05b`.
This is the real, load-bearing location of the validated 13-drive
population -- not an incidental byproduct to be cleaned up or relocated.

NOT yet wired into ``scripts/eval_streaming_npz.py``'s ``RECIPE_REGISTRY``
(``evaluation/recipe_registry.py``) -- sent for review first, matching Bonn's
and VKITTI's builders.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.video_clip_dataset import (degrade_frame_range, endpoint_degradation_seed,
                                     stratified_blur_assignment)
from evaluation.alignment_free import update_pixel_digest, write_recipe_manifest

# Captured once at process start (TOOL-09, provenance/plan_state.json), not lazily
# near the end of a long build -- a build's own recorded commit must describe the
# code that actually ran it, not whatever happens to be checked out in a possibly
# shared clone by the time this line executes minutes or hours later.
BUILDER_COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1]
).decode().strip()

BENCHMARK = "kitti"
MODE = "full"
DEFAULT_KITTI_ROOT = Path(
    "/mnt/DATA/DegradedDepth/RAMEN/results/vda_clean_repro/benchmark_data/kitti")
KITTI_ROOT = Path(os.environ.get("RAMEN_KITTI_ROOT") or DEFAULT_KITTI_ROOT)
MAX_DEPTH = 80.0
MIN_DEPTH = 1e-3


def resolve_root(root: Path | str | None = None) -> Path:
    """Resolve the on-disk KITTI root: an explicit argument wins, then
    ``$RAMEN_KITTI_ROOT``, then the crispwork constant. Read at CALL time, not
    import time, so the environment can be set after this module is imported
    (the frozen scorer imports ``evaluation.recipe_registry``, which imports this
    module, before any per-cell setup runs). Added 2026-09-18 so the frozen
    accuracy harness can score checkpoints on Isambard; with the variable unset
    the behaviour on crispwork is bit-identical."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get("RAMEN_KITTI_ROOT") or DEFAULT_KITTI_ROOT)


def drive_frame_pairs(root: Path, drive: str) -> list[tuple[Path, Path]]:
    """rgb/depth pairs for one drive, matched by FILENAME (not timestamp or sorted
    index). Missing depth for an existing rgb file raises -- matching
    build_kitti_streaming_manifest.py's own hard failure, since KITTI's
    depth-completion GT is expected to exist for every extracted frame."""
    rgb_dir, depth_dir = root / drive / "rgb", root / drive / "depth"
    rgb_files = sorted(rgb_dir.glob("*.png"))
    pairs = []
    for rgb_path in rgb_files:
        depth_path = depth_dir / rgb_path.name
        if not depth_path.exists():
            raise ValueError(f"build_kitti_recipe_manifest: missing depth GT for {rgb_path}")
        pairs.append((rgb_path, depth_path))
    return pairs


def _decode_pair(rgb_path: Path, depth_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise ValueError(f"build_kitti_recipe_manifest: failed to read {rgb_path}")
    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # (H, W, 3) in [0,1]
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise ValueError(f"build_kitti_recipe_manifest: failed to read {depth_path}")
    # copy_crop_files (VDA's own dataset_extract_kitti.py) truncated the rgb frame's
    # height/width to even numbers but left depth GT untouched -- crop depth/mask to
    # the frame's actual shape, matching build_kitti_streaming_manifest.py exactly.
    h, w = frame.shape[:2]
    depth = depth_raw[:h, :w].astype(np.float32) / 256.0
    mask = (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
    return frame.transpose(2, 0, 1), depth, mask  # frame -> (3, H, W)


def build_regenerate_range(
    condition: str,
    root: Path | None = None,
    severity: float = 1.0,
) -> Callable[[str, int, int], list]:
    """Build a ``regenerate_range(video_id, start, end)`` callback for either
    ``condition="clean"`` or ``condition="degraded"``. ``video_id`` is a bare
    drive name (e.g. ``"2011_09_26_drive_0002_sync"``). No resize -- frames stay
    at native per-drive resolution (see module docstring). Degradation (RGB
    only) uses the same shared :func:`data.video_clip_dataset.degrade_frame_range`
    / :func:`data.video_clip_dataset.stratified_blur_assignment` machinery as
    Bonn's and VKITTI's builders.
    """
    root = resolve_root(root)
    if condition not in ("clean", "degraded"):
        raise ValueError(f"condition must be 'clean' or 'degraded', got {condition!r}")

    cache: dict[str, list[tuple[Path, Path]]] = {}
    blur_by_video: dict[str, bool] = {}
    drives_cache: list[str] | None = None
    rng_state: dict[str, tuple[np.random.RandomState, int]] = {}
    pass_count: dict[str, int] = {}
    MAX_PASSES = 2

    def _pairs_for(video_id: str) -> list[tuple[Path, Path]]:
        nonlocal drives_cache
        if video_id not in cache:
            cache[video_id] = drive_frame_pairs(root, video_id)
            if not blur_by_video:
                if drives_cache is None:
                    drives_cache = sorted(p.name for p in root.iterdir() if p.is_dir())
                blur_by_video.update(stratified_blur_assignment(drives_cache))
        return cache[video_id]

    def regenerate_range(video_id: str, start: int, end: int) -> list:
        pairs = _pairs_for(video_id)
        rows = []
        for local_idx in range(start, end):
            rgb_path, depth_path = pairs[local_idx]
            frame, gt, mask = _decode_pair(rgb_path, depth_path)
            rows.append([frame, gt, mask])

        if condition == "degraded":
            if start == 0:
                pass_count[video_id] = pass_count.get(video_id, 0) + 1
                if pass_count[video_id] > MAX_PASSES:
                    raise ValueError(
                        f"build_kitti_recipe_manifest: video {video_id!r} requested a "
                        f"range starting at local frame 0 for the {pass_count[video_id]}th "
                        f"time -- RecipeManifest only ever makes {MAX_PASSES} passes "
                        "over a video (verify, then serve).")
                seed = endpoint_degradation_seed(BENCHMARK, video_id, MODE, severity)
                rng_state[video_id] = (np.random.RandomState(seed), 0)
            elif video_id not in rng_state:
                raise ValueError(
                    f"build_kitti_recipe_manifest: video {video_id!r} requested range "
                    f"[{start}, {end}) before any range starting at local frame 0.")
            rng, expected_start = rng_state[video_id]
            if start != expected_start:
                raise ValueError(
                    f"build_kitti_recipe_manifest: video {video_id!r} requested range "
                    f"[{start}, {end}) but expected to continue from {expected_start} "
                    "within the current pass.")
            # One frame at a time, not a batched stack: KITTI frames are NOT
            # guaranteed uniform shape frame-to-frame within a drive (the actual
            # reason build_kitti_streaming_manifest.py crops depth to each
            # frame's own (h, w) rather than a drive-wide constant -- see module
            # docstring). degrade_frame_range is already proven chunk-size-
            # independent down to single-frame decompositions, so this removes
            # the uniform-shape precondition at no cost (RAMEN Opus's review).
            for row in rows:
                single = torch.from_numpy(row[0]).unsqueeze(0)
                degraded, _profile = degrade_frame_range(single, rng, severity, blur_by_video[video_id])
                row[0] = degraded[0].numpy()
            rng_state[video_id] = (rng, end)
        # condition == "clean": row[0] is already numpy (set at construction above).

        return [tuple(r) for r in rows]

    return regenerate_range


def build_manifest(condition: str, out_path: Path, video_ids: list[str],
                   root: Path | None = None, severity: float = 1.0, chunk_size: int = 64) -> dict:
    root = resolve_root(root)
    regenerate_range = build_regenerate_range(condition, root=root, severity=severity)

    all_video_ids: list[str] = []
    all_frame_indices: list[float] = []
    digests: dict[str, str] = {}
    counts = {}
    for vid in video_ids:
        n = len(drive_frame_pairs(root, vid))
        counts[vid] = n
        d = __import__("hashlib").sha256()
        for lo in range(0, n, chunk_size):
            hi = min(lo + chunk_size, n)
            for frame, gt, mask in regenerate_range(vid, lo, hi):
                update_pixel_digest(d, frame, gt, mask)
        digests[vid] = d.hexdigest()
        all_video_ids.extend([vid] * n)
        all_frame_indices.extend(range(n))

    builder_commit = BUILDER_COMMIT

    write_recipe_manifest(
        out_path, all_video_ids, all_frame_indices, root=str(root), digests=digests,
        builder_commit=np.asarray([builder_commit]),
        builder_script=np.asarray(["scripts/build_kitti_recipe_manifest.py"]),
        condition=np.asarray([condition]),
        severity=np.asarray([severity if condition == "degraded" else 0.0]),
        mode=np.asarray([MODE]),
    )
    return {"video_ids": video_ids, "frame_counts": counts, "total_frames": sum(counts.values()),
            "builder_commit": builder_commit, "out_path": str(out_path)}


def main() -> int:
    print("DEPRECATED (DN-01 KITTI B/R fix, K78): this tool reads the VDA-extracted KITTI PNGs, which hold the TRUE RGB in swapped channel order (cv2.imread + BGR2RGB feeds B/R-swapped frames). It lists "
          "<clip>/rgb/ and has not been ported to the true-RGB root (<clip>/rgb_fixed/); any manifest it builds carries swapped frames. Use bench_joint's manifest of the fixed root instead.", file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, choices=("clean", "degraded"))
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    root = resolve_root(args.root)
    drives = sorted(p.name for p in root.iterdir() if p.is_dir())
    stats = build_manifest(args.condition, Path(args.out), drives, root=root, severity=args.severity)
    import json
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a recipe-locked manifest for the VKITTI-5scene endpoint population
(docs/DEGRADED_EVAL_PROTOCOL.md §3/§6), for either the ``clean`` or
``degraded_s<severity>`` condition.

VKITTI-5scene is Scene01/02/06/18/20, `clone`/`Camera_0` only, 2126 frames
total. It DELIBERATELY reads from this repo's TRAIN data root
(``/mnt/DATA/DegradedDepth/data/TRAIN/vkitti``, not TEST) because it is the
in-mix positive control per §2.1 -- not a held-out generalisation test, and
not a path mistake. Do not "fix" ``VKITTI_ROOT`` to point at a TEST tree;
that would silently swap this population for a different one the pre-
registration doesn't describe. The resolved root is recorded in every
manifest's own ``root`` field, so a later reader can always confirm which
tree a given manifest actually came from.

Unlike Bonn, VKITTI needs NO TIMESTAMP ASSOCIATION: RGB and depth share a
frame index by construction (``data/vkitti_dataset.py``'s own loader pairs
``sorted(rgb_dir.glob("rgb_*.jpg"))[i]`` with
``sorted(dep_dir.glob("depth_*.png"))[i]`` directly). Bonn's TUM tolerance
window and greedy dedup have no meaning here and are deliberately absent
rather than present-and-vacuous (RAMEN Opus's review). The one thing that
does carry over is truncating to frames with both a paired RGB and depth
file present (``n = min(len(rgb_files), len(depth_files))``), mirroring
Bonn's file-existence hard-stop for a positionally- rather than
timestamp-paired dataset.

MASK CONVENTION (verified against ``data/vkitti_dataset.py`` directly, not
assumed -- RAMEN Opus's review caught this before it became a silent
wrong-number bug): VKITTI depth is a 16-bit PNG in centimetres
(metres = value / 100); the sky / no-return sentinel is the uint16 max
(65535 cm, ~655.35 m). The mask excludes ONLY the sky sentinel and
non-positive values -- it deliberately KEEPS real geometry beyond 80 m (far
trees, hillsides) with its true depth. The 80 m ``max_depth`` cap (§3's
table) is therefore NOT part of the mask and does real work at SCORING time,
not construction time -- do not "fix" this mask to exclude >80m pixels, that
would silently change every VKITTI number. There is also a separate ``sky``
channel in the training dataset (target disparity 0 for the VDA-style sky
loss) that is NOT part of ``mask`` and does not travel into this manifest at
all -- the recipe manifest carries only (frame, gt, mask), which is correct
for scoring; sky simply has no home here, it is not silently dropped data.

The clip of GT depth to ``SKY_DEPTH_M`` (``np.clip(d_raw, 0.0, SKY_DEPTH_M)``)
is invisible to any metric (those pixels are masked out either way), but it
is NOT invisible to the locked digest -- ``gt`` is hashed as written, clipped
value included. This is a deliberate, locked choice, not an incidental
tidy-up: changing how out-of-range GT values are clipped later would
invalidate every VKITTI manifest already built under this recipe.

NOT yet wired into ``scripts/eval_streaming_npz.py``'s ``RECIPE_REGISTRY``
(``evaluation/recipe_registry.py``) -- sent for review first, matching Bonn's
builder.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("opencv-python is required to decode VKITTI depth PNGs.") from e

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

BENCHMARK = "vkitti"
MODE = "full"
IMAGE_SIZE = (518, 518)
DEFAULT_VKITTI_ROOT = Path("/mnt/DATA/DegradedDepth/data/TRAIN/vkitti")
# Host-portability override (2026-09-18): the accuracy harness must be runnable on
# Isambard, where this tree lives elsewhere. Unset -> the crispwork constant, so
# nothing changes there. The resolved root is still recorded in every manifest's
# own ``root`` field, so a reader can always tell which tree a manifest came from.
VKITTI_ROOT = Path(os.environ.get("RAMEN_VKITTI_ROOT") or DEFAULT_VKITTI_ROOT)
SCENES = ("Scene01", "Scene02", "Scene06", "Scene18", "Scene20")
VARIANT = "clone"
CAMERA = "Camera_0"
SKY_DEPTH_M = 655.0  # data/vkitti_dataset.py's own sentinel; real geometry sits below it


def resolve_root(root: Path | str | None = None) -> Path:
    """Resolve the on-disk VKITTI root: an explicit argument wins, then
    ``$RAMEN_VKITTI_ROOT``, then the crispwork constant. Read at CALL time, not
    import time, so the environment can be set after this module is imported
    (the frozen scorer imports ``evaluation.recipe_registry``, which imports this
    module, before any per-cell setup runs). Added 2026-09-18 so the frozen
    accuracy harness can score checkpoints on Isambard; with the variable unset
    the behaviour on crispwork is bit-identical."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get("RAMEN_VKITTI_ROOT") or DEFAULT_VKITTI_ROOT)


def _decode_vkitti_depth(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Could not read VKITTI depth: {path}")
    return arr.astype(np.float32) / 100.0  # cm -> m


def scene_frame_pairs(root: Path, scene: str, variant: str = VARIANT,
                      camera: str = CAMERA) -> list[tuple[Path, Path]]:
    """RGB/depth pairs for one scene, by sorted positional index -- no timestamp
    association (unlike Bonn): data/vkitti_dataset.py's own loader pairs the two
    sorted globs directly. Truncates to `min(len(rgb), len(depth))`, the
    positional equivalent of a file-existence hard-stop for a dataset that has
    no per-frame timestamp to associate on."""
    rgb_dir = root / scene / variant / "frames" / "rgb" / camera
    dep_dir = root / scene / variant / "frames" / "depth" / camera
    rgb_files = sorted(rgb_dir.glob("rgb_*.jpg"))
    dep_files = sorted(dep_dir.glob("depth_*.png"))
    n = min(len(rgb_files), len(dep_files))
    return list(zip(rgb_files[:n], dep_files[:n]))


def build_regenerate_range(
    condition: str,
    root: Path | None = None,
    severity: float = 1.0,
) -> Callable[[str, int, int], list]:
    """Build a ``regenerate_range(video_id, start, end)`` callback for either
    ``condition="clean"`` or ``condition="degraded"``. ``video_id`` is a bare
    scene name (e.g. ``"Scene02"``), matching docs/DEGRADED_EVAL_PROTOCOL.md's
    own naming for this population -- ``clone``/``Camera_0`` are fixed for the
    whole population, not part of the identifier.

    Decode matches ``data/vkitti_dataset.py`` exactly: depth is 16-bit PNG,
    centimetres -> metres (``/100.0``); mask excludes ONLY the sky sentinel
    (>=655m) and non-positive depth, keeping real geometry beyond 80m with its
    true (clipped-to-655m) value -- see this module's docstring for why. RGB is
    read as float32/255. Both are centre-cropped to a square (matching the
    dataset's own val-split centre crop, not a random train crop) then resized
    to IMAGE_SIZE. Degradation (RGB only) uses the same
    :func:`data.video_clip_dataset.degrade_frame_range` /
    :func:`stratified_blur_assignment` machinery as Bonn's builder, imported
    rather than re-implemented.
    """
    root = resolve_root(root)
    if condition not in ("clean", "degraded"):
        raise ValueError(f"condition must be 'clean' or 'degraded', got {condition!r}")

    cache: dict[str, list[tuple[Path, Path]]] = {}
    blur_by_video: dict[str, bool] = {}
    rng_state: dict[str, tuple[np.random.RandomState, int]] = {}
    pass_count: dict[str, int] = {}
    MAX_PASSES = 2

    def _pairs_for(video_id: str) -> list[tuple[Path, Path]]:
        if video_id not in cache:
            cache[video_id] = scene_frame_pairs(root, video_id)
            if not blur_by_video:
                blur_by_video.update(stratified_blur_assignment(list(SCENES)))
        return cache[video_id]

    def regenerate_range(video_id: str, start: int, end: int) -> list:
        pairs = _pairs_for(video_id)
        ih, iw = IMAGE_SIZE
        rows = []
        for local_idx in range(start, end):
            rgb_path, dep_path = pairs[local_idx]
            img = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32) / 255.0
            d_raw = _decode_vkitti_depth(dep_path)
            h, w = d_raw.shape
            off = max(0, w - h) // 2  # centre crop, matching the val split (not the random train one)
            img = img[:, off:off + h]
            d_raw = d_raw[:, off:off + h]

            valid = np.isfinite(d_raw) & (d_raw > 0) & (d_raw < SKY_DEPTH_M)
            d = np.clip(d_raw, 0.0, SKY_DEPTH_M)

            img_t = F.interpolate(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0),
                                  (ih, iw), mode="bilinear", align_corners=False).squeeze(0)
            d_t = F.interpolate(torch.from_numpy(d).unsqueeze(0).unsqueeze(0),
                                (ih, iw), mode="nearest").squeeze(0).squeeze(0)
            mask_t = F.interpolate(torch.from_numpy(valid.astype(np.float32)).unsqueeze(0).unsqueeze(0),
                                   (ih, iw), mode="nearest").squeeze(0).squeeze(0) > 0.5
            rows.append([img_t.numpy(), d_t.numpy(), mask_t.numpy()])

        if condition == "degraded":
            if start == 0:
                pass_count[video_id] = pass_count.get(video_id, 0) + 1
                if pass_count[video_id] > MAX_PASSES:
                    raise ValueError(
                        f"build_vkitti_recipe_manifest: video {video_id!r} requested a "
                        f"range starting at local frame 0 for the {pass_count[video_id]}th "
                        f"time -- RecipeManifest only ever makes {MAX_PASSES} passes "
                        "over a video (verify, then serve).")
                seed = endpoint_degradation_seed(BENCHMARK, video_id, MODE, severity)
                rng_state[video_id] = (np.random.RandomState(seed), 0)
            elif video_id not in rng_state:
                raise ValueError(
                    f"build_vkitti_recipe_manifest: video {video_id!r} requested range "
                    f"[{start}, {end}) before any range starting at local frame 0.")
            rng, expected_start = rng_state[video_id]
            if start != expected_start:
                raise ValueError(
                    f"build_vkitti_recipe_manifest: video {video_id!r} requested range "
                    f"[{start}, {end}) but expected to continue from {expected_start} "
                    "within the current pass.")
            # One frame at a time, not a batched stack -- see Bonn's builder for why
            # (real videos are not guaranteed uniform shape frame-to-frame; this
            # costs nothing since degrade_frame_range is already proven
            # chunk-size-independent down to single-frame decompositions).
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
        n = len(scene_frame_pairs(root, vid))
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
        builder_script=np.asarray(["scripts/build_vkitti_recipe_manifest.py"]),
        condition=np.asarray([condition]),
        severity=np.asarray([severity if condition == "degraded" else 0.0]),
        mode=np.asarray([MODE]),
    )
    return {"video_ids": video_ids, "frame_counts": counts, "total_frames": sum(counts.values()),
            "builder_commit": builder_commit, "out_path": str(out_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, choices=("clean", "degraded"))
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    stats = build_manifest(args.condition, Path(args.out), list(SCENES),
                           root=resolve_root(args.root), severity=args.severity)
    import json
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

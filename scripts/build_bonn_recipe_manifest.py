#!/usr/bin/env python3
"""Build a recipe-locked manifest for the Bonn RGB-D endpoint population
(docs/DEGRADED_EVAL_PROTOCOL.md §3/§6), for either the ``clean`` or
``degraded_s<severity>`` condition.

Reuses the already-verified TUM association logic developed this session
(.scratch/build_bonn_manifest.py, .scratch/count_bonn_association_v2.py --
windowed/binary-search matching against the naive O(n*m) approach that OOM'd
crispwork on ``rgbd_bonn_static``'s 10916x10000 pair) and the stratified blur
assignment pre-registered in docs/DEGRADED_EVAL_PROTOCOL.md §1.

Two populations, matching §3/§6: PRIMARY (25 sequences, excludes
``rgbd_bonn_static``) and STATIC (``rgbd_bonn_static`` alone) -- built and
written as separate manifests, since they are separate populations, not one
with an outlier flagged.

NOT yet wired into ``scripts/eval_streaming_npz.py``'s ``RECIPE_REGISTRY`` --
sent for review first (per this session's established process for every
piece of the recipe-locking mechanism). See ``build_regenerate_range``'s
docstring for the open question about how condition/severity should key into
the registry, which affects VKITTI/KITTI's builders too.
"""

from __future__ import annotations

import argparse
import os
import bisect
from pathlib import Path
import subprocess
import sys
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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

BENCHMARK = "bonn"
MODE = "full"
IMAGE_SIZE = (518, 518)
MAX_OFFSET = 0.02
DEFAULT_BONN_ROOT = Path("/mnt/DATA/DegradedDepth/data/TEST/bonn")
BONN_ROOT = Path(os.environ.get("RAMEN_BONN_ROOT") or DEFAULT_BONN_ROOT)
STATIC_VIDEO_ID = "rgbd_bonn_static"


def resolve_root(root: Path | str | None = None) -> Path:
    """Resolve the on-disk BONN root: an explicit argument wins, then
    ``$RAMEN_BONN_ROOT``, then the crispwork constant. Read at CALL time, not
    import time, so the environment can be set after this module is imported
    (the frozen scorer imports ``evaluation.recipe_registry``, which imports this
    module, before any per-cell setup runs). Added 2026-09-18 so the frozen
    accuracy harness can score checkpoints on Isambard; with the variable unset
    the behaviour on crispwork is bit-identical."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get("RAMEN_BONN_ROOT") or DEFAULT_BONN_ROOT)


def _read_tum_list(path: Path) -> list[tuple[float, str]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rows.append((float(parts[0]), parts[1]))
    return rows


def associate_sequence(seq_dir: Path, max_offset: float = MAX_OFFSET) -> list[tuple[float, str, str]]:
    """TUM associate.py's greedy nearest-offset, one-to-one matching, via binary-search
    windowed candidate generation (not the naive O(n*m) cross product, which OOM'd
    crispwork at ~86GB RSS on rgbd_bonn_static's 10916x10000 pair). A file-existence
    hard-stop runs BEFORE the tolerance check -- every one of Bonn's 26 sequences has
    depth.txt entries referencing files that don't exist on disk (2-15 for 25
    sequences, 906/10906 for rgbd_bonn_static) -- so a 'matched' pair is guaranteed
    decodable, not just within-tolerance.

    Returns ``(timestamp, rgb_relpath, depth_relpath)`` triples in chronological
    (rgb timestamp) order -- this triple sequence, plus ``seq_dir``, IS the recipe:
    the manifest never stores anything more than this.
    """
    rgb_list = _read_tum_list(seq_dir / "rgb.txt")
    depth_list = _read_tum_list(seq_dir / "depth.txt")
    exists_mask = [(seq_dir / fn).exists() for _, fn in depth_list]
    depth_ts = [dt for dt, _ in depth_list]

    candidates = []
    for i, (rt, rf) in enumerate(rgb_list):
        lo = bisect.bisect_left(depth_ts, rt - max_offset)
        hi = bisect.bisect_right(depth_ts, rt + max_offset)
        for j in range(lo, hi):
            if not exists_mask[j]:
                continue
            diff = abs(rt - depth_ts[j])
            if diff <= max_offset:
                candidates.append((diff, i, j))
    candidates.sort(key=lambda x: x[0])
    used_rgb, used_depth = set(), set()
    matches = []
    for diff, i, j in candidates:
        if i in used_rgb or j in used_depth:
            continue
        used_rgb.add(i)
        used_depth.add(j)
        matches.append((i, j))
    matches.sort(key=lambda x: x[0])  # chronological by rgb index/timestamp

    return [(rgb_list[i][0], rgb_list[i][1], depth_list[j][1]) for i, j in matches]


def build_regenerate_range(
    condition: str,
    root: Path | None = None,
    severity: float = 1.0,
    exclude_static: Optional[bool] = None,
) -> Callable[[str, int, int], list]:
    """Build a ``regenerate_range(video_id, start, end) -> [(frame, gt, mask), ...]``
    callback for either ``condition="clean"`` or ``condition="degraded"``.

    Decode/resize matches .scratch/build_bonn_manifest.py exactly: RGB read as
    float32/255, depth read as float32/5000 (TUM's metres-per-unit convention),
    bilinear resize for RGB, nearest for depth, mask = depth > 1e-3. Degradation
    (RGB only, per §1 -- GT/mask stay clean) uses :func:`degrade_frame_range`, which
    requires a persistent, video-scoped RNG advanced only by strictly increasing,
    contiguous ranges -- exactly the access pattern RecipeManifest's own verify/serve
    passes already guarantee (see degrade_frame_range's docstring for why this is a
    deliberately different draw decomposition from ``_degrade_clip``'s own whole-clip
    call, licensed by docs/DEGRADED_EVAL_PROTOCOL.md §4).

    Association is re-run per video on first access rather than cached across the
    whole benchmark up front -- association itself is cheap (timestamp lists, no
    pixels) and this keeps the callback's memory footprint at exactly one video's
    association result plus one in-flight range, not the whole benchmark's.

    ``evaluation.recipe_registry.RECIPE_REGISTRY`` (imported by
    ``scripts/eval_streaming_npz.py``, not defined there -- a non-frozen module so
    registering a builder never needs a ``benchmark_fidelity`` gate re-run) is keyed
    by ``--dataset`` alone (``dataset`` keeps meaning the dataset, not a
    dataset+condition composite -- ``--condition`` is threaded through to the
    factory as its one argument, matching this function's own ``condition``
    parameter position). Registering this builder means adding
    ``RECIPE_REGISTRY["bonn"] = lambda condition: build_regenerate_range(condition)``
    (defaults cover ``root``/``severity``) -- not done yet, this is still a draft
    pending review before it lands.
    """
    root = resolve_root(root)
    if condition not in ("clean", "degraded"):
        raise ValueError(f"condition must be 'clean' or 'degraded', got {condition!r}")

    cache: dict[str, list[tuple[float, str, str]]] = {}
    blur_by_video: dict[str, bool] = {}
    rng_state: dict[str, tuple[np.random.RandomState, int]] = {}  # video_id -> (rng, next_local_pos)
    pass_count: dict[str, int] = {}
    MAX_PASSES = 2  # RecipeManifest's own design: exactly one verify pass, then one serve
    # pass, per video, each restarting at local frame 0 -- pinning this count (rather than
    # treating every start==0 as an unconditionally valid restart) means a genuine
    # out-of-order caller that happens to land on 0 a third time is still caught, instead
    # of silently accepted as "just another fresh pass".

    def _sequence_for(video_id: str) -> list[tuple[float, str, str]]:
        if video_id not in cache:
            cache[video_id] = associate_sequence(root / video_id)
            if not blur_by_video:
                seqs = sorted(d.name for d in root.iterdir() if d.is_dir())
                primary = [s for s in seqs if s != STATIC_VIDEO_ID]
                blur_by_video.update(stratified_blur_assignment(primary))
                blur_by_video[STATIC_VIDEO_ID] = True  # §1: its own 1/1, round-up(0.7*1)=1 -> blurred
        return cache[video_id]

    def regenerate_range(video_id: str, start: int, end: int) -> list:
        matched = _sequence_for(video_id)
        seq_dir = root / video_id
        rows = []
        for local_idx in range(start, end):
            _, rf, df = matched[local_idx]
            img = np.array(Image.open(seq_dir / rf).convert("RGB")).astype(np.float32) / 255.0
            d = np.array(Image.open(seq_dir / df)).astype(np.float32) / 5000.0
            ih, iw = IMAGE_SIZE
            img_t = F.interpolate(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0),
                                  (ih, iw), mode="bilinear", align_corners=False).squeeze(0)
            d_t = F.interpolate(torch.from_numpy(d).unsqueeze(0).unsqueeze(0),
                                (ih, iw), mode="nearest").squeeze(0).squeeze(0)
            mask_t = d_t > 1e-3
            rows.append([img_t.numpy(), d_t.numpy(), mask_t.numpy()])

        if condition == "degraded":
            if start == 0:
                # RecipeManifest re-traverses a video from local frame 0 exactly twice
                # per manifest use (once to verify, once to serve) -- each such
                # restart is a fresh, independent pass over the SAME deterministic
                # seed, and must reproduce the same pixels, not continue a prior
                # pass's stream position. Capped at MAX_PASSES so a genuine
                # out-of-order call that happens to land on 0 a third time still
                # raises, instead of being silently accepted as "just another pass".
                pass_count[video_id] = pass_count.get(video_id, 0) + 1
                if pass_count[video_id] > MAX_PASSES:
                    raise ValueError(
                        f"build_bonn_recipe_manifest: video {video_id!r} requested a "
                        f"range starting at local frame 0 for the {pass_count[video_id]}th "
                        f"time -- RecipeManifest only ever makes {MAX_PASSES} passes "
                        "over a video (verify, then serve); a further restart means "
                        "this callback is being driven by something other than "
                        "RecipeManifest's own verify/serve contract.")
                seed = endpoint_degradation_seed(BENCHMARK, video_id, MODE, severity)
                rng_state[video_id] = (np.random.RandomState(seed), 0)
            elif video_id not in rng_state:
                raise ValueError(
                    f"build_bonn_recipe_manifest: video {video_id!r} requested range "
                    f"[{start}, {end}) before any range starting at local frame 0 -- "
                    "a pass over a video must begin at 0.")
            rng, expected_start = rng_state[video_id]
            if start != expected_start:
                raise ValueError(
                    f"build_bonn_recipe_manifest: video {video_id!r} requested range "
                    f"[{start}, {end}) but degrade_frame_range's shared RNG expects to "
                    f"continue from {expected_start} within the current pass -- "
                    "regenerate_range must be called with strictly increasing, "
                    "contiguous ranges within one pass over a video (exactly what "
                    "RecipeManifest's verify and serve passes each do); an "
                    "out-of-order call would silently desync the degradation RNG "
                    "stream.")
            # One frame at a time, not a batched stack: real videos are not
            # guaranteed uniform shape frame-to-frame (KITTI's own builder found
            # this the hard way in a prior draft -- RAMEN Opus's review), and
            # degrade_frame_range is already proven chunk-size-independent down
            # to single-frame decompositions, so this costs nothing.
            for row in rows:
                single = torch.from_numpy(row[0]).unsqueeze(0)  # (1, C, H, W)
                degraded, _profile = degrade_frame_range(single, rng, severity, blur_by_video[video_id])
                row[0] = degraded[0].numpy()
            rng_state[video_id] = (rng, end)
        # condition == "clean": row[0] is already numpy (set at construction above).

        return [tuple(r) for r in rows]

    return regenerate_range


def build_manifest(condition: str, out_path: Path, video_ids: list[str],
                   root: Path | None = None, severity: float = 1.0, chunk_size: int = 64) -> dict:
    """Run regenerate_range over every frame of every named video (in the SAME
    strictly-increasing per-video order the read side will use), compute each
    video's locked digest via the shared update_pixel_digest helper, and write the
    manifest. Returns build stats for the caller to log/record."""
    root = resolve_root(root)
    regenerate_range = build_regenerate_range(condition, root=root, severity=severity)

    all_video_ids: list[str] = []
    all_frame_indices: list[float] = []
    digests: dict[str, str] = {}
    counts = {}
    for vid in video_ids:
        matched = associate_sequence(root / vid)
        n = len(matched)
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
        builder_script=np.asarray(["scripts/build_bonn_recipe_manifest.py"]),
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
    ap.add_argument("--population", choices=("primary", "static", "both"), default="both",
                    help="rgbd_bonn_static (10000 frames, excluded from primary aggregation) is "
                         "the single worst case in this benchmark's memory footprint -- building it "
                         "as its own invocation means a failure there does not block or invalidate "
                         "the 25-sequence primary build.")
    ap.add_argument("--out-primary", help="required if --population is 'primary' or 'both'")
    ap.add_argument("--out-static", help="required if --population is 'static' or 'both'")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    if args.population in ("primary", "both") and not args.out_primary:
        ap.error("--out-primary is required for --population primary/both")
    if args.population in ("static", "both") and not args.out_static:
        ap.error("--out-static is required for --population static/both")

    root = resolve_root(args.root)
    seqs = sorted(d.name for d in root.iterdir() if d.is_dir())
    primary = [s for s in seqs if s != STATIC_VIDEO_ID]

    results = {}
    if args.population in ("primary", "both"):
        results["primary"] = build_manifest(args.condition, Path(args.out_primary), primary,
                                            root=root, severity=args.severity)
    if args.population in ("static", "both"):
        results["static"] = build_manifest(args.condition, Path(args.out_static), [STATIC_VIDEO_ID],
                                           root=root, severity=args.severity)

    import json
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a recipe-locked manifest for the KITTI-val single-frame supplement
(docs/DEGRADED_EVAL_PROTOCOL.md §3/§6), for either the ``clean`` or
``degraded_s1.0`` condition.

This is the "same 20 frames" §3 refers to as the trajectory layer's own
`scripts/eval_dav3_streaming.py::eval_kitti` population -- traced to its
actual source (RAMEN Opus's review): there is no separate config or wrapper,
`eval_dav3_streaming.py` reads `--kitti` (default
``.../depth_selection/val_selection_cropped``) and passes `--max-clips`
(the matrix evals used 20, per `docs/MATRIX_INTERIM_FINDINGS.md` §2.2's own
"the --max-clips 20 evals") straight to `eval_kitti`'s `max_frames`. The
population is therefore the first 20 of
``val_selection_cropped/image/*.png`` and ``groundtruth_depth/*.png``,
PAIRED POSITIONALLY -- matching `eval_kitti`'s own
``zip(sorted(image glob), sorted(groundtruth_depth glob))`` exactly, found
necessary by actually running the round-trip check against real data, not
assumed: the two directories' filenames do NOT share a naming pattern at
all (``image/`` files are named ``..._image_<idx>_image_0N.png``,
``groundtruth_depth/`` files ``..._groundtruth_depth_<idx>_image_0N.png``),
so filename matching would raise (or, worse, silently pair nothing) rather
than reproduce this population.

**A FOURTH mask convention** (verified directly against `eval_kitti`'s own
code, not assumed to match KITTI-13drive's): mask = `depth > 1e-3` with NO
upper cap at all -- neither Bonn's uncapped mask (same as this one, as it
happens) nor VKITTI's cap-excluded-but-clipped-gt convention nor
KITTI-13drive's cap-baked-into-the-mask convention. Depth scale is `/256.0`
(matching KITTI-13drive's convention -- same underlying KITTI depth-
completion format). Resize IS applied here (bilinear RGB / nearest depth to
518x518), unlike KITTI-13drive's deliberate no-resize decision -- NOT
because this population's GT is somehow less sparse than KITTI-13drive's
(it isn't: `groundtruth_depth` here is the same depth-completion sparse
LiDAR projection, sampled differently). The actual reason is comparability:
`eval_kitti` resizes, the trajectory-layer KITTI-val numbers this supplement
continues were produced that way, and matching the convention those numbers
were already produced under is exactly the same rule that keeps
KITTI-13drive at native resolution -- the rule points opposite ways for the
two populations only because their existing conventions already differ, not
because one population's GT is sparse and the other's isn't. Honest cost,
inherited from `eval_kitti` rather than introduced here: resampling sparse
depth with nearest interpolation shifts and duplicates individual point
measurements, degrading this GT relative to its source -- one more reason
this supplement column carries no streaming, temporal, or precision claim.

**Population identity is pinned, not re-derived.** A sorted-glob truncation
is fragile exactly the way §3's "Three names, not one" subsection warns
about: add a file to either ``image/`` or ``groundtruth_depth/`` and "the
same 20 frames" silently becomes a different 20 frames (or a different
pairing, since the two lists are matched positionally). The resolved 20
image filenames AND the resolved 20 depth filenames are therefore recorded
explicitly in the manifest's own ``source_image_filenames``/
``source_depth_filenames`` fields (write_recipe_manifest's ``**extra_arrays``)
-- the population is identified by WHICH files these are, not by the rule
that selected them.

**Each of the 20 frames is its own one-frame "video"** (`video_id` = the
frame's filename stem) -- KITTI-val has no video structure (§3: "n=20
independent frames"), and using :class:`evaluation.alignment_free.RecipeManifest`
here is not for memory (n=20 single frames carry zero memory risk) but for
artifact-type consistency: all four Stage-2 populations verify the same way,
rather than three RecipeManifest artifacts and one FrameManifest artifact.

**KITTI-val is a DEGENERATE case of §1's one-profile-per-video invariant,
stated explicitly here rather than left for a reader to notice.** "The
profile persists for the whole video" means, for a one-frame video,
persisting for exactly one frame -- there is no multi-frame window for a
profile to persist ACROSS. §1's sustained-domain-shift rationale ("a
robustness claim is about sustained domain shift... real lighting/focus does
not redraw every quarter-second") does not apply to a single independent
frame the way it does to a video. This is not a defect in the
population -- it is a property of a single-frame benchmark -- but
KITTI-val's degraded numbers must NOT be read as evidence about sustained
degradation robustness; that claim is what the video benchmarks (VKITTI,
Bonn, KITTI-13drive) exist to support.

**Stratified blur: 14 of 20 (§1, amended 2026-08-30 to apply to every
population this document builds, including supplements) -- no tie
(`0.7*20=14.0` exactly), first fourteen by sorted filename.**

NOT yet wired into ``scripts/eval_streaming_npz.py``'s ``RECIPE_REGISTRY``
(``evaluation/recipe_registry.py``) -- sent for review first, matching the
other three builders.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
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
    raise ImportError("opencv-python is required to decode KITTI-val depth PNGs.") from e

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

BENCHMARK = "kitti_val"
MODE = "full"
IMAGE_SIZE = (518, 518)
KITTI_VAL_ROOT = Path(
    "/mnt/DATA/DegradedDepth/data/TEST/kitti/depth_selection/val_selection_cropped")
N_FRAMES = 20
MIN_DEPTH = 1e-3

# e.g. "2011_09_26_drive_0002_sync_image_0000000005_image_02.png" ->
# ("2011_09_26_drive_0002_sync", "0000000005", "02")
_IMAGE_NAME_RE = re.compile(r"^(.+)_image_(\d+)_image_(\d+)\.png$")
_DEPTH_NAME_RE = re.compile(r"^(.+)_groundtruth_depth_(\d+)_image_(\d+)\.png$")


def _identity(pattern: re.Pattern, filename: str) -> tuple[str, str, str]:
    m = pattern.match(filename)
    if m is None:
        raise ValueError(
            f"build_kitti_val_recipe_manifest: {filename!r} does not match the expected "
            f"val_selection_cropped naming pattern ({pattern.pattern!r}).")
    return m.groups()


def resolved_pairs(root: Path = KITTI_VAL_ROOT, n: int = N_FRAMES) -> list[tuple[str, str]]:
    """First n (image_filename, depth_filename) pairs -- the population's actual
    identity. PAIRED POSITIONALLY, matching eval_kitti's own
    `zip(sorted(image glob), sorted(groundtruth_depth glob))` exactly: the two
    directories' filenames do NOT share a naming pattern at all (image files
    are named `..._image_<idx>_image_0N.png`, groundtruth_depth files
    `..._groundtruth_depth_<idx>_image_0N.png`) -- found by actually running the
    round-trip check against real data, not by inspection; an earlier draft of
    this function assumed filename matching and raised on every frame.

    **Positional pairing is exactly Bonn's original defect** (RAMEN Opus's
    review) -- Bonn's directory-position pairing produced interior, cumulative
    misalignment once a single frame went missing partway through a sequence.
    It is very likely fine here (both directories carry the same drive name
    and frame number in the same relative position, and the differing token
    -- "image" vs "groundtruth_depth" -- is constant within each directory, so
    both sort by (drive, frame, camera) and the zip aligns), but "probably, by
    an argument about sort keys" is exactly the standard that was rejected for
    Bonn. So this is CHECKED, not assumed: every pair's drive/frame/camera
    identity is extracted from BOTH filenames and asserted to correspond,
    raising with both names on any mismatch -- 20 pairs, negligible cost,
    turning an inherited assumption into an enforced property. If this ever
    raises, KITTI-val's trajectory-layer numbers have Bonn's problem too, and
    that is a finding, not a build error.

    Callers should record both filename lists, not just n and the root, since
    a later change to either directory's contents would otherwise silently
    change which 20 frames this population means.
    """
    img_files = sorted((root / "image").glob("*.png"))
    depth_files = sorted((root / "groundtruth_depth").glob("*.png"))
    if len(img_files) < n or len(depth_files) < n:
        raise ValueError(f"{root} has {len(img_files)} image files and {len(depth_files)} "
                         f"groundtruth_depth files, need {n} of each")
    pairs = []
    for i in range(n):
        img_name, depth_name = img_files[i].name, depth_files[i].name
        img_id = _identity(_IMAGE_NAME_RE, img_name)
        depth_id = _identity(_DEPTH_NAME_RE, depth_name)
        if img_id != depth_id:
            raise ValueError(
                f"build_kitti_val_recipe_manifest: positional pair at index {i} does not "
                f"correspond -- image {img_name!r} identifies as {img_id}, but depth "
                f"{depth_name!r} identifies as {depth_id}. Positional pairing "
                "(zip of two independently sorted globs) is not safe for this population.")
        pairs.append((img_name, depth_name))
    return pairs


def _video_id(image_filename: str) -> str:
    return Path(image_filename).stem


def _decode_pair(root: Path, image_filename: str,
                 depth_filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_path, depth_path = root / "image" / image_filename, root / "groundtruth_depth" / depth_filename
    img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    d_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d_raw is None:
        raise ValueError(f"build_kitti_val_recipe_manifest: failed to read {depth_path}")
    depth = d_raw.astype(np.float32) / 256.0
    ih, iw = IMAGE_SIZE
    img_t = F.interpolate(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0),
                          (ih, iw), mode="bilinear", align_corners=False).squeeze(0)
    d_t = F.interpolate(torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
                        (ih, iw), mode="nearest").squeeze(0).squeeze(0)
    mask_t = d_t > MIN_DEPTH  # no upper cap -- see module docstring
    return img_t.numpy(), d_t.numpy(), mask_t.numpy()


def build_regenerate_range(
    condition: str,
    root: Path = KITTI_VAL_ROOT,
    severity: float = 1.0,
) -> Callable[[str, int, int], list]:
    """Build a ``regenerate_range(video_id, start, end)`` callback. Each
    ``video_id`` is a one-frame "video" (a filename stem), so every real call
    is ``(video_id, 0, 1)`` -- degenerate but not special-cased: the same
    pass-counting/profile machinery as the other three builders applies
    unchanged.
    """
    if condition not in ("clean", "degraded"):
        raise ValueError(f"condition must be 'clean' or 'degraded', got {condition!r}")

    pairs = resolved_pairs(root)
    by_video_id = {_video_id(img_fn): (img_fn, depth_fn) for img_fn, depth_fn in pairs}
    # stratified_blur_assignment itself sorts internally, but pass sorted ids
    # explicitly rather than relying on dict insertion order happening to match
    # the sorted glob that produced it -- an incidental property, not a guarantee.
    blur_by_video = stratified_blur_assignment(sorted(by_video_id))
    pass_count: dict[str, int] = {}
    MAX_PASSES = 2

    def regenerate_range(video_id: str, start: int, end: int) -> list:
        if (start, end) != (0, 1):
            raise ValueError(
                f"build_kitti_val_recipe_manifest: video {video_id!r} requested range "
                f"[{start}, {end}) -- every KITTI-val video has exactly one frame, so "
                "the only valid range is (0, 1).")
        img_fn, depth_fn = by_video_id[video_id]
        frame, gt, mask = _decode_pair(root, img_fn, depth_fn)

        if condition == "degraded":
            pass_count[video_id] = pass_count.get(video_id, 0) + 1
            if pass_count[video_id] > MAX_PASSES:
                raise ValueError(
                    f"build_kitti_val_recipe_manifest: video {video_id!r} requested a "
                    f"pass for the {pass_count[video_id]}th time -- RecipeManifest only "
                    f"ever makes {MAX_PASSES} passes over a video (verify, then serve).")
            seed = endpoint_degradation_seed(BENCHMARK, video_id, MODE, severity)
            rng = np.random.RandomState(seed)
            single = torch.from_numpy(frame).unsqueeze(0)
            degraded, _profile = degrade_frame_range(single, rng, severity, blur_by_video[video_id])
            frame = degraded[0].numpy()

        return [(frame, gt, mask)]

    return regenerate_range


def build_manifest(condition: str, out_path: Path, root: Path = KITTI_VAL_ROOT,
                   severity: float = 1.0) -> dict:
    pairs = resolved_pairs(root)
    video_ids = [_video_id(img_fn) for img_fn, _ in pairs]
    regenerate_range = build_regenerate_range(condition, root=root, severity=severity)

    digests: dict[str, str] = {}
    for vid in video_ids:
        d = __import__("hashlib").sha256()
        for frame, gt, mask in regenerate_range(vid, 0, 1):
            update_pixel_digest(d, frame, gt, mask)
        digests[vid] = d.hexdigest()

    builder_commit = BUILDER_COMMIT

    write_recipe_manifest(
        out_path, video_ids, [0.0] * len(video_ids), root=str(root), digests=digests,
        builder_commit=np.asarray([builder_commit]),
        builder_script=np.asarray(["scripts/build_kitti_val_recipe_manifest.py"]),
        condition=np.asarray([condition]),
        severity=np.asarray([severity if condition == "degraded" else 0.0]),
        mode=np.asarray([MODE]),
        source_image_filenames=np.asarray([p[0] for p in pairs]),
        source_depth_filenames=np.asarray([p[1] for p in pairs]),
    )
    return {"video_ids": video_ids, "total_frames": len(video_ids),
            "builder_commit": builder_commit, "out_path": str(out_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, choices=("clean", "degraded"))
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(KITTI_VAL_ROOT))
    args = ap.parse_args()

    stats = build_manifest(args.condition, Path(args.out), root=Path(args.root),
                           severity=args.severity)
    import json
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

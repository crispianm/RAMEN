"""Extension point for a ``--model`` value's own (checkpoint, device, dtype_str,
window) -> (predict_frame, reset, checkpoint_sha256) builder -- same registry
pattern as ``evaluation.recipe_registry.RECIPE_REGISTRY``.

Deliberately NOT defined inside ``scripts/eval_streaming_npz.py`` itself, even
though that is where it is consumed: that file is one of the four frozen under
the ``benchmark_fidelity`` gate (``docs/BENCHMARK_PROTOCOL.md``'s ``BENCH-06``).
Extracted 2026-08-30 (TOOL-16, RAMEN Opus's call) for the identical reason
``RECIPE_REGISTRY`` was already extracted: a gate re-run cannot detect anything
about which model class a checkpoint dispatches to -- ``build_ramen``'s own
dispatch logic is never exercised by the gate's reference triples in a way that
would catch a routing mistake -- so adding a new architecture's routing is, by
construction, invisible to that check. Registering a new model wrapper is
therefore a plain, non-frozen commit -- reviewed and covered by its own tests,
not gated -- rather than triggering a gate re-run whose result was already
knowable in advance. This is the LAST time this specific gate reason should
apply to a model-routing change: every future wrapper is now a change here,
never to the frozen file.
"""

from __future__ import annotations

import os
import sys
from typing import Callable

import numpy as np
import torch

from evaluation.alignment_free import sha256_file


def build_ramen(checkpoint: str, device: str, dtype_str: str, window: int):
    """Model-registry entry for RAMEN. Returns (predict_frame, reset, checkpoint_sha256).

    Imports infer.py lazily: selecting a different --model (e.g. vda) should not
    require RAMEN's own model stack (models/, vendor/depth_anything_3/) to be
    importable at all, the same reasoning build_vda/build_daac already follow for
    their own baseline-specific imports.

    Dispatches on the loaded model's own type/API: VDANet (models/vda_net.py) is
    checked FIRST and explicitly (TOOL-16, 2026-08-30) -- it has no incremental
    per-frame API of its own (its forward() is a whole-clip method), so it needs
    StreamingVDA's weight-transplant wrapper, not a duck-typed fallback. OVDANet
    exposes step_stream (its cache-based streaming contract, see StreamingOVDA);
    everything else is D3Net's per_frame_encode API (StreamingDepth). All three
    wrap the model with the same (reset, step) streaming contract this script
    needs, so the architecture difference is invisible past this function.
    """
    from infer import StreamingDepth, StreamingOVDA, StreamingVDA, load
    from models.vda_net import VDANet

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[dtype_str]
    model, _ = load(checkpoint, device)
    if isinstance(model, VDANet):
        runner = StreamingVDA(model, device, dtype)
    elif hasattr(model, "step_stream"):
        runner = StreamingOVDA(model, device, dtype)
    else:
        runner = StreamingDepth(model, window, device, dtype)
    return runner.step, runner.reset, sha256_file(checkpoint)


def build_vda(checkpoint: str, device: str, dtype_str: str, window: int):
    """Model-registry entry for VDA (Video-Depth-Anything), the Tier-2 baseline
    BENCH-05b needs. Not vendored into this repo -- imports its real streaming
    class directly from its own checkout, pointed at by the ``VDA_ROOT``
    environment variable (this registry's signature is fixed, so unlike
    scripts/dump_predictions.py's own CLI there is no flag to carry this).

    ``video_depth_anything/video_depth_stream.py``'s ``VideoDepthAnything`` has a
    genuine single-frame incremental API (``infer_video_depth_one``) with an
    internal INFER_LEN=32/OVERLAP=10/INTERP_LEN=8 sliding window over cached
    hidden states, but no ``reset()`` method -- ``reset`` here clears the four
    attributes that hold that state, matching what ``__init__`` sets them to.
    Its own vits/vitb/vitl config dicts live only in its unmodified
    ``benchmark/infer/infer.py``; ``vits`` is hardcoded here since that is the
    only checkpoint BENCH-01d/BENCH-05b use.

    Its raw output is RELU'd and disparity-like -- confirmed by VDA's own
    eval.py treating raw inference output identically to disparity in its
    per-video least-squares alignment, which cannot be reused here since a
    per-video fit needs future frames and would defeat the point of a causal
    Tier-2 score. Instead this dumps ``pred = 1 / raw_disparity`` so this
    repo's own per-frame affine fit (which computes ``1/pred`` internally,
    assuming pred is depth-like) recovers the correct disparity for a
    genuinely causal, single-frame fit -- pass ``--alignment-scope per_frame``
    when scoring.

    Input convention: ``predict_frame`` expects ``frame_hwc`` HWC, ~[0,255] --
    the convention BENCH-05b's own Tier-2 gate manifest
    (scripts/build_kitti_streaming_manifest.py) stores frames in, verified by
    reading that builder directly (no transpose, no /255.0 division). This is
    NOT the convention the shared endpoint-grid manifests use (build_kitti/
    bonn/vkitti_recipe_manifest.py all store CHW, [0,1] -- verified the same
    way) -- infer.py's StreamingVDA is the entry point for THOSE manifests, and
    converts explicitly rather than reusing this function's convention
    unchanged, since a silent misscale here would not crash, just produce
    plausible-looking wrong numbers for every VDA endpoint-grid cell.
    """
    vda_root = os.environ.get("VDA_ROOT")
    if not vda_root:
        raise ValueError("VDA_ROOT must be set to the Video-Depth-Anything checkout for --model vda")
    sys.path.insert(0, vda_root)
    from video_depth_anything.video_depth_stream import VideoDepthAnything

    model = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    fp32 = dtype_str == "fp32"

    def predict_frame(frame_hwc: torch.Tensor) -> torch.Tensor:
        raw_disparity = model.infer_video_depth_one(frame_hwc.cpu().numpy(), device=device, fp32=fp32)
        pred_depth = 1.0 / np.clip(raw_disparity, 1e-3, None)
        return torch.from_numpy(pred_depth).to(device)

    def reset() -> None:
        model.transform = None
        model.frame_id_list = []
        model.frame_cache_list = []
        model.id = -1

    return predict_frame, reset, sha256_file(checkpoint)


# Extension point for the streaming (Tier 2) baselines BENCH-05b needs, and for any
# new in-repo model architecture: register a builder here with the same
# (checkpoint, device, dtype_str, window) -> (predict_frame, reset, checkpoint_sha256)
# signature. scripts/eval_streaming_npz.py only ever imports this dict -- nothing
# frozen needs to change to add an entry.
MODEL_REGISTRY: dict[str, Callable[[str, str, str, int], tuple]] = {
    "ramen": build_ramen,
    "vda": build_vda,
}

# TEACH-01 (2026-09-18): large teacher models for the zero-shot degraded-robustness
# screen (docs/RAMEN_V3_PROGRAMME.md T1). Builders live in evaluation/teacher_models.py
# (heavy imports local to each builder); registered here so the frozen scorer's
# --model flag reaches them without any change to the frozen file.
from evaluation.teacher_models import TEACHER_BUILDERS  # noqa: E402

MODEL_REGISTRY.update(TEACHER_BUILDERS)

# PRUNE-01 (2026-09-21): released / pruned relative VDA-S as a plain ``VideoDepthAnything`` state
# dict, scored on the shared ENDPOINT-GRID manifests (CHW, [0,1]). ``"vda"`` above cannot do that:
# build_vda expects its own HWC/[0,255] Tier-2 manifest and crashes in cv2.resize on these.
from evaluation.teacher_models import build_vda_stream  # noqa: E402

MODEL_REGISTRY["vda_s"] = build_vda_stream("vits", metric=False)

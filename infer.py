#!/usr/bin/env python3
"""Webcam streaming depth — run a RAMEN video-depth checkpoint live, as fast as it goes.

The model splits `encode` (frozen DINOv3 ViT — the cost) → `apply_temporal` → `decode`. The
streaming trick: cache each frame's encoder maps in a length-T ring buffer and run the backbone
ONCE per frame, then the (cheap) temporal recurrence over the window + decode only the newest
frame. Per-frame cost ≈ one ViT pass regardless of window T — vs T passes if you re-ran the whole
model on a raw T-frame clip every step.

Perf knobs: inference_mode, bf16 autocast, TF32, cudnn.benchmark, threaded capture, optional
torch.compile of the backbone. `--selftest` exercises the pipeline on synthetic frames (no camera).

  python infer.py --ckpt runs/camp_gru_renorm/final.pt          # live webcam
  python infer.py --ckpt runs/camp_gru_renorm/final.pt --selftest  # verify + FPS, no camera
"""
from __future__ import annotations
import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from models import build_model

# ponytail: global perf flags — TF32 + autotuned cudnn. Set once, helps every conv/matmul.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def _pad_to_patch_multiple(x: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, int, int]:
    """TOOL-08 / docs/DEGRADED_EVAL_PROTOCOL.md §3: RAMEN's and OVDA's ViT-family patch
    embeddings require the last two dims to be exact multiples of the backbone's own
    patch size; KITTI-13drive's native resolution (e.g. 374x1242) is not, at either
    D3Net's DINOv2 (14) or DINOv3 (16) patch size. ``patch_size`` is always read from
    the model that will consume ``x`` (``StreamingDepth`` passes ``self.m.PATCH_SIZE``),
    never hardcoded here -- a literal 14 would silently mispad a DINOv3 (16) checkpoint
    into a wrong-sized token grid, exactly the failure mode TOOL-10 showed can pass
    without an error.

    Pads with replicated edge pixels, bottom-right only, so the native content occupies
    rows [0:native_h) and columns [0:native_w) of the padded tensor unchanged -- cropping
    back after decoding is therefore an unconditional slice, never a computed one. A
    no-op (returns ``x`` unchanged, the padded branch never entered) when both dims are
    already multiples of ``patch_size``, so every already-compliant resolution --
    including every number produced before this function existed -- is untouched."""
    h, w = x.shape[-2:]
    pad_h, pad_w = (-h) % patch_size, (-w) % patch_size
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    leading = x.shape[:-2]
    flat = x.reshape(-1, 1, h, w)
    flat = torch.nn.functional.pad(flat, (0, pad_w, 0, pad_h), mode="replicate")
    return flat.reshape(*leading, h + pad_h, w + pad_w), h, w


def load(ckpt: str, device: str):
    import os

    from models import load_d3_checkpoint
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    # A checkpoint's embedded cfg pins external-repo paths (e.g. model.ovda_repo) to the
    # host it was trained on. OVDA_REPO overrides that for evaluation on a different host,
    # mirroring eval_streaming_npz.py's own VDA_ROOT override for the same class of problem.
    ovda_repo = os.environ.get("OVDA_REPO")
    if ovda_repo and str(ck["cfg"].get("model", {}).get("arch")) == "ovda":
        ck["cfg"]["model"]["ovda_repo"] = ovda_repo
    cfg = ck["cfg"]
    if cfg.get("model", {}).get("shared_temporal"):                # trainable_ema carries all of temp2; st_temp2_init is a training-host path (same override as bench_joint.load_ours)
        cfg = {**cfg, "model": {**cfg["model"], "st_temp2_init": None}}
    model = build_model(cfg).eval().to(device)
    load_d3_checkpoint(model, ck, depth_only=True)                 # streaming is depth-only: ramen.pt's display head is ignored ON PURPOSE (--save-rgb uses load_ramen)
    sz = ck["cfg"].get("data", {}).get("image_size", 518)
    res = int(sz[0] if isinstance(sz, (list, tuple)) else sz)
    return model, res


STREAMING_KNOWN_ISSUE = ("Known issue: streaming mode is not yet supported for ramen.pt / ramen_l.pt (SharedTemporalNet). "
                         "Clip mode reproduces the tables; use `--save-rgb DIR` (offline) or the scorer / bench_joint paths instead.")


class StreamingDepth:
    """Per-frame encode → ring-buffered token window → temporal → decode-newest-only.

    D3Net API: encode(frames (B,C,T,H,W)) → [(tok (B,T,N,C), cam)] per scale;
    apply_temporal(feats, H, W); decode(feats, H, W) → (B,1,T,H,W).
    """

    def __init__(self, model, window: int, device: str, dtype):
        if window < 1:
            raise ValueError("window must be at least 1")
        from models.shared_temporal import SharedTemporalNet
        if isinstance(model, SharedTemporalNet):
            raise NotImplementedError(STREAMING_KNOWN_ISSUE)
        self.m, self.win, self.dev, self.dt = model, window, device, dtype
        self.reset()

    def reset(self) -> None:
        """Forget all temporal context, normally at a video boundary."""
        self.tok_bufs = None
        self.cam_bufs = None

    @torch.inference_mode()
    def step(self, frame_chw: torch.Tensor) -> torch.Tensor:
        """Consume one frame and return its depth using at most ``window`` frames."""
        if frame_chw.ndim != 3:
            raise ValueError(f"expected a (C,H,W) frame, got {tuple(frame_chw.shape)}")
        x = frame_chw[None, :, None]                                   # (1,3,1,H,W)
        # Padding runs INSIDE this timed step, deliberately (RAMEN Opus, 2026-08-30):
        # unlike degradation synthesis (excluded from C10 timing -- a deployed system
        # never performs it, it only exists to construct the test input), a deployed
        # system fed native, non-multiple-of-patch-size frames WOULD pad every one of
        # them. The rule is "measure what deployment would actually do", not "exclude
        # all preprocessing" -- those are different rules and this is the one that applies.
        x, native_h, native_w = _pad_to_patch_multiple(x, self.m.PATCH_SIZE)
        H, W = x.shape[-2:]
        device_type = self.dev.split(":")[0]
        amp_enabled = self.dt is not None and self.dt != torch.float32
        with torch.autocast(device_type, dtype=self.dt, enabled=amp_enabled):
            cond, _ = self.m.condition(x)
            feats = self.m.encode(x, cond=cond)                         # ONE ViT pass (T=1)
            if self.tok_bufs is None:
                self.tok_bufs = [deque(maxlen=self.win) for _ in feats]
                self.cam_bufs = [deque(maxlen=self.win) for _ in feats]
            for db, cb, (tok, cam) in zip(self.tok_bufs, self.cam_bufs, feats):
                db.append(tok)                                         # tok (1,1,N,C)
                cb.append(cam)
            window = [
                (torch.cat(list(db), dim=1),                           # (1,t,N,C), t ≤ win
                 (torch.cat(list(cb), dim=1) if cb[0] is not None else None))
                for db, cb in zip(self.tok_bufs, self.cam_bufs)
            ]
            mixed = self.m.apply_temporal(window, H, W, cond=cond)     # temporal over the window
            newest = [(tok[:, -1:].contiguous(),
                       (cam[:, -1:].contiguous() if cam is not None else None))
                      for tok, cam in mixed]
            # head only decodes the newest frame; temporal already saw the window
            depth = self.m.decode_modules(None, self.m.channel, self.m.head, newest, H, W)
        # Crop back to native resolution HERE, at the full decoded pixel/depth map,
        # never at token resolution -- the caller must only ever see native shape, so
        # gt/mask alignment is exact by construction rather than by a later caller
        # remembering to crop (RAMEN Opus, 2026-08-30).
        depth = depth[0, 0, 0, :native_h, :native_w].float()           # (native_h, native_w)
        assert depth.shape == (native_h, native_w), (
            f"StreamingDepth.step must return the input's native shape "
            f"({native_h}, {native_w}), got {tuple(depth.shape)} -- a future change to "
            f"decode's own upsampling could silently return a still-padded map"
        )
        return depth

    __call__ = step


class StreamingOVDA:
    """Single-frame streaming wrapper for OVDANet, mirroring StreamingDepth's role for
    D3Net. OVDANet.forward() already runs a causal per-frame cache loop internally, but
    resets its cache state on every call; this keeps that same state (via
    OVDANet.init_stream_state/step_stream, factored out so both paths run identical
    bookkeeping) alive across separate step() calls instead.

    ``_DEPTH_RESCALE`` (TOOL-21/TOOL-23, 2026-08-30): OVDA's native disparity output can
    carry an extreme absolute scale (observed ~1e5-1e7 on a real matrix checkpoint,
    matrix_ovda_clean_t8/step8000.pt) -- 1/disp then lands in the 1e-6 to 1e-5 metre
    range, several orders of magnitude below every harness's own min_depth (1e-3).
    Confirmed via Spearman rank correlation (rho 0.85-0.95 against real GT depth on
    real Bonn frames) that this is NOT a collapsed/uninformative model -- real per-pixel
    structure survives, it is simply mis-scaled. Left unscaled, two things break, in two
    different frozen files, for reasons that have nothing to do with model quality:
    every prediction reads as invalid to DepthEvaluator's own min_depth/max_depth range
    check (RAMEN Opus's diagnosis, confirmed), and evaluation/alignment_free.py's own
    per-video affine fit (metrics.fit_scale_shift) computes its normal-equations sums in
    float32 -- squaring and summing values at this magnitude across a real video's full
    pixel count (hundreds of millions) loses so much precision the determinant becomes
    essentially random noise, confirmed directly (float32 vs float64 determinants
    differing by 8 to 43 orders of magnitude on real data at N=1 to 200 frames).

    A SINGLE FIXED CONSTANT, not a per-frame or per-video normalisation (RAMEN Opus's
    explicit correction to an earlier, wrong draft of this fix): every reported
    alignment-scope-independent scale-drift diagnostic this harness computes --
    ``scale_cv``, ``temporal_log_scale_std``, ``median_log_scale_bias`` (C12's own
    subject) -- measures how a model's OWN output scale varies across frames/videos. A
    per-frame or per-video rescale would divide out exactly that signal, silently, and
    the resulting clean-looking numbers would measure nothing. A single global constant
    is invariant on every axis that matters: the harness's own downstream affine fit
    already absorbs any constant multiplicative factor when computing aligned metrics,
    and multiplying every frame of every video by the same number preserves every
    frame-to-frame and video-to-video RATIO exactly, so it cannot mask real drift.

    Value: 1e6 (a declared power of ten, the most auditable choice, per RAMEN Opus) --
    never derived from the data being scored. Verified against 60 real consecutive Bonn
    frames from the same checkpoint: post-rescale depth ranges [2.45, 33.4] across the
    WHOLE distribution (not just the median), comfortably clear of both min_depth=1e-3
    and max_depth=80 with real margin on both ends. RAMEN's own StreamingDepth output
    (0.40-1.42 on the same real frames) and VDA's own StreamingVDA output (0.026-0.19)
    were checked and are already in a sane range -- this rescale applies to OVDA only.
    """

    _DEPTH_RESCALE = 1e6

    def __init__(self, model, device: str, dtype):
        self.m, self.dev, self.dt = model, device, dtype
        self.reset()

    def reset(self) -> None:
        """Forget all temporal context, normally at a video boundary."""
        self.state = None

    @torch.inference_mode()
    def step(self, frame_chw: torch.Tensor) -> torch.Tensor:
        if frame_chw.ndim != 3:
            raise ValueError(f"expected a (C,H,W) frame, got {tuple(frame_chw.shape)}")
        # Padding runs INSIDE this timed step, deliberately -- see StreamingDepth.step's
        # identical comment ("measure what deployment would actually do", not "exclude
        # all preprocessing"). OVDA's own vendored code (models/ovda_net.py:step_stream,
        # OnlineVideoDepthAnything.setup_cache) hardcodes patch size 14 -- it has no
        # DINOv3/16 variant, unlike D3Net -- so 14 here matches that vendored constant,
        # not an independent guess. Padded BEFORE H, W is read, so setup_cache (called
        # via init_stream_state below) and step_stream's own H//14, W//14 see the SAME
        # padded dimensions: computing them from different (padded vs native) shapes is
        # exactly TOOL-10's mismatch, reachable from the opposite direction.
        frame_chw, native_h, native_w = _pad_to_patch_multiple(frame_chw, 14)
        H, W = frame_chw.shape[-2:]
        device_type = self.dev.split(":")[0]
        amp_enabled = self.dt is not None and self.dt != torch.float32
        with torch.autocast(device_type, dtype=self.dt, enabled=amp_enabled):
            x = self._preprocess(frame_chw)                              # (1,1,3,H,W)
            if self.state is None:
                self.state = self.m.init_stream_state(H, W, self.dev)
            disp = self.m.step_stream(x, self.state)                    # (1,1,H,W)
        # Crop back to native resolution HERE, at the decoded disparity map, never at
        # token resolution -- see StreamingDepth.step's identical crop-before-return.
        # _DEPTH_RESCALE (see class docstring): a fixed multiplicative constant, applied
        # identically to every frame -- brings OVDA's own extreme native scale into a
        # numerically sane range without altering any frame-to-frame or video-to-video
        # ratio, so every scale-drift diagnostic this harness computes stays meaningful.
        return self._finish(disp, native_h, native_w)

    def _preprocess(self, frame_chw: torch.Tensor) -> torch.Tensor:
        """Exactly OVDANet.forward()'s preprocessing: the (trained) front-end, THEN mean/std.
        Until 2026-09-21 this wrapper skipped ``self.m.frontend`` while training and
        train.evaluate() applied it, so every ``use_frontend: true`` checkpoint was scored
        without a module its depth loss had trained (found by the JR-01 session; disparity
        changed by 0.2% for parity-B but 2.6% for the plastic K0 at step 10000). The front-end
        is per-frame and purely spatial, so applying it frame by frame is exact."""
        frontend = getattr(self.m, "frontend", None)
        if frontend is not None:
            frame_chw = frontend(frame_chw[None, :, None])[0, :, 0]
        mean, std = self.m._mean[0, :, 0], self.m._std[0, :, 0]          # (3,1,1) each
        return ((frame_chw - mean) / std)[None, None]

    def _finish(self, disp: torch.Tensor, native_h: int, native_w: int) -> torch.Tensor:
        depth = (self._DEPTH_RESCALE / disp.clamp(min=1e-6))[0, 0, :native_h, :native_w].float()
        assert depth.shape == (native_h, native_w), (
            f"StreamingOVDA.step must return the input's native shape "
            f"({native_h}, {native_w}), got {tuple(depth.shape)} -- a future change to "
            f"OVDA's own interpolate-back-to-input-size step could silently return a "
            f"still-padded map"
        )
        return depth

    __call__ = step


class StreamingOVDALookahead(StreamingOVDA):
    """LOOK-01 (docs/LOOKAHEAD_DESIGN.md): bounded-lookahead streaming for OVDANet.

    ``step(frame_t)`` runs the causal pass for frame t (filling the cache and the pass-2
    history) and returns the depth for frame ``t - L`` re-decoded with L future frames
    (``OVDANet.redecode``), or ``None`` for the first L frames of a stream while the
    buffer fills. ``flush()`` after the last frame returns the remaining L predictions
    (future truncated to what exists), oldest first. The one-in/one-out contract of the
    frozen scorer is therefore NOT met on purpose -- this runner is consumed by
    scripts/eval_lookahead_npz.py, which scores each prediction against its own row and
    records ``output_delay_frames = L`` separately from per-frame compute.

    ``lookahead=0`` re-decodes every frame through pass 2 with the causal layout
    (query at the newest slot, all past slots): the same attention pattern as pass 1,
    which is the equivalence gate against the frozen scorer for any OVDA checkpoint.
    """

    def __init__(self, model, device: str, dtype, lookahead: int):
        self.L = int(lookahead)
        if self.L < 0 or self.L >= int(model.cache_size):
            raise ValueError(f"lookahead={lookahead} must be in [0, cache_size-1={model.cache_size - 1}]")
        model.lookahead_enabled = True                # history bookkeeping in step_stream
        super().__init__(model, device, dtype)

    def reset(self) -> None:
        self.state = None
        self.n = 0
        self.dims: tuple[int, int, int, int] | None = None   # (H, W, native_h, native_w)

    @torch.inference_mode()
    def step(self, frame_chw: torch.Tensor) -> torch.Tensor | None:
        if frame_chw.ndim != 3:
            raise ValueError(f"expected a (C,H,W) frame, got {tuple(frame_chw.shape)}")
        frame_chw, native_h, native_w = _pad_to_patch_multiple(frame_chw, 14)
        H, W = frame_chw.shape[-2:]
        if self.dims is None:
            self.dims = (H, W, native_h, native_w)
        elif self.dims != (H, W, native_h, native_w):
            raise ValueError("frame size changed inside a stream; lookahead needs a fixed size per video")
        device_type = self.dev.split(":")[0]
        amp_enabled = self.dt is not None and self.dt != torch.float32
        with torch.autocast(device_type, dtype=self.dt, enabled=amp_enabled):
            x = self._preprocess(frame_chw)
            if self.state is None:
                self.state = self.m.init_stream_state(H, W, self.dev)
            self.m.step_stream(x, self.state)                          # pass 1
            self.n += 1
            if self.n - 1 < self.L:
                return None
            disp = self.m.redecode(self.state, self.L, self.L, H, W)   # pass 2: frame t-L
        return self._finish(disp, native_h, native_w)

    @torch.inference_mode()
    def flush(self) -> list[torch.Tensor]:
        """Predictions for the last min(L, n) frames of the stream, oldest first."""
        if self.state is None:
            return []
        H, W, native_h, native_w = self.dims
        device_type = self.dev.split(":")[0]
        amp_enabled = self.dt is not None and self.dt != torch.float32
        out = []
        with torch.autocast(device_type, dtype=self.dt, enabled=amp_enabled):
            for d in range(min(self.L, self.n) - 1, -1, -1):
                out.append(self._finish(self.m.redecode(self.state, d, self.L, H, W), native_h, native_w))
        return out

    __call__ = step


class StreamingVDA:
    """Drives an in-repo-trained VDANet checkpoint (models/vda_net.py) through the
    vendored ``video_depth_anything.video_depth_stream.VideoDepthAnything``'s genuine
    single-frame streaming API (``infer_video_depth_one``) -- the SAME streaming class
    BENCH-05b's ``build_vda`` (evaluation/model_registry.py, non-frozen) already
    validates, there for VDA's own external reference checkpoint (TOOL-16, 2026-08-30).

    ``VDANet.core`` is the vendored WHOLE-CLIP class (``video_depth_anything.video_depth.
    VideoDepthAnything``) -- confirmed by reading both classes directly (crispwork,
    2026-08-30) to build IDENTICAL ``pretrained``/``head`` submodules from the same
    constructor args, so their ``state_dict()`` keys match exactly and a strict
    transplant is safe. This class builds a streaming-class instance from the same
    ``repo`` VDANet already resolved (TOOL-11-shaped fallback already applied there)
    and loads VDANet.core's trained weights into it.

    Duplicates ``evaluation.model_registry.build_vda``'s reset()/predict_frame logic
    (the four streaming-state attributes cleared on reset; the disparity->depth
    inversion) rather than sharing code with it -- tests/test_streaming_vda.py's own
    drift test asserts this copy and build_vda's own logic agree given the same image
    in each one's own native convention, so the two cannot silently diverge unnoticed.

    INPUT CONTRACT DIFFERS FROM build_vda'S OWN: every real endpoint-grid manifest
    (scripts/build_{kitti,bonn,vkitti}_recipe_manifest.py) stores frames CHW, [0,1]
    float -- the same convention StreamingDepth/StreamingOVDA already consume, verified
    by reading each builder's own ``frame.transpose(2, 0, 1)`` / ``.permute(2, 0, 1)``
    line directly. build_vda's own ``predict_frame(frame_hwc)`` instead expects HWC,
    ~[0,255] float -- correct for ITS OWN manifest (scripts/build_kitti_streaming_
    manifest.py's BENCH-05b Tier-2 gate data, which skips both the transpose and the
    /255.0 division) but NOT the shared endpoint-grid manifests this class must work
    against. ``step()`` below converts CHW/[0,1] -> HWC/[0,255] before calling
    ``infer_video_depth_one``, which then does its own ``/255.0`` internally --
    getting this wrong would not crash, just silently rescale/misshape every input,
    exactly the kind of wrong-but-plausible-looking failure this campaign treats as
    worse than a crash.

    Wired into ``--model ramen`` via ``evaluation.model_registry.build_ramen``'s
    ``isinstance(model, VDANet)`` dispatch (TOOL-16, 2026-08-30) -- the registry's
    own extraction out of scripts/eval_streaming_npz.py (a BENCH-06 frozen file,
    mirroring RECIPE_REGISTRY's own earlier extraction) is what let this be added
    without touching frozen code.
    """

    def __init__(self, vda_net, device: str, dtype):
        from models.vda_net import _MODEL_CONFIGS

        if vda_net.repo not in sys.path:
            sys.path.insert(0, vda_net.repo)
        from video_depth_anything.video_depth_stream import VideoDepthAnything as _StreamingCore

        core = _StreamingCore(**_MODEL_CONFIGS[vda_net.encoder_name])
        core.load_state_dict(vda_net.core.state_dict(), strict=True)
        self.core = core.to(device).eval()
        self.dev = device
        self.fp32 = dtype is None or dtype == torch.float32
        # Output semantics must match evaluation.teacher_models.StreamingVDATeacher (pinned by
        # tests/test_streaming_vda.py): the metric branch emits depth, not disparity, and
        # released-lineage relative VDA-L emits disparity ~20x VDA-S's, so plain 1/raw lands
        # under the harness min_depth. VDA-S relative keeps 1/raw (build_vda's convention).
        from evaluation.teacher_models import StreamingVDATeacher
        self.metric = vda_net.metric
        self.rescale = StreamingVDATeacher.RELATIVE_RESCALE if vda_net.encoder_name == "vitl" else 1.0
        self.reset()

    def reset(self) -> None:
        """Forget all temporal context -- the same four attributes build_vda's own
        reset() clears, matching what the streaming class's __init__ sets them to."""
        self.core.transform = None
        self.core.frame_id_list = []
        self.core.frame_cache_list = []
        self.core.id = -1

    @torch.inference_mode()
    def step(self, frame_chw: torch.Tensor) -> torch.Tensor:
        """Consume one (C,H,W) frame in [0,1] (this repo's shared manifest convention)
        and return its depth at native resolution -- no separate padding step, unlike
        StreamingDepth/StreamingOVDA: the vendored class's own transform pipeline
        resizes internally (aspect-ratio-aware, to a multiple of its own patch size),
        so there is nothing for this wrapper to pad."""
        if frame_chw.ndim != 3:
            raise ValueError(f"expected a (C,H,W) frame, got {tuple(frame_chw.shape)}")
        frame_hwc = (frame_chw.permute(1, 2, 0) * 255.0).cpu().numpy()
        raw = self.core.infer_video_depth_one(frame_hwc, device=self.dev, fp32=self.fp32)
        depth = raw if self.metric else self.rescale / np.clip(raw, 1e-3, None)
        return torch.from_numpy(np.asarray(depth)).float()

    __call__ = step


def colorize(depth: np.ndarray) -> np.ndarray:
    """Inverse-depth (near = bright), robust-percentile normalized, INFERNO."""
    inv = 1.0 / np.clip(depth, 1e-3, None)
    lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
    u8 = np.clip((inv - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def to_tensor(bgr: np.ndarray, res: int, device: str) -> torch.Tensor:
    img = cv2.resize(bgr, (res, res), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).to(device, non_blocking=True)
    return t.permute(2, 0, 1).float().div_(255.0)                     # (3,res,res) in [0,1]


class Camera:
    """Threaded grabber — keeps only the latest frame so compute never waits on capture."""

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera/video {src!r}")
        self.lock = threading.Lock()
        self.frame = None
        self.run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.run:
            ok, f = self.cap.read()
            if not ok:
                self.run = False
                break
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.run = False
        self.cap.release()


def maybe_compile(model, on: bool):
    if not on:
        return
    try:
        model.net.encoder.backbone = torch.compile(model.net.encoder.backbone, mode="max-autotune")
        print("[infer] torch.compile on backbone (first frames will be slow while it warms up)")
    except Exception as e:                                            # ponytail: compile is best-effort
        print(f"[infer] compile skipped: {e}")


def selftest(streamer, res, device, n=40):
    g = torch.Generator(device="cpu").manual_seed(0)
    ts = []
    for i in range(n):
        f = torch.rand(3, res, res, generator=g).to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        d = streamer(f)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        if i >= 5:                                                    # skip warmup
            ts.append(time.perf_counter() - t0)
        assert d.shape == (res, res), d.shape
        assert torch.isfinite(d).all(), "non-finite depth"
    fps = 1.0 / float(np.median(ts))
    print(f"[selftest] OK  out={tuple(d.shape)}  window={streamer.win}  res={res}  "
          f"median {fps:.1f} FPS ({1000*np.median(ts):.1f} ms/frame on {device})")


from ramen_ckpt import default_ckpt, require_ckpt
DEFAULT_CKPT = default_ckpt()


def save_rgb_offline(ckpt: str, video: str, out_dir: str, device: str, res: int = 0, window: int = 32, max_frames: int = 0):
    """--save-rgb: offline pass with the FULL model (``load_ramen``); writes the display-head output (restored RGB, PNG) and depth (npy) per frame.
    Windows of ``window`` frames, as the scorer path (evaluation.l15_spec.ours_l15). Streaming mode never calls this."""
    from models import load_ramen
    net = load_ramen(ckpt, device)
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    frames = []
    while max_frames == 0 or len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    r = res or 518
    n = 0
    with torch.no_grad():
        for i in range(0, len(frames), window):
            chunk = frames[i:i + window]
            x = torch.stack([to_tensor(f, r, device) for f in chunk])        # (T,3,H,W)
            rgb = x.permute(1, 0, 2, 3)[None]
            r_depth, restored = net.restore_decoupled(rgb)                   # depth <- depth_restorer ; output <- rgb_restorer
            d, _ = net._depth_and_feats(net.consume(rgb, r_depth))
            for t in range(len(chunk)):
                np.save(os.path.join(out_dir, f"depth_{n:05d}.npy"), d[0, 0, t].float().cpu().numpy())
                img = (restored[0, :, t].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
                cv2.imwrite(os.path.join(out_dir, f"rgb_{n:05d}.png"), img[:, :, ::-1])
                n += 1
    print(f"[save-rgb] wrote {n} frames (depth_*.npy + rgb_*.png) to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="default checkpoints_release/ramen.pt (RAMEN); env RAMEN_CKPT overrides")
    ap.add_argument("--save-rgb", metavar="DIR", default=None, help="offline: run the video given by --camera with the full model, write display-head RGB + depth to DIR (not streaming)")
    ap.add_argument("--camera", default="0", help="webcam index or video path")
    ap.add_argument("--window", type=int, default=16, help="temporal window T (train was 16)")
    ap.add_argument("--res", type=int, default=0, help="0 = use checkpoint's train resolution")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    require_ckpt(a.ckpt)

    if a.save_rgb:
        save_rgb_offline(a.ckpt, a.camera, a.save_rgb, a.device, a.res)
        return
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    model, train_res = load(a.ckpt, a.device)
    res = a.res or train_res
    maybe_compile(model, a.compile)
    try:
        streamer = StreamingDepth(model, a.window, a.device, dt)
    except NotImplementedError as e:
        sys.exit(f"[infer] {e}")
    print(f"[infer] {a.ckpt}  res={res}  window={a.window}  {a.dtype}  {a.device}")

    if a.selftest:
        selftest(streamer, res, a.device)
        return

    src = int(a.camera) if a.camera.isdigit() else a.camera
    cam = Camera(src)
    ema_ms, alpha = None, 0.1
    try:
        while True:
            bgr = cam.read()
            if bgr is None:
                time.sleep(0.005)
                continue
            t0 = time.perf_counter()
            depth = streamer(to_tensor(bgr, res, a.device)).cpu().numpy()
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000
            ema_ms = dt_ms if ema_ms is None else (1 - alpha) * ema_ms + alpha * dt_ms

            vis = colorize(depth)
            vis = cv2.resize(vis, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            out = np.hstack([bgr, vis])
            cv2.putText(out, f"{1000/ema_ms:.1f} FPS  ({ema_ms:.0f} ms)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("RAMEN depth (q=quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

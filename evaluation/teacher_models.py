"""Teacher-model builders for the frozen streaming harness (TEACH-01, 2026-09-18).

Each builder has the ``MODEL_REGISTRY`` signature
``(checkpoint, device, dtype_str, window) -> (predict_frame, reset, checkpoint_sha256)``
so a large, slow teacher can be scored through the unmodified
``scripts/eval_streaming_npz.py`` on exactly the same recipe manifests and
frame populations as the student arms. Registering here (a non-frozen file)
is the sanctioned route -- see ``evaluation/model_registry.py``'s docstring.

Input contract, shared with ``infer.StreamingDepth``/``StreamingOVDA``/
``StreamingVDA``: ``predict_frame`` receives one ``(C, H, W)`` float frame in
``[0, 1]`` on ``device`` and returns a ``(H, W)`` float32 tensor that is
**depth-like** (larger = farther). The harness computes ``1/pred`` internally
for its disparity-space affine fit, so disparity-native teachers (DA-AC,
relative VDA) are inverted here, and metric/depth-native teachers (DepthPro,
UniDepth, DA3) are returned as-is.

Single-frame teachers are stateless: ``reset`` is a no-op and ``window`` is
ignored by the model. Pass the same ``--window`` as the student rows anyway,
because the harness uses it only to decide which frames count as
``steady_state`` -- that keeps the scored frame population identical to the
student cells (docs/RAMEN_V3_PROGRAMME.md T1).

Every heavy import is local to its builder so this module stays importable on
hosts without the baseline checkouts (tests run on the relay workstation).
External checkouts are located under ``TEACHER_ASSETS_ROOT``
(default ``/mnt/DATA/DegradedDepth``, crispwork's layout, 2026-09-18) or the
same ``VDA_ROOT`` variable ``build_vda`` already honours.
"""

from __future__ import annotations

import inspect
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from evaluation.alignment_free import sha256_file

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TEACHER_NAMES: tuple[str, ...] = (
    "da3_large",
    "da3_giant",
    "vda_l",
    "vda_l_metric",
    "depth_pro",
    "unidepth_vitl",
    "daac",
)


def assets_root() -> Path:
    return Path(os.environ.get("TEACHER_ASSETS_ROOT", "/mnt/DATA/DegradedDepth"))


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    """Bottom-right replicate-pad the last two dims to a multiple of ``multiple``.

    Same rule as ``infer._pad_to_patch_multiple`` (TOOL-08): native content stays
    at ``[:h, :w]`` so the crop back is an unconditional slice. Kept local so this
    module does not import ``infer`` (cv2, the whole model stack) at import time.
    """
    h, w = x.shape[-2:]
    pad_h, pad_w = (-h) % multiple, (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    leading = x.shape[:-2]
    flat = x.reshape(-1, 1, h, w)
    flat = torch.nn.functional.pad(flat, (0, pad_w, 0, pad_h), mode="replicate")
    return flat.reshape(*leading, h + pad_h, w + pad_w), h, w


class SingleFrameTeacher:
    """Stateless per-frame teacher behind the harness's (step, reset) contract.

    ``predict`` maps a ``(1, 3, H', W')`` batch in ``[0, 1]`` (already on
    ``device``, padded to ``patch_size`` when one is given) to a depth-like map
    of shape ``(H', W')`` or ``(1, H', W')``. The wrapper pads, runs under the
    requested autocast dtype, crops back to native shape and asserts it.
    """

    def __init__(self, predict: Callable[[torch.Tensor], torch.Tensor], patch_size: int | None,
                 device: str, dtype: torch.dtype | None):
        self.predict, self.patch, self.dev, self.dt = predict, patch_size, device, dtype

    def reset(self) -> None:
        """No temporal state to forget."""

    @torch.inference_mode()
    def step(self, frame_chw: torch.Tensor) -> torch.Tensor:
        if frame_chw.ndim != 3:
            raise ValueError(f"expected a (C,H,W) frame, got {tuple(frame_chw.shape)}")
        x = frame_chw[None].to(self.dev).float()
        if self.patch:
            x, native_h, native_w = pad_to_multiple(x, self.patch)
        else:
            native_h, native_w = x.shape[-2:]
        device_type = self.dev.split(":")[0]
        amp = self.dt is not None and self.dt != torch.float32 and device_type == "cuda"
        with torch.autocast(device_type, dtype=self.dt if amp else None, enabled=amp):
            depth = self.predict(x)
        depth = depth.reshape(depth.shape[-2:]).float()
        depth = depth[:native_h, :native_w]
        assert depth.shape == (native_h, native_w), (
            f"teacher returned {tuple(depth.shape)}, expected native ({native_h}, {native_w})")
        return depth

    __call__ = step


def _dtype(dtype_str: str) -> torch.dtype | None:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[dtype_str]


def _imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# --------------------------------------------------------------------------- DA3


def build_da3(config_name: str):
    def build(checkpoint: str, device: str, dtype_str: str, window: int):
        from models.dav3_backbone import build_official_da3

        net = build_official_da3(checkpoint, config_name, device,
                                 ignore_unexpected_prefixes=("gs_head.", "gs_adapter.")).eval()

        def predict(x: torch.Tensor) -> torch.Tensor:          # (1,3,H,W) in [0,1]
            out = net(_imagenet(x)[:, None])                    # DA3 wants (B, N, 3, H, W)
            return out.depth[0, 0]                              # (H, W), depth-like

        t = SingleFrameTeacher(predict, net.PATCH_SIZE, device, _dtype(dtype_str))
        return t.step, t.reset, sha256_file(checkpoint)

    build.__name__ = f"build_{config_name.replace('-', '_')}"
    return build


# --------------------------------------------------------------------------- VDA-L (streaming mode)


class StreamingVDATeacher:
    """Released VDA weights through VDA's own training-free streaming class,
    the same path ``infer.StreamingVDA`` uses for the matrix's VDA-S cells
    (``infer_video_depth_one``: INFER_LEN 32 / OVERLAP 10 sliding window over
    cached hidden states). Relative checkpoints emit disparity (inverted here);
    metric checkpoints emit depth (returned as-is, the convention
    ``models/vda_net.py`` verified empirically for the metric branch).
    """

    def __init__(self, checkpoint: str, encoder: str, metric: bool, device: str, dtype: torch.dtype | None):
        vda_root = os.environ.get("VDA_ROOT") or str(assets_root() / "competition" / "Video-Depth-Anything")
        if vda_root not in sys.path:
            sys.path.insert(0, vda_root)
        from video_depth_anything.video_depth_stream import VideoDepthAnything as _Core

        from models.vda_net import _MODEL_CONFIGS

        kwargs = dict(_MODEL_CONFIGS[encoder])
        if "metric" in inspect.signature(_Core.__init__).parameters:
            kwargs["metric"] = metric
        core = _Core(**kwargs)
        sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        missing, unexpected = core.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"VDA-{encoder} checkpoint missing keys: {missing[:8]}")
        if unexpected:
            raise RuntimeError(f"VDA-{encoder} checkpoint has unexpected keys: {unexpected[:8]}")
        self.core = core.to(device).eval()
        self.dev, self.metric = device, metric
        self.fp32 = dtype is None or dtype == torch.float32
        self.reset()

    def reset(self) -> None:
        self.core.transform = None
        self.core.frame_id_list = []
        self.core.frame_cache_list = []
        self.core.id = -1

    @torch.inference_mode()
    def step(self, frame_chw: torch.Tensor) -> torch.Tensor:
        if frame_chw.ndim != 3:
            raise ValueError(f"expected a (C,H,W) frame, got {tuple(frame_chw.shape)}")
        frame_hwc = (frame_chw.permute(1, 2, 0) * 255.0).cpu().numpy()
        raw = self.core.infer_video_depth_one(frame_hwc, device=self.dev, fp32=self.fp32)
        depth = raw if self.metric else self.RELATIVE_RESCALE / np.clip(raw, 1e-3, None)
        return torch.from_numpy(np.asarray(depth)).float()

    # TEACH-01 (2026-09-18): the released RELATIVE VDA-L emits raw disparity roughly 20x the
    # in-repo-trained VDA-S's (median ~450 vs ~20 on the smoke frames), so plain 1/raw lands
    # most pixels below the harness's min_depth (1e-3) -- the vkitti_5scene screen scored
    # 34-39% invalid predictions for vda_l and ~0% for every other teacher. A SINGLE FIXED
    # CONSTANT, not a per-frame/per-video normalisation, for the same reasons as
    # infer.StreamingOVDA._DEPTH_RESCALE: the harness's affine fit absorbs it and every
    # frame-to-frame ratio is preserved. 1e2 keeps the whole observed range inside
    # [1e-3, 80] (depth exceeds 80 only where raw < 1.25, i.e. sky). Declared, not tuned.
    RELATIVE_RESCALE = 1e2

    __call__ = step


def build_vda_stream(encoder: str, metric: bool):
    def build(checkpoint: str, device: str, dtype_str: str, window: int):
        t = StreamingVDATeacher(checkpoint, encoder, metric, device, _dtype(dtype_str))
        return t.step, t.reset, sha256_file(checkpoint)

    build.__name__ = f"build_vda_{encoder}{'_metric' if metric else ''}"
    return build


# --------------------------------------------------------------------------- single-frame foundation models


def build_depth_pro(checkpoint: str, device: str, dtype_str: str, window: int):
    """Apple Depth Pro (vendor/depth_pro). Metric depth, 1536^2 internally; its
    ``infer`` resizes and restores the native shape itself, so no padding."""
    import dataclasses

    from vendor.depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, create_model_and_transforms

    cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=checkpoint)
    model, _ = create_model_and_transforms(cfg, device=torch.device(device))
    model.eval()
    mdt = next(model.parameters()).dtype

    def predict(x: torch.Tensor) -> torch.Tensor:              # (1,3,H,W) in [0,1]
        return model.infer((x * 2 - 1).to(mdt))["depth"]        # (H,W) metric depth

    t = SingleFrameTeacher(predict, None, device, None)          # DepthPro manages its own precision
    return t.step, t.reset, sha256_file(checkpoint)


def build_unidepth_vitl(checkpoint: str, device: str, dtype_str: str, window: int):
    """UniDepth-V2 ViT-L, imported from the cloned repo. ``checkpoint`` is the
    weights file to hash (the HF cache's model.safetensors); the model itself is
    loaded by repo id through ``from_pretrained``."""
    repo = assets_root() / "competition" / "unidepth_repo"
    for p in (repo / "_stubs", repo):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from unidepth.models import UniDepthV2

    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(device).eval()

    def predict(x: torch.Tensor) -> torch.Tensor:              # (1,3,H,W) in [0,1]
        return model.infer(x[0], normalize=True)["depth"].squeeze()

    t = SingleFrameTeacher(predict, None, device, None)
    return t.step, t.reset, sha256_file(checkpoint)


def build_daac(checkpoint: str, device: str, dtype_str: str, window: int):
    """DepthAnything-AC (ViT-S, relative, single-frame) on the DA-V2 DPT
    architecture. Disparity out, inverted to depth-like. Patch 14."""
    repo = assets_root() / "competition" / "Depth-Anything-V2"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384]).to(device).eval()
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(sd, strict=False)                     # DA-AC's extra geo_prior_gen is skipped

    def predict(x: torch.Tensor) -> torch.Tensor:              # (1,3,H,W) in [0,1]
        disp = model(_imagenet(x))                              # (1,H,W)
        return 1.0 / disp.clamp(min=1e-3)

    t = SingleFrameTeacher(predict, 14, device, _dtype(dtype_str))
    return t.step, t.reset, sha256_file(checkpoint)


TEACHER_BUILDERS: dict[str, Callable[[str, str, str, int], tuple]] = {
    "da3_large": build_da3("da3-large"),
    "da3_giant": build_da3("da3-giant"),
    "vda_l": build_vda_stream("vitl", metric=False),
    "vda_l_metric": build_vda_stream("vitl", metric=True),
    "depth_pro": build_depth_pro,
    "unidepth_vitl": build_unidepth_vitl,
    "daac": build_daac,
}
assert tuple(TEACHER_BUILDERS) == TEACHER_NAMES

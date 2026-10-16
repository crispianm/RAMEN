"""DA-AC wrapper (protocol v3 WP23) -- Depth Anything AC (arXiv:2507.01634, degraded-trained single-image depth) fine-tuned on OUR degraded data with the SAME loop as ours.

Wraps the repo's own ``depth_anything.dpt.DepthAnything_AC`` (vits, features 64, v2 layer indices) exactly as bench_joint ``_daac`` / the frozen benchmark build it: DINOv2 is constructed
without its hub download (the checkpoint overwrites every parameter) and the one training-only key ``geo_prior_gen.weight`` is dropped before the strict load. Scope = ``DAV2Net``'s:
DINOv2 encoder frozen, the DPT ``depth_head`` trainable (same architecture and tensor count as DA-V2-S's), per frame, no temporal module.

Two variants, both initialised from the released DA-AC weights:
  * ``variant: relative`` -- DA-AC's native output, relu(disparity); ``predicts_disparity = True``, ``metric = False`` (aligned metrics only, as the released model).
  * ``variant: metric``   -- the same head read as metres (WP22a: the unchanged Conv-ReLU-Conv-ReLU stack, as Metric-VDA's); ``predicts_disparity = False``, ``metric = True``.
``forward`` returns the native output ``(B,1,T,H,W)``.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from .dav2_net import resolve_path

_DAAC_REPOS = ("/mnt/DATA/DegradedDepth/competition/DepthAnythingAC", "/projects/b5dh/benchmark/competition/DepthAnythingAC")
_DAAC_CHECKPOINTS = ("/mnt/DATA/DegradedDepth/RAMEN_aaai/baselines_ckpt/depth_anything_ac/checkpoints/depth_anything_AC_vits.pth",
                     "/projects/b5dh/benchmark/weights/depth_anything_AC_vits.pth")
DAAC_DROP = ("geo_prior_gen.weight",)   # the one training-only key (bench_joint.DAAC_DROP)
VARIANTS = ("relative", "metric")


class DAACNet(nn.Module):
    def __init__(self, repo: str | None = None, variant: str = "relative", init: str = "released", checkpoint_path: str | None = None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"DAACNet variant must be one of {VARIANTS}, got {variant!r}")
        if init not in {"released", "random"}:
            raise ValueError(f"Unknown DAAC init={init!r}")
        repo = resolve_path(repo, _DAAC_REPOS, "DAACNet repo", is_dir=True)
        if repo not in sys.path:
            sys.path.append(repo)       # APPENDED: its util/torchhub packages must never shadow VDA's
        import torchhub.facebookresearch_dinov2_main.hubconf as hub
        from depth_anything.dpt import DepthAnything_AC
        orig = hub.dinov2_vits14
        hub.dinov2_vits14 = lambda pretrained=True, **kw: orig(pretrained=False, **kw)   # looked up at build time
        try:
            self.core = DepthAnything_AC({"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384], "version": "v2"})
        finally:
            hub.dinov2_vits14 = orig
        if init == "released":
            ckpt = resolve_path(checkpoint_path, _DAAC_CHECKPOINTS, "DAACNet checkpoint")
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = sd.get("model", sd)
            self.core.load_state_dict({k: v for k, v in sd.items() if k not in DAAC_DROP}, strict=True)
        self.variant = variant
        self.metric = variant == "metric"
        self.predicts_disparity = not self.metric
        self.PATCH_SIZE = 14
        for p in self.core.pretrained.parameters():
            p.requires_grad_(False)
        self.core.pretrained.eval()
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.core.pretrained.eval()
        return self

    def trainable_modules(self) -> dict:
        return {"head": self.core.depth_head}

    def trainable_parameters(self):
        return [p for m in self.trainable_modules().values() for p in m.parameters() if p.requires_grad]

    def assert_backbone_frozen(self) -> None:
        assert not any(p.requires_grad for p in self.core.pretrained.parameters()), "DA-AC DINOv2 backbone must stay frozen"

    def parameter_summary(self) -> dict:
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.trainable_parameters())
        return {"total_m": round(tot / 1e6, 2), "trainable_m": round(tr / 1e6, 2)}

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:   # (B,C,T,H,W) in [0,1] -> (B,1,T,H,W)
        B, C, T, H, W = rgb.shape
        assert H % 14 == 0 and W % 14 == 0, "DA-AC requires H,W multiples of 14 (DINOv2 patch size)"
        x = (rgb.permute(0, 2, 1, 3, 4).flatten(0, 1) - self._mean) / self._std
        out = self.core(x)["out"]                           # (B*T,H,W): relu(head output), the encoder carries no graph (frozen)
        return out.unflatten(0, (B, T)).unsqueeze(1)

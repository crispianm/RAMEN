"""DA-V2-S metric wrapper (protocol v3 WP22) -- the per-frame DA-V2-S baseline fine-tuned on OUR degraded data with the SAME loop as ours.

Imports the vendored, unmodified ``depth_anything_v2.dpt.DepthAnythingV2`` (ViT-S, features 64). Scope matches ``VDANet`` minus its temporal part: DINOv2 encoder frozen,
the whole DPT head (``depth_head``: its 64 tensors are structurally the 64 non-temporal ``head.*`` keys of VDA-S's ``DPTHeadTemporal``) trainable, nothing else.

Two metric heads, built both ways (protocol G8). Init (``init``): ``relative`` = the released RELATIVE DA-V2-S weights (as VDA's own metric stage starts from its relative weights), or the released DA-V2 METRIC Small checkpoints
(HF depth-anything/Depth-Anything-V2-Metric-{VKITTI,Hypersim}-Small) ``metric_vkitti`` (outdoor, max_depth 80) / ``metric_hypersim`` (indoor, max_depth 20): head ``sigmoid`` only, with the checkpoint's own max_depth (another would rescale its output):
  * ``head: metres``  (WP22a) -- the DPT head unchanged (Conv-ReLU-Conv-ReLU), read as metres: what Metric-Video-Depth-Anything does (its metric DPT has the identical output stack).
  * ``head: sigmoid`` (WP22b) -- the final ReLU becomes a Sigmoid and the output is multiplied by ``max_depth`` (DA-V2's own ``metric_depth/`` head, default 80 m here): bounded in (0, max_depth).
``forward`` returns metres ``(B,1,T,H,W)``; ``predicts_disparity = False``, ``metric = True``. Every frame goes through the net independently (no temporal module, window 1 in bench_joint).
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_DAV2_REPOS = ("/mnt/DATA/DegradedDepth/competition/Depth-Anything-V2", "/projects/b5dh/benchmark/competition/Depth-Anything-V2")
_DEFAULT_DAV2_CHECKPOINT = ("/mnt/DATA/DegradedDepth/RAMEN/checkpoints/depth_anything_v2_vits.pth", "/projects/b5dh/benchmark/weights/depth_anything_v2_vits.pth",
                            "/projects/b5dh/DaBiT/weights/depth_anything_v2_vits.pth")
_METRIC_CKPT = {"metric_vkitti": (("/mnt/DATA/DegradedDepth/RAMEN/checkpoints/depth_anything_v2_metric_vkitti_vits.pth", "/projects/b5dh/benchmark/weights/depth_anything_v2_metric_vkitti_vits.pth"), 80.0),
                "metric_hypersim": (("/mnt/DATA/DegradedDepth/RAMEN/checkpoints/depth_anything_v2_metric_hypersim_vits.pth", "/projects/b5dh/benchmark/weights/depth_anything_v2_metric_hypersim_vits.pth"), 20.0)}
INITS = ("relative", "released", "metric_vkitti", "metric_hypersim", "random")   # 'released' = 'relative' (the first spelling)
_VITS = dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
HEADS = ("metres", "sigmoid")


def resolve_path(configured: str | None, fallbacks, what: str, is_dir: bool = False) -> str:
    """A cfg baked on another host (the TOOL-11 trap VDANet already handles): the configured path if it exists here, else the first existing host default."""
    ok = (lambda p: Path(p).is_dir()) if is_dir else (lambda p: Path(p).is_file())
    for p in ([configured] if configured else []) + list(fallbacks):
        if ok(p):
            if configured and p != configured:
                print(f"[{what}] configured {configured!r} does not exist on this host -- using {p}", file=sys.stderr)
            return str(p)
    raise FileNotFoundError(f"{what}: none of {([configured] if configured else []) + list(fallbacks)} exists")


class DAV2Net(nn.Module):
    predicts_disparity = False
    metric = True

    def __init__(self, repo: str | None = None, head: str = "metres", max_depth: float = 80.0, init: str = "relative",
                 checkpoint_path: str | None = None, unfreeze_last_blocks: int = 0):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"DAV2Net head must be one of {HEADS}, got {head!r}")
        if init not in INITS:
            raise ValueError(f"Unknown DAV2 init={init!r}, expected one of {INITS}")
        if init in _METRIC_CKPT and (head != "sigmoid" or float(max_depth) != _METRIC_CKPT[init][1]):
            raise ValueError(f"init={init} is DA-V2's sigmoid x max_depth head: needs head='sigmoid' and max_depth={_METRIC_CKPT[init][1]}")
        if not max_depth > 0:
            raise ValueError("max_depth must be positive")
        repo = resolve_path(repo, _DAV2_REPOS, "DAV2Net repo", is_dir=True)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from depth_anything_v2.dpt import DepthAnythingV2
        self.core = DepthAnythingV2(**_VITS)
        if init != "random":       # strict: DA-V2-S's own keys (pretrained.* + depth_head.*; the metric checkpoints have the same keys)
            ckpt = resolve_path(checkpoint_path, _METRIC_CKPT[init][0] if init in _METRIC_CKPT else _DEFAULT_DAV2_CHECKPOINT, "DAV2Net checkpoint")
            self.core.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=True)
        self.head_kind, self.max_depth = head, float(max_depth)
        if head == "sigmoid":      # the ReLU after the last 1x1 conv (parameterless) -> Sigmoid; weights unchanged
            out2 = self.core.depth_head.scratch.output_conv2
            assert isinstance(out2[3], nn.ReLU), "unexpected DA-V2 output stack"
            out2[3] = nn.Sigmoid()
        self.PATCH_SIZE = 14
        for p in self.core.pretrained.parameters():
            p.requires_grad_(False)
        # opt-in (protocol v3 WP27 dav2a_cap, default 0 = the frozen-encoder baseline, bit-identical): the last N whole DINOv2 blocks trainable (same lr as the head). The final norm stays frozen.
        n_blocks = len(self.core.pretrained.blocks)
        if not (isinstance(unfreeze_last_blocks, int) and 0 <= unfreeze_last_blocks <= n_blocks):
            raise ValueError(f"unfreeze_last_blocks must be an int in [0, {n_blocks}], got {unfreeze_last_blocks!r}")
        self.unfreeze_last_blocks = unfreeze_last_blocks
        self.unfrozen_block_idx = list(range(n_blocks - unfreeze_last_blocks, n_blocks))
        for i in self.unfrozen_block_idx:
            for p in self.core.pretrained.blocks[i].parameters():
                p.requires_grad_(True)
        self.core.pretrained.eval()
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.core.pretrained.eval()   # frozen backbone stays eval even in train() mode
        return self

    def trainable_modules(self) -> dict:
        mods = {"head": self.core.depth_head}
        if self.unfrozen_block_idx:
            mods["enc_blocks"] = nn.ModuleList([self.core.pretrained.blocks[i] for i in self.unfrozen_block_idx])
        return mods

    def trainable_parameters(self):
        return [p for m in self.trainable_modules().values() for p in m.parameters() if p.requires_grad]

    def assert_backbone_frozen(self) -> None:
        keep = {id(p) for i in self.unfrozen_block_idx for p in self.core.pretrained.blocks[i].parameters()}
        assert not any(p.requires_grad and id(p) not in keep for p in self.core.pretrained.parameters()), "DA-V2-S DINOv2 backbone must stay frozen (except the opted-in last blocks)"

    def parameter_summary(self) -> dict:
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.trainable_parameters())
        return {"total_m": round(tot / 1e6, 2), "trainable_m": round(tr / 1e6, 2)}

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:   # (B,C,T,H,W) in [0,1] -> (B,1,T,H,W) metres
        B, C, T, H, W = rgb.shape
        assert H % 14 == 0 and W % 14 == 0, "DA-V2-S requires H,W multiples of 14 (DINOv2 patch size)"
        x = (rgb.permute(0, 2, 1, 3, 4).flatten(0, 1) - self._mean) / self._std
        with (torch.no_grad() if not self.unfrozen_block_idx else contextlib.nullcontext()):   # frozen encoder (default): no graph through it
            feats = self.core.pretrained.get_intermediate_layers(x, self.core.intermediate_layer_idx[self.core.encoder],
                                                                 return_class_token=True)
        d = self.core.depth_head(feats, H // 14, W // 14)   # (B*T,1,H,W): ReLU (metres) or Sigmoid (0..1)
        d = d * self.max_depth if self.head_kind == "sigmoid" else d
        return d.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)

"""VDA-S wrapper -- trains the official Video-Depth-Anything-Small architecture (Bai et al.,
arXiv:2501.12375) inside the RAMEN trainer for VAL-01 (reproduce VDA-S on its own clean
training recipe, byte-identical architecture, to validate this repo's training pipeline
before trusting any RAMEN-specific result). Imports the vendored, unmodified
``video_depth_anything.video_depth.VideoDepthAnything`` directly rather than reimplementing
DA-V2 + DPT + temporal self-attention -- reuse-not-reimplement, the same principle
``scripts/val01_eval_vda.py`` follows for evaluation. Pin the vendored checkout's commit
wherever this wrapper is used for a real run (recorded alongside the VAL-01 config/gate
artifacts, not hardcoded here).

Initialisation (per the paper -- "weights are initialized from Depth Anything V2" -- and a
2026-08-27 code audit): the encoder (``pretrained.*``, 175 keys) strict-loads directly from
DA-V2-S since both share the identical DINOv2 architecture and key names. The head is
trickier: ``DPTHeadTemporal`` (video_depth_anything/dpt_temporal.py) subclasses DA-V2's own
``DPTHead``, so its 64 non-temporal keys are structurally identical to DA-V2's
``depth_head.*`` -- just under VDA's own ``head.`` prefix instead -- and load with a
``depth_head.`` -> ``head.`` remap. The 4 temporal self-attention layers
(``head.motion_modules.*``, 112 keys) have no DA-V2 counterpart; the vendored head already
zero-initialises them at construction (``zero_initialize=True``, AnimateDiff convention), so
this wrapper leaves them untouched rather than loading anything into them. Together this means
a freshly-built VDA-S initialised this way must reproduce DA-V2-S's output exactly at step 0
(the temporal branch is the identity function until it trains) -- verified on real crispwork
hardware, real checkpoints, real code, see
provenance/inventory/val01_vda_net_da_v2_reproduction_20260827.json.

Trainable scope (paper): encoder frozen throughout; the whole head -- spatial DPT layers AND
the temporal self-attention layers together, since "only the head is learned" and the released
checkpoint's own ``head.*`` prefix already includes ``motion_modules.*`` -- trains at one LR
(no per-group learning rates are published for VDA).

Output convention: ``forward()`` returns the model's OWN native representation unchanged --
disparity for the relative variant (VDA's actual training target), depth-like for the metric
variant -- rather than converting relative mode's output to "depth" the way OVDANet does for
its own disparity-native model. ``self.predicts_disparity`` tells train.py's compute_loss/
evaluate which one this is, so they skip their default depth->disparity inversion for the
relative case. That inversion would otherwise round-trip disparity through TWO differently-
clamped spaces (a low-end floor on disparity, then a separate low-end floor on the inverted
depth) and silently lose precision exactly at disparity==0 -- where VDA's own
sky_zero_disparity convention supervises sky pixels.

The metric branch's "depth-like" claim was, until 2026-08-27, an unverified assumption
(carried over from the relative variant's documented behaviour, never independently
checked). EMPIRICALLY CONFIRMED that day (Fable/Opus's ruling: pre-register the expected
pattern before running, then measure) by running THIS class's own forward() -- not just
the vendored infer_video_depth() convenience wrapper -- on a real KITTI clip with the
released Metric-Video-Depth-Anything-Small checkpoint
(scripts/val01_metric_units_check.py, provenance/inventory/
val01_metric_units_check_kitti_20260827.json): raw output correlates +0.887 with GT depth
and -0.826 with GT inverse depth (the depth-like prediction was "positive with depth, near
zero or negative with inverse"; disparity-like would have been the opposite), and its
numeric range (1.9-81.3) sits in the same ballpark as GT depth's own range (2.3-80.0m) --
not GT's inverse-depth range, which would be roughly 0.01-0.4. Corroborated by aligned vs.
unaligned AbsRel landing in the same neighbourhood rather than differing by an order of
magnitude, on BOTH full benchmarks (provenance/inventory/
val01_eval_vda_s_metric_{kitti,bonn}_20260827.json): KITTI 0.0948 aligned vs. 0.2832
unaligned, Bonn 0.0713 aligned vs. 0.2358 unaligned -- both ~3x, not ~10x+. The vendored
video_depth_anything/video_depth.py:135-136's own windowed-stitching loop setting
``scale, shift = 1.0, 0.0`` for metric mode (skipping the cross-window alignment fit
relative mode always computes) is *consistent* context, not independent evidence, and was
never treated as such.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_DEFAULT_VDA_REPO = "/mnt/DATA/DegradedDepth/competition/Video-Depth-Anything"
_DEFAULT_DA_V2_CHECKPOINT = "/mnt/DATA/DegradedDepth/RAMEN/checkpoints/depth_anything_v2_vits.pth"
# released VDA-S (relative, metric); byte-identical to Isambard's benchmark/weights copies (sha256 13379300/3c28432b)
_DEFAULT_RELEASED = {False: "/mnt/DATA/DegradedDepth/RAMEN/checkpoints/video_depth_anything_vits.pth",
                     True: "/mnt/DATA/DegradedDepth/RAMEN/checkpoints/metric_video_depth_anything_vits.pth"}

_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def refuse_lora(trainable: dict, where: str) -> None:
    """For eval scripts that load only head/restore tensors: a DN-01 L1 checkpoint's encoder LoRA would be
    silently dropped (scoring it as A1), so refuse it. LoRA-aware loaders use VDANet.load_lora_merged."""
    if any(k.startswith("lora.") for k in trainable):
        raise SystemExit(f"{where}: checkpoint has encoder LoRA tensors (lora.*); this script would drop them "
                         "-- use a LoRA-aware scorer (val01_eval_vda.py, dn01_eval_*.py, bench_joint.py)")


def refuse_frontend(trainable: dict, where: str) -> None:
    """DN-01 FR: a clip-level front-end (``frontend.*``) must see the whole clip before VDA's windowing; per-window
    or head-only loaders cannot apply it, so refuse rather than score without it (use bench_joint / dn01_eps_scale)."""
    if any(k.startswith("frontend.") for k in trainable):
        raise ValueError(f"{where}: checkpoint has a clip-level front-end (frontend.*) this loader cannot apply "
                         "-- score it with bench_joint.py (ours:) or dn01_eps_scale.py")


def _collect_output(module: nn.Module, _inp, out) -> None:
    # Module-level (not a closure over the model) so deepcopy'd models collect on their own copy.
    module._ramen_out.append(out)


class VDANet(nn.Module):
    def __init__(self, repo: str | None = None, encoder: str = "vits", metric: bool = False,
                 init: str = "da_v2", checkpoint_path: str | None = None,
                 da_v2_checkpoint_path: str | None = None, restoration_head: bool = False,
                 restoration_hidden: int = 32, restoration_fuse_scale: float = 1.0,
                 restoration_feat_norm: str = "none", restoration_rgb_frames: int = 1,
                 restoration_branch: str = "fuse", restoration_unet_ch: int = 32, restoration_unet_scale: float = 0.5,
                 restoration_unet_shuffle: int = 1, restoration_wiener: bool = False,
                 lora_rank: int = 0, lora_alpha: float = 16.0, input_adapter: str = "none",
                 frontend: str = "none", frontend_checkpoint: str | None = None, freeze_depth_head: bool = False,
                 post_wiener: bool = False, frontend_depth_branch: bool = False,
                 frontend_depth_grad: str = "branch", frontend_trunk_grad: float = 1.0):
        super().__init__()
        if encoder not in _MODEL_CONFIGS:
            raise ValueError(f"Unknown VDA encoder={encoder!r}")
        if init not in {"da_v2", "released", "random"}:
            raise ValueError(f"Unknown VDA init={init!r}")
        if repo and not Path(repo).is_dir():
            # TOOL-11's OVDA bug reappearing on VDA: a checkpoint's own saved cfg bakes in
            # the training machine's vda_repo path (e.g. Isambard's /projects/b5dh/...),
            # which does not exist on crispwork. Trusting it unconditionally breaks every
            # in-repo-trained VDA checkpoint's first load on a different host.
            print(f"[VDANet] configured vda_repo={repo!r} does not exist on this host "
                  f"(likely baked in from a different training machine) -- "
                  f"falling back to the local default {_DEFAULT_VDA_REPO}", file=sys.stderr)
            repo = None
        self.repo = str(repo or _DEFAULT_VDA_REPO)
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        from video_depth_anything.video_depth import VideoDepthAnything

        self.core = VideoDepthAnything(**_MODEL_CONFIGS[encoder], metric=metric)
        # infer.py's StreamingDepth.step reads PATCH_SIZE off any wrapped model to pad
        # non-patch-multiple frames (TOOL-08) -- read from the encoder itself (DA-V2/DINOv2,
        # patch_size=14 for all of vits/vitb/vitl) rather than hardcoding, same principle as
        # D3Net's own PATCH_SIZE (models/d3_net.py).
        self.PATCH_SIZE = self.core.pretrained.patch_size

        if init == "released":
            if checkpoint_path is None:
                raise ValueError("init='released' requires checkpoint_path")
            if not Path(checkpoint_path).is_file():   # TOOL-11 again: a cfg baked on another host (DN-01 FR)
                local = _DEFAULT_RELEASED[bool(metric)]
                print(f"[VDANet] configured vda_checkpoint_path={checkpoint_path!r} does not exist on this host -- "
                      f"falling back to the local released copy {local}", file=sys.stderr)
                checkpoint_path = local
            sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.core.load_state_dict(sd, strict=True)
        elif init == "da_v2":
            if da_v2_checkpoint_path and not Path(da_v2_checkpoint_path).is_file():
                # Same TOOL-11-shaped defect as vda_repo above: a checkpoint's own saved
                # cfg bakes in the training machine's da_v2_checkpoint_path (e.g. Isambard's
                # /projects/b5dh/DaBiT/weights/...), which does not exist on crispwork.
                # Loading an already-fully-trained checkpoint overwrites this init anyway
                # (load_d3_checkpoint runs right after construction) -- the exact DA-V2
                # weights used here are immaterial once real trained weights land on top,
                # so falling back to whatever local DA-V2 copy is available is safe.
                print(f"[VDANet] configured da_v2_checkpoint_path={da_v2_checkpoint_path!r} "
                      f"does not exist on this host (likely baked in from a different "
                      f"training machine) -- falling back to the local default "
                      f"{_DEFAULT_DA_V2_CHECKPOINT}", file=sys.stderr)
                da_v2_checkpoint_path = None
            self._load_da_v2_init(str(da_v2_checkpoint_path or _DEFAULT_DA_V2_CHECKPOINT))
        # init == "random": leave the freshly-constructed weights as-is (architecture tests)

        self.metric = bool(metric)
        # Relative mode's native output IS disparity (VDA's own training target); train.py's
        # compute_loss/evaluate read this to skip their default depth->disparity inversion,
        # which would otherwise round-trip through two mismatched clamp floors (1e-6 here vs
        # 1e-3 there) and silently corrupt precision right at disparity==0 -- exactly where
        # VDA's own sky_zero_disparity convention supervises sky pixels. Metric mode's output
        # is genuinely depth-like, so it keeps using the standard inversion path.
        self.predicts_disparity = not self.metric
        self.encoder_name = encoder
        for p in self.core.pretrained.parameters():
            p.requires_grad_(False)
        self.core.pretrained.eval()
        for p in self.core.head.parameters():
            p.requires_grad_(not freeze_depth_head)   # DN-01 FR: the released metric head stays frozen
        # DN-01: optional zero-init RGB restoration head on the post-temporal refinenet1
        # features. train.py's existing enh branch drives it: `enh_decoder` routes
        # model(rgb, return_recon=True) and adds enh_weight * L1(restored, clean frames).
        self.enh_decoder = bool(restoration_head)
        if self.enh_decoder:
            from .restoration import RestorationHead
            scratch = self.core.head.scratch
            self.restore = RestorationHead(scratch.output_conv1.in_channels, restoration_hidden,
                                           restoration_fuse_scale, restoration_feat_norm, restoration_rgb_frames,
                                           restoration_branch, restoration_unet_ch, restoration_unet_scale,
                                           restoration_unet_shuffle, restoration_wiener)
            # VDA's head runs refinenet1 in micro-batches of 4 frames, so the hook must
            # accumulate every chunk, not keep only the last one.
            scratch.refinenet1.register_forward_hook(_collect_output)
        # DN-01 L1: optional encoder LoRA on every DINOv2 block's attn.qkv + attn.proj (B zero-init,
        # so step 0 == the frozen encoder). Injected after the init load (it renames `qkv.weight`
        # -> `qkv.base.weight`); merge_lora() folds it back for zero-cost inference.
        self.lora = None
        if lora_rank > 0:
            from .lora import inject_lora
            self.lora = nn.ModuleList(inject_lora(self.core.pretrained, lora_rank, lora_alpha, "qkv")
                                      + inject_lora(self.core.pretrained, lora_rank, lora_alpha, "proj"))
            assert len(self.lora) == 2 * len(self.core.pretrained.blocks)
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        # DN-01 PA 2026-09-24: optional learned pre-encoder input adapter (restoration.InputAdapter), so the frozen
        # encoder sees adapted frames; the restoration head keeps reading the RAW frames. "none" = no module,
        # identical state dict and output. Built under fork_rng so every other init and the data RNG stream stay
        # exactly the adapter-off run's (the paired A/B against rf1).
        if input_adapter not in ("none", "ps_unet"):
            raise ValueError(f"input_adapter must be 'none' or 'ps_unet', got {input_adapter!r}")
        self.adapter = None
        if input_adapter == "ps_unet":
            from .restoration import InputAdapter
            with torch.random.fork_rng(devices=[]):
                self.adapter = InputAdapter()
        self.last_adapted = None                  # (B,T,3,H,W) x' of the last forward, for PA2's aux L1
        # DN-01 FR 2026-09-24: optional pretrained clip-level video denoiser in front of everything (restoration.
        # PocketFrontEnd); depth AND the restoration head read its output. "none" = no module, identical state dict.
        if frontend not in ("none", "pocketdvdnet"):
            raise ValueError(f"frontend must be 'none' or 'pocketdvdnet', got {frontend!r}")
        self.frontend = None
        # DN-01 FR2 2026-09-25 (header dn01_diag_fr2.yaml): a zero-init depth branch r on Pocket temp2's last
        # features; depth reads x_enc = y.detach() + r ("branch": depth gradient reaches the trunk only via
        # r, never temp2.output_block) or y + r ("full"); restoration = y. False = no module, same state_dict.
        if frontend_depth_branch and (frontend == "none" or post_wiener or restoration_head):
            raise ValueError("model.frontend_depth_branch needs model.frontend, "
                             "no post_wiener and no restoration_head")
        if frontend_depth_grad not in ("branch", "full"):
            raise ValueError(f"frontend_depth_grad must be 'branch' or 'full', got {frontend_depth_grad!r}")
        self.frontend_branch, self.frontend_depth_grad = bool(frontend_depth_branch), frontend_depth_grad
        # DN-01 2026-09-25 (Manager, after the FR2 verdict; build only): beta = the depth gradient's scale into Pocket's
        # shared trunk through r (1.0 = FR2 as run, 0.0 = trunk stop-grad); needs the branch.
        if float(frontend_trunk_grad) != 1.0 and not frontend_depth_branch:
            raise ValueError("model.frontend_trunk_grad needs model.frontend_depth_branch")
        if frontend == "pocketdvdnet":
            from .restoration import PocketFrontEnd
            with torch.random.fork_rng(devices=[]):
                self.frontend = PocketFrontEnd(frontend_checkpoint, depth_branch=self.frontend_branch,
                                               trunk_grad=frontend_trunk_grad)
        # DN-01 FR+W 2026-09-24: blind Wiener on the front-end's output with a pixel-based per-clip estimate
        # (restoration.PixelWiener; alpha zero-init, so step 0 is FR). False = no module, identical state dict.
        if post_wiener and self.frontend is None:
            raise ValueError("model.post_wiener needs model.frontend")
        self.post_wiener = None
        if post_wiener:
            from .restoration import PixelWiener
            with torch.random.fork_rng(devices=[]):
                self.post_wiener = PixelWiener()
        self.last_wiener_est = None               # (B*T,2) per-clip [t_b, t_n] of the last denoise(), for its L1

    def _load_da_v2_init(self, da_v2_checkpoint_path: str) -> None:
        da_v2_sd = torch.load(da_v2_checkpoint_path, map_location="cpu", weights_only=False)
        remapped = {}
        for k, v in da_v2_sd.items():
            if k.startswith("pretrained."):
                remapped[k] = v
            elif k.startswith("depth_head."):
                remapped["head." + k[len("depth_head."):]] = v
        missing, unexpected = self.core.load_state_dict(remapped, strict=False)
        assert not unexpected, f"unexpected keys loading DA-V2 init into VDA-S: {sorted(unexpected)}"
        expected_missing = {k for k in self.core.state_dict() if "motion_modules" in k}
        got_missing = set(missing)
        assert got_missing == expected_missing, (
            "DA-V2 init: missing keys diverge from the expected zero-init temporal set -- "
            f"extra missing: {sorted(got_missing - expected_missing)}, "
            f"unexpectedly present: {sorted(expected_missing - got_missing)}"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.core.pretrained.eval()   # frozen backbone stays eval even in train() mode
        return self

    def depth_head_parameters(self) -> list:
        """The trainable spatial DPT head (everything in ``core.head`` but the motion modules): the ``training.head_lr_mult`` group (protocol v3 WP21: a plain arch=vda FT Metric-VDA-S has the same group as the shared-temporal arms)."""
        return [p for n, p in self.core.head.named_parameters() if "motion_modules" not in n and p.requires_grad]

    def trainable_parameters(self):
        return [p for m in self.trainable_modules().values() for p in m.parameters() if p.requires_grad]

    def trainable_modules(self) -> dict:
        return {"head": self.core.head, **({"restore": self.restore} if self.enh_decoder else {}),
                **({"lora": self.lora} if self.lora is not None else {}),
                **({"adapter": self.adapter} if self.adapter is not None else {}),
                **({"frontend": self.frontend} if self.frontend is not None else {}),
                **({"post_wiener": self.post_wiener} if self.post_wiener is not None else {})}

    def assert_backbone_frozen(self) -> None:
        lora_ids = {id(p) for m in (self.lora or []) for p in [*m.lora_a.parameters(), *m.lora_b.parameters()]}
        assert not any(p.requires_grad for p in self.core.pretrained.parameters() if id(p) not in lora_ids), \
            "VDA-S DINOv2 backbone must stay frozen (only its LoRA adapters train)"

    def load_lora_merged(self, trainable: dict) -> None:
        """Eval loaders: load a train.py checkpoint's ``lora.*`` tensors strictly, then merge. Raises
        if the checkpoint has adapters this model was not built with (they would be dropped)."""
        lora_sd = {k[len("lora."):]: v for k, v in trainable.items() if k.startswith("lora.")}
        if self.lora is None:
            if lora_sd:
                raise ValueError("checkpoint has encoder LoRA tensors but its cfg has no model.vda_lora_rank")
            return
        self.lora.load_state_dict(lora_sd, strict=True)
        self.merge_lora()

    def denoise(self, rgb: torch.Tensor, pair: bool = False):
        """The front-end over each clip: ``(B,C,T,H,W)`` -> same (neighbours reflected at each clip's ends);
        ``pair``: (restored, depth input), which differ only for FR2."""
        out, enc, est = [], [], []
        for c in rgb.permute(0, 2, 1, 3, 4):
            y, x = self.front_pair(c)
            out.append(y)
            enc.append(x)
            est.append(None if self.post_wiener is None else self.post_wiener.last_est)
        self.last_wiener_est = None if self.post_wiener is None else torch.cat(est)
        y = torch.stack(out).permute(0, 2, 1, 3, 4)
        return (y, torch.stack(enc).permute(0, 2, 1, 3, 4)) if pair else y

    def front(self, clip: torch.Tensor) -> torch.Tensor:
        """The clip-level front path on ONE clip ``(T,3,H,W)``: the front-end, then (FR+W) the pixel-estimated Wiener.
        bench_joint applies exactly this to the whole clip before VDA's windowing. FR2: the restored y."""
        x = self.frontend(clip)
        return x if self.post_wiener is None else self.post_wiener(x)

    def front_pair(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(restored, depth input) on ONE clip: the same tensor twice, except FR2's x_enc = y.detach() + r
        ("branch") or y + r ("full"), unclamped as the header declares."""
        if not self.frontend_branch:
            x = self.front(clip)
            return x, x
        y, r = self.frontend(clip, with_branch=True)
        return y, (y.detach() if self.frontend_depth_grad == "branch" else y) + r

    def load_adapter(self, trainable: dict) -> None:
        """Head-only eval loaders: load a train.py checkpoint's ``adapter.*`` tensors strictly. Raises if the
        checkpoint has an input adapter this model was not built with (it would be silently skipped)."""
        sd = {k[len("adapter."):]: v for k, v in trainable.items() if k.startswith("adapter.")}
        if self.adapter is None:
            if sd:
                raise ValueError("checkpoint has input-adapter tensors (adapter.*) but the model was built "
                                 "without model.input_adapter")
            return
        self.adapter.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def merge_lora(self) -> None:
        """Fold each adapter into its base Linear (W += scale * B @ A) and unwrap it, leaving the
        plain VDA-S architecture: same keys, same parameter count, same inference cost."""
        from .lora import LoRALinear
        for mod in list(self.core.pretrained.modules()):
            for name, child in list(mod.named_children()):
                if isinstance(child, LoRALinear):
                    child.base.weight += child.scale * (child.lora_b[0].weight @ child.lora_a[0].weight)
                    setattr(mod, name, child.base)
        self.lora = None

    def parameter_summary(self) -> dict:
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.trainable_parameters())
        return {"total_m": round(tot / 1e6, 2), "trainable_m": round(tr / 1e6, 2)}

    @torch.no_grad()
    def encoder_tokens(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        """Patch tokens of the layers the DPT head reads, ``(B,C,T,H,W)`` in [0,1] ->
        one ``(B, T, N, C)`` tensor per layer. Teacher side of KD2 feature distillation.
        ponytail: a second encoder pass on top of forward(); hook the head's input instead
        if teacher throughput ever matters."""
        B, _C, T, H, W = rgb.shape
        x = ((rgb.permute(0, 2, 1, 3, 4) - self._mean) / self._std).flatten(0, 1)
        idx = self.core.intermediate_layer_idx[self.encoder_name]
        return [tok.reshape(B, T, *tok.shape[1:])
                for tok in self.core.pretrained.get_intermediate_layers(x, idx)]

    def forward(self, rgb: torch.Tensor, return_recon: bool = False, frontend_done: bool = False,
                frontend_only: bool = False):   # (B,C,T,H,W) in [0,1]
        # DN-01 FR: the front-end runs on the WHOLE clip given here. frontend_done: rgb is already its output
        # (bench_joint applies it to the full clip before VDA's 32-frame windowing, as for the cascade);
        # frontend_only: return just the denoised clip (the matched control's training forward).
        # FR2: depth reads x_enc and return_recon returns (depth, y); with frontend_done, rgb is x_enc.
        y = None
        if self.frontend is not None and not frontend_done:
            if self.frontend_branch and not frontend_only:
                y, rgb = self.denoise(rgb, pair=True)
            else:
                rgb = self.denoise(rgb)
        if frontend_only:
            return rgb
        B, C, T, H, W = rgb.shape
        assert H % 14 == 0 and W % 14 == 0, "VDA-S requires H,W multiples of 14 (DINOv2 patch size)"
        x = rgb.permute(0, 2, 1, 3, 4)          # -> (B,T,C,H,W), VDA's own convention
        if self.adapter is not None:
            x = self.last_adapted = self.adapter(x)
        x = (x - self._mean) / self._std       # VDA's own inference path normalises via a
                                                  # separate NormalizeImage transform before
                                                  # calling forward(); replicate that here since
                                                  # this wrapper's forward() IS that call site.
        if self.enh_decoder:
            self.core.head.scratch.refinenet1._ramen_out = []
        out = self.core(x)                        # (B,T,H,W) -- DINOv2 + DPTHeadTemporal
        out = out.unsqueeze(1)                    # -> (B,1,T,H,W), RAMEN convention
        # metric: depth-like (empirically confirmed, not assumed -- see class docstring);
        # relative: native disparity -- either way, self.predicts_disparity tells train.py's
        # loss/eval which one this is, so no conversion happens here.
        if not self.enh_decoder:
            return (out, y) if return_recon and y is not None else out
        feat = torch.cat(self.core.head.scratch.refinenet1._ramen_out, 0)   # (B*T, C, h, w)
        self.core.head.scratch.refinenet1._ramen_out = []
        frames = rgb.permute(0, 2, 1, 3, 4)                                  # (B,T,3,H,W)
        ctx = None
        if self.restore.rgb_frames > 1:
            from .restoration import neighbour_stack
            ctx = neighbour_stack(frames, self.restore.rgb_frames)          # (B*T,3k,H,W): t-k//2 .. t+k//2
        restored = self.restore(feat, frames.flatten(0, 1), ctx, clip_len=T)
        restored = restored.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)   # -> (B,3,T,H,W)
        return (out, restored) if return_recon else out

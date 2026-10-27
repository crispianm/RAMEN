"""DN-01 shared-temporal (ST) joint model (user 2026-09-26: "slightly better but mostly FASTER than a cascade").

Depth = the released Metric-VDA-S (encoder frozen; the DPT temporal head frozen for ST-F, its 4 `motion_modules`
trainable for ST-U -- trained by the depth loss only). Restoration = ONE PocketDVDNet temp2 block (full or half width),
WITHOUT temp1: it reads the raw degraded (t-1, t, t+1) at FULL resolution from the CLIP (reflected at the clip ends, the
PocketFrontEnd rule), plus (st_inject) a zero-init 1x1 projection of VDA's post-temporal refinenet1 features of frame t,
DETACHED (the restoration loss never reaches VDA), bilinearly upsampled and added after temp2's input block. VDA's
32-frame temporal attention replaces the restorer's temporal stage. y = frame_t - residual (Pocket's form), clamped.
Pocket BatchNorm stays in eval mode (stats recalibrated on the training mix by the distillation stage, then frozen).
Step 0: depth == Metric-VDA-S bit-exactly; restoration == the initial temp2 on raw triplets (NOT the Pocket cascade).

K46/K47 WS2 arm B (st_depth_inject; the mirror of the injection): a restorer PRE-PASS (temp2's input_block -> down_0 ->
down_1 on the same raw triplets, WITHOUT the VDA injection, which would be circular) gives a (C1 = down_channels_1)
feature at 1/4 resolution; per motion module k, a ZERO-INIT 1x1 projection of it, area-resized to the module input,
is ADDED to that input (a forward pre-hook). Not detached: the depth losses reach temp2's encoder through it. Zero-init
-> depth and restoration are bit-exact to the same net without the path at step 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTHS = {"full": dict(stem_channels=16, down_channels_0=32, down_channels_1=64, output_mid_channels=32),
          "slim": dict(stem_channels=8, down_channels_0=16, down_channels_1=32, output_mid_channels=16),
          # DN-01 design (c) c2: the 2x-wide SECOND restorer block (1,248,882 params; temp2 full is 321,618)
          "wide2x": dict(stem_channels=32, down_channels_0=64, down_channels_1=128, output_mid_channels=64),
          # RAMEN-L only (experimental restorer scale-up of the gated SECOND block; unused by every Tables 1/2 config)
          "wide3x": dict(stem_channels=48, down_channels_0=96, down_channels_1=192, output_mid_channels=96),
          "wide4x": dict(stem_channels=64, down_channels_0=128, down_channels_1=256, output_mid_channels=128)}


class _Gate(nn.Module):
    """A scalar, zero-initialised gate (design (c)); its own trainable module so it has its own checkpoint key and clip group."""

    def __init__(self) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.zeros(()))


def _collect(module: nn.Module, _inp, out) -> None:
    module._st_out.append(out)              # module-level (not a closure) so deepcopies collect on their own copy


def _inject_restorer(module: nn.Module, args):
    """Arm B's pre-hook on a motion module: input (B,C,T,h,w) += proj(area-resized pre-pass features). ``_st_ctx`` is
    a transient (features (B*T,C1,h1,w1), projection, B, T), set only inside depth_and_feats (a plain tuple: never a
    registered child)."""
    ctx = getattr(module, "_st_ctx", None)
    if ctx is None:
        return None
    feat, proj, B, T = ctx
    x = args[0]
    add = proj(F.interpolate(feat, size=x.shape[-2:], mode="area").to(proj.weight.dtype))
    return (x + add.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4).to(x.dtype), *args[1:])


def temp2_block(width: str) -> nn.Module:
    """A PocketDVDNet PocketDenBlock (temp2 architecture) at ``width`` (full = the published recipe)."""
    from vendor.pocketdvdnet.pocketdvdnet import PocketDenBlock, PocketDVDNetRecipe
    return PocketDenBlock(PocketDVDNetRecipe(**WIDTHS[width]))


def block2_missing(stage2_width) -> bool:
    return stage2_width is None


def triplets(clip: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) -> (N,9,H,W): frames t-1, t, t+1 reflected at the clip ends (PocketFrontEnd's rule)."""
    n = clip.shape[0]
    idx = (torch.arange(n, device=clip.device)[:, None] + torch.arange(-1, 2, device=clip.device)[None]).abs()
    idx = torch.where(idx > n - 1, 2 * (n - 1) - idx, idx).clamp(0, n - 1)
    return clip[idx].flatten(1, 2)


class _GradScale(torch.autograd.Function):
    """Forward identity (``x.view_as(x)``: the same bytes), backward multiplies the gradient by ``scale`` (L15 arm S)."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None


class SharedTemporalNet(nn.Module):
    def __init__(self, vda: nn.Module, width: str = "full", inject: bool = True, unfreeze_motion: bool = False,
                 temp2_init: str | None = None, batch: int = 8, depth_inject: bool = False, unfreeze_head: bool = False,
                 depth_input: str = "raw", stage2_width: str | None = None, decoupled_restorer: bool = False, detach_depth_restorer: bool = False,
                 depth_grad_scale: float = 1.0, stage2_aux: bool = False, rgb_stage2_width: str | None = None) -> None:
        super().__init__()
        if rgb_stage2_width is not None and not decoupled_restorer:
            raise ValueError("st_rgb_stage2_width needs st_decoupled_restorer")
        self.rgb_stage2_width = rgb_stage2_width   # L16: None = the display block2 has the same width as the depth one (every existing model); else only the display (own) block2 is built at this width
        self.vda, self.width, self.inject, self.batch = vda, width, inject, batch
        self.predicts_disparity = vda.predicts_disparity
        self.enh_decoder = False
        self.returns_recon = True                               # train.py: (depth, restored) with return_recon
        for p in vda.parameters():
            p.requires_grad_(False)
        self.unfreeze_motion = unfreeze_motion
        if unfreeze_motion:
            for p in vda.core.head.motion_modules.parameters():
                p.requires_grad_(True)
        # METRICFIX (default off: the V2 recipe keeps the released metric DPT head frozen): ``unfreeze_head`` also trains the rest of the DPT head (projects, resize_layers, the refinenets and BOTH output convs;
        # the encoder stays frozen) -- VDA's own metric stage changed 54/58 DPT and 6/6 output_conv tensors. The motion modules stay a separate module so a head-frozen checkpoint's keys still load.
        self.unfreeze_head = bool(unfreeze_head)
        if self.unfreeze_head:
            if not unfreeze_motion:
                raise ValueError("model.freeze_depth_head false (unfreeze_head) needs model.st_unfreeze_motion: the head arms train the motion modules too")
            for p in self._head_rest().parameters():
                p.requires_grad_(True)
        with torch.random.fork_rng(devices=[]):
            self.temp2 = temp2_block(width)
            self.proj = None
            if inject:
                self.proj = nn.Conv2d(64, WIDTHS[width]["stem_channels"], 1)
                nn.init.zeros_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)
            self.dproj = None
            if depth_inject:                                    # arm B: restorer -> VDA motion-module inputs
                c1 = WIDTHS[width]["down_channels_1"]
                self.dproj = nn.ModuleList(nn.Conv2d(c1, mm.temporal_transformer.norm.num_channels, 1)
                                           for mm in vda.core.head.motion_modules)
                for c in self.dproj:
                    nn.init.zeros_(c.weight)
                    nn.init.zeros_(c.bias)
        # DN-01 design (c) (default off = none of this exists, tests pin byte-identity with the base): depth CONSUMES the restored frames.
        # depth = vda(raw + coupling_scale * g * (R - raw)), g a scalar zero-init gate (step 0 == depth(raw)), R NOT detached (the depth loss reaches the restorer through g).
        # c2: R = stage 1 (temp2) refined by a SECOND 2x-wide block on stage 1's output: R = clamp(stage1 - stage2_scale * g2 * residual2), g2 a zero-init scalar (step 0 == c1).
        # coupling_scale / stage2_scale are NOT parameters: evaluation overrides (engagement measures: depth(R := raw) at coupling_scale 0, R(g2 := 0) at stage2_scale 0).
        if depth_input not in ("raw", "restored"):
            raise ValueError(f"depth_input must be 'raw' or 'restored', got {depth_input!r}")
        if depth_input == "restored" and inject:
            raise ValueError("st_depth_input='restored' needs st_inject False: the injection feeds the restorer with the depth pass's features, which would now depend on the restorer's own output")
        if stage2_width is not None and depth_input != "restored":
            raise ValueError("st_stage2_width needs st_depth_input='restored'")
        self.depth_input, self.coupling_scale, self.stage2_scale = depth_input, 1.0, 1.0
        self.gate = self.gate2 = self.block2 = None
        if depth_input == "restored":
            self.gate = _Gate()
            if stage2_width is not None:
                with torch.random.fork_rng(devices=[]):
                    self.block2 = temp2_block(stage2_width if rgb_stage2_width is None else rgb_stage2_width)
                self.gate2 = _Gate()
        # L15 DECOUPLED inference (default off = none of this exists; state_dict / forward byte-identical to the base).
        # rgb_restorer = this net's own temp2 / block2 / gate2 (the keys train.py trains and checkpoints); depth_restorer = a SECOND frozen copy of the same three modules that
        # ONLY the depth path reads (depth = vda(raw + g * (R_depth - raw))), so the depth output cannot depend on the rgb_restorer's weights. The restored RGB output is the rgb_restorer's.
        self.decoupled_restorer = bool(decoupled_restorer)
        self.detach_depth_restorer = bool(detach_depth_restorer)   # L15 option (default off): stop-grad from the depth losses into the restorer (depth sees R.detach(); the restoration loss still trains it)
        # L15 arm S (both default off = no op, forward bytes identical): depth_grad_scale s: the depth losses' gradient into the restorer is multiplied by s (forward identity, _GradScale);
        # stage2_aux: restore_batch also keeps last_stage2 = the UNGATED stage-2 output clamp(r1 - res2) (the gate g2 is not in it) for train.py's auxiliary restoration loss.
        self.depth_grad_scale = float(depth_grad_scale)
        self.stage2_aux = bool(stage2_aux)
        if self.stage2_aux and stage2_width is None:
            raise ValueError("st_stage2_aux needs st_stage2_width (a stage 2)")
        self.last_stage2 = None
        self._s2_buf = []
        self.depth_restorer = None
        self._depth_restorer_loaded = False
        if self.decoupled_restorer:
            if depth_input != "restored" or block2_missing(stage2_width) or inject:
                raise ValueError("decoupled_restorer needs st_depth_input='restored', st_stage2_width and st_inject False")
            with torch.random.fork_rng(devices=[]):
                self.depth_restorer = nn.ModuleDict({"temp2": temp2_block(width), "block2": temp2_block(stage2_width), "gate2": _Gate()})
            for p in self.depth_restorer.parameters():
                p.requires_grad_(False)
        if temp2_init:
            sd = torch.load(temp2_init, map_location="cpu", weights_only=False)
            self.temp2.load_state_dict(sd.get("temp2", sd), strict=True)
        elif width == "full":                                   # tests / timing: the released v1.0 temp2
            from .restoration import PocketFrontEnd
            self.temp2.load_state_dict(PocketFrontEnd().net.temp2.state_dict(), strict=True)
        hook_mod = vda.core.head.scratch.refinenet1
        hook_mod._st_out = []
        hook_mod.register_forward_hook(_collect)
        if self.dproj is not None:
            for mm in vda.core.head.motion_modules:
                mm._st_ctx = None
                mm.register_forward_pre_hook(_inject_restorer)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vda.train(mode)
        for m in self.temp2.modules():                          # BN frozen (stats from the distillation stage)
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        if self.depth_restorer is not None:                     # L15: the frozen depth restorer never leaves eval mode (BN buffers must not drift)
            self.depth_restorer.eval()
        if self.block2 is not None:                             # design (c) c2: block2's BN stays at its identity init stats (affine trained): one clip per forward would otherwise store the
            for m in self.block2.modules():                     # average of per-clip statistics (the distillation BN defect), and train / eval must agree
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def _head_rest(self) -> nn.ModuleDict:
        """The DPT head WITHOUT its motion modules, as a view (the same parameters, not registered as a child of this net, so no state_dict key appears twice)."""
        return nn.ModuleDict({n: c for n, c in self.vda.core.head.named_children() if n != "motion_modules"})

    def depth_head_parameters(self) -> list:
        return [p for p in self._head_rest().parameters() if p.requires_grad]

    def trainable_modules(self) -> dict:
        mods = {"temp2": self.temp2, **({"proj": self.proj} if self.proj is not None else {}),
                **({"dproj": self.dproj} if self.dproj is not None else {}),
                **({"gate": self.gate} if self.gate is not None else {}),                     # design (c)
                **({"block2": self.block2, "gate2": self.gate2} if self.block2 is not None else {})}
        if self.unfreeze_motion:
            mods["motion_modules"] = self.vda.core.head.motion_modules
        if self.unfreeze_head:
            mods["depth_head"] = self._head_rest()
        return mods

    def assert_backbone_frozen(self) -> None:
        self.vda.assert_backbone_frozen()

    def trainable_parameters(self):
        return [p for m in self.trainable_modules().values() for p in m.parameters() if p.requires_grad]

    def parameter_summary(self) -> dict:
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.trainable_parameters())
        return {"total_m": round(tot / 1e6, 2), "trainable_m": round(tr / 1e6, 2)}

    def restorer_feats(self, rgb: torch.Tensor) -> torch.Tensor:
        """Arm B's pre-pass: (B,3,T,H,W) -> temp2 down_1 features (B*T,C1,ceil(H/4),ceil(W/4)) of each clip's raw
        triplets (restore_clip's triplets + reflect pad), no injection."""
        blk, out = self.temp2, []
        for clip in rgb.permute(0, 2, 1, 3, 4):
            h, w = clip.shape[-2:]
            ph, pw = -h % 4, -w % 4
            st = triplets(F.pad(clip, (0, pw, 0, ph), mode="reflect") if ph or pw else clip)
            out += [blk.down_1(blk.down_0(blk.input_block(st[s:s + self.batch])))
                    for s in range(0, len(st), self.batch)]
        return torch.cat(out)

    def depth_and_feats(self, rgb: torch.Tensor):
        """(B,3,T,H,W) -> depth (B,1,T,H,W) and refinenet1 features (B*T,64,h,w) (detached). Design (c): depth reads the gated restored frames."""
        if self.decoupled_restorer:
            return self._depth_and_feats(self.consume(rgb, self.restore_batch(rgb, "depth")))
        if self.depth_input == "restored":
            return self._depth_and_feats(self.consume(rgb, self.restore_batch(rgb)))
        return self._depth_and_feats(rgb)

    def restore_batch(self, rgb: torch.Tensor, which: str = "rgb") -> torch.Tensor:
        """Design (c): the restored frames R of a (B,3,T,H,W) batch, NOT detached (stage 1 = temp2 on each clip's raw triplets, no injection; c2: + the gated second block)."""
        frames = rgb.permute(0, 2, 1, 3, 4)
        out = torch.stack([self.restore_two_stage(frames[b], which) for b in range(rgb.shape[0])]).permute(0, 2, 1, 3, 4)
        if self.stage2_aux and which == "rgb":
            self.last_stage2 = torch.stack(self._s2_buf).permute(0, 2, 1, 3, 4)
            self._s2_buf = []
        return out

    def _parts(self, which: str):
        """(temp2, block2, gate2) of the 'rgb' restorer (this net's own modules) or the 'depth' restorer (the frozen decoupled copy)."""
        if which == "rgb":
            return self.temp2, self.block2, self.gate2
        if which == "depth":
            if self.depth_restorer is None or not self._depth_restorer_loaded:
                raise RuntimeError("depth restorer requested but the net is not decoupled / its depth_restorer was never loaded (load_packed)")
            d = self.depth_restorer
            return d["temp2"], d["block2"], d["gate2"]
        raise ValueError(which)

    def restore_decoupled(self, rgb: torch.Tensor):
        """(B,3,T,H,W) -> (depth_input_restored R_depth [from depth_restorer], rgb_output R_rgb [from rgb_restorer])."""
        return self.restore_batch(rgb, "depth"), self.restore_batch(rgb, "rgb")

    def load_packed(self, ck: dict) -> None:
        """Load a packed decoupled checkpoint (scripts/l15_pack_decoupled.py): trainable_ema into the trainable modules, depth_restorer into the frozen copy."""
        sd = ck.get("trainable_ema", ck.get("trainable"))
        if self.rgb_stage2_width is not None:   # L16 pack v3: the display block2 has a different width, so trainable_ema's restorer keys (depth widths) cannot go into the own modules
            self._load_packed_v3(ck, sd)
            return
        torch.nn.ModuleDict(self.trainable_modules()).load_state_dict(sd, strict=True)
        if "rgb_restorer" in ck:   # pack format v2: trainable_ema is the init's (depth path + its ORIGINAL restorer); rgb_restorer replaces this net's own restorer tensors AFTER the depth copy is taken
            self.depth_restorer.load_state_dict({k: v for k, v in sd.items() if k.startswith(("temp2.", "block2.", "gate2."))}, strict=True)
            torch.nn.ModuleDict({"temp2": self.temp2, "block2": self.block2, "gate2": self.gate2}).load_state_dict(ck["rgb_restorer"], strict=True)
        else:                      # legacy format (trainable_ema carries the rgb restorer; separate depth_restorer section)
            self.depth_restorer.load_state_dict(ck["depth_restorer"], strict=True)
        self._depth_restorer_loaded = True

    def _load_packed_v3(self, ck: dict, sd: dict) -> None:
        """L16 pack v3 (mixed widths): every key of trainable_ema and rgb_restorer is consumed exactly once."""
        if "rgb_restorer" not in ck:
            raise ValueError("pack v3 (st_rgb_stage2_width) needs rgb_restorer")
        R = ("temp2.", "block2.", "gate2.")
        own = {n: m for n, m in self.trainable_modules().items() if n not in ("temp2", "block2", "gate2")}
        rest = {k: v for k, v in sd.items() if not k.startswith(R)}
        dres = {k: v for k, v in sd.items() if k.startswith(R)}
        assert len(rest) + len(dres) == len(sd)
        torch.nn.ModuleDict(own).load_state_dict(rest, strict=True)
        self.depth_restorer.load_state_dict(dres, strict=True)
        assert all(k.startswith(R) for k in ck["rgb_restorer"]), "rgb_restorer holds non-restorer keys"
        torch.nn.ModuleDict({"temp2": self.temp2, "block2": self.block2, "gate2": self.gate2}).load_state_dict(ck["rgb_restorer"], strict=True)
        n_own = sum(len(m.state_dict()) for m in own.values())
        assert n_own == len(rest) and len(self.depth_restorer.state_dict()) == len(dres), "unconsumed keys"
        self._depth_restorer_loaded = True

    def restore_two_stage(self, clip: torch.Tensor, which: str = "rgb") -> torch.Tensor:
        """ONE clip (N,3,H,W) -> R: stage 1 = restore_clip (no injection); with block2: R = clamp(stage1 - stage2_scale * g2 * residual2(triplets(stage1)))."""
        temp2, block2, gate2 = self._parts(which)
        r1 = self.restore_clip(clip, None, temp2)
        if block2 is None:
            return r1
        n, _, h, w = r1.shape
        ph, pw = -h % 4, -w % 4

        def _res2(r):
            st = triplets(F.pad(r, (0, pw, 0, ph), mode="reflect") if ph or pw else r)
            out = []
            for s in range(0, n, self.batch):
                x = st[s:s + self.batch]
                x0 = block2.input_block(x)
                x1 = block2.down_0(x0)
                x2 = block2.up_1(block2.down_1(x1))
                x1 = block2.up_0(x1 + x2)
                out.append(block2.output_block(x0 + x1)[..., :h, :w])
            return torch.cat(out)

        res = _res2(r1)
        if self.stage2_aux and which == "rgb":
            # L15 arm S (Manager): the aux loss trains ONLY block2: a second pass of block2 on r1.detach() (identical values), so no aux gradient reaches stage 1 by any path
            self._s2_buf.append((r1.detach() - _res2(r1.detach())).clamp(0, 1))
        return (r1 - self.stage2_scale * gate2.g * res).clamp(0, 1)

    def consume(self, rgb: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
        """Design (c): the depth input raw + coupling_scale * g * (R - raw) (g = 0 -> raw exactly)."""
        if self.detach_depth_restorer:
            restored = restored.detach()
        elif self.depth_grad_scale != 1.0:
            restored = _GradScale.apply(restored, self.depth_grad_scale)
        return rgb + self.coupling_scale * self.gate.g * (restored - rgb)

    def _depth_and_feats(self, rgb: torch.Tensor):
        mod = self.vda.core.head.scratch.refinenet1
        mod._st_out = []
        mms = self.vda.core.head.motion_modules
        if self.dproj is not None:
            f = self.restorer_feats(rgb)
            for mm, proj in zip(mms, self.dproj):
                mm._st_ctx = (f, proj, rgb.shape[0], rgb.shape[2])
        try:
            depth = self.vda(rgb)
        finally:
            if self.dproj is not None:
                for mm in mms:
                    mm._st_ctx = None
        feats = torch.cat(mod._st_out, 0).detach() if mod._st_out else None
        mod._st_out = []
        return depth, feats

    def restore_clip(self, clip: torch.Tensor, feats: torch.Tensor | None, temp2: nn.Module | None = None) -> torch.Tensor:
        """ONE clip (N,3,H,W) in [0,1] + its per-frame features (N,64,h,w) -> restored (N,3,H,W): triplets from the
        CLIP, padded to a multiple of 4 (reflect), temp2 (+ injection), y = frame_t - residual, clamped."""
        n, _, h, w = clip.shape
        ph, pw = -h % 4, -w % 4
        st = triplets(F.pad(clip, (0, pw, 0, ph), mode="reflect") if ph or pw else clip)
        blk, out = (self.temp2 if temp2 is None else temp2), []
        for s in range(0, n, self.batch):
            x = st[s:s + self.batch]
            x0 = blk.input_block(x)
            if self.proj is not None and feats is not None:
                x0 = x0 + F.interpolate(self.proj(feats[s:s + self.batch].to(x0.dtype)), size=x0.shape[-2:],
                                        mode="bilinear", align_corners=False)
            x1 = blk.down_0(x0)
            x2 = blk.up_1(blk.down_1(x1))
            x1 = blk.up_0(x1 + x2)
            out.append((x[:, 3:6] - blk.output_block(x0 + x1)).clamp(0, 1)[..., :h, :w])
        return torch.cat(out)

    def forward(self, rgb: torch.Tensor, return_recon: bool = False):   # (B,3,T,H,W) in [0,1]
        if self.decoupled_restorer:             # L15: depth reads the frozen depth_restorer; the returned RGB is the rgb_restorer's
            r_depth, restored = self.restore_decoupled(rgb)
            depth, _ = self._depth_and_feats(self.consume(rgb, r_depth))
            return (depth, restored) if return_recon else depth
        if self.depth_input == "restored":      # design (c): R is computed ONCE and both consumed by depth and returned for the restoration loss
            restored = self.restore_batch(rgb)
            depth, _ = self._depth_and_feats(self.consume(rgb, restored))
            return (depth, restored) if return_recon else depth
        depth, feats = self.depth_and_feats(rgb)
        if not return_recon:
            return depth
        B, _, T = rgb.shape[:3]
        frames = rgb.permute(0, 2, 1, 3, 4)
        restored = torch.stack([self.restore_clip(frames[b], None if feats is None else feats[b * T:(b + 1) * T])
                                for b in range(B)]).permute(0, 2, 1, 3, 4)
        return depth, restored

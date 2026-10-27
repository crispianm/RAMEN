"""
RAMEN models package (public release: headline architecture + per-frame baselines)
=================

    build_model         — dispatch on cfg['model']['arch']: 'vda' (RAMEN / depth-only fine-tune),
                          'dav2' and 'daac' (per-frame baselines)
    load_d3_checkpoint  — load a RAMEN training checkpoint into a model in place
    trainable_modules   — the checkpointed sub-modules of a run
"""

from __future__ import annotations

import torch



def load_ramen(path, device: str = "cpu"):
    """Load a composed RAMEN / RAMEN-L file (ramen.pt, ramen_l.pt; pack v2/v3) -> eval-mode net with depth path AND display head (``load_packed``, strict)."""
    ck = torch.load(path, map_location="cpu", weights_only=False) if isinstance(path, (str, bytes)) or hasattr(path, "__fspath__") else path
    if "rgb_restorer" not in ck and "depth_restorer" not in ck:
        raise ValueError(f"{path} is not a composed RAMEN checkpoint (no rgb_restorer / depth_restorer section)")
    m = dict(ck["cfg"]["model"])
    m["st_temp2_init"] = None                       # trainable_ema carries all of temp2; the init path is a training-host path
    m["st_decoupled_restorer"] = True
    net = build_model({**ck["cfg"], "model": m})
    net.load_packed(ck)
    return net.to(device).eval()


def load_d3_checkpoint(model, ckpt, use_ema: bool = True, depth_only: bool = False) -> None:
    """Load a RAMEN training checkpoint into ``model`` (in place).

    Handles every checkpoint format the trainer emits:
      * frozen run  → ``trainable`` / ``trainable_ema`` (head **+** temporal mixers)
      * unfrozen run → ``model`` / ``model_ema`` (whole net)
      * legacy frozen run → ``head`` / ``head_ema`` (head only; temporal stays identity)

    ``ckpt`` may be a path or an already-loaded dict. ``use_ema`` prefers the EMA
    weights (cleaner) when present.

    A composed checkpoint (RAMEN / RAMEN-L: has ``rgb_restorer`` / ``depth_restorer``) carries a display head
    this function would silently drop, so it RAISES unless ``depth_only=True`` (depth path only, display
    head ignored on purpose). Use :func:`load_ramen` for the full model.
    """
    if isinstance(ckpt, (str, bytes)):
        ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not depth_only and ("rgb_restorer" in ckpt or "depth_restorer" in ckpt):
        raise ValueError("composed checkpoint (rgb_restorer/depth_restorer present): load_d3_checkpoint would silently drop the "
                         "display head. Use models.load_ramen(path), or pass depth_only=True for the depth path only.")

    def pick(primary, fallback):
        return ckpt[primary] if (use_ema and primary in ckpt) else ckpt[fallback]

    if "model" in ckpt or "model_ema" in ckpt:                 # unfrozen: whole net
        model.load_state_dict(pick("model_ema", "model"), strict=False)
    elif "trainable" in ckpt or "trainable_ema" in ckpt:       # frozen: head + temporal (+ channel …)
        tgt = torch.nn.ModuleDict(trainable_modules(model))
        src = pick("trainable_ema", "trainable")
        dst = tgt.state_dict()
        # Back-compat: pre-MoLoRA checkpoints (big2..big16) named adapters lora_{a,b}.weight;
        # the current ModuleList wrapper expects lora_{a,b}.0.weight. Remap so old checkpoints
        # still load (no-op for current-format checkpoints whose keys already match).
        src = {(k.replace(".lora_a.weight", ".lora_a.0.weight").replace(".lora_b.weight", ".lora_b.0.weight")
                if k not in dst else k): v for k, v in src.items()}
        # Back-compat, same reasoning as the remap above: an OVDA checkpoint trained
        # before EXPT-07 has no `lora.*` keys at all if its embedded cfg carries a
        # dormant lora_rank inherited from ramen_relative_base.yaml (harmless before
        # OVDANet read that key, live now that it does) -- a freshly-built model with
        # that stale cfg would then have a `lora` submodule the saved weights never
        # trained. Missing keys there are expected and left at their fresh init;
        # unexpected keys would mean genuine corruption and must still raise.
        missing, unexpected = tgt.load_state_dict(src, strict=False)
        assert not unexpected, f"unrecognised checkpoint keys: {unexpected}"
    elif "head" in ckpt or "head_ema" in ckpt:                 # legacy: head only
        model.head.load_state_dict(pick("head_ema", "head"))
    else:
        raise KeyError(f"Unrecognised RAMEN checkpoint keys: {list(ckpt)[:8]}")


def trainable_modules(model) -> dict:
    """The trainable sub-modules of a frozen-backbone RAMEN run, keyed for checkpointing.
    Single source of truth shared by the trainer's EMA target and checkpoint loading."""
    if hasattr(type(model), "trainable_modules"):   # DPVideoNet supplies its own mapping
        return model.trainable_modules()
    if getattr(model, "trainable_scope", "head") == "temporal_only":
        temporal = getattr(model, "temporal", None)
        if temporal is None:
            raise ValueError("trainable_scope=temporal_only requires a temporal module")
        return {"temporal": temporal}
    mods = {"head": model.head}
    if getattr(model, "temporal", None) is not None:
        mods["temporal"] = model.temporal
    if getattr(model, "channel", None) is not None and any(p.numel() for p in model.channel.parameters()):
        mods["channel"] = model.channel
    for key in ("frontend", "profile", "lora"):  # D3v2 restoration + LoRA baseline
        if getattr(model, key, None) is not None:
            mods[key] = getattr(model, key)
    return mods


def build_model(cfg: dict):
    """Dispatch on ``cfg['model']['arch']``: ``'vda'`` (RAMEN / depth-only fine-tune), ``'dav2'``, ``'daac'``."""
    m = cfg.get("model", cfg)
    arch = str(m.get("arch", "vda"))
    if arch == "vda":
        from .vda_net import VDANet
        net = VDANet(repo=m.get("vda_repo"),
                      encoder=str(m.get("vda_encoder", "vits")),
                      metric=bool(m.get("metric", False)),
                      init=str(m.get("vda_init", "da_v2")),
                      checkpoint_path=m.get("vda_checkpoint_path"),
                      da_v2_checkpoint_path=m.get("da_v2_checkpoint_path"),
                      restoration_head=bool(m.get("restoration_head", False)),
                      restoration_hidden=int(m.get("restoration_hidden", 32)),
                      restoration_fuse_scale=float(m.get("restoration_fuse_scale", 1.0)),
                      restoration_feat_norm=str(m.get("restoration_feat_norm", "none")),
                      restoration_rgb_frames=int(m.get("restoration_rgb_frames", 1)),
                      restoration_branch=str(m.get("restoration_branch", "fuse")),
                      restoration_unet_ch=int(m.get("restoration_unet_ch", 32)),
                      restoration_unet_scale=float(m.get("restoration_unet_scale", 0.5)),
                      restoration_unet_shuffle=int(m.get("restoration_unet_shuffle", 1)),
                      restoration_wiener=bool(m.get("restoration_wiener", False)),
                      # vda_-prefixed: every VDA arm inherits a dormant lora_rank: 32 from
                      # ramen_relative_base.yaml, so `lora_rank` itself must not switch this on.
                      lora_rank=int(m.get("vda_lora_rank", 0)),
                      lora_alpha=float(m.get("vda_lora_alpha", 16.0)),
                      input_adapter=str(m.get("input_adapter", "none")),
                      frontend=str(m.get("frontend", "none")), frontend_checkpoint=m.get("frontend_checkpoint"),
                      freeze_depth_head=bool(m.get("freeze_depth_head", False)),
                      post_wiener=bool(m.get("post_wiener", False)),
                      frontend_depth_branch=bool(m.get("frontend_depth_branch", False)),
                      frontend_depth_grad=str(m.get("frontend_depth_grad", "branch")),
                      frontend_trunk_grad=float(m.get("frontend_trunk_grad", 1.0)),
                      checkpoint_sha256=m.get("vda_checkpoint_sha256"),            # RAMEN-L: refuse a released file whose sha256 differs
                      encoder_grad_ckpt=bool(m.get("vda_encoder_grad_ckpt", False)))  # RAMEN-L: recompute the frozen trunk in backward
        if m.get("shared_temporal"):   # DN-01 final-v2 shared-temporal arms (models/shared_temporal.py)
            from .shared_temporal import SharedTemporalNet
            return SharedTemporalNet(net, width=str(m.get("st_width", "full")), inject=bool(m.get("st_inject", True)),
                                     unfreeze_motion=bool(m.get("st_unfreeze_motion", False)),
                                     temp2_init=m.get("st_temp2_init"),
                                     depth_inject=bool(m.get("st_depth_inject", False)),
                                     unfreeze_head=(m.get("freeze_depth_head", True) is False),   # METRICFIX: an EXPLICIT freeze_depth_head: false trains the DPT head too (absent / true = the old frozen head)
                                     depth_input=str(m.get("st_depth_input", "raw")),                    # DN-01 design (c): default off
                                     stage2_width=m.get("st_stage2_width"),
                                     decoupled_restorer=bool(m.get("st_decoupled_restorer", False)),
                                     detach_depth_restorer=bool(m.get("st_detach_depth_restorer", False)),
                                     depth_grad_scale=float(m.get("st_depth_grad_scale", 1.0)), stage2_aux=bool(m.get("st_stage2_aux", False)),
                                     rgb_stage2_width=m.get("st_rgb_stage2_width"))
        return net
    if arch == "dav2":    # protocol v3 WP22: per-frame DA-V2-S, frozen encoder + trainable DPT head, metric head 'metres' (a) or 'sigmoid' (b)
        from .dav2_net import DAV2Net
        return DAV2Net(repo=m.get("dav2_repo"), head=str(m.get("dav2_head", "metres")), max_depth=float(m.get("dav2_max_depth", 80.0)),
                       init=str(m.get("dav2_init", "relative")), checkpoint_path=m.get("dav2_checkpoint_path"),
                       unfreeze_last_blocks=int(m.get("dav2_unfreeze_last_blocks", 0)))
    if arch == "daac":    # protocol v3 WP23: per-frame DA-AC, same scope; 'relative' (disparity) or 'metric' (metres)
        from .daac_net import DAACNet
        return DAACNet(repo=m.get("daac_repo"), variant=str(m.get("daac_variant", "relative")),
                       init=str(m.get("daac_init", "released")), checkpoint_path=m.get("daac_checkpoint_path"))
    raise ValueError(f"Unknown model.arch={arch!r}: this release supports 'vda', 'dav2' and 'daac'")


__all__ = ["build_model", "load_d3_checkpoint", "load_ramen", "trainable_modules"]

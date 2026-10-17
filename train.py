#!/usr/bin/env python3
"""
RAMEN training
===========

Trains the lightweight temporal head on top of the frozen DAv3 (da3-small)
backbone for relative video depth.  The frozen/trainable boundary is enforced
(the optimizer only ever sees head params; the backbone is asserted frozen).

Pipeline: bf16 autocast, gradient accumulation + clipping, cosine LR, EMA of the
head, periodic aligned validation (AbsRel / δ1 / RMSE via the VDA disparity
protocol), best/last checkpointing, JSONL logging.

Usage::

    python train.py --config configs/d3_local_train.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import re as _re
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import VideoDepthClipDataset
from metrics import compute_depth_metrics
from models import build_model, load_d3_checkpoint, trainable_modules


SEED_SCHEMES = ("add", "x1000")


def seed_everything(seed: int, rank: int = 0, scheme: str = "add") -> int:
    """Seed every RNG used by the trainer and return the rank-local seed. ``scheme`` (``training.seed_scheme``, E4, opt-in): "add" (default, unchanged) = seed + rank, which ALIASES across
    runs (seed 43 rank 1 == seed 44 rank 0: every rank-k stream of one run is another run's rank-(k-1) stream); "x1000" = seed * 1000 + rank (distinct for any seed and < 1000 ranks)."""
    if scheme not in SEED_SCHEMES:
        raise ValueError(f"training.seed_scheme must be one of {SEED_SCHEMES}, got {scheme!r}")
    local_seed = int(seed) * 1000 + int(rank) if scheme == "x1000" else int(seed) + int(rank)
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(local_seed)
    return local_seed


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# --------------------------------------------------------------------------- #
# Optimizer — only trainable (head) params; backbone must contribute zero.
# --------------------------------------------------------------------------- #
def build_optimizer(model, cfg: dict, verbose: bool = True) -> torch.optim.Optimizer:
    """AdamW over trainable params. Frozen → head only (with freeze assertion);
    unfrozen → discriminative LR (head at ``learning_rate``, backbone at
    ``backbone_lr``, default 0.1× to protect the pretrained features)."""
    t = cfg.get("training", {})
    head_lr = float(t.get("learning_rate", 1e-4))
    wd = float(t.get("weight_decay", 0.01))

    if getattr(model, "backbone_trainable", False):
        head_p = [p for p in model.head.parameters() if p.requires_grad]
        bb_p = [p for p in model.backbone.parameters() if p.requires_grad]
        assert bb_p, "unfreeze_backbone=True but the backbone has no trainable params."
        bb_lr = float(t.get("backbone_lr", head_lr * 0.1))
        opt = torch.optim.AdamW(
            [{"params": head_p, "lr": head_lr}, {"params": bb_p, "lr": bb_lr}],
            weight_decay=wd, betas=(0.9, 0.999),
        )
    else:
        model.assert_backbone_frozen()
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert sum(p.numel() for p in trainable) == sum(
            p.numel() for p in model.trainable_parameters()
        ), "Optimizer param set != model trainable params."
        assert trainable, "No trainable parameters — the head was not registered."
        tscale = float(t.get("temporal_lr_scale", 1.0))   # ⑤ smaller LR for the temporal block
        hmult = t.get("head_lr_mult")                     # METRICFIX (default off): the trainable DPT depth head (model.freeze_depth_head false) in its OWN param group at learning_rate * head_lr_mult
        rscale = float(t.get("renorm_lr_scale", 1.0))     # boosted LR for the gated renorm branch
        temporal = getattr(model, "temporal", None)
        if hmult is not None:
            hp = list(model.depth_head_parameters()) if hasattr(model, "depth_head_parameters") else []
            if not hp:
                raise ValueError("training.head_lr_mult needs a model with a trainable DPT depth head (model.freeze_depth_head: false)")
            hids = {id(p) for p in hp}
            opt = torch.optim.AdamW(
                [{"params": [p for p in trainable if id(p) not in hids], "lr": head_lr}, {"params": hp, "lr": head_lr * float(hmult)}],
                weight_decay=wd, betas=(0.9, 0.999))
        elif rscale != 1.0 and temporal is not None:
            # Gate + renorm params train through a (near-)zero gate → tiny effective
            # gradients at the base LR (ar1: gate moved 5e-4 in 4k steps). Boost them.
            rids = {id(p) for n, p in temporal.named_parameters()
                    if p.requires_grad and (".renorm." in n or n.endswith(".gate") or ".icsa." in n
                                            or ".gate_mlp." in n or ".rel_score." in n or ".rel_norm." in n)}
            ren_p = [p for p in trainable if id(p) in rids]
            rest_p = [p for p in trainable if id(p) not in rids]
            assert ren_p, "renorm_lr_scale set but no renorm/gate params found"
            opt = torch.optim.AdamW(
                [{"params": rest_p, "lr": head_lr}, {"params": ren_p, "lr": head_lr * rscale}],
                weight_decay=wd, betas=(0.9, 0.999))
        elif tscale != 1.0 and temporal is not None:
            tids = {id(p) for p in temporal.parameters() if p.requires_grad}
            temp_p = [p for p in trainable if id(p) in tids]
            rest_p = [p for p in trainable if id(p) not in tids]
            opt = torch.optim.AdamW(
                [{"params": rest_p, "lr": head_lr}, {"params": temp_p, "lr": head_lr * tscale}],
                weight_decay=wd, betas=(0.9, 0.999))
        elif float(t.get("gate_lr_scale", 1.0)) != 1.0 and getattr(model, "gate", None) is not None:
            # DN-01 design (c): the scalar zero-init gates (depth input g, c2's second block g2) train in their OWN group at learning_rate x gate_lr_scale (absent / 1.0 = the single-group optimizer above, unchanged):
            # a scalar gate at the base LR can move at most ~lr per step and may never engage the coupling within 4000 steps.
            gids = {id(p) for m in (model.gate, getattr(model, "gate2", None)) if m is not None for p in m.parameters()}
            opt = torch.optim.AdamW(
                [{"params": [p for p in trainable if id(p) not in gids], "lr": head_lr},
                 {"params": [p for p in trainable if id(p) in gids], "lr": head_lr * float(t["gate_lr_scale"])}],
                weight_decay=wd, betas=(0.9, 0.999))
        elif t.get("encoder_lr") is not None and getattr(model, "lora", None) is not None:
            # DN-01 L1: encoder LoRA at its own LR. Head group FIRST so a LoRA-less checkpoint's
            # single group maps onto it on resume (_load_resume_trainable).
            lids = {id(p) for p in model.lora.parameters()}
            opt = torch.optim.AdamW(
                [{"params": [p for p in trainable if id(p) not in lids], "lr": head_lr},
                 {"params": [p for p in trainable if id(p) in lids], "lr": float(t["encoder_lr"])}],
                weight_decay=wd, betas=(0.9, 0.999))
        else:
            opt = torch.optim.AdamW(trainable, lr=head_lr, weight_decay=wd, betas=(0.9, 0.999))
        if hmult is not None and float(t.get("gate_lr_scale", 1.0)) != 1.0 and getattr(model, "gate", None) is not None:
            # protocol v3 WP17 (design (c) on the a1h base): the head_lr_mult branch above is taken FIRST (a1h sets it), which would silently leave the gates at the base LR; carve them out into their own
            # group at learning_rate x gate_lr_scale here (same rule as the gate branch above; absent gate_lr_scale / no gate = nothing happens)
            gids = {id(p) for m in (model.gate, getattr(model, "gate2", None)) if m is not None for p in m.parameters()}
            for g in opt.param_groups:
                g["params"] = [p for p in g["params"] if id(p) not in gids]
            opt.add_param_group({"params": [p for p in trainable if id(p) in gids], "lr": head_lr * float(t["gate_lr_scale"])})

    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    if verbose:
        s = model.parameter_summary()
        print(
            f"[params] total={s['total_m']}M trainable={s['trainable_m']}M "
            f"(unfrozen_backbone={getattr(model, 'backbone_trainable', False)})",
            flush=True,
        )
    return opt


# --------------------------------------------------------------------------- #
# EMA of the trainable head.
# --------------------------------------------------------------------------- #
class EMA:
    """Exponential moving average of a module's parameters/buffers."""

    def __init__(self, module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module) -> None:
        for k, v in module.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().to(s.dtype), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, module) -> None:
        module.load_state_dict(self.shadow, strict=True)


# --------------------------------------------------------------------------- #
# Data.
# --------------------------------------------------------------------------- #
_NO_GT_SOURCE_TYPES = ("video_only", "paired_video")   # teacher-only; never valid as a val source


def _make_source(spec: dict, split: str, common: dict):
    """``_make_source_inner`` + the K80 option-B per-source ``metric_supervision`` flag (absent = nothing set, byte-identical): the dataset then emits ``metric_sup`` on every clip and SiLog runs only on the
    sources that say true. Only the VDA-semantics DEGRADED classes carry it (a stale loader fails loudly rather than silently supervising a source)."""
    ds = _make_source_inner(spec, split, common)
    if "metric_supervision" in spec:
        if not isinstance(spec["metric_supervision"], bool):
            raise ValueError(f"data.sources[].metric_supervision must be a bool, got {spec['metric_supervision']!r}")
        if not hasattr(ds, "metric_supervision"):
            raise ValueError("data.sources[].metric_supervision needs resize_mode: vda with degrade on (the VDA-semantics degraded loaders)")
        ds.metric_supervision = spec["metric_supervision"]
    if "sparse_gt" in spec:                        # protocol v3 WP13: the source's GT is sparse LiDAR: the dataset emits ``sparse`` (1.0 / 0.0) and the VDA loss uses the sparse-safe masked SSI for it; absent = nothing set
        if not isinstance(spec["sparse_gt"], bool):
            raise ValueError(f"data.sources[].sparse_gt must be a bool, got {spec['sparse_gt']!r}")
        if not hasattr(ds, "sparse_gt"):
            raise ValueError("data.sources[].sparse_gt is only implemented for the protocol-v3 real sources (data/v3_clip.py) and the VDA-semantics degraded loaders (resize_mode: vda with degrade on)")
        ds.sparse_gt = spec["sparse_gt"]
    if "silog_max_depth" in spec:                  # METRICFIX: a PER-SOURCE SiLog far ceiling (m); absent = nothing set, byte-identical
        v = spec["silog_max_depth"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
            raise ValueError(f"data.sources[].silog_max_depth must be a positive number, got {v!r}")
        if spec.get("metric_supervision") is not True:
            raise ValueError("data.sources[].silog_max_depth needs metric_supervision: true on the same source (it bounds that source's SiLog term)")
        ds.silog_max_depth = float(v)
    return ds


def _make_source_inner(spec: dict, split: str, common: dict):
    """Instantiate one training/val source from a declarative spec (dispatch on `type`).

    `resize_mode: vda` (per-source or common) swaps in the VDA-semantics variant
    (data/vda_semantics_dataset.py: short-edge resize + square crop, matching VDA's own
    transform) in place of the source's existing stretch/bespoke resize.

    `degrade=True` combined with `resize_mode='vda'` (matrix item 3, Opus/Fable,
    2026-08-28) swaps in the further `VDASemanticsDegraded*` variant, which degrades
    AFTER resize+crop -- reusing the same `_degrade_clip` degradation every other
    degraded dataset class already uses, not a separate implementation. `pseudo_gt`
    has `resize_mode='vda'` support since K80 option B (VDASemanticsPseudoGTDataset and
    its degraded sibling); without it the legacy stretch loader is used.

    `type: video_only` (DIST-01 arm (c), 2026-09-18) is the unlabeled-real-video source:
    supported under BOTH loaders, all-zero depth/mask, online-teacher-only supervision.
    """
    from data import (VKITTIDataset, VideoDepthClipDataset, IRSDataset,
                      PointOdysseyDataset, PseudoGTVideoDataset, VideoOnlyDataset)
    from data.vda_semantics_dataset import (
        VDASemanticsIRSDataset, VDASemanticsPointOdysseyDataset,
        VDASemanticsTartanAirDataset, VDASemanticsVKITTIDataset,
        VDASemanticsDegradedIRSDataset, VDASemanticsDegradedPointOdysseyDataset,
        VDASemanticsDegradedTartanAirDataset, VDASemanticsDegradedVKITTIDataset,
        VDASemanticsVideoOnlyDataset, VDASemanticsDegradedVideoOnlyDataset,
        VDASemanticsPairedVideoDataset,
    )
    t = spec["type"]
    if spec.get("max_depth") is not None:     # METRICFIX (default off: absent = the common data.max_depth, byte-identical): a PER-SOURCE validity upper bound / clip (m) for this source's GT, so the 80 m training cap can be lifted per source
        common = {**common, "max_depth": float(spec["max_depth"])}
    vf = float(spec.get("val_fraction", 0.1))
    vda = spec.get("resize_mode", common.get("resize_mode")) == "vda"
    vda_degrade = vda and bool(spec.get("degrade", common.get("degrade")))
    if t == "vkitti":
        cls = (VDASemanticsDegradedVKITTIDataset if vda_degrade
              else VDASemanticsVKITTIDataset if vda else VKITTIDataset)
        extra = {k: spec[k] for k in ("cameras", "scene_holdout", "no_holdout") if k in spec}   # K80 option B: both cameras, whole-scene holdout (absent = the old kwargs)
        ds = cls(root=spec["root"], split=split, val_fraction=vf,
                 variants=spec.get("variants", ["clone"]),
                 degrade=spec.get("degrade", common["degrade"]),
                 **({"exclude_eval_overlap": True} if spec.get("exclude_eval_overlap") else {}),      # ultrareview: drop VKITTI Scene01/Scene20 (twins of KITTI eval drives); absent = unchanged. Set it on the val source too
                 **extra,
                 **{k: v for k, v in common.items() if k not in ("degrade", "resize_mode")})
        if spec.get("vkitti_max_depth") is not None:   # DN-01 D0 metric cap; absent = unchanged
            if not vda:
                raise ValueError("vkitti_max_depth is only implemented for resize_mode: vda")
            ds.vkitti_max_depth = float(spec["vkitti_max_depth"])
        if spec.get("vkitti_sky_as_valid_depth"):   # Add.44 arm VK655; absent = unchanged. Needs the cap raised above 655.35, else the sentinel stays excluded
            if float(spec.get("vkitti_max_depth") or 0.0) <= 655.35:
                raise ValueError("vkitti_sky_as_valid_depth needs vkitti_max_depth > 655.35 (e.g. 656)")
            ds.sky_as_valid_depth = True
        return ds
    if t == "tartanair":
        cls = (VDASemanticsDegradedTartanAirDataset if vda_degrade
              else VDASemanticsTartanAirDataset if vda else VideoDepthClipDataset)
        ds = cls(root=spec["root"], split=split, val_fraction=vf,
                 max_trajectories=spec.get("max_trajectories"), envs=spec.get("envs"),
                 **{k: v for k, v in common.items() if k != "resize_mode"})
        if spec.get("ts2_sky"):   # outdoor-debug TS2 sky/mask recipe (data/tartanair_ts2.py); absent = unchanged (released rule)
            if not vda:
                raise ValueError("ts2_sky is only implemented for resize_mode: vda")
            if not spec.get("ts2_seg_root"):
                raise ValueError("ts2_sky needs ts2_seg_root (dir holding <env>/<difficulty>/seg_lcam_front.zip)")
            from data.tartanair_ts2 import TS2Labeller
            ds.ts2 = TS2Labeller(spec["ts2_seg_root"], spec.get("ts2_env_table"), max_depth=ds.max_depth)
        elif spec.get("ts2_seg_root") or spec.get("ts2_env_table"):
            raise ValueError("ts2_seg_root / ts2_env_table given without ts2_sky: true")
        return ds
    if t == "irs":
        cls = (VDASemanticsDegradedIRSDataset if vda_degrade
              else VDASemanticsIRSDataset if vda else IRSDataset)
        return cls(root=spec["root"], split=split, val_fraction=vf,
                   test_fraction=float(spec.get("test_fraction", 0.0)), families=spec.get("families"),
                   **{k: v for k, v in common.items() if k != "resize_mode"})
    if t == "dynamic_replica":      # protocol v3 WP5 dr-loader: Dynamic Replica left RGB + metric depth (data/dynamic_replica_dataset.py; metric_supervision / sampling_mass come from the spec like irs)
        from data.dynamic_replica_dataset import DynamicReplicaDataset, _vda_classes
        vcls, vdcls = _vda_classes()
        cls = vdcls if vda_degrade else vcls if vda else DynamicReplicaDataset
        return cls(root=spec["root"], split=split, val_fraction=vf,
                   test_fraction=float(spec.get("test_fraction", 0.0)),
                   **{k: v for k, v in common.items() if k != "resize_mode"})
    if t == "pointodyssey":
        cls = (VDASemanticsDegradedPointOdysseyDataset if vda_degrade
              else VDASemanticsPointOdysseyDataset if vda else PointOdysseyDataset)
        extra = {"keep_list": spec["po_keep_list"], "keep_sha256": spec.get("po_keep_sha256")} if spec.get("po_keep_list") else {}   # K80 option B: the committed background-depth filter
        return cls(root=spec["root"], split=split,
                   val_fraction=float(spec.get("val_fraction", 0.0)),
                   **extra,
                   **{k: v for k, v in common.items() if k != "resize_mode"})
    if t == "video_only":
        # DIST-01 arm (c), 2026-09-18: unlabeled REAL video, RGB only, no depth tree read.
        # Works under both loaders (legacy stretch resize, and resize_mode='vda' short-edge
        # 518 + crop with the same post-resize degradation as the other VDA-semantics
        # sources). depth/mask come back all-zero, so the GT term is exactly zero and the
        # online teacher is the only supervision -- see data/video_only_dataset.py.
        cls = (VDASemanticsDegradedVideoOnlyDataset if vda_degrade
              else VDASemanticsVideoOnlyDataset if vda else VideoOnlyDataset)
        return cls(rgb_root=spec["rgb_root"], split=split, val_fraction=vf,
                   max_trajectories=spec.get("max_trajectories"),
                   **{k: v for k, v in common.items() if k != "resize_mode"})
    if t == "paired_video":
        # DIST-01 arm (e), 2026-09-20: REAL bright/dark twins (BVI-RLV). Train-only, no GT,
        # no synthetic degradation -- see VDASemanticsPairedVideoDataset.
        if not vda:
            raise ValueError("paired_video sources require resize_mode='vda'")
        return VDASemanticsPairedVideoDataset(
            rgb_root=spec["rgb_root"], dark_root=spec["dark_root"], split=split, val_fraction=vf,
            max_trajectories=spec.get("max_trajectories"),
            **{k: v for k, v in common.items() if k != "resize_mode"})
    if t == "pseudo_gt":
        if vda:      # K80 option B: the VDA-semantics pseudo-GT loader (short-edge resize + crop, the shared degradation); absent resize_mode keeps the legacy stretch loader
            from data.vda_semantics_dataset import VDASemanticsDegradedPseudoGTDataset, VDASemanticsPseudoGTDataset
            cls = VDASemanticsDegradedPseudoGTDataset if vda_degrade else VDASemanticsPseudoGTDataset
            return cls(rgb_root=spec["rgb_root"], depth_root=spec["depth_root"], split=split, val_fraction=vf,
                       max_trajectories=spec.get("max_trajectories"),
                       **{k: v for k, v in common.items() if k != "resize_mode"})
        return PseudoGTVideoDataset(rgb_root=spec["rgb_root"], depth_root=spec["depth_root"],
                                    split=split, val_fraction=vf,
                                    **{k: v for k, v in common.items() if k != "resize_mode"})
    if t in V3_REAL_SOURCES:    # protocol v3 WP13 (kitti_raw; WP14 / WP12 add their types to V3_REAL_SOURCES): real-data clip sources on data/v3_clip.py, always VDA-semantics
        if not vda:
            raise ValueError(f"{t} sources require resize_mode='vda'")
        from data.v3_clip import with_degradation
        cls = V3_REAL_SOURCES[t]()
        cls = with_degradation(cls) if vda_degrade else cls
        kw = {k: spec[k] for k in V3_REAL_SPEC_KEYS.get(t, ()) if k in spec}
        merged = {k: v for k, v in common.items() if k != "resize_mode"}
        merged.update(kw)     # protocol v3 WP27 F1: a per-source spec key (frame_stride) overrides the data-level `common` value
        return cls(split=split, **merged)
    raise ValueError(f"unknown source type {t!r}")


def _kitti_raw_cls():
    from data.kitti_raw_train import KITTIRawTrainDataset
    return KITTIRawTrainDataset


def _nyu_train_cls():
    from data.nyu_train import NYUTrainDataset
    return NYUTrainDataset


def _scannetpp_train_cls():
    from data.scannetpp_train import ScanNetPPTrainDataset
    return ScanNetPPTrainDataset


def _spring_cls():
    from data.spring_dataset import SpringDataset
    return SpringDataset


V3_REAL_SOURCES = {"kitti_raw": _kitti_raw_cls, "nyu_train": _nyu_train_cls, "scannetpp_train": _scannetpp_train_cls, "spring": _spring_cls}   # type -> lazy class getter (protocol v3 WP13 / WP14 / WP12)
V3_REAL_SPEC_KEYS = {"kitti_raw": ("root", "gt_root", "manifest", "val_fraction", "allow_missing", "frame_stride"),             # the spec keys each type's constructor takes
                     "nyu_train": ("root", "val_fraction", "exclude_video_test"), "scannetpp_train": ("root", "splits", "exclude_splits", "allow_missing", "extra_root", "extra_scene_list", "extra_scene_list_sha256", "frame_stride"), "spring": ("root", "sequences", "val_fraction")}


def effective_depth_bound(ds) -> float:
    """METRICFIX: the effective TRAINING-side upper bound (metres) of a source's GT validity mask (a pixel is valid iff 0 < depth < bound). VKITTI: the 655 m sky sentinel (strict), lowered by ``vkitti_max_depth`` when
    set; TartanAir: ``max_depth`` (its sky rule, non-finite or > 1e4 m, is separate); IRS / PointOdyssey / pseudo-GT: ``max_depth``. Printed per source at start-up so a cap that was meant to be lifted is visible."""
    if hasattr(ds, "vkitti_max_depth"):
        from data.vkitti_dataset import _SKY_DEPTH_M
        cap = getattr(ds, "vkitti_max_depth", None)
        if getattr(ds, "sky_as_valid_depth", False):   # Add.44 VK655: the sentinel (655.35 m) is inside the bound
            return float(cap)
        return float(_SKY_DEPTH_M if cap is None else min(_SKY_DEPTH_M, cap))
    return float(getattr(ds, "max_depth", float("nan")))


SPLIT_SEED_SOURCES = ("vkitti", "tartanair", "irs", "pointodyssey", "pseudo_gt")      # the source types whose dataset takes `split_seed`
SPLIT_SEED_SOURCES = SPLIT_SEED_SOURCES + ("kitti_raw", "nyu_train", "spring", "scannetpp_train")                              # protocol v3 WP13 (the real sources whose drive / scene hold-out takes `split_seed`)
SPLIT_SEED_SOURCES = SPLIT_SEED_SOURCES + ("dynamic_replica",)   # protocol v3 WP5 dr-loader


def _with_split_seed(common: dict, d: dict) -> None:
    """E4 (opt-in ``data.split_seed``): seeds ONLY the held-out scene / trajectory shuffle of the sources; ``data.seed`` keeps driving everything else (val degradation, torch RNG,
    samplers). Absent = nothing is added, so the datasets see exactly the old kwargs (byte-identical)."""
    if d.get("split_seed") is None:
        return
    bad = sorted({s.get("type") for s in list(d.get("sources", [])) + list(d.get("val_sources", [])) if s.get("type") not in SPLIT_SEED_SOURCES})
    if bad:
        raise ValueError(f"data.split_seed is not implemented for source types {bad} (supported: {SPLIT_SEED_SOURCES})")
    common["split_seed"] = int(d["split_seed"])


def _build_multisource(cfg: dict, split: str, world_size: int, rank: int):
    """Declarative multi-dataset loader: uniform per-dataset sampling for train (each source
    contributes equally, VDA recipe); equal-sized held-out subsets per source for val, interleaved
    so any evaluated prefix stays dataset-balanced."""
    from torch.utils.data import ConcatDataset, Subset, WeightedRandomSampler
    d = cfg.get("data", {})
    num_frames = int(d.get("num_frames", 8))
    clip_stride = num_frames if split == "val" else int(d.get("clip_stride", 8))
    common = dict(
        num_frames=num_frames,
        image_size=tuple(d.get("image_size", (518, 518))),
        clip_stride=clip_stride,
        frame_stride=int(d.get("frame_stride", 1)),
        max_depth=float(d.get("max_depth", 80.0)),
        seed=int(d.get("seed", 42)),
        degrade=bool(d.get("val_degrade", d.get("degrade", True)) if split == "val" else d.get("degrade", True)),
        degrade_mode=d.get("degrade_mode", "full"),
        degrade_severity=(d.get("val_degrade_severity", d.get("degrade_severity")) if split == "val"
                          else d.get("degrade_severity")),
        max_clips_per_video=d.get("max_clips_per_video"),
        resize_mode=d.get("resize_mode"),
    )
    _with_split_seed(common, d)

    if split == "val":
        specs = d.get("val_sources", d["sources"])
        n = int(d.get("val_clips_per_source", 20))
        caps = []
        vid2src = {}   # video_id -> source type, for per-source val reporting
        _assert_no_video_only_val_source({"data": {"val_sources": specs}})   # also checked at load_config
        for s in specs:
            p = _make_source(s, "val", common)
            L = len(p)
            k = min(n, L)
            idx = [int(i * L / k) for i in range(k)] if k else []   # evenly-spaced subset
            caps.append(Subset(p, idx))
            for j in idx:
                vid = p.clips[j][2]
                if s["type"] == "vkitti":          # VKITTIDataset prefixes its ids at __getitem__
                    vid = f"vkitti/{vid}"
                vid2src[vid] = s["type"]
        cat = ConcatDataset(caps)
        offs = [0]
        for p in caps:
            offs.append(offs[-1] + len(p))
        order = []                                                  # round-robin interleave
        for i in range(max((len(p) for p in caps), default=0)):
            for k, p in enumerate(caps):
                if i < len(p):
                    order.append(offs[k] + i)
        dataset = Subset(cat, order)
        weights = None
    else:
        parts = [_make_source(s, "train", common) for s in d["sources"]]
        if any(getattr(p, "silog_max_depth", None) is not None for p in parts):   # METRICFIX per-source SiLog ceiling: every clip must carry the key (collate), so the sources without one get +inf (= no ceiling)
            for p in parts:
                if getattr(p, "silog_max_depth", None) is None:
                    p.silog_max_depth = float("inf")
        _depth_bounds = [(s["type"], effective_depth_bound(p)) for s, p in zip(d["sources"], parts)]
        dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
        # DIST-01 arm (c): make the GT / no-GT split of the training mix visible at startup.
        vo = sum(len(p) for s, p in zip(d["sources"], parts) if s.get("type") in _NO_GT_SOURCE_TYPES)
        _source_clip_counts = (sum(len(p) for p in parts) - vo, vo)
        weights = None
        sampling = d.get("sampling", "equal" if d.get("uniform_sampler", True) else "off")
        if sampling == "equal":                      # per-clip 1/len -> each source sampled equally
            weights = []
            for p in parts:
                weights += [1.0 / max(len(p), 1)] * len(p)
        elif sampling == "proportional":
            # NOT part of VDA's own recipe -- corrected 2026-08-27, verified against the
            # paper's supplementary (verbatim: "we employ a uniform sampler to ensure that
            # each dataset contributes equally during training"), contradicting an earlier
            # inference this mode was built from. Left in place as a generically-tested,
            # unused option rather than reverted -- do not enable it for a VDA config.
            # Each source's total sampling mass is proportional to its raw frame count
            # (not its clip count, and not equal per source). frame_count is declared per
            # source in the config rather than computed at runtime: it comes from a
            # recorded, hashed dataset-inventory pass (avoids re-scanning the whole corpus
            # on every launch -- the same directory walk that OOMs Isambard's login node --
            # and keeps the exact numbers actually used auditable in the config itself,
            # not implicit).
            weights = []
            for s, p in zip(d["sources"], parts):
                fc = s.get("frame_count")
                if not isinstance(fc, (int, float)) or fc <= 0:
                    raise ValueError(
                        f"sampling='proportional' requires a positive data.sources[].frame_count "
                        f"declared per source; got {fc!r} for source type {s.get('type')!r}"
                    )
                weights += [float(fc) / max(len(p), 1)] * len(p)
        elif sampling == "mass":
            # K80 option B: each source's total sampling mass = its declared ``sampling_mass`` (VDA's OWN published frame counts, tartanair 0.31M : vkitti 0.04M : pointodyssey 0.1M : irs 0.1M, and the
            # pseudo-GT sources' share of the total), NOT this host's inventory: so the inventory cross-check of 'proportional' does not apply, and the mass is never read from ``frame_count``.
            weights = []
            for s, p in zip(d["sources"], parts):
                sm = s.get("sampling_mass")
                if isinstance(sm, bool) or not isinstance(sm, (int, float)) or sm <= 0:
                    raise ValueError(f"sampling='mass' requires a positive data.sources[].sampling_mass; got {sm!r} for source type {s.get('type')!r}")
                weights += [float(sm) / max(len(p), 1)] * len(p)
        elif sampling != "off":
            raise ValueError(f"unknown data.sampling {sampling!r} (expected 'equal', 'proportional', 'mass', or 'off')")

    bs = int(cfg.get("training", {}).get("batch_size", 1))
    bs = bs if split == "train" else max(1, int(d.get("val_batch_size", bs)))
    sampler = None
    if split == "train" and weights is not None:
        # Weighted sampling is WITH replacement, so under DDP nothing needs sharding: every rank
        # draws from the same per-source weights with its own generator, 1/world_size of the
        # epoch each. This keeps the recipe's per-source balance on multi-GPU runs (KD2, 2026-09-20).
        gen = None
        if world_size > 1:
            # the sampler generator is seeded seed*1000+rank; under training.seed_scheme "x1000" the global RNG gets the SAME integer, so the sampler is moved to +250 (E4) to keep the two streams apart
            gen = torch.Generator().manual_seed(int(d.get("seed", 0)) * 1000 + rank
                                                 + (250 if str(cfg.get("training", {}).get("seed_scheme", "add")) == "x1000" else 0))
        sampler = WeightedRandomSampler(weights, num_samples=max(1, len(weights) // world_size),
                                        replacement=True, generator=gen)
    elif world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=(split == "train"), drop_last=(split == "train"))
    nw = int(d.get("num_workers", 0))
    loader = DataLoader(dataset, batch_size=bs, shuffle=(split == "train" and sampler is None),
                        sampler=sampler, num_workers=nw, pin_memory=True,
                        drop_last=(split == "train"), persistent_workers=nw > 0)
    if split == "val":
        loader.vid2src = vid2src   # evaluate() picks this up for the per-source breakdown
    else:
        loader.gt_clips, loader.video_only_clips = _source_clip_counts
        loader.depth_bounds = _depth_bounds
    return loader, sampler


def build_dataloader(cfg: dict, split: str, world_size: int = 1, rank: int = 0):
    """Build a clip DataLoader. Train = TartanAir (+ optional VKITTI, Scene18 excluded);
    val = held-out VKITTI Scene18 (the KITTI-domain eval proxy) when a vkitti_root is set,
    else TartanAir val. DDP shards both train and val via DistributedSampler."""
    from torch.utils.data import ConcatDataset
    from data import VKITTIDataset

    d = cfg.get("data", {})
    if d.get("sources") is not None:   # declarative multi-dataset path (uniform sampler + 4-split val)
        return _build_multisource(cfg, split, world_size, rank)
    num_frames = int(d.get("num_frames", 8))
    # Val tiles the sequence exactly (stride = clip length) so every frame is evaluated
    # once; a smaller stride skips frames (small T) or double-counts them (large T) in both
    # the per-video alignment fit and the metric average, making cross-T numbers incomparable.
    # Train keeps its overlapping stride.
    clip_stride = num_frames if split == "val" else int(d.get("clip_stride", 8))
    common = dict(
        num_frames=num_frames,
        image_size=tuple(d.get("image_size", (518, 518))),
        clip_stride=clip_stride,
        frame_stride=int(d.get("frame_stride", 1)),
        max_depth=float(d.get("max_depth", 80.0)),
        seed=int(d.get("seed", 42)),
        degrade=bool(d.get("val_degrade", d.get("degrade", True)) if split == "val"
                     else d.get("degrade", True)),  # val_degrade lets train+val differ (clean-train control)
        degrade_mode=d.get("degrade_mode", "full"),  # str or list[str] — list → sampled per clip
        degrade_severity=(d.get("val_degrade_severity", d.get("degrade_severity")) if split == "val"
                          else d.get("degrade_severity")),  # fixed s for eval sweeps; None = sample.
                          # val_degrade_severity pins val to a severe point while train still curricula.
        teacher_cache_dir=d.get("teacher_cache_dir"),  # offline teacher distillation targets
        teacher_name=str(d.get("teacher_name", "depth_pro")),  # which cached teacher (e.g. video_depth_anything)
        max_clips_per_video=d.get("max_clips_per_video"),  # cap per-trajectory clips (113k-frame TA trajs)
    )
    _with_split_seed(common, d)
    vkitti_root = d.get("vkitti_root")
    held_out = d.get("val_scenes", ["Scene18"])
    _tartanair_n = 0  # tracks TartanAir clip count for weighted sampler

    if split == "val" and vkitti_root:
        # Held-out VKITTI Scene18 (clean variant) — the primary eval proxy.
        dataset = VKITTIDataset(root=vkitti_root, split="val", scenes=held_out,
                                variants=d.get("vkitti_variants", ["clone"]),
                                val_fraction=1.0, **common)
    else:
        parts = []
        if d.get("root"):
            parts.append(VideoDepthClipDataset(
                root=d["root"], split=split, val_fraction=float(d.get("val_fraction", 0.1)),
                max_trajectories=d.get("max_trajectories"), envs=d.get("envs"), **common))
            _tartanair_n = len(parts[-1])
        if vkitti_root:  # train: all VKITTI scenes except the held-out eval scene
            parts.append(VKITTIDataset(
                root=vkitti_root, split=split, val_fraction=float(d.get("val_fraction", 0.1)),
                variants=d.get("vkitti_variants", ["clone"]), exclude_scenes=held_out, **common))
            natural_variants = d.get("vkitti_natural_variants", [])
            if natural_variants and split == "train":
                natural_common = {**common, "degrade": False}
                parts.append(VKITTIDataset(
                    root=vkitti_root, split=split, val_fraction=float(d.get("val_fraction", 0.1)),
                    variants=natural_variants, exclude_scenes=held_out, **natural_common))
        if not parts:
            raise ValueError("No dataset configured (set data.root and/or data.vkitti_root).")
        dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)

    bs = int(cfg.get("training", {}).get("batch_size", 1))
    bs = bs if split == "train" else max(1, int(d.get("val_batch_size", bs)))
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=(split == "train"), drop_last=(split == "train"),
        )
    elif split == "train" and _tartanair_n > 0 and float(d.get("vkitti_sample_weight", 0)) > 0:
        from torch.utils.data import WeightedRandomSampler
        _vkitti_n = len(dataset) - _tartanair_n
        _vkitti_w = float(d.get("vkitti_sample_weight", 1.0))
        _weights = [1.0] * _tartanair_n + [_vkitti_w] * _vkitti_n
        sampler = WeightedRandomSampler(_weights, num_samples=len(_weights), replacement=True)
    nw = int(d.get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=nw,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=nw > 0,
    )
    return loader, sampler


def severity_schedule(step: int, hold: int, start: float, cap: float, warmup: int) -> float:
    """Train-time severity cap at optimizer step ``step``: 0 for ``step < hold`` (clean),
    then linear ``start -> cap`` over ``warmup`` steps. hold=0, start=0.3 is the legacy
    curriculum exactly."""
    if step < hold:
        return 0.0
    return start + (cap - start) * min(1.0, (step - hold) / warmup)


def _start_state(dataset, t: dict, gstep: int, rank: int, resumed: bool) -> None:
    """Called once, after any resume. Sets the severity cap for the FIRST step from ``gstep``
    (a run resumed at step 8000 starts at step 8000's cap, not step 0's clean one) and, on a
    resume, applies ``training.resume_reseed`` (DN-01 C seed replicates: same checkpoint,
    different post-resume data order) after the checkpoint's RNG state was restored."""
    if resumed and t.get("resume_reseed") is not None:
        seed_everything(int(t["resume_reseed"]), rank)
    sev_warmup = int(t.get("severity_warmup_steps", 0))
    sev_cap = float(t.get("severity_max_cap", 1.0))
    if sev_warmup > 0:
        _set_severity_max(dataset, severity_schedule(gstep, int(t.get("severity_hold_steps", 0)),
                                                     float(t.get("severity_warmup_start", 0.3)), sev_cap,
                                                     sev_warmup))
    elif sev_cap != 1.0:
        _set_severity_max(dataset, sev_cap)            # no warmup: apply the static cap immediately


def _load_resume_trainable(ema_target, optimizer, ema, resume: dict) -> list[str]:
    """Frozen-backbone resume of trainable weights, optimizer and EMA shadow. Strict, except that
    a model with LoRA may resume from a LoRA-less checkpoint (DN-01 L1 from A1 step8000): the
    ``lora.*`` keys keep their zero-B init, the saved optimizer groups are reused as-is and the
    LoRA group (built last) starts fresh, and the EMA shadow takes the live LoRA weights.
    Returns the missing (LoRA) keys."""
    strict = getattr(ema_target, "lora", None) is None
    missing, unexpected = ema_target.load_state_dict(resume.get("trainable", resume.get("trainable_ema")),
                                                     strict=strict)
    osd, shadow = resume["optimizer"], dict(resume["ema"])
    if missing:
        assert not unexpected and all(k.startswith("lora.") for k in missing), (missing, unexpected)
        cur = optimizer.state_dict()["param_groups"]
        assert len(cur) == len(osd["param_groups"]) + 1 and \
            len(cur[-1]["params"]) == sum(p.requires_grad for p in ema_target.lora.parameters()), \
            "resuming a LoRA-less checkpoint needs training.encoder_lr (LoRA in its own last group)"
        osd = {**osd, "param_groups": osd["param_groups"] + [cur[-1]]}
        live = ema_target.state_dict()
        shadow.update({k: live[k].detach().clone() for k in missing})
    else:
        assert not unexpected, unexpected
    optimizer.load_state_dict(osd)
    for g in optimizer.param_groups:   # load_state_dict never checks shapes; a param-order slip would
        for p in g["params"]:          # otherwise attach another tensor's Adam moments silently
            st = optimizer.state.get(p, {})
            assert "exp_avg" not in st or st["exp_avg"].shape == p.shape, "optimizer state/param mismatch"
    ema.shadow = shadow
    return sorted(missing)


def _set_dataset_attr(dataset, name: str, value) -> None:
    """Set ``name`` on every leaf of a dataset tree (ConcatDataset / Subset, nested: the val loader is
    Subset(ConcatDataset(Subset, ...))) that has it (before workers fork)."""
    from torch.utils.data import ConcatDataset, Subset
    if isinstance(dataset, ConcatDataset):
        for p in dataset.datasets:
            _set_dataset_attr(p, name, value)
    elif isinstance(dataset, Subset):
        _set_dataset_attr(dataset.dataset, name, value)
    elif hasattr(dataset, name):
        setattr(dataset, name, value)


def _set_severity_max(dataset, value: float) -> None:
    """Set the train-time degradation severity cap (curriculum). The cap lives in a
    shared-memory tensor on each dataset, so forked persistent workers see updates."""
    from torch.utils.data import ConcatDataset
    parts = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    for p in parts:
        sm = getattr(p, "severity_max", None)
        if sm is not None:
            sm[0] = value


# --------------------------------------------------------------------------- #
# Loss — scale-shift-invariant disparity + temporal stability.
# --------------------------------------------------------------------------- #
def _scale_shift_align(pred, target, mask):
    """Per-frame least-squares align of ``pred`` to ``target``. Tensors ``(N,H,W)``."""
    N = pred.shape[0]
    p, g, m = pred.reshape(N, -1), target.reshape(N, -1), mask.reshape(N, -1)
    a00 = (m * p * p).sum(1)
    a01 = (m * p).sum(1)
    a11 = m.sum(1)
    b0 = (m * p * g).sum(1)
    b1 = (m * g).sum(1)
    det = a00 * a11 - a01 * a01
    eps = 1e-6
    ok = det.abs() > eps
    s = torch.where(ok, (a11 * b0 - a01 * b1) / det.clamp_min(eps), torch.ones_like(det))
    t = torch.where(ok, (a00 * b1 - a01 * b0) / det.clamp_min(eps), torch.zeros_like(det))
    return s[:, None, None] * pred + t[:, None, None]


def _gradient_loss_multiscale(pred, target, mask, scales: int = 4) -> torch.Tensor:
    """VDA's multi-scale spatial gradient matching (``loss.py::GradientLoss``), ported.

    ``pred``/``target``/``mask`` are ``(N, H, W)`` in the normalised disparity space.
    Uses average pooling for scale decimation to prevent aliasing.
    """
    total = pred.new_zeros(())
    for scale in range(scales):
        step = 2 ** scale
        if step > 1:
            # Prevent aliasing by proper average pooling instead of striding
            p = F.avg_pool2d(pred.unsqueeze(1), kernel_size=step, stride=step).squeeze(1)
            g = F.avg_pool2d(target.unsqueeze(1), kernel_size=step, stride=step).squeeze(1)
            m = F.avg_pool2d(mask.unsqueeze(1), kernel_size=step, stride=step).squeeze(1)
        else:
            p, g, m = pred, target, mask
            
        diff = m * (p - g)
        gx = (diff[:, :, 1:] - diff[:, :, :-1]).abs() * (m[:, :, 1:] * m[:, :, :-1])
        gy = (diff[:, 1:, :] - diff[:, :-1, :]).abs() * (m[:, 1:, :] * m[:, :-1, :])
        total = total + (gx.sum() + gy.sum()) / m.sum().clamp_min(1.0)
    return total


_VDA_LOSS_MODULE_CACHE: dict = {}


def _import_vda_loss_module(repo: str | None = None):
    """Lazily import the vendored loss/loss.py from the VDA repo (same on-demand-import
    pattern as models/vda_net.py / models/ovda_net.py). Cached per repo path so repeated
    training steps don't re-import."""
    from models.vda_net import _DEFAULT_VDA_REPO
    repo = str(repo or _DEFAULT_VDA_REPO)
    if repo not in _VDA_LOSS_MODULE_CACHE:
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from loss import loss as vda_loss_module
        _VDA_LOSS_MODULE_CACHE[repo] = vda_loss_module
    return _VDA_LOSS_MODULE_CACHE[repo]


_VDA_VIDEO_DEPTH_LOSS_CACHE: dict = {}


def _get_vda_video_depth_loss(repo: str | None, cfg_loss: dict):
    """VideoDepthLoss(alpha=0.5, scales=4, trim=0.0, stable_scale=10) is VDA's own
    COMPOSED training loss (loss/loss.py) -- not TrimmedProcrustesLoss's own standalone
    defaults (trim=0.2), which are never actually used this way. Confirmed by reading the
    composition itself: spatial (SSI) term is per-FRAME (B*T,H,W), untrimmed; a SEPARATE
    per-clip least-squares scale+shift (not per-frame) feeds a single-scale
    (temp_grad_scales=1) TemporalGradientMatchingLoss at weight stable_scale=10 -- this is
    why VAL-01 imports the class directly rather than re-deriving the composition by hand."""
    vda_loss_module = _import_vda_loss_module(repo)
    alpha = float(cfg_loss.get("gradient_weight", 0.5))
    scales = int(cfg_loss.get("ssi_scales", 4))
    trim = float(cfg_loss.get("ssi_trim", 0.0))
    stable_scale = float(cfg_loss.get("tgm_weight", 10.0))
    key = (repo, alpha, scales, trim, stable_scale)
    if key not in _VDA_VIDEO_DEPTH_LOSS_CACHE:
        _VDA_VIDEO_DEPTH_LOSS_CACHE[key] = vda_loss_module.VideoDepthLoss(
            alpha=alpha, scales=scales, trim=trim, stable_scale=stable_scale)
    return _VDA_VIDEO_DEPTH_LOSS_CACHE[key]


def distillation_loss(pred_disp: torch.Tensor, teacher_disp: torch.Tensor, mask: torch.Tensor,
                      grad_w: float = 0.5) -> torch.Tensor:
    """Scale-shift-invariant L1 between student and teacher disparity, plus VDA's multiscale
    gradient term when ``grad_w > 0``. All tensors ``(N, H, W)``; ``mask`` is float 0/1.

    The student is aligned to the teacher per frame (``_scale_shift_align``), so the
    teacher's absolute scale never enters -- only its relative structure is distilled.
    Shared by the cached-teacher path (``compute_loss``) and the online-teacher path
    (``online_distillation_term``, DIST-01) so the two cannot drift apart.
    """
    if mask.sum() <= 0:
        return pred_disp.new_zeros(())
    pd_al = _scale_shift_align(pred_disp, teacher_disp, mask)
    loss = (mask * (pd_al - teacher_disp).abs()).sum() / mask.sum().clamp_min(1.0)
    if grad_w > 0:
        loss = loss + _gradient_loss_multiscale(pd_al, teacher_disp, mask)
    return loss


def online_distillation_term(pred: torch.Tensor, teacher_out: torch.Tensor, teacher_is_disparity: bool,
                             batch: dict, cfg: dict, pred_is_disparity: bool) -> torch.Tensor:
    """DIST-01 (2026-09-18): distillation term against a teacher run on the SAME batch.

    ``pred``/``teacher_out`` are ``(B, 1, T, H, W)``; each is converted to disparity
    according to its own ``predicts_disparity`` convention. The mask is every finite,
    positive teacher pixel (``loss.distill_mask: all``, default) or additionally the
    ground-truth validity mask (``loss.distill_mask: gt``); invalid frames
    (``batch["frame_valid"]``) are always excluded. Unlike the cached path this term is
    added OUTSIDE ``compute_loss`` so it also applies under ``loss.loss_mode: vda``,
    which returns early from ``compute_loss`` (the parity-B recipe every student arm uses).
    """
    lc = cfg.get("loss", {})
    grad_w = float(lc.get("gradient_weight", 0.5))
    B, _, T, H, W = pred.shape
    pd = pred.reshape(B * T, H, W).float()
    if not pred_is_disparity:
        pd = 1.0 / pd.clamp_min(1e-3)
    td = teacher_out.reshape(B * T, H, W).float()
    tdisp = td if teacher_is_disparity else 1.0 / td.clamp_min(1e-3)
    tm = torch.isfinite(tdisp) & (tdisp > 0)
    dmask = str(lc.get("distill_mask", "all"))
    if dmask == "gt":
        depth = batch["depth"].to(pred.device, non_blocking=True).reshape(B * T, H, W)
        tm = tm & (batch["mask"].to(pred.device, non_blocking=True).reshape(B * T, H, W) > 0) & (depth > 0)
    elif dmask == "no_gt":
        # KD2 (2026-09-20): disjoint supervision pools, the DA-V2 / VDA pattern -- exact GT where it
        # exists, the teacher ONLY on clips that have none (video_only / paired_video sources).
        has_gt = batch["mask"].to(pred.device, non_blocking=True).reshape(B, -1).sum(dim=1) > 0
        tm = tm & ~has_gt.repeat_interleave(T)[:, None, None]
    elif dmask != "all":
        raise ValueError(f"loss.distill_mask must be 'all', 'gt' or 'no_gt', got {dmask!r}")
    if "frame_valid" in batch:
        fv = batch["frame_valid"].to(pred.device, non_blocking=True).reshape(B * T)
        tm = tm & (fv[:, None, None] > 0.5)
    tdisp = torch.where(tm, tdisp, torch.zeros_like(tdisp))
    # Per-frame median/MAD normalisation of the teacher disparity (VDA's own SSI
    # convention), so the residual after aligning the student to it is O(1) and the
    # term is on the same scale as the GT spatial term whatever the teacher's native
    # units are. Without this a raw-disparity teacher (released VDA-L emits values in
    # the hundreds) made the term ~50x the GT loss in the 2026-09-18 smoke.
    tdisp = _normalize_disparity_per_frame(tdisp, tm.float())
    return distillation_loss(pd, tdisp, tm.float(), grad_w)


def feature_distillation_loss(student_tokens: list, teacher_tokens: list, batch: dict,
                              distill_mask: str = "all") -> torch.Tensor:
    """KD2 feature distillation: mean (1 - cosine) between the student's PROJECTED patch tokens
    and the frozen teacher's, per layer, each ``(B, T, N, D)``. Scale-free, so the teacher's
    outlier token norms never enter. ``distill_mask: no_gt`` keeps the disjoint-pool rule --
    clips that carry ground truth contribute exactly zero (but stay in the graph)."""
    keep = None
    if distill_mask == "no_gt":
        m = batch["mask"]
        keep = (m.reshape(m.shape[0], -1).sum(dim=1) == 0).float()
    total = student_tokens[0].new_zeros(())
    for s_tok, t_tok in zip(student_tokens, teacher_tokens):
        if s_tok.shape != t_tok.shape:
            raise ValueError(f"student/teacher token shapes differ: {tuple(s_tok.shape)} vs "
                             f"{tuple(t_tok.shape)} -- same H x W and patch size are required")
        d = 1.0 - F.cosine_similarity(s_tok.float(), t_tok.detach().float(), dim=-1)   # (B, T, N)
        d = d.mean(dim=(1, 2))
        total = total + (d if keep is None else d * keep.to(d.device)).mean()
    return total / max(len(student_tokens), 1)


def inject_teacher_pseudo_gt(batch: dict, teacher_out: torch.Tensor, teacher_is_disparity: bool) -> dict:
    """KD2 ``loss.distill_mode: pseudo_gt``: clips with NO ground truth get the frozen teacher's
    prediction as their depth target (mask = finite, positive teacher pixels), so they run
    through the SAME loss as labelled clips -- under ``loss_mode: vda`` that includes the
    temporal gradient matching term the per-frame distillation term lacks. Clips that have GT
    are untouched. Returns a shallow copy; ``teacher_out`` is ``(B, 1, T, H, W)``."""
    mask = batch["mask"]
    no_gt = mask.reshape(mask.shape[0], -1).sum(dim=1) == 0
    if not bool(no_gt.any()):
        return batch
    t = teacher_out.detach().float()
    valid = torch.isfinite(t) & (t > 1e-6)
    depth_t = torch.where(valid, (1.0 / t.clamp_min(1e-6)) if teacher_is_disparity else t, torch.zeros_like(t))
    sel = no_gt.to(t.device).view(-1, 1, 1, 1, 1)
    out = dict(batch)
    out["depth"] = torch.where(sel, depth_t, batch["depth"].to(t.device).float())
    out["mask"] = torch.where(sel, valid.float(), mask.to(t.device).float())
    return out


def _normalize_disparity_per_frame(disp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``(N,H,W)`` -> per-frame ``(disp - median) / mean|disp - median|`` over ``mask``;
    masked-out pixels return 0. Frames with no valid pixel are left at 0."""
    N = disp.shape[0]
    d, m = disp.reshape(N, -1), mask.reshape(N, -1) > 0
    out = torch.zeros_like(d)
    for i in range(N):
        v = d[i][m[i]]
        if v.numel() == 0:
            continue
        med = v.median()
        mad = (v - med).abs().mean().clamp_min(1e-6)
        out[i][m[i]] = (v - med) / mad
    return out.reshape(disp.shape)


def check_lowlight_target(cfg: dict) -> None:
    """data.lowlight.target 'exposed' (DN-01, provenance/dn01_lowlight_defect/) makes batch['frames'] the clean frame
    at the INPUT's exposure, so it may only feed the restoration losses. The clean-twin consumers (consistency, the PA2
    adapter L1, a non-degraded online teacher) would silently see a darkened frame: refuse them."""
    target = (cfg.get("data", {}).get("lowlight") or {}).get("target")
    if target not in (None, "exposed"):
        raise ValueError(f"data.lowlight.target must be absent or 'exposed', got {target!r}")
    lc, ot = cfg.get("loss", {}), cfg.get("model", {}).get("online_teacher")
    if target == "exposed" and (float(lc.get("consistency_weight", 0.0)) > 0
                                or float(lc.get("adapter_l1_weight", 0.0)) > 0
                                or (ot and ot.get("input", "degraded") != "degraded")):
        raise ValueError("data.lowlight.target 'exposed' feeds only restoration losses: no consistency, adapter L1 "
                         "or clean online teacher")


def check_seeding(cfg: dict) -> None:
    """E4 opt-in seeding keys: ``training.seed_scheme`` in SEED_SCHEMES (absent = "add", the old scheme); ``data.split_seed`` an int (absent = the scene split follows ``data.seed``)."""
    scheme = cfg.get("training", {}).get("seed_scheme", "add")
    if scheme not in SEED_SCHEMES:
        raise ValueError(f"training.seed_scheme must be one of {SEED_SCHEMES}, got {scheme!r}")
    ss = cfg.get("data", {}).get("split_seed")
    if ss is not None and (isinstance(ss, bool) or not isinstance(ss, int)):
        raise ValueError(f"data.split_seed must be an int, got {ss!r}")


def check_static_regime(cfg: dict) -> None:
    """DN-01 WS2 arm S (K49): ``data.static_prob`` in [0, 1] (absent = off); > 0 needs every degrade mode to have an
    independent per-frame draw (awgn / awgn_blur / lowlight) and a valid ``data.static_noise_mult`` [lo, hi]
    (0 < lo <= hi) / ``data.static_ev_jitter`` >= 0."""
    d = cfg.get("data", {})
    p = float(d.get("static_prob", 0.0))
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"data.static_prob must be in [0, 1], got {p}")
    if p == 0.0:
        if d.get("static_exempt_shares", False):
            raise ValueError("data.static_exempt_shares needs data.static_prob > 0")
        if d.get("static_lowlight_only", False):
            raise ValueError("data.static_lowlight_only needs data.static_prob > 0")
        return
    if d.get("static_lowlight_only", False):                     # E4 arm c: static clips are always 'lowlight' (needs that mode + its parameters)
        modes = d.get("degrade_mode", "full")
        if "lowlight" not in (modes if isinstance(modes, list) else [modes]) or not d.get("lowlight"):
            raise ValueError("data.static_lowlight_only needs 'lowlight' in data.degrade_mode and a data.lowlight block")
    from data.video_clip_dataset import STATIC_MODES
    modes = d.get("degrade_mode", "full")                    # the loaders' default
    modes = modes if isinstance(modes, list) else [modes]
    if not modes or any(m not in STATIC_MODES for m in modes):
        raise ValueError(f"data.static_prob > 0 needs data.degrade_mode within {STATIC_MODES}, got {modes!r}")
    lo, hi = d.get("static_noise_mult", (0.5, 1.5))
    if not 0.0 < float(lo) <= float(hi):
        raise ValueError(f"data.static_noise_mult must be [lo, hi] with 0 < lo <= hi, got {[lo, hi]}")
    if float(d.get("static_ev_jitter", 0.15)) < 0:
        raise ValueError("data.static_ev_jitter must be >= 0")


def online_teacher_input(mode: str, rgb: torch.Tensor, batch: dict, device) -> torch.Tensor:
    """Frames the frozen teacher sees. ``degraded``: the student's own input. ``clean``: the
    clean twin. ``degraded_or_bright_twin`` (DIST-01 arm (e)): the student's input, except for
    ``paired_video`` clips (``degrade_params`` all -1), whose REAL bright twin the teacher sees
    while the student gets the dark one -- per sample, so synthetic clips in the same mixed
    batch keep arm m's ``degraded`` routing exactly."""
    if mode == "degraded":
        return rgb
    clean = batch["frames"].to(device, non_blocking=True)
    if mode == "clean":
        return clean
    paired = (batch["degrade_params"].to(device) < 0).all(dim=1)
    return torch.where(paired.view(-1, 1, 1, 1, 1), clean, rgb)


def build_online_teacher(cfg: dict, device: str):
    """DIST-01: build the frozen teacher named by ``model.online_teacher``, or ``None``.

    Two forms. ``{checkpoint: runs/<arm>/stepN.pt, input: degraded|clean}`` rebuilds an
    in-repo-trained (adapted) teacher from the checkpoint's own embedded cfg, exactly as
    ``infer.load`` does; ``{arch: vda, vda_encoder: vitl, vda_init: released,
    vda_checkpoint_path: ..., input: ...}`` builds a zero-shot teacher through
    ``build_model``. ``input`` selects which frames the teacher sees: the student's own
    degraded frames (default -- the teacher's degraded-input behaviour is what is
    distilled) or the clean twin (the classic clean-teacher control).

    Returns ``(teacher, input, teacher_predicts_disparity)``.
    """
    ot = cfg.get("model", {}).get("online_teacher")
    if not ot:
        return None
    if float(cfg.get("loss", {}).get("distill_weight", 0.0)) <= 0:
        raise ValueError("model.online_teacher is set but loss.distill_weight is 0")
    if str(cfg.get("training", {}).get("causal_training", "masked")) == "prefix_newest":
        raise ValueError("model.online_teacher is incompatible with causal_training=prefix_newest (no pred tensor)")
    inp = str(ot.get("input", "degraded"))
    if inp not in {"degraded", "clean", "degraded_or_bright_twin"}:
        raise ValueError("model.online_teacher.input must be 'degraded', 'clean' or "
                         f"'degraded_or_bright_twin', got {inp!r}")
    # Null-valued keys are dropped, not passed through: `extends` merges the online_teacher
    # block rather than replacing it, so a child arm that swaps an adapted teacher for a
    # zero-shot one must be able to cancel the inherited `checkpoint:` with an explicit
    # `checkpoint: null` (DIST-01 arm (c0), 2026-09-18) without that null reaching build_model.
    spec = {k: v for k, v in ot.items() if k != "input" and v is not None}
    if spec.get("checkpoint"):
        ck = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
        mcfg = dict(ck["cfg"]["model"])
        mcfg["device"] = device
        teacher = build_model({"model": mcfg}).to(device)
        load_d3_checkpoint(teacher, ck)
    else:
        spec.setdefault("device", device)
        teacher = build_model({"model": spec}).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, inp, bool(getattr(teacher, "predicts_disparity", False))


def build_scale_anchor_teacher(cfg: dict, device: str):
    """SKYSCALE F3 (default off: ``loss.scale_anchor_weight`` 0/absent returns None and nothing is loaded): the frozen reference model of the scale anchor, named by ``model.scale_anchor_teacher`` (the same spec forms as
    ``model.online_teacher``: ``{arch: vda, vda_encoder: vits, metric: true, vda_checkpoint_path: <released Metric-VDA-S>, input: degraded|clean}``). Returns ``(teacher, input, predicts_disparity)`` or None."""
    if float(cfg.get("loss", {}).get("scale_anchor_weight", 0.0)) <= 0:
        return None
    spec = cfg.get("model", {}).get("scale_anchor_teacher")
    if not spec:
        raise ValueError("loss.scale_anchor_weight > 0 needs model.scale_anchor_teacher")
    sub = {"model": {"online_teacher": spec}, "loss": {"distill_weight": 1.0}, "training": cfg.get("training", {})}
    return build_online_teacher(sub, device)


def scale_anchor_term(pred: torch.Tensor, teacher_out: torch.Tensor, teacher_is_disparity: bool, batch: dict, cfg: dict,
                      pred_is_disparity: bool = False) -> torch.Tensor:
    """SKYSCALE F3: per clip, r = mean over the pixels where the frozen teacher's depth is finite and in [0.1, 80] m (and the frame is valid) of log(pred) - log(teacher); Huber(r, delta) (quadratic inside delta, slope delta
    beyond: bounded, never absorbing), averaged over the clips that count. ``loss.scale_anchor_nonmetric_only`` (default true) keeps only the clips with ``metric_sup`` < 0.5 (SiLog already anchors the others; a batch with
    no ``metric_sup`` key has SiLog on every clip, so no clip counts). The student side is the RAW prediction (``clamp_min(1e-3)`` only), so a prediction at 1e5 m still gets the full restoring gradient. No counting clip gives
    an exact 0 that stays on the graph."""
    lc = cfg.get("loss", {})
    delta = float(lc.get("scale_anchor_huber", 0.1))
    if pred_is_disparity:
        raise ValueError("loss.scale_anchor_weight needs a depth-native (metric) student")
    B, _, T, H, W = pred.shape
    pd = pred.reshape(B, T, H, W).float()
    td = teacher_out.detach().reshape(B, T, H, W).float()
    if teacher_is_disparity:
        td = 1.0 / td.clamp_min(1e-6)
    valid = torch.isfinite(td) & (td >= 0.1) & (td <= 80.0)
    if "frame_valid" in batch:
        valid = valid & (batch["frame_valid"].to(pred.device, non_blocking=True).reshape(B, T, 1, 1) > 0.5)
    if bool(lc.get("scale_anchor_nonmetric_only", True)):
        sup = batch.get("metric_sup")
        clip_ok = torch.zeros(B, dtype=torch.bool, device=pred.device) if sup is None else (sup.to(pred.device, non_blocking=True).reshape(B) < 0.5)
        valid = valid & clip_ok.reshape(B, 1, 1, 1)
    n = valid.reshape(B, -1).sum(dim=1)
    keep = n > 0
    if not bool(keep.any()):
        return pd.reshape(-1)[0] * 0.0
    lr = torch.log(pd.clamp_min(1e-3)) - torch.log(td.clamp_min(1e-6))
    r = torch.where(valid, lr, torch.zeros_like(lr)).reshape(B, -1).sum(dim=1)[keep] / n[keep]      # where, not a product: a NaN teacher pixel outside the valid mask must not leak
    a = r.abs()
    return torch.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta)).mean()


def _no_gt_zero(pred: torch.Tensor, batch: dict) -> torch.Tensor | None:
    """DIST-01 arm (c), 2026-09-18: a finite zero for a clip with NO valid GT pixel.

    ``type: video_only`` clips (unlabeled real video, supervised only by the online teacher)
    carry an all-zero ``mask``. Every GT term below is then a 0/0 -- and VDA's own
    ``loss/loss.py::VideoDepthLoss`` cannot be inspected here (the vendored repo is not
    present on this workstation), and its per-frame SSI / TGM paths normalise by their own
    mask sums, so it is not safe to assume it returns 0 rather than NaN. Guard at OUR call
    site instead: skip the GT term entirely and return zero.

    The zero is built from ``pred`` (``pred.reshape(-1)[0] * 0``), NOT ``pred.new_zeros(())``,
    so it keeps a grad_fn: the trainer calls ``loss.backward()`` unconditionally on any
    finite loss, and a detached constant would raise when ``loss.distill_weight`` is 0.
    Keeping it finite is also what stops the trainer's ``torch.isfinite(loss)`` guard from
    silently DROPPING these batches -- their gradient comes from the distillation term,
    which is added outside this function.

    Returns ``None`` when the batch does have GT (the normal path)."""
    mask = batch.get("mask")
    if mask is None:
        return None
    depth = batch.get("depth")
    valid = mask if depth is None else mask * (depth > 0)
    if float(valid.sum()) > 0:
        return None
    return pred.reshape(-1)[0] * 0.0


_TERM_ACC: dict = {}     # METRICFIX loss.log_terms: name -> [sum, count] since the last flush (per process)


def _term_add(name: str, value) -> None:
    if value is None:
        return
    v = float(value.detach().float()) if torch.is_tensor(value) else float(value)
    a = _TERM_ACC.setdefault(name, [0.0, 0])
    a[0] += v
    a[1] += 1


def term_flush() -> dict:
    """Mean of every accumulated term since the last flush ({} when none); resets the accumulator."""
    out = {k: (a[0] / a[1] if k not in ("n_clips", "n_silog_clips") else a[0]) for k, a in _TERM_ACC.items() if a[1]}
    _TERM_ACC.clear()
    return out


def _compute_loss_vda(pred: torch.Tensor, batch: dict, cfg: dict, pred_is_disparity: bool) -> torch.Tensor:
    """VAL-01, Fable's ruling 2026-08-27: reuse VDA's own loss/loss.py::VideoDepthLoss
    directly rather than this file's hand-rolled SSI/TGM, which diverges from the real
    recipe in ways structural enough that re-deriving them by hand risks getting subtly
    wrong (per-frame SSI vs per-clip TGM alignment; untrimmed despite
    TrimmedProcrustesLoss's own -- unused here -- trim=0.2 default; single-scale TGM, not
    four). VDA's own loss operates on raw disparity space directly -- no additional
    median/MAD normalisation here (VideoDepthLoss's own TrimmedProcrustesLoss normalises
    internally -- pre-normalising would train quietly against a different loss than the
    one being reused). VDA's own loss has no sky concept: sky pixels are left excluded
    from mask, matching sky_zero_disparity=false (enforced at config-load time for
    arch=vda -- see _assert_vda_uses_vda_loss_mode).

    Matrix item 2 (Opus, 2026-08-28), extending this beyond arch=vda: ``pred_is_disparity``
    (a per-model, not per-arch, property -- see each model's own ``predicts_disparity``)
    controls whether ``pred`` is used as disparity directly (VDA's own relative mode;
    OVDANet after removing its old depth-wrapper conversion, models/ovda_net.py) or
    converted from depth first, ONCE, with the same clamp floor train.py's own non-vda
    `compute_loss` already uses elsewhere (D3Net, verified depth-native -- see
    models/d3_net.py's own note on why, not a round-trip: there is exactly one necessary
    conversion here, not two mismatched ones)."""
    zero = _no_gt_zero(pred, batch)
    if zero is not None:   # video_only clip: no GT pixel anywhere, skip VDA's loss entirely
        return zero
    # DN-01 amendment 2026-09-23 (Manager): loss.metric_depth_clamp [lo, hi] clamps a metric model's
    # predicted depth to [lo, hi] m before the disparity inversion and in SiLog. Absent = unchanged.
    rng = cfg.get("loss", {}).get("metric_depth_clamp")
    rng = None if rng is None or pred_is_disparity else (float(rng[0]), float(rng[1]))
    pd, gd, m = _vda_disp(pred, batch, pred_is_disparity, rng)
    loss_fn = _get_vda_video_depth_loss(cfg.get("model", {}).get("vda_repo"), cfg.get("loss", {}))
    # SKY FIX (DN-01 sky hunt, 2026-10-01; both opt-in, default off = byte-identical): the loaders drop the sky (VKITTI 655 m) from ``mask`` and emit it as ``sky``, which no loss read, so the sky had zero gradient and the
    # motion modules drifted it to ~30-55 m (released: 400-650 m). Fix 2 ``loss.sky_vda_zero_disparity``: VideoDepthLoss (SSI / gradient / TGM) sees mask | sky, the sky at target disparity 0 (``gd`` is already 0 off-mask), as in the
    # June D3 fix 19b783b; SiLog keeps the ORIGINAL mask (log of a zero disparity is -inf: the 5ad56f5 NaN). Fix 1 ``loss.sky_hinge_weight``: w * relu(log(sky_hinge_min_depth) - log(pred[sky])).mean(), one-sided, zero beyond the cap.
    lc_sky = cfg.get("loss", {})
    sky_zd, sky_hw = bool(lc_sky.get("sky_vda_zero_disparity", False)), float(lc_sky.get("sky_hinge_weight", 0.0))
    sky = None
    if sky_zd or sky_hw > 0:
        sky = sky_pixels(batch, m)
    m_loss = (m | sky) if sky_zd else m
    sp = batch.get("sparse")                      # protocol v3 WP13 (absent = byte-identical): a sparse-LiDAR sample uses the sparse-safe masked SSI instead of VideoDepthLoss's dense terms, PER SAMPLE
    sp_idx = None if sp is None else (sp.float().reshape(-1).to(pd.device) > 0.5)
    if sp_idx is not None and bool(sp_idx.any()):
        n_sp, n_all = int(sp_idx.sum()), int(sp_idx.numel())
        parts = [sparse_ssi_loss(pd[sp_idx], gd[sp_idx], m_loss[sp_idx]) * (n_sp / n_all)]
        if n_sp < n_all:                          # the dense samples: VDA's own loss on exactly those samples, weighted by their share of the batch
            dn = ~sp_idx
            parts.append(loss_fn(prediction=pd[dn], target=gd[dn], mask=m_loss[dn])["total_loss"] * ((n_all - n_sp) / n_all))
        out = {"total_loss": sum(parts), "spatial_loss": None, "stable_loss": None}
    else:
        out = loss_fn(prediction=pd, target=gd, mask=m_loss)
    total = out["total_loss"]
    log_terms = bool(lc_sky.get("log_terms", False))     # METRICFIX (default off: nothing below runs and the graph is untouched): per-term values -> _TERM_ACC, flushed to metrics.jsonl by the trainer
    if log_terms:
        _term_add("vda_total", total)
        _term_add("vda_spatial", out.get("spatial_loss"))
        _term_add("vda_tgm", out.get("stable_loss"))
    if sky_hw > 0:
        # SKYSCALE F1 (default off = byte-identical): ``sky_hinge_metric_only`` scopes the hinge to the metric-supervised clips (SiLog is their counter-force; the sky_vda_zero_disparity mask above is NOT scoped);
        # ``sky_hinge_max_depth`` adds the upper side of a two-sided band on the raw prediction.
        sky_h = sky_hinge_scope(sky, batch) if bool(lc_sky.get("sky_hinge_metric_only", False)) else sky
        hi = lc_sky.get("sky_hinge_max_depth")
        hinge = sky_hw * (sky_hinge(pred, sky_h, float(lc_sky.get("sky_hinge_min_depth", 80.0))) if hi is None
                          else sky_hinge(pred, sky_h, float(lc_sky.get("sky_hinge_min_depth", 80.0)), float(hi)))
        total = total + hinge
        if log_terms:
            _term_add("sky_hinge_weighted", hinge)
    silog_w = float(cfg.get("loss", {}).get("metric_silog_weight", 0.0))
    if silog_w <= 0:
        return total
    # DN-01 Phase D: VideoDepthLoss is scale/shift-invariant by construction, so a metric
    # model needs a term that pins absolute scale. DA-V2 metric_depth/util/loss.py's
    # SiLogLoss verbatim (lambd=0.5 keeps half the log-scale error; lambd=1 would be fully
    # scale-invariant), on the model's NATIVE depth vs GT depth -- never on disparity.
    if pred_is_disparity:
        raise ValueError("loss.metric_silog_weight needs a depth-native (metric) model, "
                         "but pred_is_disparity=True")
    B, _, T, H, W = pred.shape
    depth = batch["depth"].to(pred.device, non_blocking=True)
    pd_raw = pred.reshape(B, T, H, W).float()
    lambd = float(cfg.get("loss", {}).get("metric_silog_lambda", 0.5))
    m_si = silog_mask(m, batch)
    if m_si is not None and batch.get("silog_ceil") is not None:
        # METRICFIX (default off: no source sets silog_max_depth = no key in the batch = byte-identical): the PER-SOURCE SiLog far ceiling (one value per clip, +inf = none). Only the sources that set it lose their far band from the
        # log-space term; every other source's SiLog mask is untouched.
        ceil = batch["silog_ceil"].to(depth.device, non_blocking=True).float().reshape(-1, 1, 1, 1)
        m_si = m_si & (depth.reshape(B, T, H, W) < ceil)
        if not bool(m_si.any()):
            m_si = None
    if m_si is not None and cfg.get("loss", {}).get("metric_silog_max_depth") is not None:
        # METRICFIX (default off: absent = byte-identical): a SiLog-ONLY far ceiling -- GT at or beyond it (a TartanAir / PointOdyssey far band that the lifted training cap now keeps in `mask`) does not enter the log-space term
        m_si = m_si & (depth.reshape(B, T, H, W) < float(cfg["loss"]["metric_silog_max_depth"]))
        if not bool(m_si.any()):
            m_si = None
    if log_terms:
        sup = batch.get("metric_sup")
        _term_add("n_clips", float(B))
        _term_add("n_silog_clips", float(B) if sup is None else float((sup.float() > 0.5).sum()))
    if m_si is None:   # K80 option B: no clip of this batch is metric-supervised (a TartanAir / PointOdyssey / pseudo-GT clip): SSI / gradient / TGM only; the zero keeps the graph
        return total
    pd_d = pd_raw[m_si].clamp_min(1e-3) if rng is None else pd_raw[m_si].clamp(*rng)   # ReLU output can be exactly 0
    diff_log = torch.log(depth.reshape(B, T, H, W).float()[m_si]) - torch.log(pd_d)
    silog = torch.sqrt((diff_log.pow(2).mean() - lambd * diff_log.mean().pow(2)).clamp_min(1e-12))
    if log_terms:
        _term_add("silog", silog)
        _term_add("silog_mean_log_err", diff_log.mean())      # + = prediction BELOW the GT, - = ABOVE (the signed scale error the term sees)
    return total + silog_w * silog


def silog_mask(m: torch.Tensor, batch: dict) -> torch.Tensor | None:
    """K80 option B, per-source metric supervision: the SiLog mask. Without ``metric_sup`` in the batch (every config but option B) this IS ``m`` (byte-identical); with it, ``m`` restricted to the clips whose
    source is metric-supervised (``metric_sup`` 1.0 per clip), or None when none of the batch's clips is (or none has a valid pixel), so the caller skips the term."""
    sup = batch.get("metric_sup")
    if sup is None:
        return m
    m_si = m & (sup.to(m.device, non_blocking=True).reshape(-1, 1, 1, 1) > 0.5)
    return m_si if bool(m_si.any()) else None


def sky_pixels(batch: dict, m: torch.Tensor) -> torch.Tensor:
    """``(B,T,H,W)`` bool: the loader's sky label (> 0.5 after the resize) outside the GT mask ``m``. A missing ``sky`` key raises (stale loaders), never silently no-ops (as the June fix 19b783b)."""
    if "sky" not in batch:
        raise KeyError("loss.sky_vda_zero_disparity / loss.sky_hinge_weight need a 'sky' mask in the batch -- stale data loaders?")
    return (batch["sky"].to(m.device, non_blocking=True).reshape(m.shape).float() > 0.5) & ~m


def sky_hinge_scope(sky: torch.Tensor, batch: dict) -> torch.Tensor:
    """SKYSCALE F1: ``sky`` restricted to the clips whose source is metric-supervised (``metric_sup`` > 0.5). A missing ``metric_sup`` raises (the flag is meaningless without per-source supervision flags)."""
    sup = batch.get("metric_sup")
    if sup is None:
        raise KeyError("loss.sky_hinge_metric_only needs data.sources[].metric_supervision on every train source (no 'metric_sup' in the batch)")
    return sky & (sup.to(sky.device, non_blocking=True).reshape(-1, 1, 1, 1) > 0.5)


def sky_hinge(pred: torch.Tensor, sky: torch.Tensor, min_depth: float, max_depth: float | None = None) -> torch.Tensor:
    """Fix 1: mean over sky pixels of relu(log(min_depth) - log(depth)): pushes the sky out to at least ``min_depth`` metres, zero gradient beyond it; a clip with no sky gives an exact 0 that stays on the graph.
    SKYSCALE F1 (``max_depth`` None = off = the exact original expression): adds relu(log(depth) - log(max_depth)) on the RAW prediction (no clamp above, so a prediction far past any ceiling still gets a restoring gradient):
    a two-sided band whose inside has zero gradient."""
    B, _, T, H, W = pred.shape
    d = pred.reshape(B, T, H, W).float()
    if not bool(sky.any()):
        return d.reshape(-1)[0] * 0.0
    if max_depth is None:
        return torch.relu(math.log(min_depth) - torch.log(d[sky].clamp_min(1e-3))).mean()
    ld = torch.log(d[sky].clamp_min(1e-3))
    return (torch.relu(math.log(min_depth) - ld) + torch.relu(ld - math.log(max_depth))).mean()


def check_metric_supervision(cfg: dict) -> None:
    """K80 option B: ``data.sources[].metric_supervision`` (bool) is all-or-none across the TRAIN sources (the batches' key set must agree under the collate; a missing key would silently mean
    'supervised'), needs ``loss.loss_mode`` 'vda' and a metric SiLog term to act on (else it is a no-op the config author should not believe in). Absent everywhere = off (byte-identical)."""
    specs = (cfg.get("data") or {}).get("sources") or []
    flags = [s.get("metric_supervision") for s in specs]
    if all(f is None for f in flags):
        return
    if any(f is None for f in flags) or not all(isinstance(f, bool) for f in flags):
        raise ValueError("data.sources[].metric_supervision must be set (true / false) on EVERY train source or on none")
    if float(cfg.get("loss", {}).get("metric_silog_weight", 0.0)) <= 0:
        raise ValueError("data.sources[].metric_supervision needs loss.metric_silog_weight > 0 (it restricts the SiLog term)")
    if not any(flags):
        raise ValueError("data.sources[].metric_supervision: at least one source must be metric-supervised")


def check_sparse_gt(cfg: dict) -> None:
    """Protocol v3 WP13: ``data.sources[].sparse_gt`` (bool) is set on EVERY train source or none (a dense source says ``false``; the collate needs one key set and a missing key must not silently mean 'dense'), needs
    ``loss.loss_mode`` 'vda' (the sparse-safe masked SSI replaces VideoDepthLoss's dense SSI / gradient / TGM for the flagged SAMPLES, a mixed dense + sparse batch is handled per sample). Absent everywhere = off (byte-identical)."""
    specs = (cfg.get("data") or {}).get("sources") or []
    flags = [s.get("sparse_gt") for s in specs]
    if all(f is None for f in flags):
        return
    if any(f is None for f in flags) or not all(isinstance(f, bool) for f in flags):
        raise ValueError("data.sources[].sparse_gt must be set (true / false) on EVERY train source or on none")
    if str(cfg.get("loss", {}).get("loss_mode")) != "vda":
        raise ValueError("data.sources[].sparse_gt needs loss.loss_mode 'vda'")


def sparse_ssi_loss(pd: torch.Tensor, gd: torch.Tensor, m: torch.Tensor, min_valid: int = 16) -> torch.Tensor:
    """Sparse-safe scale-shift-invariant L1 on disparity ``(B,T,H,W)``: per FRAME, the median (shift) and mean absolute deviation (scale) of prediction and target are taken over the VALID pixels only (VDA's
    ``normalize_prediction_robust`` takes the median over the whole frame, masked-out zeros included, which is only right for a dense mask); the loss is the mean |pred_n - target_n| over the valid pixels of
    frames with >= ``min_valid`` of them (VDA's untrimmed data term, reduction batch-based). No gradient / TGM term (they need neighbouring valid pixels, a LiDAR frame has none). Zero (graph kept) if no frame qualifies."""
    B, T, H, W = pd.shape
    p, g, mf = pd.reshape(-1, H * W), gd.reshape(-1, H * W), m.reshape(-1, H * W).bool()
    n = mf.sum(1)
    ok = n >= min_valid
    if not bool(ok.any()):
        return pd.sum() * 0.0
    p, g, mf, n = p[ok], g[ok], mf[ok], n[ok].float()

    def norm(x):
        med = torch.where(mf, x, torch.full_like(x, float("nan"))).nanmedian(dim=1).values
        xc = x - med[:, None]
        s = ((xc.abs() * mf).sum(1) / n).clamp_min(1e-6)
        return xc / s[:, None]
    diff = (norm(p) - norm(g)).abs()
    return (diff * mf).sum() / mf.sum()


def _sha256_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_metricfix(cfg: dict) -> None:
    """METRICFIX opt-in keys (every one absent = off = byte-identical to the base): ``loss.metric_silog_max_depth`` (SiLog-only far ceiling, m), ``loss.log_terms`` (per-term loss rows), ``data.sources[].max_depth`` (per-source
    validity bound, m), ``data.sources[].silog_max_depth`` (per-source SiLog far ceiling, m), ``training.head_lr_mult`` (LR multiplier of the trainable DPT head; needs ``model.freeze_depth_head: false``), ``model.init_from_sha256`` ('record' or a hex sha256 pin of ``model.init_from``),
    ``eval.outdoor_monitor`` (the outdoor VKITTI scale monitor)."""
    lc, t, m = cfg.get("loss", {}), cfg.get("training", {}), cfg.get("model", {})

    def pos(v, name):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
            raise ValueError(f"{name} must be a positive number, got {v!r}")
    if lc.get("metric_silog_max_depth") is not None:
        pos(lc["metric_silog_max_depth"], "loss.metric_silog_max_depth")
        if float(lc.get("metric_silog_weight", 0.0)) <= 0:
            raise ValueError("loss.metric_silog_max_depth needs loss.metric_silog_weight > 0 (it bounds the SiLog term)")
    if "log_terms" in lc and not isinstance(lc["log_terms"], bool):
        raise ValueError(f"loss.log_terms must be a bool, got {lc['log_terms']!r}")
    for s in (cfg.get("data") or {}).get("sources") or []:
        if s.get("max_depth") is not None:
            pos(s["max_depth"], "data.sources[].max_depth")
        if s.get("silog_max_depth") is not None:
            pos(s["silog_max_depth"], "data.sources[].silog_max_depth")
            if s.get("metric_supervision") is not True:
                raise ValueError("data.sources[].silog_max_depth needs metric_supervision: true on the same source")
            if float(lc.get("metric_silog_weight", 0.0)) <= 0:
                raise ValueError("data.sources[].silog_max_depth needs loss.metric_silog_weight > 0 (it bounds that source's SiLog term)")
    if t.get("head_lr_mult") is not None:
        pos(t["head_lr_mult"], "training.head_lr_mult")
        if m.get("freeze_depth_head", True) is not False or not (m.get("shared_temporal") or m.get("arch") == "vda"):   # protocol v3 WP21: a plain arch=vda (FT Metric-VDA-S) has depth_head_parameters() too
            raise ValueError("training.head_lr_mult needs model.shared_temporal (or a plain arch=vda) and an explicit model.freeze_depth_head: false")
    sha = m.get("init_from_sha256")
    if sha is not None:
        if not m.get("init_from"):
            raise ValueError("model.init_from_sha256 needs model.init_from")
        if sha != "record" and not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
            raise ValueError("model.init_from_sha256 must be 'record' or a 64-hex sha256")
    mc = (cfg.get("eval") or {}).get("outdoor_monitor")
    if mc is not None:
        if not isinstance(mc, dict) or not mc.get("root"):
            raise ValueError("eval.outdoor_monitor must be a mapping with a VKITTI 'root'")
        for k in ("every_steps", "clips"):
            if mc.get(k) is not None and (isinstance(mc[k], bool) or not isinstance(mc[k], int) or mc[k] < 1):
                raise ValueError(f"eval.outdoor_monitor.{k} must be a positive int, got {mc[k]!r}")
        if not m.get("metric", False):
            raise ValueError("eval.outdoor_monitor reports median(pred)/median(GT) in metres: needs model.metric: true")


class OutdoorMonitor:
    """METRICFIX (``eval.outdoor_monitor``, default off): the OUTDOOR scale-drift monitor, logged to the run's metrics.jsonl every ``every_steps`` (default 100) so the step at which the metric scale jumps is visible in EVERY arm.
    A few Virtual KITTI clips (Scene01 / Scene20 are ALWAYS excluded: KITTI-eval replicas, K85), the centre crop, a fixed degradation draw per clip (its own seed, ``seed + idx``): every arm sees the same inputs. IN-SAMPLE by construction
    (the training sources use every remaining scene / variant / frame, and the monitor must not change train membership), so it is a DRIFT monitor, never a held-out score and never used for selection. Per clip it takes
    median(pred) / median(GT) over the pixels with 0 < GT < 80 m (the evaluation cap), and reports the median over clips: 1.0 = right scale, > 1 = the model predicts too far. ``clean_too`` (default true) adds the same on the undegraded
    frames. The global RNG state is saved and restored around every run (no draw of the training stream moves)."""

    def __init__(self, cfg: dict) -> None:
        mc = dict(cfg["eval"]["outdoor_monitor"])
        d = cfg.get("data", {})
        self.every, self.n, self.clean_too = int(mc.get("every_steps", 100)), int(mc.get("clips", 4)), bool(mc.get("clean_too", True))
        nf = int(d.get("num_frames", 8))
        common = dict(num_frames=nf, image_size=tuple(d.get("image_size", (518, 518))), clip_stride=nf, frame_stride=int(d.get("frame_stride", 1)), max_depth=80.0, seed=int(mc.get("seed", 1234)),
                      degrade=True, degrade_mode=d.get("degrade_mode", "full"), degrade_severity=d.get("val_degrade_severity"), max_clips_per_video=None, resize_mode="vda")
        spec = {"type": "vkitti", "root": mc["root"], "variants": list(mc.get("variants", ["clone", "overcast"])), "cameras": list(mc.get("cameras", ["Camera_0"])),
                "no_holdout": True, "exclude_eval_overlap": True, "resize_mode": "vda"}
        self.ds_deg = self._build(spec, common, d)
        self.ds_clean = self._build(spec, {**common, "degrade": False}, d) if self.clean_too else None
        for ds in (self.ds_deg, self.ds_clean):
            if ds is not None and {c[2].split("/")[0] for c in ds.clips} & {"Scene01", "Scene20"}:
                raise RuntimeError("outdoor monitor: VKITTI Scene01 / Scene20 (KITTI-eval replicas) reached the monitor source")
        L = len(self.ds_deg)
        k = min(self.n, L)
        self.idx = [int(i * L / k) for i in range(k)] if k else []
        if not self.idx:
            raise ValueError(f"eval.outdoor_monitor: no VKITTI clip under {mc['root']}")

    @staticmethod
    def _build(spec: dict, common: dict, d: dict):
        ds = _make_source(spec, "train", common)
        ds.split = "val"                       # centre crop + the deterministic per-clip degradation (the `val` branch of the loaders); the sequence list was built with no_holdout, so it is every kept scene
        if hasattr(ds, "lowlight"):
            ds.lowlight = d.get("lowlight")
        return ds

    @torch.no_grad()
    def run(self, core, device, amp_dtype, step: int) -> dict:
        rng_state = _rng_state()
        was_training = core.training
        core.eval()
        out: dict = {"event": "outdoor_monitor", "step": int(step)}
        try:
            for suffix, ds in (("", self.ds_deg), ("_clean", self.ds_clean)):
                if ds is None:
                    continue
                ratios, pms, gms = [], [], []
                for i in self.idx:
                    it = ds[i]
                    x = it.get("degraded_frames", it["frames"])[None].to(device)
                    with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None):
                        pred = core(x)
                    pred = (pred[0] if isinstance(pred, (tuple, list)) else pred).float().reshape(-1).cpu()
                    depth, mask = it["depth"].reshape(-1), it["mask"].reshape(-1)
                    v = (mask > 0.5) & (depth > 0) & (depth < 80.0) & torch.isfinite(pred)
                    if not bool(v.any()):
                        continue
                    pm, gm = float(pred[v].median()), float(depth[v].median())
                    pms.append(pm)
                    gms.append(gm)
                    ratios.append(pm / gm)
                if ratios:
                    out[f"outdoor_scale_ratio{suffix}"] = float(np.median(ratios))
                    out[f"outdoor_pred_median{suffix}"] = float(np.median(pms))
                    out[f"outdoor_gt_median{suffix}"] = float(np.median(gms))
                    out[f"outdoor_n_clips{suffix}"] = len(ratios)
                    out[f"outdoor_ratio_per_clip{suffix}"] = [float(r) for r in ratios]
        finally:
            core.train(was_training)
            _restore_rng_state(rng_state)
        return out


VDA_LOSS_ARCHES = ("vda", "dav2", "daac", "da3")   # protocol v3 WP22/23: the per-frame baselines train under VDA's own loss + the same loss-key guards as arch=vda


def check_scale_anchor(cfg: dict) -> None:
    """SKYSCALE F3 opt-in keys (absent = off): ``loss.scale_anchor_weight`` (>= 0), ``loss.scale_anchor_huber`` (> 0, log units), ``loss.scale_anchor_nonmetric_only`` (bool), ``model.scale_anchor_teacher`` (required when the weight is > 0)."""
    lc, m = cfg.get("loss", {}), cfg.get("model", {})
    w, hu, nm = lc.get("scale_anchor_weight", 0.0), lc.get("scale_anchor_huber", 0.1), lc.get("scale_anchor_nonmetric_only", True)
    if isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0:
        raise ValueError(f"loss.scale_anchor_weight must be a number >= 0, got {w!r}")
    if isinstance(hu, bool) or not isinstance(hu, (int, float)) or hu <= 0:
        raise ValueError(f"loss.scale_anchor_huber must be a positive number, got {hu!r}")
    if not isinstance(nm, bool):
        raise ValueError(f"loss.scale_anchor_nonmetric_only must be a bool, got {nm!r}")
    if not w > 0:
        return
    if not m.get("scale_anchor_teacher"):
        raise ValueError("loss.scale_anchor_weight > 0 needs model.scale_anchor_teacher")
    if not bool(m.get("metric", False)):
        raise ValueError("loss.scale_anchor_weight needs a metric (depth-native) student: model.metric=true")
    if str(cfg.get("training", {}).get("causal_training", "masked")) == "prefix_newest":
        raise ValueError("loss.scale_anchor_weight is incompatible with causal_training=prefix_newest (no pred tensor)")
    if nm and any(x.get("metric_supervision") is None for x in cfg.get("data", {}).get("sources", [])):
        raise ValueError("loss.scale_anchor_nonmetric_only needs data.sources[].metric_supervision set on every train source")


def check_sky_loss(cfg: dict) -> None:
    """Validate the opt-in sky-fix keys (absent = off). Fix 2 needs arch 'vda' + loss_mode 'vda' (VideoDepthLoss); the hinge needs a depth-native (metric) VDA. The OLD ``loss.sky_zero_disparity`` stays banned for arch=vda."""
    lc = cfg.get("loss", {})
    zd, hw, md = lc.get("sky_vda_zero_disparity", False), lc.get("sky_hinge_weight", 0.0), lc.get("sky_hinge_min_depth", 80.0)
    if not isinstance(zd, bool):
        raise ValueError(f"loss.sky_vda_zero_disparity must be a bool, got {zd!r}")
    if isinstance(hw, bool) or not isinstance(hw, (int, float)) or hw < 0:
        raise ValueError(f"loss.sky_hinge_weight must be a number >= 0, got {hw!r}")
    if isinstance(md, bool) or not isinstance(md, (int, float)) or md <= 0:
        raise ValueError(f"loss.sky_hinge_min_depth must be a positive number, got {md!r}")
    mo, hi = lc.get("sky_hinge_metric_only", False), lc.get("sky_hinge_max_depth")
    if not isinstance(mo, bool):
        raise ValueError(f"loss.sky_hinge_metric_only must be a bool, got {mo!r}")
    if hi is not None and (isinstance(hi, bool) or not isinstance(hi, (int, float)) or hi <= md):
        raise ValueError(f"loss.sky_hinge_max_depth must be null or a number > loss.sky_hinge_min_depth ({md}), got {hi!r}")
    if (mo or hi is not None) and not hw > 0:
        raise ValueError("loss.sky_hinge_metric_only / sky_hinge_max_depth need loss.sky_hinge_weight > 0")
    if mo:
        flags = [x.get("metric_supervision") for x in cfg.get("data", {}).get("sources", [])]
        if flags and any(f is None for f in flags):
            raise ValueError("loss.sky_hinge_metric_only needs data.sources[].metric_supervision set on every train source")
    if not (zd or hw > 0):
        return
    if cfg.get("model", {}).get("arch") not in VDA_LOSS_ARCHES:
        raise ValueError("loss.sky_vda_zero_disparity / sky_hinge_weight need model.arch='vda' (or a protocol-v3 per-frame arch: %s)" % (VDA_LOSS_ARCHES[1:],))
    if zd and str(lc.get("loss_mode")) != "vda":
        raise ValueError("loss.sky_vda_zero_disparity needs loss.loss_mode='vda' (VideoDepthLoss)")
    if hw > 0 and not bool(cfg.get("model", {}).get("metric", False)):
        raise ValueError("loss.sky_hinge_weight needs a metric (depth-native) VDA: model.metric=true")


def _vda_disp(pred: torch.Tensor, batch: dict, pred_is_disparity: bool,
              depth_range: tuple[float, float] | None = None):
    """``(pd, gd, m)``, each ``(B,T,H,W)``: predicted disparity, GT disparity (0 off-mask) and
    the bool mask -- exactly the tensors ``_compute_loss_vda`` feeds VideoDepthLoss, so the
    DN-01 joint temporal term sees the same depth as the depth loss. ``depth_range`` (metric
    models, loss.metric_depth_clamp) clamps predicted depth before inversion; None = 1e-3 floor."""
    device = pred.device
    depth = batch["depth"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True) * (depth > 0)
    B, _, T, H, W = pred.shape
    pd_raw = pred.reshape(B, T, H, W).float()
    if pred_is_disparity:
        pd = pd_raw
    else:
        pd = 1.0 / (pd_raw.clamp_min(1e-3) if depth_range is None else pd_raw.clamp(*depth_range))
    m = mask.reshape(B, T, H, W).bool()
    gd = torch.where(m, 1.0 / depth.reshape(B, T, H, W).clamp_min(1e-3), torch.zeros_like(pd))
    assert torch.isfinite(gd).all(), "non-finite target disparity — NaN/Inf depth upstream?"
    assert torch.isfinite(m.float()).all(), "non-finite mask — corrupt batch?"
    return pd, gd, m

# --------------------------------------------------------------------------- #
# DN-01 Phase C -- joint temporal loss (docs/dn01_joint_temporal_loss.tex).
# --------------------------------------------------------------------------- #
JOINT_TEMPORAL_MODES = ("lse", "sum", "max", "task_lse")
JOINT_TEMPORAL_DEPTH_SPACES = ("disparity_affine", "metric_disparity", "metric_depth")
METRIC_EPS_MIN_DEPTH, METRIC_EPS_MAX_DEPTH = 0.1, 80.0   # metric eval range; see depth_flicker's amendment


def crop_batch(batch: dict, size: int, gen: torch.Generator) -> dict:
    """DN-01 4090 proxy screen (training.frontend_crop): one random ``size``^2 box per step, applied to every
    ``(B,C,T,H,W)`` tensor of the batch (clean and degraded frames alike), AFTER the loader's full-resolution
    degradation, so blur and noise keep their full-frame statistics. ``gen`` is seeded from data.seed."""
    H, W = batch["frames"].shape[-2:]
    y0 = int(torch.randint(0, H - size + 1, (), generator=gen))
    x0 = int(torch.randint(0, W - size + 1, (), generator=gen))
    return {k: v[..., y0:y0 + size, x0:x0 + size]
            if torch.is_tensor(v) and v.dim() == 5 and tuple(v.shape[-2:]) == (H, W) else v
            for k, v in batch.items()}


def clip_gradients(core, clip: float, per_group: bool, split_branch: bool = False) -> None:
    """Global grad-norm clip over all trainable parameters (the default), or -- DN-01 amendment
    2026-09-23, training.grad_clip_per_group -- one clip per trainable module group (head, restore,
    ...). Metric depth losses give the head ~5x the gradient of relative ones (grad probe,
    fp_h8_blur vs A1 step6000), so one global clip starved the zero-init restoration head.
    ``split_branch`` (DN-01 FR3, training.grad_clip_branch_separate): the front-end's FR2 depth branch is clipped as its
    own group, apart from Pocket (provenance/dn01_fr2_mechanism/: the branch's depth gradient otherwise sets the shared
    clip factor for Pocket's restoration-only gradient)."""
    if not per_group:
        torch.nn.utils.clip_grad_norm_(core.trainable_parameters(), clip)
        return
    for name, mod in trainable_modules(core).items():
        parts = [mod.net, mod.branch] if split_branch and name == "frontend" else [mod]
        for part in parts:
            torch.nn.utils.clip_grad_norm_([p for p in part.parameters() if p.requires_grad], clip)


def _clip_median(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return torch.stack([x[i][m[i]].median() if m[i].any() else x.new_ones(())
                        for i in range(x.shape[0])]).clamp_min(1e-6).view(-1, 1, 1, 1)


def depth_flicker(pd: torch.Tensor, gd: torch.Tensor, m: torch.Tensor, vda_repo: str | None = None,
                  space: str = "disparity_affine"):
    """Signed error flicker of depth, ``eps_d = |E_t - E_{t+1}|``, and the pair mask
    ``omega``; both ``(B,T-1,H,W)``. ``pd``/``gd`` are disparities (``_vda_disp``). ``space``:
      disparity_affine  E = (s*pd + b - gd) / med(gd)  (relative models; the default)
      metric_disparity  E = (pd - gd) / med(gd)        (metric: no fit, disparity units)
      metric_depth      E = (1/pd - 1/gd) / med(1/gd)  (metric: no fit, depth units, on m)

    (s, b) is VDA's own per-clip ``compute_scale_and_shift`` on the same flatten
    VideoDepthLoss uses for its TGM (loss.py:253), not detached, as there. ``omega`` is
    VDA's TGM mask verbatim (loss.py:216-227), from GT disparity in every space: valid in
    both frames and ``|dgd| < 0.05 * (max - min)`` of the LATER frame's valid GT disparity
    -- a per-frame relative threshold, not an absolute 0.05.

    AMENDMENT 2026-09-23 (Manager, before any final-twin result): in the metric spaces the
    predicted depth is clamped to the metric eval range [METRIC_EPS_MIN_DEPTH, METRIC_EPS_MAX_DEPTH]
    = [0.1, 80] m before use, and pixels predicted below 0.1 m in either frame leave omega. A
    ReLU head outputting ~0 m otherwise becomes disparity ~1000 and dominates eps_d (fp_h8_awgn
    step8000: metric_disparity eps_d mean 7.56 from a <1% IRS tail). Identical to the unclamped
    definition whenever every prediction is inside the range."""
    near = None
    if space == "disparity_affine":
        s, b = _import_vda_loss_module(vda_repo).compute_scale_and_shift(
            pd.flatten(1, 2), gd.flatten(1, 2), m.flatten(1, 2))
        aligned = s.view(-1, 1, 1, 1) * pd + b.view(-1, 1, 1, 1)
        err = (aligned - gd) / _clip_median(gd, m)
    elif space in ("metric_disparity", "metric_depth"):
        near = pd > 1.0 / METRIC_EPS_MIN_DEPTH                           # predicted depth < 0.1 m
        pd = pd.clamp(1.0 / METRIC_EPS_MAX_DEPTH, 1.0 / METRIC_EPS_MIN_DEPTH)
        if space == "metric_disparity":
            err = (pd - gd) / _clip_median(gd, m)
        else:
            dh = 1.0 / pd                                                # = the model's depth, in [0.1, 80]
            dg = torch.where(m, 1.0 / gd.clamp_min(1e-6), torch.zeros_like(gd))
            err = torch.where(m, (dh - dg) / _clip_median(dg, m), torch.zeros_like(dh))
    else:
        raise ValueError(f"loss.joint_temporal_depth_space must be one of {JOINT_TEMPORAL_DEPTH_SPACES}, "
                         f"got {space!r}")
    eps_d = (err[:, 1:] - err[:, :-1]).abs()
    lo = torch.where(m, gd, torch.inf).amin((-2, -1))
    hi = torch.where(m, gd, -torch.inf).amax((-2, -1))
    th = ((hi - lo) * 0.05)[:, 1:, None, None]
    omega = m[:, 1:] & m[:, :-1] & ((gd[:, 1:] - gd[:, :-1]).abs() < th)
    if near is not None:
        omega = omega & ~near[:, 1:] & ~near[:, :-1]
    return eps_d, omega


def restore_flicker(recon: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Signed error flicker of restoration: channel mean of ``|E_t - E_{t+1}|`` with
    ``E = recon - clean``. ``(B,3,T,H,W)`` -> ``(B,T-1,H,W)``."""
    err = recon.float() - clean.float()
    return (err[:, :, 1:] - err[:, :, :-1]).abs().mean(1)


def joint_temporal_loss(eps_d: torch.Tensor, eps_r: torch.Tensor, omega: torch.Tensor,
                        mode: str, tau: float) -> torch.Tensor:
    """L3, the mean over ``omega`` of a per-pixel coupling of the two flicker maps:
    ``lse`` (C1) smooth max, ``sum`` (C0) separable, ``max`` (C2) hard max, ``task_lse``
    (C0b) smooth max of the two per-clip means. LSE is taken minus ``tau*log 2`` -- no
    gradient change, but L3 = 0 at zero flicker and ``lse -> sum/2`` as ``tau -> inf``.
    An empty ``omega`` gives a zero that keeps its grad_fn."""
    w = omega.float()

    def lse(a, b):
        return tau * torch.logsumexp(torch.stack([a, b]) / tau, 0) - tau * math.log(2.0)

    if mode == "task_lse":
        n = w.flatten(1).sum(1)
        a = (eps_d * w).flatten(1).sum(1) / n.clamp_min(1.0)
        b = (eps_r * w).flatten(1).sum(1) / n.clamp_min(1.0)
        has = (n > 0).float()
        return (lse(a, b) * has).sum() / has.sum().clamp_min(1.0)
    if mode == "lse":
        v = lse(eps_d, eps_r)
    elif mode == "sum":
        v = eps_d + eps_r
    elif mode == "max":
        v = torch.maximum(eps_d, eps_r)
    else:
        raise ValueError(f"loss.joint_temporal_mode must be one of {JOINT_TEMPORAL_MODES}, got {mode!r}")
    return (v * w).sum() / w.sum().clamp_min(1.0)


def joint_temporal_schedule(step: int, lc: dict, total_steps: int) -> tuple[float, float]:
    """``(lambda, tau)`` at optimizer step ``step``. lambda ramps linearly 0 ->
    ``joint_temporal_weight`` over ``joint_temporal_ramp`` (default [8000, 10000]); tau
    anneals by cosine ``joint_temporal_tau`` -> tau/4 from the ramp start to ``total_steps``.
    Weight 0 or absent -> lambda is exactly 0.0 at every step."""
    w = float(lc.get("joint_temporal_weight", 0.0))
    if w <= 0:
        return 0.0, 0.0
    start, end = lc.get("joint_temporal_ramp", [8000, 10000])
    lam = w * min(1.0, max(0.0, (step - start) / max(end - start, 1)))
    frac = min(1.0, max(0.0, (step - start) / max(total_steps - start, 1)))
    tau = float(lc.get("joint_temporal_tau", 0.0)) * (0.25 + 0.375 * (1.0 + math.cos(math.pi * frac)))
    return lam, tau


def enhancement_error(recon: torch.Tensor, clean: torch.Tensor, batch: dict, lc: dict) -> torch.Tensor:
    """The restoration term before ``enh_weight``. Default (``enh_loss: l1``, no FFT term) = L1, op for op.

    DN-01 R2 (docs/DN01_FIX_PROPOSALS.md), both opt-in:
      ``enh_loss: relative``: per frame, L1(recon, clean) / max(L1(degraded input, clean), ``enh_rel_floor``),
        averaged. Every frame counts by its RELATIVE improvement, so the small-error blur-only frames are no longer
        swamped by noise. The floor (default 0.02 = the L1 of AWGN sigma ~6/255) caps the weight of clean and
        near-clean frames: they stay in the loss as a minor term (damaging a clean frame costs err/floor).
        The ratio is O(1), ~30x the L1's scale, so an arm using it rescales enh_weight.
      ``enh_fft_weight`` w > 0: + w * mean |FFT(recon - clean)| (ortho) over ``enh_fft_band`` cycles/px (default
        0.05-0.15: the band a sigma <= 3 px Gaussian attenuates but leaves invertible), after MIMO-UNet's
        frequency loss (Cho et al., ICCV 2021).
      ``enh_gain_weight: true`` (K44 proxy arm (a); after the ELVIS research brief): display-space L1, i.e.
        L1(g * recon, g * target) with g = 1 / degrade_params[:, 2] (= 2^-EV on a darkened lowlight clip, 1 on
        awgn_blur / clean clips; non-positive params, e.g. paired real clips, get 1). With the exposed lowlight
        target the restorer works at native exposure, where a dark clip's residual error is ~g times smaller than
        what the display gain will show; this weights it as shown.
    Shapes (B, 3, T, H, W)."""
    diff = recon - clean
    if lc.get("enh_gain_weight", False):
        p = batch["degrade_params"][:, 2].to(recon.device).float()
        g = torch.where(p > 0, 1.0 / p.clamp_min(1.0 / 16.0), torch.ones_like(p))
        diff = diff * g.view(-1, *([1] * (diff.dim() - 1)))
    mode = str(lc.get("enh_loss", "l1"))
    if mode == "l1":
        err = diff.abs().mean()
    elif mode == "relative":
        y = batch["degraded_frames"].to(recon.device, non_blocking=True)
        den = (y - clean).abs().mean((1, 3, 4)).clamp_min(float(lc.get("enh_rel_floor", 0.02)))
        err = (diff.abs().mean((1, 3, 4)) / den).mean()
    else:
        raise ValueError(f"loss.enh_loss must be 'l1' or 'relative', got {mode!r}")
    w_fft = float(lc.get("enh_fft_weight", 0.0))
    if w_fft > 0:
        lo, hi = (float(v) for v in lc.get("enh_fft_band", (0.05, 0.15)))
        H, W = diff.shape[-2:]
        f = torch.sqrt(torch.fft.fftfreq(H, device=diff.device)[:, None] ** 2
                       + torch.fft.rfftfreq(W, device=diff.device)[None, :] ** 2)
        band = (f >= lo) & (f <= hi)
        err = err + w_fft * torch.fft.rfft2(diff.float(), norm="ortho").abs()[..., band].mean()
    return err


# --------------------------------------------------------------------------- #
# DN-01 K46/K47 WS2 arm A -- restoration-guided temporal consistency of depth (L_rg): GT-free, NON-separable.
# --------------------------------------------------------------------------- #
RG_GUIDES = ("restored", "raw", "none")


def rg_gain(x: torch.Tensor) -> torch.Tensor:
    """bench_joint.blind_gain per clip, on device: ``(B,3,T,H,W)`` -> ``(B,)`` g = clip(GAIN_REF / median over frames
    of the frame's q99.5 luma, 1, GAIN_MAX) -- the v2 display gain every pipeline shares (fitted on training clips)."""
    import bench_joint as bj
    luma = (x.detach().double() * x.new_tensor(bj.LUMA, dtype=torch.float64).view(1, 3, 1, 1, 1)).sum(1)
    q = torch.stack([torch.quantile(c.flatten(1), float(bj.GAIN_STAT[1:]) / 100.0, dim=1) for c in luma])
    stat = torch.quantile(q, 0.5, dim=1)                                  # = np.median (mean of the middle pair)
    return (bj.GAIN_REF / stat.clamp_min(1e-4)).clamp(1.0, bj.GAIN_MAX).float()


def rg_guide(kind: str, rgb: torch.Tensor, recon: torch.Tensor | None) -> torch.Tensor | None:
    """L_rg's guide G, DETACHED: ``restored`` = the model's restored clip, ``raw`` = the degraded input, each x the
    INPUT clip's blind gain (the same g for both guides) and clamped to [0, 1]; ``none`` -> None (weight 1)."""
    if kind == "none":
        return None
    src = recon if kind == "restored" else rgb
    return (src.detach().float() * rg_gain(rgb).view(-1, 1, 1, 1, 1)).clamp(0, 1)


def rg_loss(pred: torch.Tensor, guide: torch.Tensor | None, sigma: float,
            static: torch.Tensor | None = None) -> torch.Tensor:
    """L_rg = mean over consecutive frame pairs and valid pixels of |D_t - D_{t-1}| / med(D) * exp(-|G_t - G_{t-1}|
    / sigma). D = predicted METRIC DISPARITY 1 / clamp(depth, 0.1, 80 m) (the L3 metric_disparity range; pixels
    predicted < 0.1 m in either frame are dropped, as depth_flicker's 2026-09-23 amendment), divided by the clip's
    median predicted disparity (DETACHED: scale-free, needs no GT). |dG| = channel mean of the guide's frame
    difference (``guide`` None -> weight 1, arm A-none). ``static`` (B,) 1 = a K49 STATIC clip (one still x T; arm S):
    the scene is known not to change, so its weight is 1 whatever the guide (no guide is needed, and a noisy raw
    guide must not excuse flicker there). ``pred`` (B,1,T,H,W) metric depth; ``guide`` (B,3,T,H,W)."""
    B, _, T, H, W = pred.shape
    d = pred.reshape(B, T, H, W).float()
    near = d < METRIC_EPS_MIN_DEPTH
    pd = 1.0 / d.clamp(METRIC_EPS_MIN_DEPTH, METRIC_EPS_MAX_DEPTH)
    scale = pd.detach().flatten(1).median(1).values.clamp_min(1e-6).view(B, 1, 1, 1)
    dd = (pd[:, 1:] - pd[:, :-1]).abs() / scale
    if guide is not None:
        w = torch.exp(-(guide[:, :, 1:] - guide[:, :, :-1]).abs().mean(1) / sigma)
        if static is not None:
            st = static.to(w.device, w.dtype).view(-1, 1, 1, 1)
            w = st + (1.0 - st) * w
        dd = dd * w
    valid = (~(near[:, 1:] | near[:, :-1])).float()
    return (dd * valid).sum() / valid.sum().clamp_min(1.0)


def rg_unlabeled_step(model, urgb: torch.Tensor, kind: str, w: float, sigma: float, accum: int, amp_dtype) -> float:
    """One unlabeled clip: backward of w * L_rg / accum and NOTHING else (no depth / restoration / L3 term; the guide is
    detached). Returns L_rg (unweighted) for the log."""
    with torch.autocast(device_type=urgb.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        pred, rec = model(urgb, return_recon=True) if kind == "restored" else (model(urgb), None)
        lrg = rg_loss(pred, rg_guide(kind, urgb, rec), sigma)
    if torch.isfinite(lrg):
        (w * lrg / accum).backward()
    return float(lrg.detach())


def rg_unlabeled_batches(cfg: dict, world_size: int, rank: int):
    """Endless batches (batch 1) of ``data.rg_unlabeled`` (VDASemanticsClipListDataset over the committed list), its
    sources sampled with EQUAL mass, from generators of its own: the labelled stream's sampling and worker seeds are
    untouched (tested), so the labelled data equal the control's."""
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from data.vda_semantics_dataset import VDASemanticsClipListDataset
    d = cfg["data"]
    u = d["rg_unlabeled"]
    lst = Path(u["list"]) if Path(u["list"]).is_absolute() else Path(__file__).resolve().parent / u["list"]
    ds = VDASemanticsClipListDataset(str(lst), u["roots"], int(d.get("num_frames", 32)), int(u.get("clip_stride", 16)),
                                     tuple(d.get("image_size", (518, 518))))
    n = {s: ds.source.count(s) for s in set(ds.source)}
    seed = int(d.get("seed", 42)) * 1000 + 500 + rank
    sampler = WeightedRandomSampler([1.0 / (len(n) * n[s]) for s in ds.source], num_samples=len(ds),
                                    replacement=True, generator=torch.Generator().manual_seed(seed))
    nw = int(u.get("num_workers", 2))
    loader = DataLoader(ds, batch_size=1, sampler=sampler, num_workers=nw, pin_memory=True,
                        persistent_workers=nw > 0, generator=torch.Generator().manual_seed(seed + 1))
    while True:
        yield from loader


def restoration_loss(pred: torch.Tensor, recon: torch.Tensor, batch: dict, cfg: dict, step: int,
                     total_steps: int, pred_is_disparity: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``enh_weight * L1(recon, clean) + lambda(step) * L3`` and the detached L3 (None when
    lambda is 0). With ``joint_temporal_weight`` 0/absent this is A1's term op for op: the
    flicker maps are never computed."""
    lc = cfg.get("loss", {})
    clean_px = batch["frames"].to(recon.device, non_blocking=True)
    out = float(lc.get("enh_weight", 0.0)) * enhancement_error(recon, clean_px, batch, lc)
    lam, tau = joint_temporal_schedule(step, lc, total_steps)
    if lam == 0.0:
        return out, None
    pd, gd, m = _vda_disp(pred, batch, pred_is_disparity)
    eps_d, omega = depth_flicker(pd, gd, m, cfg.get("model", {}).get("vda_repo"),
                                 str(lc.get("joint_temporal_depth_space", "disparity_affine")))
    l3 = joint_temporal_loss(eps_d, restore_flicker(recon, clean_px), omega,
                             str(lc.get("joint_temporal_mode", "lse")), tau)
    return out + lam * l3, l3.detach()


def compute_loss(pred: torch.Tensor, batch: dict, cfg: dict, pred_is_disparity: bool = False) -> torch.Tensor:
    """VDA losses on the depth-output model: disparity-space SSI + spatial gradient
    matching + Temporal Gradient Matching (TGM) + SI-Log.

    ``pred_is_disparity=True`` (VDA's own native output space, e.g. models/vda_net.py's
    relative mode) skips the depth->disparity inversion below entirely, taking ``pred`` as
    disparity directly. Round-tripping VDA's own disparity through depth and back
    (1/disp.clamp(min=1e-6) then here 1/depth.clamp_min(1e-3)) uses two DIFFERENT clamp
    floors in two different spaces -- for near-zero disparity (VDA's own sky_zero_disparity
    convention supervises sky pixels at exactly disparity 0) that mismatch silently corrupts
    precision exactly where it matters. Native disparity needs no floor at all: the model's
    own ReLU already guarantees pred >= 0.

    ``loss.loss_mode: "vda"`` (VAL-01) bypasses everything below and reuses VDA's own
    loss/loss.py directly instead of this function's hand-rolled SSI/TGM -- see
    _compute_loss_vda's docstring for exactly how and why they diverge otherwise.
    """
    lc = cfg.get("loss", {})
    if str(lc.get("loss_mode", "custom")) == "vda":
        return _compute_loss_vda(pred, batch, cfg, pred_is_disparity)
    zero = _no_gt_zero(pred, batch)
    if zero is not None:   # video_only clip: every term below is 0/0 (SSI align -> NaN)
        return zero
    beta = float(lc.get("disp_weight", lc.get("ssi_weight", 1.0)))
    alpha = float(lc.get("temporal_weight", lc.get("tgm_weight", 0.1)))
    grad_w = float(lc.get("gradient_weight", 0.5))
    silog_w = float(lc.get("silog_weight", 0.0))
    metric_w = float(lc.get("metric_weight", 0.0))   # absolute (metric) scale supervision
    distill_w = float(lc.get("distill_weight", 0.0)) # Depth Pro clean-teacher distillation

    device = pred.device
    depth = batch["depth"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True) * (depth > 0)

    B, _, T, H, W = pred.shape
    m = mask.reshape(B * T, H, W)
    if pred_is_disparity:
        pd = pred.reshape(B * T, H, W).float()
    else:
        pd = (1.0 / pred.reshape(B * T, H, W).float().clamp_min(1e-3))         # pred disparity
    gd = torch.where(m.bool(), 1.0 / depth.reshape(B * T, H, W).clamp_min(1e-3), torch.zeros_like(pd))

    if bool(lc.get("sky_zero_disparity", False)):
        # DA2/VDA sky handling: sky pixels are supervised at disparity 0 and take part in
        # the SSI fit, gradient and TGM terms. gd is already 0 there (sky is outside
        # `mask`), so widening m is the whole change. Metrics/eval masks are untouched.
        if "sky" not in batch:
            raise KeyError("sky_zero_disparity=true but the batch has no 'sky' mask — stale data loaders?")
        m = (m + batch["sky"].to(device, non_blocking=True).reshape(B * T, H, W)).clamp(max=1.0)

    # SSI alignment: per-frame (default, scale-drift-blind) or per-clip (one (s,t) over the T
    # frames → penalises frame-to-frame scale drift, matching the per-video eval). Erosion was
    # diagnosed as temporal scale drift that per-frame SSI cannot see.
    if str(lc.get("ssi_align", "perframe")) == "perclip":
        aligned = _scale_shift_align(
            pd.reshape(B, T * H, W), gd.reshape(B, T * H, W), m.reshape(B, T * H, W)
        ).reshape(B * T, H, W)
    else:
        aligned = _scale_shift_align(pd, gd, m)                                # per-FRAME align pred→gt disparity
    loss_ssi = (m * (aligned - gd).abs()).sum() / m.sum().clamp_min(1.0)

    loss_grad = pred.new_zeros(())
    if grad_w > 0:
        loss_grad = _gradient_loss_multiscale(aligned, gd, m)

    loss_tgm = pred.new_zeros(())
    if T > 1 and alpha > 0:
        # ONE (s,t) per clip (shared over T) — VDA's per-clip align before differencing.
        # Without this, the per-frame align above removes the very scale/shift flicker TGM
        # is meant to penalise (verified: per-frame TGM == 0 on flickered predictions).
        aligned_clip = _scale_shift_align(
            pd.reshape(B, T * H, W), gd.reshape(B, T * H, W), m.reshape(B, T * H, W)
        ).reshape(B, T, H, W)
        a = aligned_clip
        g = gd.reshape(B, T, H, W)
        mt = m.reshape(B, T, H, W)
        mm = mt[:, 1:] * mt[:, :-1]
        dg, da = g[:, 1:] - g[:, :-1], a[:, 1:] - a[:, :-1]
        scale = gd[m.bool()].abs().mean().clamp_min(1e-6) if m.any() else gd.new_tensor(1.0)
        small_motion = (dg.abs() < 0.05 * scale).float()                       # exclude depth edges
        tgm_mask = mm * small_motion
        loss_tgm = (tgm_mask * (da - dg).abs()).sum() / tgm_mask.sum().clamp_min(1.0)  # signed diff

    loss_metric = pred.new_zeros(())
    if metric_w > 0:
        # Absolute supervision in inverse-depth, NO scale-shift align — pins the metric
        # scale (VKITTI/TartanAir GT are metric). Headline-metric path; 0 = pure relative.
        loss_metric = (m * (pd - gd).abs()).sum() / m.sum().clamp_min(1.0)

    loss_distill = pred.new_zeros(())
    if distill_w > 0 and "teacher_depth" in batch:
        # Distil the DEGRADED-input student toward Depth Pro's CLEAN-frame depth — a strong
        # external clean-teacher for robustness. Scale-shift-invariant (align pred→teacher in
        # disparity) so the teacher's absolute scale errors don't bite; + gradient term for the
        # teacher's sharp boundaries. Masked to teacher-valid frames/pixels (TartanAir only in v1).
        tdep = batch["teacher_depth"].to(device, non_blocking=True).reshape(B * T, H, W)
        tval = batch["teacher_valid"].to(device, non_blocking=True).reshape(B * T)
        tm = ((tval[:, None, None] > 0.5) & (tdep > 1e-3) & (tdep < 80.0) & torch.isfinite(tdep)).float()
        if tm.sum() > 0:
            tdisp = torch.where(tm.bool(), 1.0 / tdep.clamp_min(1e-3), torch.zeros_like(tdep))
            loss_distill = distillation_loss(pd, tdisp, tm, grad_w)

    loss_silog = pred.new_zeros(())
    if silog_w > 0:
        # Scale-Invariant Log Loss (per-frame). Sky / zero-disparity pixels must NOT enter
        # the log term: log-depth ratio is undefined at infinite depth (gd→0), so with
        # sky_zero_disparity on, log(gd=0) = -inf → NaN. `sm` keeps only positive-disparity
        # valid pixels; identical to `m` when sky_zero_disparity is off (gd>1e-6 wherever m).
        sm = m.bool() & (gd > 1e-6)
        ld = torch.zeros_like(pd)
        ld[sm] = torch.log(pd[sm]) - torch.log(gd[sm])
        m_sum = sm.reshape(B * T, -1).sum(1)
        valid = m_sum > 1
        if valid.any():
            ld_mean = ld.reshape(B * T, -1).sum(1)[valid] / m_sum[valid]
            ld_sq_mean = (ld**2).reshape(B * T, -1).sum(1)[valid] / m_sum[valid]
            loss_silog = (ld_sq_mean - ld_mean**2).mean()

    return (beta * loss_ssi + grad_w * loss_grad + alpha * loss_tgm
            + metric_w * loss_metric + distill_w * loss_distill + silog_w * loss_silog)


def _slice_batch_time(batch: dict, start: int, stop: int, total_t: int) -> dict:
    """Slice tensors carrying the clip time dimension while preserving metadata."""
    out = {}
    time_keys = {"frames", "degraded_frames", "depth", "mask", "sky", "teacher_depth",
                 "teacher_valid", "frame_valid"}
    for key, value in batch.items():
        if key in time_keys and torch.is_tensor(value):
            if value.ndim >= 5 and value.shape[2] == total_t:
                out[key] = value[:, :, start:stop]
            elif value.ndim >= 2 and value.shape[1] == total_t:
                out[key] = value[:, start:stop]
            else:
                out[key] = value
        else:
            out[key] = value
    return out


def compute_prefix_newest_loss(model, rgb: torch.Tensor, batch: dict, cfg: dict) -> torch.Tensor:
    """Train from independently evaluated prefixes, supervising newest predictions only."""
    pid = getattr(model, "predicts_disparity", False)
    total_t = rgb.shape[2]
    if total_t < 2:
        return compute_loss(model(rgb), batch, cfg, pred_is_disparity=pid)
    endpoint = int(torch.randint(2, total_t + 1, (), device=rgb.device).item())
    current = model(rgb[:, :, :endpoint])[:, :, -1:]
    previous = model(rgb[:, :, :endpoint - 1])[:, :, -1:]

    spatial_cfg = copy.deepcopy(cfg)
    spatial_cfg.setdefault("loss", {})["tgm_weight"] = 0.0
    current_batch = _slice_batch_time(batch, endpoint - 1, endpoint, total_t)
    spatial = compute_loss(current, current_batch, spatial_cfg, pred_is_disparity=pid)

    temporal_cfg = copy.deepcopy(cfg)
    temporal_cfg.setdefault("loss", {}).update({
        "ssi_weight": 0.0, "disp_weight": 0.0, "gradient_weight": 0.0,
        "silog_weight": 0.0, "metric_weight": 0.0, "distill_weight": 0.0,
    })
    pair = torch.cat((previous, current), dim=2)
    pair_batch = _slice_batch_time(batch, endpoint - 2, endpoint, total_t)
    return spatial + compute_loss(pair, pair_batch, temporal_cfg, pred_is_disparity=pid)


def consistency_loss(pred_deg, pred_clean, batch):
    """Clean↔degraded consistency: the degraded-input prediction should match the (stop-grad)
    clean-input prediction in structure. Both are robustly normalised (median/MAD) per frame and
    compared with a masked L1 — decouples predicted structure from input sharpness. Robust
    normalisation avoids the ill-conditioned scale-shift fit that explodes on rough early predictions.
    """
    device = pred_deg.device
    depth = batch["depth"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True) * (depth > 0)
    B, _, T, H, W = pred_deg.shape
    m = mask.reshape(B * T, H, W)
    nd = _robust_norm_stable(1.0 / pred_deg.reshape(B * T, H, W).float().clamp_min(1e-3), m)
    nc = _robust_norm_stable(1.0 / pred_clean.reshape(B * T, H, W).float().clamp_min(1e-3), m)
    return (m * (nd - nc.detach()).abs()).sum() / m.sum().clamp_min(1.0)


def feature_consistency_loss(feats_deg, feats_clean, gram_weight: float = 0.0):
    """Feature-space clean↔degraded consistency: match the post-temporal tokens of the
    degraded path to the (stop-grad) clean path. Both sides are layer-normed first —
    raw DINOv2 tokens carry outlier dimensions in the hundreds and the clean target is
    produced by the same trainable mixer, so an unnormalized loss has an unanchored
    scale mode that diverges (observed: loss 168→NaN in 180 steps). Normalizing makes
    the loss scale-invariant and O(1), matching feature-distillation practice (DVT).

    gram_weight > 0: also match Gram matrices (feature covariance topology) — a
    domain-invariance regulariser that doesn't require spatial alignment."""
    total = feats_deg[0][0].float().new_zeros(())
    for (td, _), (tc, _) in zip(feats_deg, feats_clean):
        td = F.layer_norm(td.float(), td.shape[-1:])
        tc = F.layer_norm(tc.float(), tc.shape[-1:])
        total = total + F.smooth_l1_loss(td, tc.detach())
        if gram_weight > 0.0:
            # td/tc: (B, T, N, C) → flatten BT → (BT, N, C)
            flat_d = td.flatten(0, -3)  # (BT, N, C)
            flat_c = tc.flatten(0, -3)
            N = flat_d.shape[-2]
            Gd = torch.bmm(flat_d.transpose(-1, -2), flat_d) / N   # (BT, C, C)
            Gc = torch.bmm(flat_c.transpose(-1, -2), flat_c) / N
            total = total + gram_weight * F.smooth_l1_loss(Gd, Gc.detach())
    return total / max(len(feats_deg), 1)


def _robust_norm_stable(disp, mask, eps=1e-2):
    """Per-frame median-centred, MAD-scaled disparity normalisation (MiDaS SSI), with the
    median/MAD *detached* and the MAD floor raised, so a low-variance degraded prediction
    can't drive ``1/MAD`` to a NaN."""
    out = torch.zeros_like(disp)
    for i in range(disp.shape[0]):
        v = mask[i].bool()
        if v.sum() < 2:
            continue
        d = disp[i][v]
        t = d.median().detach()
        s = (d - t).abs().mean().clamp_min(eps).detach()
        out[i] = (disp[i] - t) / s
    return out


def output_consistency_loss(pred_deg, pred_teacher, batch, conf=None):
    """Output-space consistency (EMA-teacher / siamese): stabilized robust-norm L1 between
    the degraded prediction and the (stop-grad) clean-teacher prediction, optionally
    weighted per-pixel by the teacher's depth confidence."""
    device = pred_deg.device
    depth = batch["depth"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True) * (depth > 0)
    B, _, T, H, W = pred_deg.shape
    m = mask.reshape(B * T, H, W)
    nd = _robust_norm_stable(1.0 / pred_deg.reshape(B * T, H, W).float().clamp_min(1e-3), m)
    nc = _robust_norm_stable(1.0 / pred_teacher.reshape(B * T, H, W).float().clamp_min(1e-3), m)
    w = m
    if conf is not None:
        c = conf.detach().float()
        while c.dim() > 4:
            c = c.squeeze(-1)
        w = m * c.reshape(B * T, H, W)
    return (w * (nd - nc.detach()).abs()).sum() / w.sum().clamp_min(1.0)


# --------------------------------------------------------------------------- #
# Validation — canonical per-VIDEO disparity-SSI metrics.
# Matches Video-Depth-Anything's benchmark (benchmark/eval/eval.py): ONE scale+shift
# PER VIDEO over all its valid pixels (NOT per-clip), then per-frame depth metrics
# averaged. Per-clip alignment gives each 4/16-frame clip its own affine, which
# over-fits, flatters the metric, and hides cross-clip scale drift (a model that is
# not globally/temporally consistent still scores well). Per-video alignment is the
# protocol the field reports and reflects real video consistency, so it is the
# correct signal for best.pt selection. Stats are keyed by ``video_id`` so the
# protocol stays correct when val spans more than one sequence.
# --------------------------------------------------------------------------- #
def _allreduce_sum(values, device):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        t = torch.tensor(values, device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.tolist()
    return list(values)


def _solve_per_video_alignment(stats: dict, device: str) -> dict:
    """Solve one disparity (s, t) per video from accumulated lstsq stats
    ``{vid: [a00, a01, a11, b0, b1]}``. Under DDP a video's pixels are sharded across
    ranks, so the per-video sums are all-reduced over a rank-consistent video ordering."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, sorted(stats.keys()))
        all_vids = sorted({v for g in gathered for v in g})
        mat = torch.zeros(len(all_vids), 5, device=device, dtype=torch.float64)
        for idx, vid in enumerate(all_vids):
            if vid in stats:
                mat[idx] = torch.tensor(stats[vid], device=device, dtype=torch.float64)
        dist.all_reduce(mat, op=dist.ReduceOp.SUM)
        rows = {vid: mat[idx].tolist() for idx, vid in enumerate(all_vids)}
    else:
        rows = stats

    out = {}
    for vid, (a00, a01, a11, b0, b1) in rows.items():
        det = a00 * a11 - a01 * a01
        if abs(det) < 1e-12:
            out[vid] = (1.0, 0.0)
        else:
            out[vid] = ((a11 * b0 - a01 * b1) / det, (-a01 * b0 + a00 * b1) / det)
    return out


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches=None) -> dict:
    model.eval()
    pred_is_disparity = getattr(model, "predicts_disparity", False)
    # ---- Pass 1: forward once, accumulate PER-VIDEO disparity lstsq stats, cache preds. ----
    stats: dict = {}  # video_id -> [a00, a01, a11, b0, b1]
    cache = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        frames = batch.get("degraded_frames", batch["frames"]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(frames)  # (B,1,T,H,W) depth, or disparity if pred_is_disparity
        depth = batch["depth"].to(device)
        mask = batch["mask"].to(device)
        if pred_is_disparity:
            pred_disp = pred.float()
        else:
            pred_disp = 1.0 / pred.float().clamp(min=1e-3)  # model emits depth -> disparity
        gt_disp = 1.0 / depth.clamp(min=1e-3)
        valid = mask.bool() & (depth > 1e-3) & (depth < 80.0) & torch.isfinite(pred_disp)
        vids = list(batch["video_id"])
        for b in range(pred_disp.shape[0]):
            vb = valid[b]
            if vb.sum() == 0:
                continue
            pd, gd = pred_disp[b][vb], gt_disp[b][vb]
            acc_v = stats.setdefault(vids[b], [0.0, 0.0, 0.0, 0.0, 0.0])
            acc_v[0] += (pd * pd).sum().item(); acc_v[1] += pd.sum().item(); acc_v[2] += float(pd.numel())
            acc_v[3] += (pd * gd).sum().item(); acc_v[4] += gd.sum().item()
        cache.append((pred_disp.half().cpu(), depth.cpu(), mask.cpu(), vids))

    # ---- One scale+shift PER VIDEO (s*pred_disp + t ~= gt_disp), shared across ranks. ----
    align = _solve_per_video_alignment(stats, device)

    # ---- Pass 2: per-frame metrics, each sample using its own video's alignment. ----
    acc = {"frames": 0.0, "abs_rel": 0.0, "abs_rel_metric": 0.0, "abs_rel_perframe": 0.0,
           "sq_rel": 0.0, "rmse": 0.0, "d1": 0.0, "silog": 0.0}
    vid_scales: dict = {}   # video_id -> list of per-frame optimal disparity scales (drift diagnostic)
    vid2src = getattr(loader, "vid2src", None)  # multi-source val: per-source breakdown
    src_acc: dict = {}      # source -> [frames, abs_rel_sum] (single-rank; not allreduced)
    for pred_disp, depth, mask, vids in cache:
        pred_disp = pred_disp.float().to(device); depth = depth.to(device); mask = mask.to(device)
        valid = mask.bool() & (depth > 1e-3) & (depth < 80.0)
        gt_disp = 1.0 / depth.clamp(min=1e-3)
        B, _, T, H, W = pred_disp.shape
        for b in range(B):
            s, t = align.get(vids[b], (1.0, 0.0))
            dep = (1.0 / (s * pred_disp[b] + t).clamp(min=1e-3)).clamp(1e-3, 80.0)  # (1,T,H,W)
            dep_m = (1.0 / pred_disp[b].clamp(min=1e-3)).clamp(1e-3, 80.0)          # metric, no align
            df = dep.reshape(T, H, W); dfm = dep_m.reshape(T, H, W)
            pdf = pred_disp[b].reshape(T, H, W); gdf = gt_disp[b].reshape(T, H, W)
            gf = depth[b].reshape(T, H, W); vf = valid[b].reshape(T, H, W)
            for j in range(T):
                v = vf[j]
                n = v.sum().item()
                if n == 0:
                    continue
                p, g = df[j][v], gf[j][v]
                diff = p - g; ld = torch.log(p) - torch.log(g); ratio = torch.max(p / g, g / p)
                acc["frames"] += 1.0
                acc["abs_rel"] += (diff.abs() / g).mean().item()
                if vid2src is not None:
                    sa = src_acc.setdefault(vid2src.get(vids[b], "?"), [0.0, 0.0])
                    sa[0] += 1.0
                    sa[1] += (diff.abs() / g).mean().item()
                acc["abs_rel_metric"] += ((dfm[j][v] - g).abs() / g).mean().item()
                acc["sq_rel"] += ((diff**2) / g).mean().item()
                acc["rmse"] += (diff**2).mean().sqrt().item()
                acc["d1"] += (ratio < 1.25).float().mean().item()
                acc["silog"] += ((ld**2).mean() - ld.mean() ** 2).item()
                # --- per-FRAME alignment (diagnostic): own (s_f, t_f) per frame ---
                pj, gj = pdf[j][v], gdf[j][v]
                a00 = (pj * pj).sum(); a01 = pj.sum(); a11 = pj.new_tensor(float(n))
                bb0 = (pj * gj).sum(); bb1 = gj.sum(); det = a00 * a11 - a01 * a01
                if det.abs() > 1e-9:
                    s_f = ((a11 * bb0 - a01 * bb1) / det); t_f = ((a00 * bb1 - a01 * bb0) / det)
                    depf = (1.0 / (s_f * pj + t_f).clamp(min=1e-3)).clamp(1e-3, 80.0)
                    acc["abs_rel_perframe"] += ((depf - g).abs() / g).mean().item()
                    vid_scales.setdefault(vids[b], []).append(float(s_f))
                else:
                    acc["abs_rel_perframe"] += (diff.abs() / g).mean().item()

    keys = list(acc.keys())
    vals = _allreduce_sum([acc[k] for k in keys], device)
    acc = {k: v for k, v in zip(keys, vals)}
    f = max(acc["frames"], 1.0)
    # Scale-drift diagnostic: mean over videos of the per-frame scale CV (std/mean).
    # Computed locally (single-rank diagnostic); high values => temporal scale drift.
    import numpy as _np
    cvs = [float(_np.std(v) / (abs(_np.mean(v)) + 1e-9)) for v in vid_scales.values() if len(v) > 1]
    out = {
        "abs_rel": acc["abs_rel"] / f,
        "abs_rel_metric": acc["abs_rel_metric"] / f,
        "abs_rel_perframe": acc["abs_rel_perframe"] / f,
        "scale_cv": float(_np.mean(cvs)) if cvs else 0.0,
        "sq_rel": acc["sq_rel"] / f,
        "rmse": acc["rmse"] / f,
        "delta1": acc["d1"] / f,
        "si_log": acc["silog"] / f,
        "num_frames": int(acc["frames"]),
    }
    if src_acc:
        out["per_source"] = {k: v[1] / max(v[0], 1.0) for k, v in sorted(src_acc.items())}
    return out


def _checkpoint_gate(gstep: int, val_every: int, checkpoint_every: int,
                      checkpoint_at_steps: set[int]) -> tuple[bool, bool]:
    """Whether this step runs a validation pass and/or saves a resumability checkpoint.

    Decoupled (Fable, 2026-08-26) so relaxing val_every for contention/overhead reasons
    doesn't widen the step-window a TIMEOUT can lose: checkpoint_every governs a cheap
    weights-only last.pt save on its own cadence. A checkpoint_at_steps hit always forces
    both -- it is the permanent stepN.pt snapshot, which needs a real eval.
    """
    do_val = gstep % val_every == 0 or gstep in checkpoint_at_steps
    do_checkpoint = gstep % checkpoint_every == 0 or gstep in checkpoint_at_steps
    return do_val, do_checkpoint


# --------------------------------------------------------------------------- #
# Train loop.
# --------------------------------------------------------------------------- #
def _assert_and_log_git_state(expected_commit: str | None, *, under_slurm: bool) -> None:
    """Fable's ruling, VAL-01, 2026-08-27 (twice-amended after a real failure and a real
    loophole, both found by actually running this): never assume submission-time commit
    == start-time commit -- a queued job can wait hours, and the checkout can change
    underneath it in that time (this exact repo had a job queue that long while more
    fixes landed).

    ``under_slurm`` (SLURM_JOB_ID present in the environment), not merely
    ``expected_commit`` being set, is what makes enforcement MANDATORY: under Slurm,
    VAL01_EXPECTED_COMMIT must be present, git must succeed, the working tree must be
    clean, and HEAD must match -- any of those failing raises. This closes a loophole in
    an earlier version keyed on ``expected_commit`` alone: a submitter who simply forgot
    to set the env var would have silently gotten the permissive branch instead of a
    failure. Under Slurm, forgetting it is now itself a hard failure, not a bypass.

    When NOT under Slurm (smoke tests, local dev, a deployment that is a plain rsync snapshot
    rather than a git checkout), this is a plain log line, not a requirement: treating a missing
    checkout as an error would make every such smoke fail before it even builds the model. Its
    own commit provenance then rests on the copied tree plus whatever artifact records the run."""
    if not under_slurm:
        print("[git] verification skipped (not under Slurm)", flush=True)
        return
    if expected_commit is None:
        raise RuntimeError(
            "SLURM_JOB_ID is set but VAL01_EXPECTED_COMMIT is not -- a Slurm job must "
            "declare the commit it expects to run at submission time; this is not "
            "optional under Slurm"
        )
    repo_root = Path(__file__).resolve().parent
    try:
        head = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                                       text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            f"VAL01_EXPECTED_COMMIT={expected_commit!r} is set (under Slurm) but train.py "
            f"could not determine its own git commit at {repo_root} -- not a git checkout, "
            "or git is unavailable"
        ) from e
    print(f"[git] HEAD={head}", flush=True)
    porcelain = subprocess.check_output(["git", "-C", str(repo_root), "status", "--porcelain"], text=True)
    if porcelain.strip():
        raise RuntimeError(
            f"VAL01_EXPECTED_COMMIT is set but the working tree at {repo_root} is dirty:\n"
            f"{porcelain}\nA commit hash proves nothing about code with uncommitted changes on top of it."
        )
    if head != expected_commit:
        raise RuntimeError(
            f"VAL01_EXPECTED_COMMIT={expected_commit!r} but the checked-out HEAD is {head!r} -- "
            "the code changed between job submission and start; resubmit against the code "
            "actually intended to run"
        )


def train(cfg: dict) -> None:
    _assert_and_log_git_state(os.environ.get("VAL01_EXPECTED_COMMIT"),
                              under_slurm="SLURM_JOB_ID" in os.environ)
    t = cfg.get("training", {})

    # Silence DA3 backbone per-forward INFO spam (95% "Selecting reference view" noise).
    logging.getLogger("depth_anything_3").setLevel(logging.WARNING)

    # ---- Distributed setup (torchrun sets LOCAL_RANK / WORLD_SIZE) ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    seed = seed_everything(int(cfg.get("data", {}).get("seed", 42)), rank, str(cfg.get("training", {}).get("seed_scheme", "add")))
    ddp = world_size > 1
    backend = os.environ.get("D3_DIST_BACKEND", "nccl")  # gloo for CPU/local testing
    if ddp and backend == "nccl" and torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        dev_idx = local_rank % max(1, torch.cuda.device_count())  # share GPU if fewer GPUs than ranks
        torch.cuda.set_device(dev_idx)
        device = f"cuda:{dev_idx}"
    else:
        if ddp:
            dist.init_process_group(backend=backend)
        device = cfg.get("model", {}).get("device", "cuda")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
    is_main = rank == 0
    amp_dtype = torch.bfloat16 if (device.startswith("cuda") and t.get("precision", "bf16") == "bf16") else None

    ckpt_dir = Path(t.get("checkpoint_dir", "checkpoints/d3"))
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "metrics.jsonl"

    # Mandate guard: a config may pin its temporal module so the paper line can't silently
    # drift (e.g. DyFN → restoration — the swap that orphaned the contribution). Opt-in: only
    # fires when require_temporal_type is set, so all existing configs are byte-for-byte unaffected.
    _req_tt = cfg.get("model", {}).get("require_temporal_type")
    if _req_tt and cfg.get("model", {}).get("temporal_type") != _req_tt:
        raise SystemExit(f"[mandate] config requires temporal_type={_req_tt!r} but got "
                         f"{cfg.get('model', {}).get('temporal_type')!r} — refusing to train.")

    # Build model on this rank's device; optimizer + freeze-assert on the core module.
    core = build_model(cfg).to(device)
    init_from = cfg.get("model", {}).get("init_from")
    if init_from:  # warm-start trainable weights (head + temporal) from a converged run;
        ck = torch.load(str(init_from), map_location="cpu", weights_only=False)  # shape-tolerant:
        src = ck.get("trainable_ema", ck.get("trainable", ck.get("model_ema", ck.get("model", {}))))
        tgt = torch.nn.ModuleDict(trainable_modules(core))
        dst = tgt.state_dict()
        # Back-compat: pre-MoLoRA checkpoints (big2..big16) named adapters `lora_{a,b}.weight`;
        # the current ModuleList wrapper expects `lora_{a,b}.0.weight`. Remap so warm-start from
        # those checkpoints loads all 304 LoRA tensors instead of silently reinitialising them.
        src = {(k.replace(".lora_a.weight", ".lora_a.0.weight").replace(".lora_b.weight", ".lora_b.0.weight")
                if k not in dst else k): v for k, v in src.items()}
        # dyfn_attn_renorm graft: a restoration checkpoint's `temporal.N.<k>` weights live at
        # `temporal.N.attn.<k>` in the new mixer. Remap so the trained attention warm-starts
        # (zero renorm gate then makes the graft EXACTLY the source model at step 0).
        src = {(_re.sub(r"^(temporal\.\d+\.)", r"\1attn.", k)
                if k not in dst and _re.sub(r"^(temporal\.\d+\.)", r"\1attn.", k) in dst else k): v
               for k, v in src.items()}
        no_lora = bool(cfg.get("model", {}).get("init_from_no_lora", False))
        skip = list(cfg.get("model", {}).get("init_from_skip", []))  # e.g. ['.gate'] to keep a fresh gate init
        keep = {k: v for k, v in src.items()
                if k in dst and dst[k].shape == v.shape
                and not (no_lora and k.startswith("lora."))
                and not any(s in k for s in skip)}
        init_sha = cfg.get("model", {}).get("init_from_sha256")
        if init_sha is not None:
            # METRICFIX (default off): a 2-STAGE arm pins its stage-1 checkpoint BY SHA256 ('record' = hash it and write init_from.json next to the run's checkpoints; a hex = refuse a different file), and every
            # stage-1 tensor must land in a stage-2 tensor (a silently dropped key would restart that module from the released init).
            got = _sha256_file(init_from)
            if init_sha != "record" and got != init_sha:
                raise ValueError(f"model.init_from sha256 mismatch: {init_from} is {got}, the config pins {init_sha}")
            missed = sorted(k for k in src if k not in keep)
            if missed:
                raise ValueError(f"model.init_from {init_from}: {len(missed)} stage-1 tensors did not load (absent or shape-changed in this model), e.g. {missed[:3]}")
            if is_main:
                Path(cfg.get("training", {}).get("checkpoint_dir", "checkpoints/d3")).mkdir(parents=True, exist_ok=True)
                (Path(cfg.get("training", {}).get("checkpoint_dir", "checkpoints/d3")) / "init_from.json").write_text(
                    json.dumps({"init_from": str(init_from), "sha256": got, "tensors_loaded": len(keep), "tensors_in_model": len(dst)}) + "\n")
        tgt.load_state_dict(keep, strict=False)
        if is_main:
            print(f"[init] warm-started {len(keep)}/{len(dst)} tensors from {init_from}", flush=True)
    optimizer = build_optimizer(core, cfg, verbose=is_main)
    unfrozen = getattr(core, "backbone_trainable", False)
    # Frozen runs train the head AND the temporal mixers — both must be EMA'd and
    # saved, else the checkpoint can't reproduce the model (temporal mixers would
    # reload at their identity init).
    if unfrozen:
        ema_target = core
    else:
        ema_target = torch.nn.ModuleDict(trainable_modules(core))
    ema = EMA(ema_target, decay=float(t.get("ema_decay", 0.999)))
    if ddp and device.startswith("cuda"):
        di = torch.cuda.current_device()
        model = DDP(core, device_ids=[di], output_device=di, broadcast_buffers=False,
                    find_unused_parameters=True)  # DualDPT ray-head branch is unused
    elif ddp:
        model = DDP(core, broadcast_buffers=False, find_unused_parameters=True)  # CPU/gloo
    else:
        model = core

    train_loader, train_sampler = build_dataloader(cfg, "train", world_size, rank)
    val_loader, _ = build_dataloader(cfg, "val", world_size, rank)
    # Diagnostic: also track CLEAN val during degraded training (forgetting vs degradation-specific).
    clean_val_loader = None
    if bool(cfg.get("eval", {}).get("track_clean", False)) and bool(cfg.get("data", {}).get("degrade", False)):
        clean_cfg = copy.deepcopy(cfg); clean_cfg["data"]["degrade"] = False
        clean_val_loader, _ = build_dataloader(clean_cfg, "val", world_size, rank)
    monitor = OutdoorMonitor(cfg) if (cfg.get("eval") or {}).get("outdoor_monitor") else None   # METRICFIX (default off)
    log_terms = bool(cfg.get("loss", {}).get("log_terms", False))
    terms_every = int(cfg.get("loss", {}).get("log_terms_every", 100))
    if monitor is not None and is_main:
        print(f"[monitor] outdoor VKITTI scale monitor: {len(monitor.idx)} clips (in-sample drift monitor, never used for selection), every {monitor.every} steps, clean_too={monitor.clean_too}", flush=True)
    if is_main:
        print(f"[data] train clips={len(train_loader.dataset)} val clips={len(val_loader.dataset)} "
              f"world_size={world_size} device={device} seed={seed}", flush=True)
        if getattr(train_loader, "depth_bounds", None):      # METRICFIX: always printed (a log line only)
            print("[data] effective train depth upper bound per source (m, valid iff depth < bound): " + ", ".join(f"{n}={b:g}" for n, b in train_loader.depth_bounds), flush=True)
        if getattr(train_loader, "video_only_clips", 0):
            # DIST-01 arm (c): the video-only clips have NO ground truth -- their only
            # supervision is the online teacher, so this split must be obvious in the log.
            print(f"[data] GT clips={train_loader.gt_clips} "
                  f"video-only (teacher-supervised, no GT) clips={train_loader.video_only_clips}",
                  flush=True)

    max_epochs = int(t.get("max_epochs", 1))
    max_steps = int(t.get("max_steps", 0))
    accum = int(t.get("grad_accum", 1))
    clip = float(t.get("grad_clip", 1.0))
    split_branch = bool(t.get("grad_clip_branch_separate", False))
    if split_branch and not (t.get("grad_clip_per_group", False) and getattr(core, "frontend_branch", False)):
        raise ValueError("training.grad_clip_branch_separate needs grad_clip_per_group and model.frontend_depth_branch")
    warmup = int(t.get("warmup_steps", 100))
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = max_steps or steps_per_epoch * max_epochs
    val_every = int(t.get("val_every_steps", 0)) or steps_per_epoch
    val_max_batches = t.get("val_max_batches")
    # Resumability shouldn't cost a validation pass: checkpoint_every_steps saves last.pt
    # (weights only, no eval) on its own cadence, independent of val_every. Fable,
    # 2026-08-26: relaxing val_every to 1000+ for future screens to cut contention-amplified
    # validation overhead must not widen the window a TIMEOUT can lose -- default keeps the
    # old coupled behavior (0 = same as val_every) so this is opt-in per config.
    checkpoint_every = int(t.get("checkpoint_every_steps", 0)) or val_every
    # Distinguishes pretrained capability / early adaptation / catastrophic forgetting /
    # scratch recovery (docs/CVPR_2027_RESUBMISSION_PLAN.md's checkpoint-schedule
    # requirement) -- these steps get their own permanent stepN.pt in addition to
    # whatever last.pt/best.pt would do at that tick, and force an eval+save even
    # when they don't land on the periodic val_every cadence.
    checkpoint_at_steps = {int(s) for s in t.get("checkpoint_at_steps", [])}
    if is_main:
        # Opus/Fable's audit, 2026-08-27 (job 6155634): checkpoint_every_steps only ever
        # governs last.pt, which gets overwritten every cadence -- permanent step{N}.pt
        # snapshots require checkpoint_at_steps explicitly, a config key easy to forget
        # since nothing fails loudly when it's unset (val01_vda_s_relative.yaml never set
        # it; the entire run's intermediate curve was unrecoverable once this was noticed
        # mid-run, worked around only by a separate, out-of-band snapshot watcher). Print
        # the actual plan so this is visible in every job's own log, not just discoverable
        # by reading the config or losing data first.
        if checkpoint_at_steps:
            print(f"[checkpoint] permanent step{{N}}.pt snapshots at: {sorted(checkpoint_at_steps)}", flush=True)
        else:
            print("[checkpoint] permanent step{N}.pt snapshots: none (checkpoint_at_steps unset) -- "
                  "only last.pt (overwritten every checkpoint_every_steps) will exist", flush=True)
    lc = cfg.get("loss", {})
    cons_w = float(lc.get("consistency_weight", 0.0))                 # clean↔degraded (opt-in)
    gram_w = float(lc.get("gram_weight", 0.0))                       # Gram topology term (domain invariance)
    cons_mode = str(lc.get("consistency_teacher", "feature"))        # feature | ema | siamese
    cons_warmup = int(lc.get("consistency_warmup", warmup))          # γ ramps 0→cons_w over this
    conf_weight = bool(lc.get("confidence_weighting", False))        # weight by teacher depth_conf
    prof_w = float(lc.get("profile_weight", 0.0))                    # degradation-profile regression
    fe_w = float(lc.get("frontend_weight", 0.0))                     # pixel L1 for the curve front-end
    enh_w = float(lc.get("enh_weight", 0.0))                         # ELVIS enhancement-decoder recon L1
    est_w = float(lc.get("restoration_est_weight", 0.0))             # DN-01 R1: Wiener (blur, noise) estimate L1
    ad_w = float(lc.get("adapter_l1_weight", 0.0))                   # DN-01 PA2: L1(input adapter's x', clean)
    # DN-01 FR matched control: train the front-end ALONE on the restoration loss (no depth forward or loss)
    fe_only = bool(t.get("frontend_only", False))
    if fe_only and (getattr(core, "frontend", None) is None or enh_w <= 0
                    or float(lc.get("joint_temporal_weight", 0.0)) > 0):
        raise ValueError("training.frontend_only needs model.frontend, enh_weight > 0 and no L3")
    fe_crop = int(t.get("frontend_crop", 0))                         # DN-01 4090 proxy screen (crop_batch)
    if fe_crop and not fe_only:
        raise ValueError("training.frontend_crop needs training.frontend_only")
    check_lowlight_target(cfg)
    check_static_regime(cfg)
    check_seeding(cfg)
    check_sky_loss(cfg)
    check_scale_anchor(cfg)
    check_metric_supervision(cfg)
    check_sparse_gt(cfg)         # protocol v3 WP13 (absent = nothing checked)
    check_metricfix(cfg)
    crop_gen = torch.Generator().manual_seed(int(cfg.get("data", {}).get("seed", 42))) if fe_crop else None
    if ad_w > 0 and getattr(core, "adapter", None) is None:
        raise ValueError("loss.adapter_l1_weight needs model.input_adapter")
    if float(lc.get("joint_temporal_weight", 0.0)) > 0:              # DN-01 Phase C joint temporal L3
        jt_mode = str(lc.get("joint_temporal_mode", "lse"))
        if enh_w <= 0 or jt_mode not in JOINT_TEMPORAL_MODES:
            raise ValueError("loss.joint_temporal_weight needs enh_weight > 0 (the restored frames) and "
                             f"joint_temporal_mode in {JOINT_TEMPORAL_MODES}, got {jt_mode!r}")
        if jt_mode in ("lse", "task_lse") and float(lc.get("joint_temporal_tau", 0.0)) <= 0:
            raise ValueError("loss.joint_temporal_tau must be > 0 for lse/task_lse")
        jt_space = str(lc.get("joint_temporal_depth_space", "disparity_affine"))
        if jt_space not in JOINT_TEMPORAL_DEPTH_SPACES or \
                (jt_space.startswith("metric") and not cfg.get("model", {}).get("metric", False)):
            raise ValueError(f"loss.joint_temporal_depth_space={jt_space!r}: must be one of "
                             f"{JOINT_TEMPORAL_DEPTH_SPACES}, and metric_* needs model.metric: true")
    jt_l3 = None
    rg_w = float(lc.get("rg_weight", 0.0))                           # DN-01 WS2 arm A: L_rg (rg_loss)
    rg_kind, rg_sigma = str(lc.get("rg_guide", "restored")), float(lc.get("rg_sigma", 0.0))
    rg_unl = cfg.get("data", {}).get("rg_unlabeled")
    if rg_w > 0:
        if rg_kind not in RG_GUIDES or (rg_kind != "none" and rg_sigma <= 0):
            raise ValueError(f"loss.rg_guide must be one of {RG_GUIDES} and loss.rg_sigma > 0 unless 'none'")
        if not cfg.get("model", {}).get("metric", False) or getattr(core, "predicts_disparity", False):
            raise ValueError("loss.rg_weight is defined on metric depth (model.metric: true)")
        if rg_kind == "restored" and (enh_w <= 0 or not getattr(core, "returns_recon", False)):
            raise ValueError("loss.rg_guide 'restored' needs a model that returns the restored clip and enh_weight > 0")
    elif rg_unl:
        raise ValueError("data.rg_unlabeled needs loss.rg_weight > 0")
    rg_log = {}
    causal_training = str(t.get("causal_training", "masked"))
    if causal_training not in {"masked", "prefix_newest"}:
        raise ValueError("training.causal_training must be 'masked' or 'prefix_newest'")
    if causal_training == "prefix_newest" and any(w > 0 for w in (cons_w, prof_w, fe_w, enh_w)):
        raise ValueError("prefix_newest requires all auxiliary losses to be disabled")
    sev_warmup = int(t.get("severity_warmup_steps", 0))              # curriculum: s_max 0.3→cap (0 = off)
    sev_cap = float(t.get("severity_max_cap", 1.0))                  # extend >1.0 for heavy-tail (BBC extreme-lowlight) robustness
    sev_hold = int(t.get("severity_hold_steps", 0))                  # DN-01: s_max=0 (clean) for the first N steps
    sev_start = float(t.get("severity_warmup_start", 0.3))           # DN-01: ramp start (0.3 = legacy)
    _set_dataset_attr(train_loader.dataset, "clean_prob", float(cfg.get("data", {}).get("clean_prob", 0.0)))
    _set_dataset_attr(train_loader.dataset, "seeded_noise", bool(cfg.get("data", {}).get("seeded_noise", False)))
    _set_dataset_attr(train_loader.dataset, "blur_only_prob", float(cfg.get("data", {}).get("blur_only_prob", 0.0)))
    _set_dataset_attr(train_loader.dataset, "blur_disk_prob", float(cfg.get("data", {}).get("blur_disk_prob", 0.0)))
    # WS2 arm S: static clips are TRAIN-only (val keeps the class default 0.0)
    _set_dataset_attr(train_loader.dataset, "static_prob", float(cfg.get("data", {}).get("static_prob", 0.0)))
    _set_dataset_attr(train_loader.dataset, "static_exempt_shares",
                      bool(cfg.get("data", {}).get("static_exempt_shares", False)))
    _set_dataset_attr(train_loader.dataset, "static_lowlight_only",
                      bool(cfg.get("data", {}).get("static_lowlight_only", False)))
    _set_dataset_attr(train_loader.dataset, "static_noise_mult",
                      tuple(float(v) for v in cfg.get("data", {}).get("static_noise_mult", (0.5, 1.5))))
    _set_dataset_attr(train_loader.dataset, "static_ev_jitter",
                      float(cfg.get("data", {}).get("static_ev_jitter", 0.15)))
    _set_dataset_attr(train_loader.dataset, "lowlight", cfg.get("data", {}).get("lowlight"))
    _set_dataset_attr(val_loader.dataset, "lowlight", cfg.get("data", {}).get("lowlight"))   # val shares degrade_mode
    ema_decoder = None
    if cons_w > 0 and cons_mode == "ema" and not unfrozen:
        ema_decoder = copy.deepcopy(ema_target).to(device).eval()    # EMA-weight teacher decoder
        for p in ema_decoder.parameters():
            p.requires_grad_(False)
    if (cons_w > 0 or prof_w > 0 or fe_w > 0) and ddp and is_main:
        print("[warn] consistency/profile/frontend losses run through the unwrapped module "
              "(single-GPU screen assumption).", flush=True)
    distill_w = float(lc.get("distill_weight", 0.0))
    distill_mode = str(lc.get("distill_mode", "term"))   # term | pseudo_gt (KD2)
    feat_w = float(lc.get("feature_distill_weight", 0.0))   # KD2 feature distillation (needs model.feature_kd_dim)
    if feat_w > 0 and getattr(core, "kd_proj", None) is None:
        raise ValueError("loss.feature_distill_weight > 0 needs model.feature_kd_dim > 0 (student-side projectors)")
    if feat_w > 0 and not cfg.get("model", {}).get("online_teacher"):
        raise ValueError("loss.feature_distill_weight > 0 needs model.online_teacher")
    if distill_mode not in ("term", "pseudo_gt"):
        raise ValueError(f"loss.distill_mode must be 'term' or 'pseudo_gt', got {distill_mode!r}")
    # loss.gt_weight (DIST-01, 2026-09-18): multiplier on the ground-truth loss. 0.0 = the
    # "distillation-only" student (EXPT-12's S0 rule consequence: no GT fine-tuning, the
    # online teacher is the only adaptation signal). Validation is unaffected.
    gt_w = float(lc.get("gt_weight", 1.0))
    if gt_w < 0:
        raise ValueError("loss.gt_weight must be >= 0")
    if gt_w == 0.0 and distill_w <= 0:
        raise ValueError("loss.gt_weight=0 needs a distillation signal (loss.distill_weight > 0)")
    online_teacher = build_online_teacher(cfg, device)                # DIST-01 (opt-in), frozen
    anchor_teacher = build_scale_anchor_teacher(cfg, device)          # SKYSCALE F3 (opt-in, default None), frozen
    anchor_w = float(lc.get("scale_anchor_weight", 0.0))
    if online_teacher is not None and is_main:
        t_model, t_input, t_disp = online_teacher
        print(f"[teacher] online teacher {type(t_model).__name__} on {t_input} frames, "
              f"predicts_disparity={t_disp}, distill_weight={distill_w}, "
              f"params={sum(p.numel() for p in t_model.parameters()) / 1e6:.1f}M (frozen)", flush=True)

    # lr_decay_steps overrides the cosine length (default = total_steps). Setting it >> total_steps
    # holds LR ~constant (used to expose over-training erosion fast on a small repro).
    lr_decay_steps = int(t.get("lr_decay_steps", 0)) or total_steps

    def lr_scale(step):  # in [0, 1]; multiplies each param group's initial LR
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, lr_decay_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    best = float("inf")
    gstep = 0
    micro = 0
    nf_skips = 0      # K80 option B: non-finite-loss micro-batches skipped (see the end-of-run nonfinite_skips.json)
    start_epoch = 0
    resume_from = t.get("resume_from")
    if resume_from:
        resume = torch.load(str(resume_from), map_location="cpu", weights_only=False)
        if "optimizer" not in resume or "rng" not in resume:
            raise ValueError("resume_from requires a RAMEN resumable checkpoint, not a warm-start checkpoint")
        if unfrozen:
            core.load_state_dict(resume.get("model", resume.get("model_ema")), strict=True)
            optimizer.load_state_dict(resume["optimizer"])
            ema.shadow = resume["ema"]
        else:
            new_keys = _load_resume_trainable(ema_target, optimizer, ema, resume)
            if new_keys and is_main:
                print(f"[resume] {len(new_keys)} LoRA tensors absent from the checkpoint: zero-B init, "
                      "fresh LoRA optimizer group; head optimizer state kept", flush=True)
        # optimizer.load_state_dict moves most per-param state to match each param's own
        # device -- but NOT every key. Caught 2026-08-26 on real GPU hardware, first real
        # resume smoke after the false-positive assertion fix: AdamW's per-param "step"
        # entry stayed on cpu after load_state_dict (a known PyTorch behavior -- "step" is
        # handled outside the normal per-tensor device-cast path), so the resume-time
        # assertion below correctly caught a SECOND, genuine device mismatch, not another
        # false positive. Sweep every optimizer.state tensor explicitly rather than trust
        # load_state_dict to have moved all of them.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        # resume was loaded with map_location="cpu" -- shadow tensors must move to the
        # training device explicitly; EMA.update()'s in-place add_ has no such remapping
        # of its own (unlike optimizer.load_state_dict, which does move state to match
        # each param's own device). Caught 2026-08-26: every one of EXPT-02's 5 resumed
        # arms crashed on the first post-resume EMA update with a cuda/cpu device
        # mismatch -- the CPU-only resume-exactness test never exercises this, since
        # there "device" is cpu everywhere and the bug can't trigger.
        ema.shadow = {k: v.to(device) for k, v in ema.shadow.items()}
        # Fable's follow-up (2026-08-26): fail loudly at resume time, not mid-run, if any
        # restored tensor ended up off `device` -- cheaper than a crash after minutes of
        # training, and a clearer message than EMA.update()'s bare device-mismatch error.
        # BUG (caught 2026-08-26 on real GPU hardware, all 6 OVDA-arch resumes): torch.device
        # ("cuda") != torch.device("cuda:0") even on a single-GPU machine -- a tensor moved via
        # .to("cuda") reports .device as "cuda:0", so comparing against the bare torch.device
        # (device) false-positived on every single tensor. Canonicalize via an actual allocation
        # on `device` instead of parsing the string, so both sides use the same concrete form.
        resume_device = torch.empty(0, device=device).device
        off_device = [("ema.shadow", k) for k, v in ema.shadow.items() if v.device != resume_device]
        for state in optimizer.state.values():
            off_device += [("optimizer.state", k) for k, v in state.items()
                           if torch.is_tensor(v) and v.device != resume_device]
        # Opus's audit of the optimizer.state fix (2026-08-26): three device bugs in one day
        # in this same resume path -- rather than wait for a fourth, also walk the live
        # model's own parameters/buffers directly, catching any future off-device component
        # automatically at resume time instead of after a wasted queue cycle. load_state_dict
        # copies in-place and so preserves each tensor's existing device, but this is a
        # backstop against that assumption breaking for some future submodule, not a
        # duplicate of the checks above (ema.shadow/optimizer.state are never part of
        # core's own parameters/buffers).
        off_device += [("core.parameters", n) for n, p in core.named_parameters() if p.device != resume_device]
        off_device += [("core.buffers", n) for n, b in core.named_buffers() if b.device != resume_device]
        if off_device:
            raise RuntimeError(f"resume left tensors off {device} after loading: {off_device[:5]}")
        gstep = int(resume["step"])
        micro = int(resume.get("micro", gstep * accum))
        start_epoch = int(resume.get("epoch", -1)) + 1
        best = float(resume.get("best", best))
        # rng is CPU state by construction (torch.cuda.set_rng_state_all needs CPU
        # ByteTensors) and must NOT be swept to device like everything else here --
        # do not "complete" this block by adding it to a device sweep.
        _restore_rng_state(resume["rng"])
        if is_main:
            print(f"[resume] {resume_from} epoch={start_epoch} step={gstep} best={best:.4f}", flush=True)
    _start_state(train_loader.dataset, t, gstep, rank, bool(resume_from))

    def _monitor_tick(step: int) -> None:
        """METRICFIX: run the outdoor monitor on EVERY rank (identical work, no collective) and let rank 0 append the row; the EMA weights are scored when an EMA is on (all METRICFIX arms: ema_decay 0 = the live weights)."""
        backup = None
        if ema.decay > 0:
            backup = copy.deepcopy(ema_target.state_dict())
            ema.copy_to(ema_target)
        try:
            row = monitor.run(core, device, amp_dtype, step)
        finally:
            if backup is not None:
                ema_target.load_state_dict(backup)
        if is_main:
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[monitor] step{step} outdoor median(pred)/median(GT) = {row.get('outdoor_scale_ratio', float('nan')):.3f} (pred {row.get('outdoor_pred_median', float('nan')):.2f} m, GT {row.get('outdoor_gt_median', float('nan')):.2f} m)", flush=True)

    if monitor is not None and not resume_from:
        _monitor_tick(gstep)                       # step 0 (or the stage's first step): the init's own outdoor scale
    if 0 in checkpoint_at_steps and not resume_from:
        # Pretrained/initial-state snapshot before any optimizer step -- the
        # "step 0" baseline the checkpoint schedule needs to read forgetting and
        # early adaptation against. Primary val metrics only (skips the optional
        # panel/clean-val diagnostics the periodic loop below also runs).
        # EVERY rank validates (evaluate() all-reduces, exactly like the periodic val below);
        # only main writes. Guarding the whole block on is_main deadlocked the first DDP run.
        metrics0 = _validate_with_ema(core, ema_target, ema, val_loader, device, amp_dtype, val_max_batches)
        if is_main:
            metrics0.update({"epoch": start_epoch, "step": 0, "train_loss": float("nan")})
            with open(log_path, "a") as f:
                f.write(json.dumps(metrics0) + "\n")
            _save(ckpt_dir / "step0.pt", core, ema_target, ema, optimizer, cfg,
                  start_epoch, 0, 0, best, metrics0, unfrozen)
            print(f"[val] step0 (pretrained baseline) AbsRel={metrics0['abs_rel']:.4f}", flush=True)
    unl_iter = None
    if rg_unl:
        if resume_from:                  # the unlabeled stream's position is not checkpointed
            raise ValueError("data.rg_unlabeled runs are not resumable (the unlabeled stream restarts)")
        unl_iter = rg_unlabeled_batches(cfg, world_size, rank)
    t0 = time.time()
    throughput_log_path = ckpt_dir / "throughput.jsonl"
    if is_main and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    stop = False

    last_metrics = {}
    last_epoch = start_epoch
    for epoch in range(start_epoch, max_epochs):
        last_epoch = epoch
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        core.train()  # backbone stays eval() via its train() override
        run_loss = 0.0
        ep_micro = 0  # per-epoch window for run_loss; `micro` is cumulative (drives grad-accum)
        for batch in train_loader:
            if fe_crop:
                batch = crop_batch(batch, fe_crop, crop_gen)
            if unl_iter is not None and micro % accum == 0:
                # WS2 arm A: ONE unlabeled real clip per optimizer step (per rank), L_rg ONLY, accumulated locally
                # (no_sync) and all-reduced with the group's last labelled micro-batch. Never counted in `micro`.
                urgb = next(unl_iter)["frames"].to(device, non_blocking=True)
                with (model.no_sync() if ddp else nullcontext()):
                    rg_log["Lrg_u"] = rg_unlabeled_step(model, urgb, rg_kind, rg_w, rg_sigma, accum, amp_dtype)
            rgb = batch.get("degraded_frames", batch.get("frames")).to(device, non_blocking=True)
            if "static" in batch:                 # WS2 arm S: the static-clip share of this labelled batch (log only)
                rg_log["static"] = float(batch["static"].float().mean())
            sync = (micro + 1) % accum == 0
            ctx = model.no_sync() if (ddp and not sync) else nullcontext()
            with ctx:
                with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None):
                    have_cons = cons_w > 0 and "degraded_frames" in batch
                    prof_pred = None
                    recon = None
                    if hasattr(core, "set_true_profile"):   # EXPT-08 oracle-routing arm (opt-in)
                        true_profile = batch.get("degrade_params")
                        if true_profile is not None:
                            true_profile = true_profile.to(device, non_blocking=True)
                        core.set_true_profile(true_profile)
                    if hasattr(core, "set_step"):   # EXPT-09 weighting's lambda warmup (opt-in)
                        core.set_step(gstep)
                    if have_cons:
                        # encode/decode split: the frozen clean tokens are cached and reused
                        # by the teacher (single-GPU screen path; bypasses the DDP wrapper).
                        _B, _C, _T, H, W = rgb.shape
                        cond_deg, prof_pred = core.condition(rgb)   # (None, None) without estimator
                        feats_deg = core.encode(rgb, cond=cond_deg)
                        pred = core.decode(feats_deg, H, W, cond=cond_deg)
                    else:
                        if fe_only:
                            pred, recon = None, model(rgb, frontend_only=True)
                        elif enh_w > 0 and (getattr(core, "enh_decoder", False) or getattr(core, "returns_recon", False)
                                            or getattr(core, "frontend_branch", False)):   # FR2 / ST: recon = y
                            pred, recon = model(rgb, return_recon=True)   # ELVIS enhancement branch
                        else:
                            pred = None if causal_training == "prefix_newest" else model(rgb)
                        if getattr(core, "profile", None) is not None and not ddp and "degrade_params" in batch:
                            _, prof_pred = core.condition(rgb)
                    t_out = None
                    if online_teacher is not None:
                        # DIST-01/KD2: the frozen teacher sees the same batch (view chosen by
                        # online_teacher.input). pseudo_gt mode turns its output into the target
                        # of unlabeled clips BEFORE the loss; term mode adds a separate term after.
                        t_model, t_input, t_disp = online_teacher
                        t_src = online_teacher_input(t_input, rgb, batch, device)
                        with torch.no_grad():
                            t_out = t_model(t_src)
                        if distill_mode == "pseudo_gt":
                            batch = inject_teacher_pseudo_gt(batch, t_out, t_disp)
                    loss = (compute_prefix_newest_loss(model, rgb, batch, cfg)
                            if causal_training == "prefix_newest" else
                            recon.new_zeros(()) if fe_only else
                            compute_loss(pred, batch, cfg,
                                        pred_is_disparity=getattr(core, "predicts_disparity", False)))
                    if gt_w != 1.0:
                        loss = gt_w * loss                          # keeps grad_fn at gt_w == 0
                    if feat_w > 0 and t_out is not None and getattr(core, "_kd_tokens", None) is not None:
                        loss = loss + feat_w * feature_distillation_loss(
                            core._kd_tokens, t_model.encoder_tokens(t_src), batch,
                            str(lc.get("feature_distill_mask", lc.get("distill_mask", "all"))))
                    if anchor_teacher is not None and pred is not None:      # SKYSCALE F3: log-ratio scale anchor to the frozen reference, non-metric clips only
                        a_model, a_input, a_disp = anchor_teacher
                        with torch.no_grad():
                            a_out = a_model(online_teacher_input(a_input, rgb, batch, device))
                        loss = loss + anchor_w * scale_anchor_term(pred, a_out, a_disp, batch, cfg,
                                                                   pred_is_disparity=getattr(core, "predicts_disparity", False))
                    if t_out is not None and pred is not None and distill_mode == "term":
                        loss = loss + distill_w * online_distillation_term(
                            pred, t_out, t_disp, batch, cfg,
                            pred_is_disparity=getattr(core, "predicts_disparity", False))
                    if recon is not None:   # (B) reconstruct the CLEAN frame from degraded input
                        rest, jt_l3 = restoration_loss(pred, recon, batch, cfg, gstep, total_steps,
                                                       getattr(core, "predicts_disparity", False))
                        loss = loss + rest
                        # R1: the restoration head's Wiener estimate; FR+W: the pre-encoder PixelWiener's
                        est_mod = getattr(core, "post_wiener", None) or getattr(core, "restore", None)
                        est = getattr(core, "last_wiener_est", None) if getattr(core, "post_wiener", None) \
                            else getattr(est_mod, "last_est", None)
                        if est_w > 0 and est is not None and "degrade_params" in batch:
                            # R1: supervise the Wiener step's per-clip (blur, noise) estimate with the true
                            # synthesis profile; paired clips (degrade_params all -1) carry no truth
                            gp = batch["degrade_params"].to(device, non_blocking=True).float()
                            keep = gp[:, 0] >= 0
                            if keep.any():
                                tgt = est_mod.wiener.targets(gp)                         # (B,2)
                                est = est.float().view(gp.shape[0], -1, 2)[keep]           # (b,T,2)
                                loss = loss + est_w * F.l1_loss(est, tgt[keep, None].expand_as(est))
                    if rg_w > 0 and pred is not None:     # WS2 arm A: L_rg on every labelled clip too
                        lrg = rg_loss(pred, rg_guide(rg_kind, rgb, recon), rg_sigma, batch.get("static"))
                        loss = loss + rg_w * lrg
                        rg_log["Lrg"] = float(lrg.detach())
                    if ad_w > 0:
                        # PA2: direct denoising pressure on the pre-encoder adapter against the GT clean frames
                        adapted = core.last_adapted.permute(0, 2, 1, 3, 4)            # (B,3,T,H,W)
                        loss = loss + ad_w * (adapted - batch["frames"].to(device, non_blocking=True)).abs().mean()
                    if prof_w > 0 and prof_pred is not None and "degrade_params" in batch:
                        gp = batch["degrade_params"].to(device, non_blocking=True).float()
                        # Per-clip truth (B,5); some estimators (EXPT-08's causal one) predict
                        # per-frame (B,T,5) instead of D3Net's per-clip (B,5) -- broadcast the
                        # single truth across T explicitly rather than relying on F.l1_loss's
                        # own implicit broadcasting, which warns (and is easy to get wrong by
                        # accident) whenever shapes don't match exactly even when the intended
                        # broadcast is this one.
                        pp = prof_pred.float()
                        if pp.dim() == gp.dim() + 1:
                            gp = gp.unsqueeze(1).expand_as(pp)
                        loss = loss + prof_w * F.l1_loss(pp, gp)
                    if fe_w > 0 and core.frontend is not None and not ddp and "degraded_frames" in batch:
                        enhanced = core.frontend(rgb)               # pixel-supervised: no backbone grads
                        clean_px = batch["frames"].to(device, non_blocking=True)
                        loss = loss + fe_w * (enhanced - clean_px).abs().mean()
                    if have_cons:
                        gamma = cons_w * min(1.0, gstep / max(cons_warmup, 1))  # ramp γ from 0
                        if gamma > 0:
                            clean = batch["frames"].to(device, non_blocking=True)
                            with torch.no_grad():
                                cond_cln, _ = core.condition(clean)
                                feats_cln = core.encode(clean, cond=cond_cln)
                            if cons_mode == "feature":
                                sd = core.apply_temporal(feats_deg, H, W, cond_deg)
                                with torch.no_grad():
                                    tc = core.apply_temporal(feats_cln, H, W, cond_cln)
                                loss = loss + gamma * feature_consistency_loss(sd, tc, gram_w)
                            elif cons_mode == "ema" and ema_decoder is not None:
                                ema_decoder.load_state_dict(ema.shadow)  # refresh teacher → EMA weights
                                tmix = ema_decoder["temporal"] if "temporal" in ema_decoder else None
                                tchan = ema_decoder["channel"] if "channel" in ema_decoder else None
                                with torch.no_grad():
                                    pred_t, extra_t = core.decode_modules(
                                        tmix, tchan, ema_decoder["head"], feats_cln, H, W,
                                        return_extra=True, cond=cond_cln)
                                conf = extra_t["head_out"].get("depth_conf") if conf_weight else None
                                loss = loss + gamma * output_consistency_loss(pred, pred_t, batch, conf)
                            else:  # siamese: current weights on cached clean tokens
                                with torch.no_grad():
                                    pred_clean = core.decode(feats_cln, H, W, cond=cond_cln)
                                loss = loss + gamma * consistency_loss(pred, pred_clean, batch)
                    if hasattr(core, "anchor_penalty"):   # temporal-weight anchoring (opt-in via anchor_lambda)
                        loss = loss + core.anchor_penalty()
                    loss = loss / accum
                if torch.isfinite(loss):
                    loss.backward()
                else:  # one bad batch must not poison an unattended run
                    print(f"[warn] non-finite loss at step {gstep}, skipping micro-batch", flush=True)
                    nf_skips += 1
                    loss = loss.detach().new_zeros(())
            run_loss += loss.item() * accum
            micro += 1
            ep_micro += 1

            if micro % accum == 0:
                sc = lr_scale(gstep)
                for g in optimizer.param_groups:
                    g["lr"] = g["initial_lr"] * sc
                if clip > 0:
                    clip_gradients(core, clip, bool(t.get("grad_clip_per_group", False)), split_branch)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(ema_target)
                gstep += 1
                if sev_warmup > 0:  # severity curriculum: hold clean, then ramp start -> sev_cap
                    _set_severity_max(train_loader.dataset,
                                      severity_schedule(gstep, sev_hold, sev_start, sev_cap, sev_warmup))

                if log_terms and gstep % terms_every == 0:
                    terms = term_flush()
                    if is_main and terms:
                        with open(log_path, "a") as f:
                            f.write(json.dumps({"event": "loss_terms", "step": gstep, **terms}) + "\n")
                if monitor is not None and (gstep % monitor.every == 0 or gstep in checkpoint_at_steps):      # every `every_steps` AND at every permanent snapshot (so 250 / 1500 ... are on the curve)
                    _monitor_tick(gstep)
                if is_main and gstep % 20 == 0:
                    sps = micro / (time.time() - t0)
                    jt = ""
                    if jt_l3 is not None:
                        lam, tau = joint_temporal_schedule(gstep, lc, total_steps)
                        jt = f" L3={float(jt_l3):.4f} lam={lam:.3g} tau={tau:.3g}"
                    jt += "".join(f" {k}={v:.4f}" for k, v in rg_log.items())
                    if sev_warmup > 0:   # DN-01: log the degradation cap (0 = clean clips)
                        jt += f" sev={severity_schedule(gstep, sev_hold, sev_start, sev_cap, sev_warmup):.3g}"
                    print(f"e{epoch} step{gstep}/{total_steps} loss={run_loss/ep_micro:.4f} "
                          f"lr={optimizer.param_groups[0]['lr']:.2e} {sps:.1f}it/s{jt}", flush=True)
                    if device.startswith("cuda") and torch.cuda.is_available():
                        # Fable's ask, VAL-01: the real run IS the throughput measurement --
                        # log per-optimizer-step wall time + peak memory into the run dir
                        # rather than a separate synthetic-tensor job.
                        with open(throughput_log_path, "a") as f:
                            f.write(json.dumps({
                                "gstep": gstep,
                                "wall_seconds_since_start": time.time() - t0,
                                "optimizer_steps_per_second": gstep / (time.time() - t0),
                                "peak_mem_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
                                "peak_mem_reserved_gb": torch.cuda.max_memory_reserved() / (1024 ** 3),
                            }) + "\n")

                do_val, do_checkpoint = _checkpoint_gate(gstep, val_every, checkpoint_every, checkpoint_at_steps)
                if is_main and do_checkpoint and not do_val:
                    # Cheap resumability point: weights/optimizer/EMA/RNG only, no eval pass --
                    # last_metrics is stale (from the last real val) by design.
                    _save(ckpt_dir / "last.pt", core, ema_target, ema, optimizer, cfg,
                          epoch, gstep, micro, best, last_metrics, unfrozen)
                if do_val:
                    metrics = _validate_with_ema(core, ema_target, ema, val_loader, device, amp_dtype, val_max_batches)
                    if clean_val_loader is not None:
                        cm = _validate_with_ema(core, ema_target, ema, clean_val_loader, device, amp_dtype, val_max_batches)
                        metrics.update({"clean_abs_rel": cm["abs_rel"], "clean_abs_rel_perframe": cm["abs_rel_perframe"],
                                        "clean_scale_cv": cm["scale_cv"], "clean_delta1": cm["delta1"]})
                    # ---- Multi-combo val_panel (Part C harness) ----
                    panel_result = None
                    if is_main and cfg.get("val_panel"):
                        from scripts.val_panel import run_panel
                        # Swap in EMA weights for the panel run (mirrors _validate_with_ema).
                        backup = copy.deepcopy(ema_target.state_dict())
                        ema.copy_to(ema_target)
                        try:
                            panel_result = run_panel(
                                core, cfg, gstep, device, amp_dtype, ckpt_dir,
                                viz=True,
                            )
                        finally:
                            ema_target.load_state_dict(backup)
                    if is_main:
                        metrics.update({"epoch": epoch, "step": gstep, "train_loss": run_loss / max(ep_micro, 1)})
                        last_metrics = dict(metrics)
                        # Keyed metrics.jsonl: embed the panel combos alongside the default metrics.
                        log_row = dict(metrics)
                        if panel_result is not None:
                            log_row["combos"] = panel_result["combos"]
                            log_row["panel_mean"] = panel_result["mean"]
                        print(f"[val] step{gstep} AbsRel={metrics['abs_rel']:.4f} perframe={metrics['abs_rel_perframe']:.4f} "
                              f"metric={metrics['abs_rel_metric']:.3f} scaleCV={metrics['scale_cv']:.4f} "
                              f"d1={metrics['delta1']:.3f}" + (f" | clean={metrics['clean_abs_rel']:.4f}" if clean_val_loader is not None else "")
                              + (f" | panel_mean_abs_rel={panel_result['mean'].get('abs_rel', float('nan')):.4f}" if panel_result else ""), flush=True)
                        if metrics.get("per_source"):
                            print("[val/src] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics["per_source"].items()), flush=True)
                        with open(log_path, "a") as f:
                            f.write(json.dumps(log_row) + "\n")
                        _save(ckpt_dir / "last.pt", core, ema_target, ema, optimizer, cfg,
                              epoch, gstep, micro, best, metrics, unfrozen)
                        if gstep in checkpoint_at_steps:   # permanent snapshot; last.pt gets overwritten
                            _save(ckpt_dir / f"step{gstep}.pt", core, ema_target, ema, optimizer, cfg,
                                  epoch, gstep, micro, best, metrics, unfrozen)
                        # best.pt selection: prefer panel mean degraded abs_rel when available,
                        # else fall back to the default abs_rel.
                        selection_absrel = (
                            panel_result["mean"].get("abs_rel", metrics["abs_rel"])
                            if panel_result else metrics["abs_rel"]
                        )
                        if selection_absrel < best and not math.isnan(selection_absrel):
                            best = selection_absrel
                            _save(ckpt_dir / "best.pt", core, ema_target, ema, optimizer, cfg,
                                  epoch, gstep, micro, best, metrics, unfrozen)
                            print(f"  -> new best AbsRel={best:.4f} (saved best.pt)", flush=True)
                    core.train()

                if max_steps and gstep >= max_steps:
                    stop = True
                    break
        if stop:
            break

    if is_main:   # K80 option B: the isfinite-loss guard's skipped micro-batches (rank 0), a residual coupling between branches; must be 0 in every stage-2 run (read by read_vm_screen)
        (ckpt_dir / "nonfinite_skips.json").write_text(json.dumps({"skipped_microbatches": nf_skips, "optimizer_steps": gstep}) + "\n")
        _save(ckpt_dir / "final.pt", core, ema_target, ema, optimizer, cfg,
              last_epoch, gstep, micro, best, last_metrics, unfrozen)
        print(f"[done] best AbsRel={best:.4f} in {(time.time()-t0)/60:.1f} min", flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def _validate_with_ema(core, ema_target, ema, loader, device, amp_dtype, max_batches):
    backup = copy.deepcopy(ema_target.state_dict())
    ema.copy_to(ema_target)
    try:
        return evaluate(core, loader, device, amp_dtype, max_batches)
    finally:
        ema_target.load_state_dict(backup)


def _save(path, core, ema_target, ema, optimizer, cfg, epoch, step, micro, best, metrics, unfrozen):
    payload = {
        "cfg": cfg, "epoch": epoch, "step": step, "micro": micro,
        "best": best, "metrics": metrics, "optimizer": optimizer.state_dict(),
        "ema": ema.shadow, "rng": _rng_state(),
    }
    if unfrozen:  # backbone fine-tuned → save the whole model
        payload["model"] = core.state_dict()
        payload["model_ema"] = ema.shadow
    else:  # frozen backbone → trainable head + temporal mixers (small checkpoint)
        payload["trainable"] = ema_target.state_dict()   # live weights (head + temporal)
        payload["trainable_ema"] = ema.shadow            # EMA weights (head + temporal)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RAMEN.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    train(cfg)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


_FRAME_COUNT_INVENTORY = (
    Path(__file__).resolve().parent / "provenance" / "inventory"
    / "val01_dataset_inventory_20260827.json"
)


def _assert_declared_frame_counts_match_inventory(cfg: dict) -> None:
    """Opus's hardening request, 2026-08-27: data.sampling='proportional' must not trust a
    hand-typed frame_count -- it has to match the hashed, real dataset-inventory artifact
    (evidence over assertion, same pattern as the gate-file checks). The config keeps its own
    frame_count for readability (so a reader doesn't need to cross-reference a separate file
    to see the sampling ratio), but this cross-check makes a stale or hand-edited number a
    hard failure at config-load time rather than a silent divergence."""
    d = cfg.get("data", {})
    if d.get("sampling") != "proportional":
        return
    with _FRAME_COUNT_INVENTORY.open() as f:
        declared = json.load(f)["frame_counts"]
    for s in d.get("sources", []):
        t = s.get("type")
        fc = s.get("frame_count")
        if t not in declared:
            continue   # not one of the four inventoried types (e.g. pseudo_gt) -- nothing to check
        if fc != declared[t]:
            raise ValueError(
                f"data.sources[].frame_count for type {t!r} is {fc!r}, but "
                f"{_FRAME_COUNT_INVENTORY.name} records {declared[t]!r} -- "
                "sampling='proportional' requires the declared count to match the hashed "
                "inventory exactly (re-run the inventory if the corpus has genuinely changed)"
            )


def _assert_vda_uses_vda_loss_mode(cfg: dict) -> None:
    """Fable's ruling, 2026-08-27: compute_loss's own hand-rolled SSI/TGM diverges from
    VDA's real recipe in ways structural enough (per-frame-vs-per-clip alignment split,
    trimming, TGM scale count) that reproducing it requires reusing VDA's own
    loss/loss.py::VideoDepthLoss directly (loss.loss_mode='vda'), not matching it with our
    own weights -- superseding an earlier ssi_align='perclip' requirement, which assumed
    the hand-rolled path. sky_zero_disparity is RAMEN's own convention (supervise sky at
    disparity 0); VDA's own loss has no sky concept, so it must stay off here -- sky
    pixels are simply excluded from the mask, not force-included."""
    check_scale_anchor(cfg)
    check_sky_loss(cfg)          # the NEW opt-in sky keys are validated here; the OLD sky_zero_disparity ban below is unchanged
    check_metric_supervision(cfg)
    check_sparse_gt(cfg)         # protocol v3 WP13 (absent = nothing checked)
    check_metricfix(cfg)         # METRICFIX opt-in keys (absent = nothing checked)
    if cfg.get("model", {}).get("arch") not in VDA_LOSS_ARCHES:
        return
    lc = cfg.get("loss", {})
    # DN-01 Phase D carve-out: a METRIC VDA may use the custom path (e.g. metric_weight L1),
    # because VideoDepthLoss alone cannot pin absolute scale. Relative VDA stays hard-locked.
    if str(lc.get("loss_mode")) != "vda" and not bool(cfg["model"].get("metric", False)):
        raise ValueError(
            f"model.arch='vda' requires loss.loss_mode='vda' (reuse VDA's own "
            f"loss/loss.py::VideoDepthLoss), got {lc.get('loss_mode')!r} -- this file's own "
            "hand-rolled SSI/TGM diverges from VDA's real recipe in structural ways a "
            "weight override cannot fix"
        )
    if bool(lc.get("sky_zero_disparity", False)):
        raise ValueError(
            "model.arch='vda' requires loss.sky_zero_disparity=false -- VDA's own loss has "
            "no sky concept; RAMEN's disparity-0 sky convention does not apply to it"
        )


def load_config(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """Load YAML with explicit, recursive ``extends`` inheritance."""
    path = path.resolve()
    if path in seen:
        raise ValueError(f"cyclic config inheritance: {' -> '.join(map(str, seen + (path,)))}")
    with path.open() as handle:
        cfg = yaml.safe_load(handle) or {}
    parent = cfg.pop("extends", None)
    if parent is None:
        resolved = cfg
    else:
        parent_path = (path.parent / parent).resolve()
        resolved = _deep_merge(load_config(parent_path, seen + (path,)), cfg)
    if not seen:   # only the original top-level call sees the fully-merged config
        _assert_no_video_only_val_source(resolved)
        _assert_declared_frame_counts_match_inventory(resolved)
        _assert_vda_uses_vda_loss_mode(resolved)
        _warn_if_checkpoint_every_without_checkpoint_at_steps(resolved)
    return resolved


def _assert_no_video_only_val_source(cfg: dict) -> None:
    """DIST-01 arm (c), 2026-09-18: ``type: video_only`` has NO ground truth -- its depth and
    mask are all zero by construction. Validating on it would report metrics computed over an
    empty mask (AbsRel over zero pixels), silently polluting every val curve and the
    best-checkpoint rule with a meaningless number. This is a hard failure at config-load
    time, not a warning: there is no legitimate configuration in which it belongs in
    ``data.val_sources``."""
    specs = (cfg.get("data") or {}).get("val_sources")
    if not specs:
        return
    bad = [i for i, s in enumerate(specs) if isinstance(s, dict) and s.get("type") in _NO_GT_SOURCE_TYPES]
    if bad:
        raise ValueError(
            f"data.val_sources[{bad}] has type 'video_only', which carries no ground-truth "
            "depth (all-zero depth/mask) and therefore cannot be validated on. Use video_only "
            "in data.sources only; keep val_sources on the GT sources."
        )


def _warn_if_checkpoint_every_without_checkpoint_at_steps(cfg: dict) -> None:
    """A warning, not a hard failure (Opus/Fable's audit, 2026-08-27, job 6155634): plenty
    of legitimate configs only want cheap last.pt resumability and never need a permanent
    snapshot curve, so this must not break them. But forgetting checkpoint_at_steps when
    you DID want intermediate snapshots fails silently otherwise -- checkpoint_every_steps
    only ever governs last.pt (overwritten every cadence), never step{N}.pt; that requires
    checkpoint_at_steps explicitly. val01_vda_s_relative.yaml hit exactly this: nothing
    failed, nothing warned, and the entire run's intermediate curve was unrecoverable once
    noticed mid-run."""
    t = cfg.get("training", {})
    if t.get("checkpoint_every_steps") and not t.get("checkpoint_at_steps"):
        print(
            "[load_config] WARNING: training.checkpoint_every_steps is set but "
            "training.checkpoint_at_steps is not -- only last.pt (overwritten every "
            "checkpoint_every_steps) will ever be written; no permanent step{N}.pt "
            "snapshots will exist for this run. If you want an intermediate checkpoint "
            "curve (e.g. for scripts/val01_eval_cycle.py), set checkpoint_at_steps "
            "explicitly. If you only need cheap resumability, this warning is expected "
            "and safe to ignore.",
            flush=True,
        )


if __name__ == "__main__":
    main()

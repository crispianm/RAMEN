#!/usr/bin/env python3
"""
val_panel — Multi-combo validation harness with visual outputs
==============================================================

Called from ``train.py``'s validation path.  Loops over a configured panel
of ``(env/scene × degrade_mode × severity)`` combos, running existing
``train.evaluate`` per combo, adding temporal metrics (tOF + OPW), and
optionally dumping image grids + MP4s for a fixed deterministic clip set.

Maximises reuse; does **not** reimplement evaluate, build_dataloader,
colorize, or the temporal-metrics functions.

# ponytail: fixed clip set + percentile norm so frames are comparable
# across steps; reuse the two existing colorizers, don't write a third.
"""

from __future__ import annotations

import copy
import math
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Allow import from project root when running as a script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train import build_dataloader, evaluate  # noqa: E402
from metrics_temporal import clip_temporal_metrics  # noqa: E402


# --------------------------------------------------------------------------- #
# Colorization / visualisation helpers (reuse existing patterns).
# --------------------------------------------------------------------------- #

def _colorize_depth(stack: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """``(T,H,W)`` depth → ``(T,H,W,3)`` BGR via inverse-depth + INFERNO.

    ONE global robust range across the whole clip prevents flicker.
    Reuses the ``scripts/demo_video.py::colorize`` pattern verbatim.
    """
    inv = 1.0 / np.clip(stack, 1e-3, None)
    v = np.isfinite(inv) if valid is None else (valid > 0.5)
    if v.sum() == 0:
        return np.zeros((*stack.shape, 3), dtype=np.uint8)
    lo, hi = float(np.percentile(inv[v], 2)), float(np.percentile(inv[v], 98))
    u8 = np.clip((inv - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    out = np.stack([cv2.applyColorMap(u8[t], cv2.COLORMAP_INFERNO) for t in range(u8.shape[0])])
    if valid is not None:
        out[valid <= 0.5] = 0
    return out


def _rgb_to_bgr_u8(rgb: np.ndarray) -> np.ndarray:
    """``(T,3,H,W)`` float [0,1] RGB → ``(T,H,W,3)`` uint8 BGR for cv2."""
    T, C, H, W = rgb.shape
    out = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    out = out.transpose(0, 2, 3, 1)  # (T,H,W,3) RGB
    return out[..., ::-1].copy()  # RGB → BGR


def _save_grid(
    rgb_bgr: np.ndarray, gt_depth: np.ndarray, pred_depth: np.ndarray,
    valid: np.ndarray | None, path: Path,
) -> None:
    """Save a single image grid for the middle frame: ``input | GT | pred``.

    Reuses the ``scripts/make_qual_grid.py::disp_img`` percentile-norm pattern.
    """
    t = rgb_bgr.shape[0] // 2  # middle frame
    inp = rgb_bgr[t]  # (H,W,3) BGR
    gt_c = _colorize_depth(gt_depth[t:t + 1], valid[t:t + 1] if valid is not None else None)[0]
    pr_c = _colorize_depth(pred_depth[t:t + 1], valid[t:t + 1] if valid is not None else None)[0]
    grid = np.hstack([inp, gt_c, pr_c])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)


def _save_video(
    rgb_bgr: np.ndarray, gt_depth: np.ndarray, pred_depth: np.ndarray,
    valid: np.ndarray | None, path: Path, fps: int = 8,
) -> None:
    """Save an MP4: ``degraded RGB | pred depth | GT depth``, per-frame."""
    gt_c = _colorize_depth(gt_depth, valid)
    pr_c = _colorize_depth(pred_depth, valid)
    T, H, W, _ = rgb_bgr.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (3 * W, H))
    for t in range(T):
        frame = np.hstack([rgb_bgr[t], pr_c[t], gt_c[t]])
        vw.write(frame)
    vw.release()


# --------------------------------------------------------------------------- #
# Combo builder.
# --------------------------------------------------------------------------- #

def _build_combos(cfg: dict) -> List[Tuple[str, dict]]:
    """Build ``(combo_key, combo_cfg)`` pairs from the ``val_panel`` config.

    Each combo_cfg is a full training-style config dict with data overrides
    for one evaluation slice (env × degradation).
    """
    panel = cfg.get("val_panel", {})
    envs = panel.get("envs", ["NordicHarbor", "Hospital"])
    vkitti_scene = panel.get("vkitti_scene", "Scene18")
    degradations = panel.get("degradations", [
        {"mode": "clean"},
        {"mode": "elvis", "severity": 0.5},
        {"mode": "elvis", "severity": 0.9},
    ])

    combos: list[tuple[str, dict]] = []
    for env in envs:
        for deg in degradations:
            mode = deg.get("mode", "clean")
            sev = deg.get("severity")
            key = f"{env}/{mode}" + (f"@{sev}" if sev is not None else "")
            c = copy.deepcopy(cfg)
            d = c.setdefault("data", {})
            # TartanAir env: filter to this single environment.
            d["envs"] = [env]
            d.pop("vkitti_root", None)  # exclude VKITTI for TartanAir combos
            if mode == "clean":
                d["degrade"] = False
            else:
                d["degrade"] = True
                d["degrade_mode"] = mode
                if sev is not None:
                    d["val_degrade_severity"] = sev
                    d["degrade_severity"] = sev
            combos.append((key, c))

    # VKITTI combos (Scene18 only, using the vkitti_root from the original cfg).
    vkitti_root = cfg.get("data", {}).get("vkitti_root")
    if vkitti_root and vkitti_scene:
        for deg in degradations:
            mode = deg.get("mode", "clean")
            sev = deg.get("severity")
            key = f"{vkitti_scene}/{mode}" + (f"@{sev}" if sev is not None else "")
            c = copy.deepcopy(cfg)
            d = c.setdefault("data", {})
            d.pop("root", None)  # exclude TartanAir
            d.pop("envs", None)
            d["vkitti_root"] = vkitti_root
            d["val_scenes"] = [vkitti_scene]
            if mode == "clean":
                d["degrade"] = False
            else:
                d["degrade"] = True
                d["degrade_mode"] = mode
                if sev is not None:
                    d["val_degrade_severity"] = sev
                    d["degrade_severity"] = sev
            combos.append((key, c))

    return combos


# --------------------------------------------------------------------------- #
# Temporal metrics collection.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _temporal_metrics(
    model, loader, device: str, amp_dtype, max_batches: int | None,
) -> Tuple[float, float]:
    """Compute mean tOF and OPW across clips in ``loader``."""
    model.eval()
    tofs, opws = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        frames_in = batch.get("degraded_frames", batch["frames"]).to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None,
        ):
            pred = model(frames_in)  # (B,1,T,H,W) depth
        pred = pred.float().cpu()
        for b in range(pred.shape[0]):
            pd = pred[b, 0]                       # (T,H,W)
            gd = batch["depth"][b, 0]              # (T,H,W)
            mk = batch["mask"][b, 0]               # (T,H,W)
            rgb = batch["frames"][b].permute(1, 0, 2, 3)  # (3,T,H,W) → (T,3,H,W)
            tm = clip_temporal_metrics(gd, pd, rgb, mk)
            tofs.append(tm["tof"])
            opws.append(tm["opw"])
    mean_tof = float(np.nanmean(tofs)) if tofs else float("nan")
    mean_opw = float(np.nanmean(opws)) if opws else float("nan")
    return mean_tof, mean_opw


# --------------------------------------------------------------------------- #
# Visual dump for one combo.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _capture_viz_batch(
    model, loader, device: str, amp_dtype, clip_idx: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Capture the ``clip_idx``-th batch for visualisation.

    Returns ``(rgb_bgr, gt_depth, pred_depth, valid)`` as numpy arrays, or
    ``None`` if the loader has fewer clips.
    """
    model.eval()
    for i, batch in enumerate(loader):
        if i != clip_idx:
            continue
        frames_in = batch.get("degraded_frames", batch["frames"]).to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None,
        ):
            pred = model(frames_in)
        # Take the first sample in the batch.
        b = 0
        pred_np = pred[b, 0].float().cpu().numpy()        # (T,H,W)
        gt_np = batch["depth"][b, 0].numpy()               # (T,H,W)
        mask_np = batch["mask"][b, 0].numpy()               # (T,H,W)
        # degraded RGB for the input panel
        rgb_in = batch.get("degraded_frames", batch["frames"])[b].numpy()  # (3,T,H,W)
        rgb_in = rgb_in.transpose(1, 0, 2, 3)  # (T,3,H,W) for _rgb_to_bgr_u8
        rgb_bgr = _rgb_to_bgr_u8(rgb_in)       # (T,H,W,3) BGR uint8
        return rgb_bgr, gt_np, pred_np, mask_np
    return None


# --------------------------------------------------------------------------- #
# Main entry point.
# --------------------------------------------------------------------------- #

def run_panel(
    model,
    cfg: dict,
    step: int,
    device: str,
    amp_dtype,
    output_dir: Path,
    viz: bool = True,
) -> dict:
    """Run the multi-combo validation panel.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate (should already be in eval mode / EMA-swapped).
    cfg : dict
        Full training config (must include ``val_panel`` section).
    step : int
        Current training step (for keying outputs).
    device : str
        Torch device string.
    amp_dtype
        AMP dtype (``torch.bfloat16`` or ``None``).
    output_dir : Path
        Base output directory (e.g. ``runs/dyfn_fullres_local``).
    viz : bool
        Whether to dump image grids and MP4s.

    Returns
    -------
    dict
        ``{"step": step, "combos": {combo_key: metrics}, "mean": mean_metrics}``
        where ``mean`` averages over degraded combos (excludes clean).
    """
    panel_cfg = cfg.get("val_panel", {})
    max_batches = cfg.get("training", {}).get("val_max_batches", 40)
    fixed_clip_idx = int(panel_cfg.get("fixed_clip_idx", 0))

    combos = _build_combos(cfg)
    results: dict[str, dict] = {}

    for key, combo_cfg in combos:
        try:
            loader, _ = build_dataloader(combo_cfg, "val")
        except (ValueError, FileNotFoundError) as exc:
            print(f"[val_panel] skip {key}: {exc}", flush=True)
            continue

        # ---- Standard metrics via evaluate() ----
        metrics = evaluate(model, loader, device, amp_dtype, max_batches)

        # ---- Temporal metrics (tOF + OPW) ----
        loader2, _ = build_dataloader(combo_cfg, "val")  # fresh iterator
        tof_val, opw_val = _temporal_metrics(model, loader2, device, amp_dtype, max_batches)
        metrics["tof"] = tof_val
        metrics["opw"] = opw_val

        results[key] = metrics
        print(f"[val_panel] {key}: abs_rel={metrics['abs_rel']:.4f} d1={metrics['delta1']:.3f} "
              f"tof={tof_val:.4f} opw={opw_val:.4f}", flush=True)

        # ---- Visual outputs ----
        if viz:
            loader3, _ = build_dataloader(combo_cfg, "val")  # fresh iterator
            viz_data = _capture_viz_batch(model, loader3, device, amp_dtype, fixed_clip_idx)
            if viz_data is not None:
                rgb_bgr, gt_depth, pred_depth, valid = viz_data
                safe_key = key.replace("/", "_").replace("@", "_at_")
                viz_dir = output_dir / "viz" / f"step{step}"
                _save_grid(rgb_bgr, gt_depth, pred_depth, valid,
                           viz_dir / f"{safe_key}_grid.png")
                _save_video(rgb_bgr, gt_depth, pred_depth, valid,
                            viz_dir / f"{safe_key}.mp4")

    # ---- Mean over DEGRADED combos (best.pt selection criterion) ----
    deg_keys = [k for k in results if "clean" not in k]
    mean_metrics: dict = {}
    if deg_keys:
        all_metric_names = list(results[deg_keys[0]].keys())
        for mname in all_metric_names:
            vals = []
            for k in deg_keys:
                v = results[k].get(mname, float("nan"))
                if isinstance(v, (int, float)) and not math.isnan(v):
                    vals.append(v)
            mean_metrics[mname] = float(np.mean(vals)) if vals else float("nan")

    return {"step": step, "combos": results, "mean": mean_metrics}

"""Unified alignment-free depth evaluation and serialisable result schema.

Raw scores never use ground-truth fitting.  A per-video affine disparity score is
included under a visibly separate ``aligned_reference`` key for comparison with
the historical D3 protocol imported into RAMEN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from metrics import METRIC_KEYS, compute_depth_metrics, fit_median_scale, fit_scale_shift

ALIGNMENT_SCOPES = ("per_frame", "per_video")
ALIGNMENT_MODES = ("affine", "median_scale", "none")


@dataclass
class EvaluationMetadata:
    model_name: str
    dataset_name: str
    split: str
    checkpoint_sha256: str | None = None
    manifest_sha256: str | None = None
    output_semantics: str = "metric_depth"
    condition: str = "clean"
    min_depth: float = 1e-3
    max_depth: float = 80.0
    streaming: dict[str, Any] = field(default_factory=lambda: {
        "mode": "stateless_clip", "window": None, "warmup_scored": True,
        "state_reset": "on_video_boundary", "frame_order": "chronological",
    })


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable UTF-8 representation used for provenance hashes."""
    return json.dumps(finite_json(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_result_payload(value: Any) -> Any:
    """Strip volatile/self-referential fields recursively before result hashing."""
    if isinstance(value, dict):
        return {k: canonical_result_payload(v) for k, v in value.items()
                if k not in {"created_utc", "result_sha256"}}
    if isinstance(value, list):
        return [canonical_result_payload(v) for v in value]
    return value


class _MicroAccumulator:
    """Streaming equivalent of ``compute_depth_metrics(..., invalid_pred='penalize')``
    pooled over every pixel seen so far, as running sums -- no per-pixel tensor is
    ever retained, so pixel_micro stays memory-bounded across an arbitrarily large
    dataset sweep."""

    def __init__(self, min_depth: float, max_depth: float):
        self.lo, self.hi = min_depth, max_depth
        self.n = 0
        self.sum_abs_rel = self.sum_sq_rel = self.sum_sq_diff = 0.0
        self.sum_log_diff = self.sum_log_diff_sq = 0.0
        self.delta1 = self.delta2 = self.delta3 = 0

    def update(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> None:
        lo, hi = self.lo, self.hi
        pred, gt = pred.float().flatten(), gt.float().flatten()
        mask = mask.flatten().bool()
        valid = mask & torch.isfinite(gt) & (gt > lo) & (gt < hi)
        if not bool(valid.any()):
            return
        safe = torch.where(torch.isfinite(pred) & (pred > lo), pred, pred.new_tensor(lo))
        p, g = safe[valid], gt[valid]
        diff = p - g
        log_diff = torch.log(p) - torch.log(g)
        ratio = torch.max(p / g, g / p)
        self.n += int(p.numel())
        self.sum_abs_rel += float((diff.abs() / g).sum())
        self.sum_sq_rel += float(((diff**2) / g).sum())
        self.sum_sq_diff += float((diff**2).sum())
        self.sum_log_diff += float(log_diff.sum())
        self.sum_log_diff_sq += float((log_diff**2).sum())
        self.delta1 += int((ratio < 1.25).sum())
        self.delta2 += int((ratio < 1.25**2).sum())
        self.delta3 += int((ratio < 1.25**3).sum())

    def result(self) -> dict[str, float]:
        if self.n == 0:
            return {k: float("nan") for k in METRIC_KEYS}
        n = self.n
        mean_log_diff = self.sum_log_diff / n
        si_log = self.sum_log_diff_sq / n - mean_log_diff**2
        return {
            "abs_rel": self.sum_abs_rel / n, "sq_rel": self.sum_sq_rel / n,
            "rmse": math.sqrt(self.sum_sq_diff / n), "rmse_log": math.sqrt(self.sum_log_diff_sq / n),
            "si_log": si_log, "delta1": self.delta1 / n, "delta2": self.delta2 / n, "delta3": self.delta3 / n,
            "si_log_pct": math.sqrt(max(si_log, 0.0)) * 100.0,
        }


def validate_video_id_contiguity(video_ids: Iterable[Any]) -> None:
    """Reject a ``video_id`` that reappears after a different one was seen.

    DepthEvaluator.add_frame enforces this too, but only once it reaches the
    offending frame -- possibly deep into a long stream. Callers that hold the
    whole array up front (any cache-file scorer, not just the streaming ones)
    should check it first and fail with one clear message instead.
    """
    seen: set[str] = set()
    current: str | None = None
    for row, vid_raw in enumerate(video_ids):
        vid = str(vid_raw)
        if vid != current:
            if vid in seen:
                raise ValueError(f"video {vid!r} is non-contiguous at row {row}; "
                                 "frames for one video must be grouped together")
            seen.add(vid)
            current = vid


class DepthEvaluator:
    """Accumulate named video frames and produce raw plus reference-aligned scores.

    Frames for one ``video_id`` must be added contiguously (matching the
    chronological-manifest contract every caller in this repo already follows):
    once a different ``video_id`` is seen, the previous video is finalised --
    its per-frame metric rows and fit are kept, but its raw ``pred``/``gt``/``mask``
    tensors are discarded. This bound held for those raw tensors and for
    ``frame_biases`` (one value per frame) in every run this project has ever made
    -- but it was FALSE for the per-pixel scale-log accumulation ``scale_logs``
    feeds (TOOL-15, 2026-08-30): that list held one Python float per VALID PIXEL
    across the WHOLE DATASET, not per video, and was never discarded. No prior run
    was ever actually bounded by video size; every one was simply small enough not
    to notice, until Bonn-primary's ~5.1 billion valid pixels (518x518 x 19064
    frames) OOM-killed the host at 109.8GB RSS. ``track_pixel_scale_logs`` (below)
    now defaults this accumulation OFF, closing the gap rather than merely
    describing it. Re-adding frames for an already-finalised video raises.
    """

    def __init__(self, metadata: EvaluationMetadata, alignment_scope: str = "per_video",
                 alignment_mode: str = "affine", *, track_pixel_scale_logs: bool = False):
        if alignment_scope not in ALIGNMENT_SCOPES:
            raise ValueError(f"alignment_scope must be one of {ALIGNMENT_SCOPES}")
        if alignment_mode not in ALIGNMENT_MODES:
            raise ValueError(f"alignment_mode must be one of {ALIGNMENT_MODES}")
        self.metadata = metadata
        self.alignment_scope = alignment_scope
        self.alignment_mode = alignment_mode
        # Keyword-only with a default so both frozen callers (scripts/eval_streaming_npz.py,
        # scripts/eval_alignment_free.py), which construct this positionally, are unaffected.
        self._track_pixel_scale_logs = track_pixel_scale_logs
        self._current_video_id: str | None = None
        self._current_frames: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._video_ids_seen: list[str] = []
        self._closed: dict[str, dict[str, Any]] = {}
        self._micro_raw = _MicroAccumulator(metadata.min_depth, metadata.max_depth)
        self._micro_aligned = _MicroAccumulator(metadata.min_depth, metadata.max_depth)
        self._valid_pixels = self._invalid_pred_pixels = 0
        self._fit_attempted = self._fit_solved = 0

    def add_frame(self, video_id: str, pred: torch.Tensor, gt: torch.Tensor,
                  mask: torch.Tensor | None = None) -> None:
        pred, gt = torch.as_tensor(pred).float().cpu(), torch.as_tensor(gt).float().cpu()
        if pred.shape != gt.shape:
            raise ValueError(f"pred/gt shape mismatch: {pred.shape} != {gt.shape}")
        mk = torch.ones_like(gt, dtype=torch.bool) if mask is None else torch.as_tensor(mask).bool().cpu()
        if mk.shape != gt.shape:
            raise ValueError(f"mask/gt shape mismatch: {mk.shape} != {gt.shape}")
        vid = str(video_id)
        if vid in self._closed:
            raise ValueError(f"video {vid!r} was already finalised; frames for one video "
                             "must be added contiguously")
        if self._current_video_id is not None and vid != self._current_video_id:
            self._close_current_video()
        if self._current_video_id is None:
            self._current_video_id = vid
            self._video_ids_seen.append(vid)
        self._current_frames.append((pred, gt, mk))

    def add_video(self, video_id: str, pred: torch.Tensor, gt: torch.Tensor,
                  mask: torch.Tensor | None = None) -> None:
        pred, gt = torch.as_tensor(pred), torch.as_tensor(gt)
        if pred.shape != gt.shape or pred.ndim < 1:
            raise ValueError("video pred and gt must share a leading time dimension")
        masks = torch.ones_like(gt, dtype=torch.bool) if mask is None else torch.as_tensor(mask).bool()
        for t in range(pred.shape[0]):
            self.add_frame(video_id, pred[t], gt[t], masks[t])

    def _close_current_video(self) -> None:
        if self._current_video_id is None:
            return
        self._closed[self._current_video_id] = self._finalize_video(self._current_frames)
        self._current_video_id, self._current_frames = None, []

    def _finalize_video(self, frames) -> dict[str, Any]:
        """One pass over one video's buffered frames: raw + aligned metric rows,
        the fit(s), scale-drift logs, and running micro/pixel-count updates. The
        raw tensors are only ever touched here, then dropped by the caller.

        Called only from ``_close_current_video`` (``add_frame``'s video-boundary
        path and ``result()``), never from inside a caller's own per-frame timing
        boundary (``scripts/eval_streaming_npz.py``'s ``timed_predict`` returns its
        prediction to ``add_frame`` before this runs) -- TOOL-15's memory fix below
        cannot affect any already-recorded C10 latency number."""
        lo, hi = self.metadata.min_depth, self.metadata.max_depth
        video_fit = None
        if self.alignment_mode != "none" and self.alignment_scope == "per_video":
            video_fit = self._fit_group(frames, lo, hi)
            self._fit_attempted += 1
            self._fit_solved += int(video_fit.solved)

        raw_rows, aligned_rows, fits, scale_logs, frame_biases = [], [], [], [], []
        for pred, gt, mask in frames:
            gt_valid = mask & torch.isfinite(gt) & (gt > lo) & (gt < hi)
            bad = gt_valid & (~torch.isfinite(pred) | (pred <= lo))
            self._valid_pixels += int(gt_valid.sum())
            self._invalid_pred_pixels += int(bad.sum())

            raw_rows.append(compute_depth_metrics(pred, gt, mask, lo, hi, invalid_pred="penalize"))
            self._micro_raw.update(pred, gt, mask)

            fit, evaluated = None, pred
            if self.alignment_mode != "none":
                if self.alignment_scope == "per_video":
                    fit = video_fit
                else:
                    fit = self._fit_group([(pred, gt, mask)], lo, hi)
                    self._fit_attempted += 1
                    self._fit_solved += int(fit.solved)
                evaluated = self._apply_fit(pred, fit, lo, hi)
                fits.append(asdict(fit))
            aligned_rows.append(compute_depth_metrics(evaluated, gt, mask, lo, hi, invalid_pred="penalize"))
            self._micro_aligned.update(evaluated, gt, mask)

            scale_valid = gt_valid & torch.isfinite(pred) & (pred > lo)
            if bool(scale_valid.any()):
                vals = torch.log(pred[scale_valid] / gt[scale_valid]).numpy()
                if self._track_pixel_scale_logs:
                    scale_logs.extend(vals.tolist())
                frame_biases.append(float(np.median(vals)))

        return {"raw_rows": raw_rows, "aligned_rows": aligned_rows, "fits": fits,
                "video_fit": asdict(video_fit) if video_fit is not None else None,
                "scale_logs": scale_logs, "frame_biases": frame_biases, "frame_count": len(frames)}

    def _fit_group(self, frames, lo: float, hi: float):
        """One fit over pooled pred/gt/mask across `frames`, per ``self.alignment_mode``."""
        ps, gs, ms = zip(*frames)
        pred = torch.cat([p.reshape(-1) for p in ps])
        gt = torch.cat([g.reshape(-1) for g in gs])
        mask = torch.cat([m.reshape(-1) for m in ms])
        valid = mask & torch.isfinite(gt) & (gt > lo) & (gt < hi) & torch.isfinite(pred) & (pred > lo)
        if self.alignment_mode == "affine":
            return fit_scale_shift(1 / pred.clamp(min=lo), 1 / gt.clamp(min=lo), valid)
        if self.alignment_mode == "median_scale":
            return fit_median_scale(pred, gt, valid)
        raise ValueError(f"unknown alignment_mode {self.alignment_mode!r}")

    def _apply_fit(self, pred: torch.Tensor, fit, lo: float, hi: float) -> torch.Tensor:
        if fit is None or not fit.solved:
            return pred
        if self.alignment_mode == "affine":
            disp = fit.scale / pred.clamp(min=lo) + fit.shift
            return (1 / disp.clamp(min=1 / hi)).clamp(lo, hi)
        if self.alignment_mode == "median_scale":
            return (fit.scale * pred).clamp(lo, hi)
        raise ValueError(f"unknown alignment_mode {self.alignment_mode!r}")

    def _aggregate(self, aligned: bool) -> dict[str, Any]:
        row_key = "aligned_rows" if aligned else "raw_rows"
        frame_rows = []
        video_rows: dict[str, list[dict[str, float]]] = {}
        fits: dict[str, Any] = {}
        for vid in self._video_ids_seen:
            summary = self._closed[vid]
            rows = summary[row_key]
            frame_rows.extend(rows)
            video_rows[vid] = rows
            if aligned and self.alignment_mode != "none":
                fits[vid] = summary["video_fit"] if self.alignment_scope == "per_video" else summary["fits"]
        macro = {k: float(np.nanmean([r[k] for r in frame_rows])) for k in METRIC_KEYS} if frame_rows else {}
        micro_acc = self._micro_aligned if aligned else self._micro_raw
        micro = micro_acc.result() if micro_acc.n > 0 else {}
        per_video_metrics = {
            vid: {k: float(np.nanmean([row[k] for row in rows])) for k in METRIC_KEYS}
            for vid, rows in video_rows.items()
        }
        video_macro = ({k: float(np.nanmean([row[k] for row in per_video_metrics.values()]))
                        for k in METRIC_KEYS} if per_video_metrics else {})
        return {"frame_macro": macro, "video_macro": video_macro, "pixel_micro": micro,
                "per_video": per_video_metrics, "fits": fits}

    def _scale_metrics(self) -> dict[str, Any]:
        logs, per_video = [], {}
        total_frames = covered_frames = covered_pixels = 0
        for vid in self._video_ids_seen:
            summary = self._closed[vid]
            total_frames += summary["frame_count"]
            covered_frames += len(summary["frame_biases"])
            covered_pixels += len(summary["scale_logs"])
            logs.extend(summary["scale_logs"])
            if summary["frame_biases"]:
                per_video[vid] = {"median_log_scale_bias": float(np.median(summary["frame_biases"])),
                                  "temporal_log_scale_std": float(np.std(summary["frame_biases"]))}
        arr = np.asarray(logs)
        # TOOL-15: the pixel-pooled fields below are None (not NaN/0) when
        # track_pixel_scale_logs is off, never silently absent -- "not measured" and
        # "measured as zero/empty" are different claims (same principle as §4.7's
        # unevaluable-vs-evaluated-and-found-nothing distinction), and a 0
        # pixel_fraction reads as a real, alarming finding rather than a deliberate
        # default. The marker is stated in BOTH states so an artifact is
        # self-describing regardless of when it was produced.
        if self._track_pixel_scale_logs:
            pixel_scale_tracking = "enabled"
            median_log_scale_bias = float(np.median(arr)) if arr.size else None
            scale_factor_to_gt = float(np.exp(-np.median(arr))) if arr.size else None
            median_abs_log_scale_error = float(np.median(np.abs(arr))) if arr.size else None
            pixels_with_valid_scale: int | None = covered_pixels
            pixel_fraction: float | None = covered_pixels / max(self._valid_pixels, 1)
        else:
            pixel_scale_tracking = "disabled"
            median_log_scale_bias = scale_factor_to_gt = median_abs_log_scale_error = None
            pixels_with_valid_scale = pixel_fraction = None
        return {
            "pixel_scale_tracking": pixel_scale_tracking,
            "median_log_scale_bias": median_log_scale_bias,
            "scale_factor_to_gt": scale_factor_to_gt,
            "median_abs_log_scale_error": median_abs_log_scale_error,
            "video_macro_temporal_log_scale_std": float(np.mean([v["temporal_log_scale_std"] for v in per_video.values()])) if per_video else float("nan"),
            "coverage": {"frames_total": total_frames, "frames_with_valid_scale": covered_frames,
                         "frame_fraction": covered_frames / max(total_frames, 1),
                         "gt_valid_pixels": self._valid_pixels, "pixels_with_valid_scale": pixels_with_valid_scale,
                         "pixel_fraction": pixel_fraction},
            "per_video": per_video,
        }

    def result(self) -> dict[str, Any]:
        self._close_current_video()
        raw, aligned = self._aggregate(False), self._aggregate(True)
        frames = sum(summary["frame_count"] for summary in self._closed.values())
        result = {
            "schema_version": "1.0.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "metadata": asdict(self.metadata),
            "aggregation": {"primary": "frame_macro", "also_reported": ["video_macro", "pixel_micro"],
                            "alignment_scope": f"none_for_raw; {self.alignment_scope} {self.alignment_mode} "
                                                "fit for aligned_reference"},
            "counts": {"videos": len(self._closed), "frames_scored": frames,
                       "valid_pixels": self._valid_pixels, "invalid_pred_pixels": self._invalid_pred_pixels,
                       "invalid_pred_fraction": self._invalid_pred_pixels / max(self._valid_pixels, 1)},
            "metrics": {"raw": {"frame_macro": raw["frame_macro"], "video_macro": raw["video_macro"],
                                "pixel_micro": raw["pixel_micro"]},
                        "scale": self._scale_metrics(),
                        "aligned_reference": {"mode": f"{self.alignment_scope}_{self.alignment_mode}",
                                              "frame_macro": aligned["frame_macro"],
                                              "video_macro": aligned["video_macro"],
                                              "pixel_micro": aligned["pixel_micro"], "fits": aligned["fits"],
                                              "fit_counts": {"attempted": self._fit_attempted,
                                                             "solved": self._fit_solved,
                                                             "unsolved": self._fit_attempted - self._fit_solved}}},
        }
        result["provenance_sha256"] = canonical_sha256(result["metadata"])
        # Timestamp is intentionally excluded: identical evidence must hash identically.
        result["result_sha256"] = canonical_sha256(canonical_result_payload(result))
        return result

    def write_json(self, path: str | Path) -> dict[str, Any]:
        result = self.result()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(finite_json(result), indent=2, allow_nan=False) + "\n")
        return result


def finite_json(value):
    """Recursively replace non-finite floats with JSON-safe ``None`` values."""
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_prediction_cache(path: str | Path, rows: Iterable[tuple[Any, Any, Any, Any]]) -> None:
    """Write a Tier-1 prediction cache: ``(video_id, pred, gt, mask)`` rows, one per frame.

    Frames are stored as flat, concatenated buffers plus a per-frame ``shape`` array
    rather than one stacked ``(N, H, W)`` ndarray, so frames at different native
    resolutions -- e.g. raw KITTI camera size varies a few pixels between drives --
    can share one file with no padding and no ``allow_pickle`` object array.
    """
    video_ids: list[str] = []
    shapes: list[tuple[int, int]] = []
    pred_chunks, gt_chunks, mask_chunks = [], [], []
    for video_id, pred, gt, mask in rows:
        pred = np.asarray(pred, dtype=np.float32)
        gt = np.asarray(gt, dtype=np.float32)
        if pred.shape != gt.shape:
            raise ValueError(f"pred/gt shape mismatch for video {video_id!r}: {pred.shape} != {gt.shape}")
        mask = np.ones(gt.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if mask.shape != gt.shape:
            raise ValueError(f"mask/gt shape mismatch for video {video_id!r}: {mask.shape} != {gt.shape}")
        video_ids.append(str(video_id))
        shapes.append(pred.shape)
        pred_chunks.append(pred.reshape(-1))
        gt_chunks.append(gt.reshape(-1))
        mask_chunks.append(mask.reshape(-1))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        video_id=np.asarray(video_ids),
        shape=np.asarray(shapes, dtype=np.int64) if shapes else np.zeros((0, 2), dtype=np.int64),
        pred=np.concatenate(pred_chunks) if pred_chunks else np.zeros(0, dtype=np.float32),
        gt=np.concatenate(gt_chunks) if gt_chunks else np.zeros(0, dtype=np.float32),
        mask=np.concatenate(mask_chunks) if mask_chunks else np.zeros(0, dtype=bool),
    )


def read_prediction_cache(path: str | Path) -> Iterable[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Read a cache written by :func:`write_prediction_cache`.

    Validates video_id contiguity eagerly, before any frame is reconstructed, so a bad
    cache fails fast with a clear message rather than mid-stream.
    """
    data = np.load(path, allow_pickle=False)
    video_ids = [str(v) for v in data["video_id"]]
    validate_video_id_contiguity(video_ids)
    shapes = data["shape"]
    pred_flat, gt_flat, mask_flat = data["pred"], data["gt"], data["mask"]

    def _rows():
        offset = 0
        for video_id, (h, w) in zip(video_ids, shapes):
            n = int(h) * int(w)
            pred = pred_flat[offset:offset + n].reshape(int(h), int(w))
            gt = gt_flat[offset:offset + n].reshape(int(h), int(w))
            mask = mask_flat[offset:offset + n].reshape(int(h), int(w))
            offset += n
            yield video_id, pred, gt, mask

    return _rows()


def write_frame_manifest(path: str | Path,
                         rows: Iterable[tuple[Any, float, Any, Any, Any]]) -> None:
    """Write a streaming-eval frame manifest: ``(video_id, frame_index, frame, gt, mask)``
    rows, one per frame, chronological within each video.

    ``frame`` and ``gt``/``mask`` are stored as flat, concatenated buffers plus their own
    per-frame ``shape`` arrays -- the same ragged-safe technique as
    :func:`write_prediction_cache` -- because real video frames vary in native resolution
    (e.g. raw KITTI camera size differs a few pixels per drive) and ``frame`` additionally
    carries a channel dimension ``gt``/``mask`` don't, so they need separate shape arrays.
    """
    video_ids: list[str] = []
    frame_indices: list[float] = []
    frame_shapes: list[tuple[int, ...]] = []
    gt_shapes: list[tuple[int, ...]] = []
    frame_chunks, gt_chunks, mask_chunks = [], [], []
    for video_id, frame_index, frame, gt, mask in rows:
        frame = np.asarray(frame, dtype=np.float32)
        gt = np.asarray(gt, dtype=np.float32)
        mask = np.ones(gt.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if mask.shape != gt.shape:
            raise ValueError(f"mask/gt shape mismatch for video {video_id!r}: {mask.shape} != {gt.shape}")
        video_ids.append(str(video_id))
        frame_indices.append(float(frame_index))
        frame_shapes.append(frame.shape)
        gt_shapes.append(gt.shape)
        frame_chunks.append(frame.reshape(-1))
        gt_chunks.append(gt.reshape(-1))
        mask_chunks.append(mask.reshape(-1))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        video_id=np.asarray(video_ids),
        frame_index=np.asarray(frame_indices, dtype=np.float64),
        frame_shape=np.asarray(frame_shapes, dtype=np.int64),
        gt_shape=np.asarray(gt_shapes, dtype=np.int64),
        frame=np.concatenate(frame_chunks) if frame_chunks else np.zeros(0, dtype=np.float32),
        gt=np.concatenate(gt_chunks) if gt_chunks else np.zeros(0, dtype=np.float32),
        mask=np.concatenate(mask_chunks) if mask_chunks else np.zeros(0, dtype=bool),
    )


class FrameManifest:
    """Lazy, ragged-safe reader for a manifest written by :func:`write_frame_manifest`.

    Reconstructs one frame/gt/mask at a time from the flat buffers -- nothing beyond the
    small per-frame shape/index arrays is loaded until a specific row is asked for.
    """

    def __init__(self, path: str | Path):
        data = np.load(path, allow_pickle=False)
        self.video_id: list[str] = [str(v) for v in data["video_id"]]
        self.frame_index: np.ndarray = data["frame_index"]
        self._frame_shapes = data["frame_shape"]
        self._gt_shapes = data["gt_shape"]
        self._frame_flat = data["frame"]
        self._gt_flat = data["gt"]
        self._mask_flat = data["mask"]
        self._frame_offsets = np.concatenate(([0], np.cumsum(np.prod(self._frame_shapes, axis=1))))
        self._gt_offsets = np.concatenate(([0], np.cumsum(np.prod(self._gt_shapes, axis=1))))
        if not (len(self.video_id) == len(self.frame_index) == len(self._frame_shapes) == len(self._gt_shapes)):
            raise ValueError("frame manifest arrays have inconsistent leading lengths")

    def __len__(self) -> int:
        return len(self.video_id)

    def frame(self, row: int) -> np.ndarray:
        shape = tuple(int(x) for x in self._frame_shapes[row])
        o0, o1 = self._frame_offsets[row], self._frame_offsets[row + 1]
        return self._frame_flat[o0:o1].reshape(shape)

    def gt(self, row: int) -> np.ndarray:
        shape = tuple(int(x) for x in self._gt_shapes[row])
        o0, o1 = self._gt_offsets[row], self._gt_offsets[row + 1]
        return self._gt_flat[o0:o1].reshape(shape)

    def mask(self, row: int) -> np.ndarray:
        shape = tuple(int(x) for x in self._gt_shapes[row])
        o0, o1 = self._gt_offsets[row], self._gt_offsets[row + 1]
        return self._mask_flat[o0:o1].reshape(shape)


def write_recipe_manifest(path: str | Path, video_ids: Iterable[Any],
                          frame_indices: Iterable[float], root: str,
                          digests: Mapping[str, str], **extra_arrays: np.ndarray) -> None:
    """Write a recipe-locked manifest: identity/ordering (``video_id``, ``frame_index``)
    plus ``root`` (the path prefix every source path elsewhere is relative to, so the
    manifest is not machine-bound) and each video's locked pixel digest -- never the
    pixels themselves. See :class:`RecipeManifest` for why (materialising a large
    population's pixels can exceed available memory/disk; provenance/plan_state.json's
    MATRIX-03 note, 2026-08-30, records the OOM this replaces).

    Per-video recipe fields (source paths, degradation seed/mode/severity/blur
    assignment) are deliberately NOT this function's concern -- dataset-specific
    decode/degradation logic does not belong in this frozen, dataset-agnostic file,
    mirroring ``eval_streaming_npz.py``'s own model-registry callback pattern. A
    per-benchmark builder script passes its own extra fields via ``**extra_arrays``
    so everything lands in one atomic ``npz`` write; this function only fixes the
    contract :class:`RecipeManifest` depends on, so the two halves cannot drift
    apart silently.

    Validates at write time, not read time, that every distinct ``video_id`` has a
    digest -- the manifest being written IS the artifact being hash-locked, so it
    must never be possible to write an incomplete one and only discover the gap
    mid-run.
    """
    video_ids = [str(v) for v in video_ids]
    digest_video_id = sorted(digests)
    if not digest_video_id:
        raise ValueError("write_recipe_manifest: digests must be non-empty")
    missing = set(video_ids) - set(digest_video_id)
    if missing:
        raise ValueError(f"write_recipe_manifest: no digest recorded for video(s) {sorted(missing)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        recipe_marker=np.asarray([1]),
        video_id=np.asarray(video_ids),
        frame_index=np.asarray(list(frame_indices), dtype=np.float64),
        root=np.asarray([str(root)]),
        digest_video_id=np.asarray(digest_video_id),
        digest_sha256=np.asarray([digests[v] for v in digest_video_id]),
        **extra_arrays,
    )


def update_pixel_digest(digest: "hashlib._Hash", frame: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> None:
    """Feed one ``(frame, gt, mask)`` row into ``digest`` in the exact dtype/order that
    a recipe manifest's locked digest is defined over. Shared by :meth:`RecipeManifest._verify`
    and every per-benchmark builder script that computes a digest to lock at write time --
    the two sides must hash identically or every video mismatches with an error that blames
    the wrong thing. Do not reimplement this inline; call it from both places instead, so a
    forgotten ``ascontiguousarray`` or a reordered field cannot drift the two apart silently."""
    digest.update(np.ascontiguousarray(frame, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(gt, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(mask, dtype=bool).tobytes())


class RecipeManifest:
    """Regenerate-and-verify reader for a manifest written by :func:`write_recipe_manifest`.

    Stores per-frame source paths and a locked per-video pixel digest instead of
    materialised pixels. Never holds more than ``chunk_size`` frames' worth of
    pixels in memory at once -- a whole-video cache would reintroduce the exact
    OOM this class exists to avoid on a long video (rgbd_bonn_static's 10000
    frames at ~4.4MB/frame is ~44GB; see provenance/plan_state.json's MATRIX-03
    note, 2026-08-30).

    Two passes per video, both bounded to ``chunk_size``:

    1. VERIFY (once, the first time a video is touched, before any of its frames
       are returned to a caller): regenerate the whole video in ``chunk_size``
       pieces, feed each piece into a running digest, discard it immediately.
       Compare the final digest to the locked one; raise, naming the video, on
       any mismatch (missing/changed source, code change, parameter drift).
    2. SERVE: once verified, regenerate one ``chunk_size`` window at a time as
       ``frame``/``gt``/``mask`` are called, caching only that window.

    This regenerates every frame twice (once to verify, once to serve) rather
    than once -- a deliberate trade of compute for the "verified before scored"
    property under a hard memory bound; regeneration is comparatively cheap
    (decode/resize/degrade) and, per the call site's own structure, always runs
    outside the timed region (see ``scripts/eval_streaming_npz.py``'s
    ``video_samples``/``timed_predict`` split), so this does not touch latency
    numbers -- only wall-clock time to complete a run.

    Exposes the identical ``(len, video_id, frame_index, frame(row), gt(row),
    mask(row))`` interface as :class:`FrameManifest`, so any existing consumer
    written against ``FrameManifest`` works unmodified against this class too --
    see :func:`load_frame_source` for the by-content dispatch between the two.
    Consumption must be chronological within each video and video-to-video in
    the manifest's own order (exactly what ``ContinuousStreamConsumer`` already
    does); revisiting an earlier, already-evicted video raises rather than
    silently re-verifying and re-serving it, since nothing in this repo's own
    consumers should ever do that and a silent re-run would be an O(n^2) cliff
    hiding as correct behaviour.

    ``regenerate_range(video_id, start, end) -> list[(frame, gt, mask)]`` is
    supplied by the caller, not this class: a pure, deterministic function of
    the recipe this manifest recorded for that video (source paths relative to
    ``self.root``, degradation seed/mode/severity/blur assignment), returning
    exactly ``end - start`` rows for local frame indices ``[start, end)`` of
    that video, IN THAT ORDER -- the same order the manifest's own
    ``frame_index`` gives that video's frames. Dataset-specific decode logic
    (Bonn's TUM depth convention, VKITTI's sky sentinel, KITTI's own scale)
    lives in that callback, not in this frozen, dataset-agnostic reader.
    """

    def __init__(self, path: str | Path, regenerate_range: Callable[[str, int, int], list],
                 chunk_size: int = 64):
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        data = np.load(path, allow_pickle=False)
        if "recipe_marker" not in data.files:
            raise ValueError(f"{path} is not a recipe manifest (no recipe_marker key)")
        self.video_id: list[str] = [str(v) for v in data["video_id"]]
        self.frame_index: np.ndarray = data["frame_index"]
        self.root: str = str(data["root"][0])
        validate_video_id_contiguity(self.video_id)
        digest_video_id = [str(v) for v in data["digest_video_id"]]
        self._locked_digest: dict[str, str] = dict(zip(digest_video_id,
                                                        (str(h) for h in data["digest_sha256"])))
        self._regenerate_range = regenerate_range
        self._chunk_size = chunk_size
        self._row_range: dict[str, tuple[int, int]] = {}
        start = 0
        for i in range(1, len(self.video_id) + 1):
            if i == len(self.video_id) or self.video_id[i] != self.video_id[start]:
                self._row_range[self.video_id[start]] = (start, i)
                start = i
        self._verified_video_id: str | None = None
        self._evicted_video_ids: set[str] = set()
        self._chunk_range: tuple[int, int] | None = None  # (start, end), LOCAL to the verified video
        self._chunk_rows: list | None = None

    def __len__(self) -> int:
        return len(self.video_id)

    def _regenerate_local(self, vid: str, lo: int, hi: int, n_total: int) -> list:
        """Call the caller's callback for local range [lo, hi) of ``vid`` and
        validate it returned exactly that many rows, in order -- an undercount
        or overcount here is a callback bug that must fail loudly at the call
        site, not as a mysterious IndexError or a silently mislabeled frame."""
        rows = self._regenerate_range(vid, lo, hi)
        rows = list(rows)
        if len(rows) != hi - lo:
            raise ValueError(
                f"recipe manifest: regenerate_range({vid!r}, {lo}, {hi}) returned "
                f"{len(rows)} rows, expected {hi - lo} (video has {n_total} frames "
                "total). Rows must be returned in the manifest's own frame_index "
                "order for this video.")
        return rows

    def _verify(self, vid: str) -> None:
        if vid in self._evicted_video_ids:
            raise ValueError(
                f"recipe manifest: video {vid!r} was already consumed and evicted; "
                "revisiting an earlier video is not supported (consumption must be "
                "chronological, video-to-video in the manifest's own order -- "
                "exactly what ContinuousStreamConsumer already does).")
        locked = self._locked_digest.get(vid)
        if locked is None:
            raise ValueError(f"recipe manifest: video {vid!r} has no locked digest recorded")
        start, end = self._row_range[vid]
        n_total = end - start
        digest = hashlib.sha256()
        for lo in range(0, n_total, self._chunk_size):
            hi = min(lo + self._chunk_size, n_total)
            rows = self._regenerate_local(vid, lo, hi, n_total)
            for frame, gt, mask in rows:
                update_pixel_digest(digest, frame, gt, mask)
            del rows  # discarded before the next chunk is regenerated -- never accumulated
        realised = digest.hexdigest()
        if realised != locked:
            raise ValueError(
                f"recipe manifest DIGEST MISMATCH for video {vid!r}: locked={locked} "
                f"realised={realised} -- regeneration no longer matches the locked "
                "recipe (a source file, the degradation code, or a parameter changed). "
                "Refusing to score this video.")
        if self._verified_video_id is not None:
            self._evicted_video_ids.add(self._verified_video_id)
        self._verified_video_id = vid
        self._chunk_range, self._chunk_rows = None, None

    def _local_row(self, row: int) -> tuple:
        vid = self.video_id[row]
        if vid != self._verified_video_id:
            self._verify(vid)
        start, end = self._row_range[vid]
        local = row - start
        if self._chunk_range is None or not (self._chunk_range[0] <= local < self._chunk_range[1]):
            lo = (local // self._chunk_size) * self._chunk_size
            hi = min(lo + self._chunk_size, end - start)
            self._chunk_rows = self._regenerate_local(vid, lo, hi, end - start)
            self._chunk_range = (lo, hi)
        return self._chunk_rows[local - self._chunk_range[0]]

    def frame(self, row: int) -> np.ndarray:
        return self._local_row(row)[0]

    def gt(self, row: int) -> np.ndarray:
        return self._local_row(row)[1]

    def mask(self, row: int) -> np.ndarray:
        return self._local_row(row)[2]


def load_frame_source(path: str | Path, regenerate_range: Callable[[str, int, int], list] | None = None,
                      chunk_size: int = 64):
    """Detect whether ``path`` is a pixel-materialised manifest
    (:func:`write_frame_manifest`) or a recipe-locked one (:func:`write_recipe_manifest`)
    BY CONTENT -- the presence of the ``recipe_marker`` key -- never by a caller-supplied
    flag, so an existing call site that only ever sees pixel manifests is unaffected
    even if this function is introduced underneath it. Returns the matching reader;
    both expose the identical interface FrameManifest and RecipeManifest share.
    ``regenerate_range``/``chunk_size`` are ignored for a pixel manifest.
    """
    with np.load(path, allow_pickle=False) as data:
        is_recipe = "recipe_marker" in data.files
    if is_recipe:
        if regenerate_range is None:
            raise ValueError(f"{path} is a recipe manifest; regenerate_range must be supplied")
        return RecipeManifest(path, regenerate_range, chunk_size=chunk_size)
    return FrameManifest(path)


class ContinuousStreamConsumer:
    """Consume chronological frames while preserving model state within each video.

    ``predict_frame(frame)`` owns model state; ``reset()`` is called exactly at a
    video boundary.  Chunk boundaries are deliberately absent from this interface.
    """

    def __init__(self, evaluator: DepthEvaluator, predict_frame: Callable[[Any], torch.Tensor],
                 reset: Callable[[], None]):
        self.evaluator, self.predict_frame, self.reset = evaluator, predict_frame, reset

    def consume(self, videos: Mapping[str, Iterable[tuple[Any, torch.Tensor, torch.Tensor | None]]]) -> None:
        for video_id, samples in videos.items():
            self.reset()
            for frame, gt, mask in samples:
                self.evaluator.add_frame(video_id, self.predict_frame(frame), gt, mask)

"""VAL-01: VDA-semantics variants of the four training dataset classes.

Each subclass reuses its parent's directory scanning / clip indexing (``__init__``,
``self.clips``) unchanged -- the frozen, paused pipeline's file discovery logic is not
touched -- but overrides ``__getitem__`` to use ``data/vda_resize.py``'s short-edge
resize + square crop instead of the parent's own resize (a direct anisotropic stretch for
TartanAir/IRS/PointOdyssey, or VKITTI's own bespoke square-crop-then-resize -- Fable's
2026-08-27 decision: implement ONE uniform transform for all four, do not reuse VKITTI's
path, so the code is identical and the audit stays checkable).

The clean-only classes below (``VDASemantics*Dataset``) have NO degradation support --
this was VAL-01's own scope ("no degradation, no RAMEN modules, no LoRA") and stays
frozen exactly as VAL-01 used it; do not add degradation to these classes. Do not use
them for anything that needs ``degraded_frames`` in its output.

The crop position is chosen ONCE per clip (first frame) and reused for every frame in that
clip -- a clip must stay spatially consistent across time, matching every other dataset
class's existing pattern (e.g. VKITTIDataset's own x_off, computed once per clip).

``VDASemanticsDegraded*Dataset`` (matrix item 3, Opus/Fable, 2026-08-28): separate
sibling classes, one per source type, for the 8-cell training-comparison matrix's
degraded cells -- built alongside, not by modifying, the frozen clean classes above.
Each degrades AFTER resize+crop (the ``_VDADegradedMixin`` below wraps the clean
parent's already-assembled ``result["frames"]``, at the target resolution, e.g. 518^2 --
NOT the raw pre-resize image), reusing ``video_clip_dataset._degrade_clip`` directly
(the same ELVIS-style low-light+blur+noise synthesis, same ``degrade_mode``/
``severity_max`` curriculum-cap machinery every existing degraded screen already uses --
not reimplemented) so the matrix's degraded cells are comparable to prior degraded
screens. GT depth/mask/sky stay clean, matching every other degraded dataset class's
convention -- only ``degraded_frames`` (the student's input) is degraded; ``frames``
remains the clean, undegraded reference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image

from .irs_dataset import IRSDataset, _read_exr_depth
from .pointodyssey_dataset import PointOdysseyDataset, _PO_DEPTH_SCALE
from .video_clip_dataset import (STATIC_EV_JITTER, STATIC_NOISE_MULT, VideoDepthClipDataset, _decode_depth,
                                 _degrade_clip, _degrade_static_clip)
from .vkitti_dataset import VKITTIDataset, _SKY_DEPTH_M, _SKY_VALID_DEPTH_M, _decode_vkitti_depth
from .vda_resize import vda_crop_box, vda_resize_depth_like, vda_resize_frame
from .pseudo_gt_dataset import PseudoGTVideoDataset
from .video_only_dataset import VideoOnlyDataset

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("opencv-python is required for the PointOdyssey VDA-semantics variant.") from e


def _vda_frame_and_crop(raw_img: Image.Image, target: int, split: str):
    """Shared first step for every VDA-semantics __getitem__: short-edge resize the frame,
    pick one crop box (random train / centre val), return the (uncropped) resized frame,
    the crop box, and the resized (w, h) every depth/mask array in this clip must also be
    resized to before applying the SAME crop box."""
    resized = vda_resize_frame(raw_img, target=target)
    rh, rw = resized.shape[:2]
    x0, y0 = vda_crop_box(rw, rh, target=target, split=split)
    return resized, (x0, y0), (rw, rh)


def _crop(arr: np.ndarray, x0: int, y0: int, target: int) -> np.ndarray:
    return arr[y0:y0 + target, x0:x0 + target]


class VDASemanticsTartanAirDataset(VideoDepthClipDataset):
    ts2 = None   # opt-in TS2 sky/mask recipe: a data.tartanair_ts2.TS2Labeller (set by train.py from `ts2_sky: true`); None = the released rule

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_files, dep_files, video_id, start = self.clips[idx]
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]
        H, W = self.image_size
        assert H == W, "VDA-semantics mode requires a square target (e.g. 518x518)"

        frames, depths, masks, skies = [], [], [], []
        crop_xy = rwh = raw_wh = None
        for i in indices:
            raw_img = Image.open(img_files[i]).convert("RGB")
            if raw_wh is None:
                raw_wh = raw_img.size
            else:
                assert raw_img.size == raw_wh, (
                    "every frame in a clip must share the same raw resolution -- the crop "
                    "box is chosen once (from the first frame) and reused for the whole "
                    f"clip, so a size change mid-clip would silently mis-crop later frames: "
                    f"{video_id} frame {i} is {raw_img.size}, clip started at {raw_wh}"
                )
            resized, crop_xy_i, rwh_i = _vda_frame_and_crop(raw_img, H, self.split)
            if crop_xy is None:
                crop_xy, rwh = crop_xy_i, rwh_i
            x0, y0 = crop_xy
            frames.append(torch.from_numpy(_crop(resized, x0, y0, H).copy()).permute(2, 0, 1))

            d_raw = _decode_depth(str(dep_files[i]))  # (H0, W0) metres; inf for sky
            d_resized = vda_resize_depth_like(np.nan_to_num(d_raw, nan=0.0, posinf=1e9), *rwh)
            if self.ts2 is None:       # released rule (default; byte-identical)
                valid = np.isfinite(d_raw) & (d_raw > 0) & (d_raw < self.max_depth)
                sky = ~np.isfinite(d_raw) | (d_raw > 1.0e4)
            else:                      # opt-in TS2 (data/tartanair_ts2.py): seg-based sky / uncertain at RAW resolution; masks resized below exactly as released
                sky, valid, _unc = self.ts2.label(dep_files[i], d_raw)
            valid_r = vda_resize_depth_like(valid.astype(np.float32), *rwh)
            sky_r = vda_resize_depth_like(sky.astype(np.float32), *rwh)
            d_c = _crop(np.clip(d_resized, 0.0, self.max_depth), x0, y0, H)
            depths.append(torch.from_numpy(d_c.copy()).unsqueeze(0).float())
            masks.append(torch.from_numpy(_crop(valid_r, x0, y0, H).copy()).unsqueeze(0))
            skies.append(torch.from_numpy(_crop(sky_r, x0, y0, H).copy()).unsqueeze(0))

        return {
            "frames": torch.stack(frames, 1).contiguous(),
            "depth": torch.stack(depths, 1),
            "mask": torch.stack(masks, 1),
            "sky": torch.stack(skies, 1),
            "video_id": video_id,
            "start_frame": start,
        }


class VDASemanticsVKITTIDataset(VKITTIDataset):
    # DN-01 D0: optional metric cap. None (default) masks only the 655 m sky sentinel, exactly as
    # every relative run was trained; a value additionally drops d >= cap from `mask` (depth untouched).
    vkitti_max_depth: float | None = None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_files, dep_files, sid, start = self.clips[idx]
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]
        H, W = self.image_size
        assert H == W, "VDA-semantics mode requires a square target (e.g. 518x518)"

        frames, depths, masks, skies = [], [], [], []
        crop_xy = rwh = raw_wh = None
        for i in indices:
            raw_img = Image.open(rgb_files[i]).convert("RGB")
            if raw_wh is None:
                raw_wh = raw_img.size
            else:
                assert raw_img.size == raw_wh, (
                    "every frame in a clip must share the same raw resolution -- the crop "
                    "box is chosen once (from the first frame) and reused for the whole "
                    f"clip, so a size change mid-clip would silently mis-crop later frames: "
                    f"vkitti/{sid} frame {i} is {raw_img.size}, clip started at {raw_wh}"
                )
            resized, crop_xy_i, rwh_i = _vda_frame_and_crop(raw_img, H, self.split)
            if crop_xy is None:
                crop_xy, rwh = crop_xy_i, rwh_i
            x0, y0 = crop_xy
            frames.append(torch.from_numpy(_crop(resized, x0, y0, H).copy()).permute(2, 0, 1))

            d_raw = _decode_vkitti_depth(str(dep_files[i]))
            valid = np.isfinite(d_raw) & (d_raw > 0) & (d_raw < _SKY_DEPTH_M)
            if self.vkitti_max_depth is not None:
                valid &= d_raw < self.vkitti_max_depth
            sky = d_raw >= _SKY_DEPTH_M
            d_hi = _SKY_DEPTH_M
            if getattr(self, "sky_as_valid_depth", False):   # Add.44 VK655 (default off = byte-identical): the 65535 sentinel is a VALID 655.35 m target, sky=0
                valid = np.isfinite(d_raw) & (d_raw > 0)
                if self.vkitti_max_depth is not None:
                    valid &= d_raw < self.vkitti_max_depth
                sky = np.zeros_like(sky)
                d_hi = _SKY_VALID_DEPTH_M
            d_resized = vda_resize_depth_like(np.clip(d_raw, 0.0, d_hi), *rwh)
            valid_r = vda_resize_depth_like(valid.astype(np.float32), *rwh)
            sky_r = vda_resize_depth_like(sky.astype(np.float32), *rwh)
            depths.append(torch.from_numpy(_crop(d_resized, x0, y0, H).copy()).unsqueeze(0).float())
            masks.append(torch.from_numpy(_crop(valid_r, x0, y0, H).copy()).unsqueeze(0))
            skies.append(torch.from_numpy(_crop(sky_r, x0, y0, H).copy()).unsqueeze(0))

        return {
            "frames": torch.stack(frames, 1).contiguous(),
            "depth": torch.stack(depths, 1),
            "mask": torch.stack(masks, 1),
            "sky": torch.stack(skies, 1),
            "video_id": f"vkitti/{sid}",
            "start_frame": start,
        }


class VDASemanticsIRSDataset(IRSDataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_files, dep_files, video_id, start = self.clips[idx]
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]
        H, W = self.image_size
        assert H == W, "VDA-semantics mode requires a square target (e.g. 518x518)"

        frames, depths, masks = [], [], []
        crop_xy = rwh = raw_wh = None
        for i in indices:
            raw_img = Image.open(img_files[i]).convert("RGB")
            if raw_wh is None:
                raw_wh = raw_img.size
            else:
                assert raw_img.size == raw_wh, (
                    "every frame in a clip must share the same raw resolution -- the crop "
                    "box is chosen once (from the first frame) and reused for the whole "
                    f"clip, so a size change mid-clip would silently mis-crop later frames: "
                    f"{video_id} frame {i} is {raw_img.size}, clip started at {raw_wh}"
                )
            resized, crop_xy_i, rwh_i = _vda_frame_and_crop(raw_img, H, self.split)
            if crop_xy is None:
                crop_xy, rwh = crop_xy_i, rwh_i
            x0, y0 = crop_xy
            frames.append(torch.from_numpy(_crop(resized, x0, y0, H).copy()).permute(2, 0, 1))

            d_raw = _read_exr_depth(dep_files[i])
            valid = np.isfinite(d_raw) & (d_raw > 0) & (d_raw < self.max_depth)
            d_resized = vda_resize_depth_like(np.clip(d_raw, 0.0, self.max_depth), *rwh)
            valid_r = vda_resize_depth_like(valid.astype(np.float32), *rwh)
            depths.append(torch.from_numpy(_crop(d_resized, x0, y0, H).copy()).unsqueeze(0).float())
            masks.append(torch.from_numpy(_crop(valid_r, x0, y0, H).copy()).unsqueeze(0))

        mask = torch.stack(masks, 1)
        return {
            "frames": torch.stack(frames, 1).contiguous(),
            "depth": torch.stack(depths, 1),
            "mask": mask,
            "sky": torch.zeros_like(mask),
            "video_id": video_id,
            "start_frame": start,
        }


class VDASemanticsPointOdysseyDataset(PointOdysseyDataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_files, dep_files, video_id, start = self.clips[idx]
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]
        H, W = self.image_size
        assert H == W, "VDA-semantics mode requires a square target (e.g. 518x518)"

        frames, depths, masks = [], [], []
        crop_xy = rwh = raw_wh = None
        for i in indices:
            raw_img = Image.open(img_files[i]).convert("RGB")
            if raw_wh is None:
                raw_wh = raw_img.size
            else:
                assert raw_img.size == raw_wh, (
                    "every frame in a clip must share the same raw resolution -- the crop "
                    "box is chosen once (from the first frame) and reused for the whole "
                    f"clip, so a size change mid-clip would silently mis-crop later frames: "
                    f"{video_id} frame {i} is {raw_img.size}, clip started at {raw_wh}"
                )
            resized, crop_xy_i, rwh_i = _vda_frame_and_crop(raw_img, H, self.split)
            if crop_xy is None:
                crop_xy, rwh = crop_xy_i, rwh_i
            x0, y0 = crop_xy
            frames.append(torch.from_numpy(_crop(resized, x0, y0, H).copy()).permute(2, 0, 1))

            raw = cv2.imread(str(dep_files[i]), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(f"Could not read PO depth: {dep_files[i]}")
            d_raw = raw.astype(np.float32) * _PO_DEPTH_SCALE
            valid = np.isfinite(d_raw) & (d_raw > 0) & (d_raw < self.max_depth)
            d_resized = vda_resize_depth_like(np.clip(d_raw, 0.0, self.max_depth), *rwh)
            valid_r = vda_resize_depth_like(valid.astype(np.float32), *rwh)
            depths.append(torch.from_numpy(_crop(d_resized, x0, y0, H).copy()).unsqueeze(0).float())
            masks.append(torch.from_numpy(_crop(valid_r, x0, y0, H).copy()).unsqueeze(0))

        mask = torch.stack(masks, 1)
        return {
            "frames": torch.stack(frames, 1).contiguous(),
            "depth": torch.stack(depths, 1),
            "mask": mask,
            "sky": torch.zeros_like(mask),
            "video_id": video_id,
            "start_frame": start,
        }


class VDASemanticsPseudoGTDataset(PseudoGTVideoDataset):
    """K80 option B: the pseudo-GT real-video source (YouTube-VOS / BVI-AOM with released Metric-VDA-L pseudo-depth, ``<depth_root>/<seq>/depth/depth_*.npz``) under VDA semantics: the same short-edge
    resize + one random square crop per clip as every other VDA-semantics source (the legacy PseudoGTVideoDataset STRETCHES to 518 x 518), the pseudo depth resized like the other sources' depth, valid =
    finite and 0 < depth < max_depth. No sky label (a pseudo-label has none). Degradation, severity cap, clean / blur-only / static draws: the shared mixin below, as for the synthetic sources."""

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_files, dep_files, video_id, start = self.clips[idx]
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]
        H, W = self.image_size
        assert H == W, "VDA-semantics mode requires a square target (e.g. 518x518)"

        frames, depths, masks = [], [], []
        crop_xy = rwh = raw_wh = None
        for i in indices:
            raw_img = Image.open(img_files[i]).convert("RGB")
            if raw_wh is None:
                raw_wh = raw_img.size
            else:
                assert raw_img.size == raw_wh, (
                    "every frame in a clip must share the same raw resolution -- the crop "
                    "box is chosen once (from the first frame) and reused for the whole "
                    f"clip, so a size change mid-clip would silently mis-crop later frames: "
                    f"{video_id} frame {i} is {raw_img.size}, clip started at {raw_wh}"
                )
            resized, crop_xy_i, rwh_i = _vda_frame_and_crop(raw_img, H, self.split)
            if crop_xy is None:
                crop_xy, rwh = crop_xy_i, rwh_i
            x0, y0 = crop_xy
            frames.append(torch.from_numpy(_crop(resized, x0, y0, H).copy()).permute(2, 0, 1))

            d_raw = np.load(dep_files[i])["depth"].astype(np.float32)     # (H0, W0) metres
            valid = np.isfinite(d_raw) & (d_raw > 0) & (d_raw < self.max_depth)
            d_resized = vda_resize_depth_like(np.clip(np.nan_to_num(d_raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, self.max_depth), *rwh)
            valid_r = vda_resize_depth_like(valid.astype(np.float32), *rwh)
            depths.append(torch.from_numpy(_crop(d_resized, x0, y0, H).copy()).unsqueeze(0).float())
            masks.append(torch.from_numpy(_crop(valid_r, x0, y0, H).copy()).unsqueeze(0))

        mask = torch.stack(masks, 1)
        return {
            "frames": torch.stack(frames, 1).contiguous(),
            "depth": torch.stack(depths, 1),
            "mask": mask,
            "sky": torch.zeros_like(mask),
            "video_id": video_id,
            "start_frame": start,
        }


class _VDADegradedMixin:
    """Adds ``degraded_frames`` to a clean ``VDASemantics*Dataset``'s already-assembled
    result, AFTER resize+crop -- matrix item 3 (Opus/Fable, 2026-08-28). Mixed in ahead
    of the clean class in each ``VDASemanticsDegraded*Dataset`` below (MRO: this class's
    ``__getitem__`` runs first, calls ``super().__getitem__`` to get the clean parent's
    full result, then degrades ``result["frames"]`` in place) so none of the four
    source-specific resize/crop/depth-decode implementations are duplicated.

    Same rng/severity/mode selection as every existing degraded dataset class
    (``video_clip_dataset.VideoDepthClipDataset``, ``VKITTIDataset``, ``IRSDataset``,
    ``PointOdysseyDataset``): deterministic per-clip seed for val (comparable metrics
    across runs), free-running RNG + ``self.severity_max`` curriculum cap for train.
    ``self.degrade``/``self.degrade_mode``/``self.degrade_severity``/``self.severity_max``
    are already set by the clean parent's own ``__init__`` (inherited unchanged, same
    constructor kwargs every non-VDA degraded class already accepts) -- this mixin adds
    no new constructor logic, only the post-resize degradation step."""

    clean_prob = 0.0   # set per dataset by train.py (data.clean_prob), before workers fork
    blur_only_prob = 0.0   # data.blur_only_prob (DN-01 R2): awgn_blur clips kept blurred but noise-free
    blur_disk_prob = 0.0   # data.blur_disk_prob (DN-01 proxy screen): blurred awgn_blur clips use a disk PSF
    lowlight = None        # data.lowlight {ev: [lo, hi], kappa} (DN-01 final-v2 'lowlight' mode)
    seeded_noise = False   # data.seeded_noise: train noise seeded from torch's RNG (reproducible)
    # DN-01 WS2 arm S (K49; data.static_prob / static_noise_mult / static_ev_jitter): this fraction of TRAIN clips
    # becomes ONE still frame + its GT repeated T times, each frame degraded independently (_degrade_static_clip).
    # The coin comes from its OWN RNG, so every non-static clip is the control's clip (tested).
    static_prob = 0.0
    # DN-01 recipe screen R7 (data.static_exempt_shares, default False = byte-identical): a static clip is exempt from the clean_prob
    # share (sev = 0 all run) and the blur_only_prob share (noise dropped), so EVERY static clip carries independent per-frame noise.
    static_exempt_shares = False
    # E4 arm c (data.static_lowlight_only, default False = byte-identical): a static clip's degrade mode is forced to 'lowlight' AFTER the normal mode draw (so every non-static clip's RNG
    # stream is untouched). The static dose is unchanged; the overall lowlight share of the mix rises by the static clips that would have been awgn / awgn_blur.
    static_lowlight_only = False
    static_noise_mult = STATIC_NOISE_MULT
    static_ev_jitter = STATIC_EV_JITTER
    # K80 option B (per-source metric supervision; the source spec's ``metric_supervision``): None (default) emits nothing, so every other config's batches are byte-identical; True / False emits
    # ``metric_sup`` (1.0 / 0.0) on EVERY clip of the source, and train.py's SiLog term then runs only on the clips with 1.0 (SSI / gradient / TGM stay on every source).
    metric_supervision = None
    # METRICFIX per-source SiLog far ceiling (the source spec's ``silog_max_depth``, m): None (default) emits nothing (byte-identical); a float (+inf = no ceiling) emits ``silog_ceil`` on EVERY clip of the source and
    # train.py's SiLog term drops the pixels with GT >= that value for these clips only.
    silog_max_depth = None
    # protocol v3 WP13 (the source spec's ``sparse_gt``): None (default) emits nothing (byte-identical); True / False emits ``sparse`` (1.0 / 0.0) on EVERY clip of the source, so a mixed dense + sparse-LiDAR batch carries a
    # per-SAMPLE flag through the ordinary collate and train.py's VDA loss applies the sparse-safe masked SSI to the flagged samples only (dense sources set ``sparse_gt: false``).
    sparse_gt = None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = super().__getitem__(idx)   # type: ignore[misc]
        if self.sparse_gt is not None:
            result["sparse"] = torch.tensor(1.0 if self.sparse_gt else 0.0)
        if self.metric_supervision is not None:
            result["metric_sup"] = torch.tensor(1.0 if self.metric_supervision else 0.0)
        if self.silog_max_depth is not None:
            result["silog_ceil"] = torch.tensor(float(self.silog_max_depth))
        if not self.degrade:                # type: ignore[attr-defined]
            return result
        frames = result["frames"]           # (3, T, H, W), clean, AT TARGET RESOLUTION
        static_rng = None                   # the static coin's own stream (never the degradation rng)
        if self.split != "train":           # type: ignore[attr-defined]
            rng = np.random.RandomState(self.seed + idx)   # type: ignore[attr-defined]
        elif self.seeded_noise:             # DN-01 (opt-in): torch's RNG is seeded per DataLoader worker from
            seed0 = int(torch.randint(0, 2 ** 31 - 1, ()).item())   # the run's seed
            rng, static_rng = np.random.RandomState(seed0), np.random.RandomState(seed0 + 1)
        else:                               # default: OS entropy -- NOT reproducible across runs
            rng, static_rng = np.random.RandomState(), np.random.RandomState()
        is_static = bool(static_rng is not None and self.static_prob > 0
                         and static_rng.uniform() < self.static_prob)
        sev = self.degrade_severity         # type: ignore[attr-defined]
        sev_b = None                        # awgn_blur only: blur severity, drawn independently of the noise's
        disk = False
        mode = (rng.choice(self.degrade_mode) if isinstance(self.degrade_mode, list)  # type: ignore[attr-defined]
               else self.degrade_mode)      # type: ignore[attr-defined]
        if is_static and self.static_lowlight_only:       # E4 arm c (see the class attribute)
            mode = "lowlight"
        if sev is None and self.split == "train" and mode != "elvis":  # type: ignore[attr-defined]
            s_max = float(self.severity_max[0])   # type: ignore[attr-defined]
            sev = float(rng.uniform(0.0, s_max))
            exempt = is_static and self.static_exempt_shares   # R7: a static clip keeps its noise (no clean / blur-only draw)
            if self.clean_prob > 0 and not exempt and rng.uniform() < self.clean_prob:   # DN-01: keep a clean fraction (s=0) all run
                sev = 0.0
            if mode == "awgn_blur":         # drawn last, so every other mode's RNG stream is unchanged
                sev_b = 0.0 if sev == 0 else float(rng.uniform(0.0, s_max))
                # DN-01 R2 (opt-in): blur only ever comes with noise above, so noise-free blur (the b3s0 cell)
                # is ~absent from training; this keeps the blur and drops the noise for a fraction of clips.
                # No extra draw at 0, so the old RNG stream is byte-identical.
                if self.blur_only_prob > 0 and sev_b > 0 and not exempt and rng.uniform() < self.blur_only_prob:
                    sev = 0.0
                # Proxy screen (opt-in): drawn last, and only when on, so the RNG stream at 0 is byte-identical.
                disk = self.blur_disk_prob > 0 and sev_b > 0 and rng.uniform() < self.blur_disk_prob
            elif mode == "lowlight":        # final-v2: the blur severity drawn as awgn_blur's (no blur-only / disk)
                sev_b = 0.0 if sev == 0 else float(rng.uniform(0.0, s_max))
        evs = None                          # static lowlight clips: each frame's exposure in stops
        if is_static:                       # ONE still: the middle frame and every per-frame GT tensor, x T
            T = frames.shape[1]
            for k, v in list(result.items()):
                if torch.is_tensor(v) and v.ndim == 4 and v.shape[1] == T:
                    result[k] = v[:, T // 2:T // 2 + 1].expand(-1, T, -1, -1).contiguous()
            frames = result["frames"]
            degraded, params, evs = _degrade_static_clip(
                frames.permute(1, 0, 2, 3), rng, mode, sev, sev_b, disk, self.lowlight,
                self.static_noise_mult, self.static_ev_jitter)
        else:
            degraded, params = _degrade_clip(frames.permute(1, 0, 2, 3), rng, mode, severity=sev, blur_severity=sev_b,
                                             blur_disk=disk, lowlight=self.lowlight)
        result["degraded_frames"] = degraded.permute(1, 0, 2, 3).contiguous()
        result["degrade_params"] = torch.from_numpy(params)
        if self.static_prob > 0:            # regime on: the collate key-set contract (train.py sets it on every leaf);
            result["static"] = torch.tensor(1.0 if is_static else 0.0)   # off: the control's key set, unchanged
        # Opt-in exposure-equivariant target (data.lowlight.target 'exposed'; provenance/dn01_lowlight_defect/): a
        # darkened lowlight clip's target is the clean frame at the SAME exposure (params[2] = 2^ev), so the restorer
        # is never asked to infer global exposure from local patches (which taught it to lift shadows). A static
        # clip's target follows each frame's own (jittered) exposure.
        if mode == "lowlight" and (self.lowlight or {}).get("target") == "exposed" and float(params[0]) > 0:
            from .elvis_noise import lowlight_reference
            clean = frames.permute(1, 0, 2, 3).numpy()
            if evs is None:
                ref = lowlight_reference(clean, float(np.log2(np.float64(params[2]))))
            else:
                ref = np.concatenate([lowlight_reference(clean[t:t + 1], float(evs[t])) for t in range(len(evs))])
            result["frames"] = torch.from_numpy(ref).permute(1, 0, 2, 3).contiguous()
        return result


class VDASemanticsDegradedTartanAirDataset(_VDADegradedMixin, VDASemanticsTartanAirDataset):
    pass


class VDASemanticsDegradedVKITTIDataset(_VDADegradedMixin, VDASemanticsVKITTIDataset):
    pass


class VDASemanticsDegradedIRSDataset(_VDADegradedMixin, VDASemanticsIRSDataset):
    pass


class VDASemanticsDegradedPointOdysseyDataset(_VDADegradedMixin, VDASemanticsPointOdysseyDataset):
    pass


class VDASemanticsDegradedPseudoGTDataset(_VDADegradedMixin, VDASemanticsPseudoGTDataset):
    pass


class VDASemanticsVideoOnlyDataset(VideoOnlyDataset):
    """DIST-01 arm (c), 2026-09-18: unlabeled REAL video under VDA semantics.

    Same short-edge resize + one-crop-per-clip transform as the four synthetic
    VDA-semantics classes above (``_vda_frame_and_crop`` / ``_crop``), applied to RGB only --
    there is no depth, mask or sky to resize. ``depth``/``mask``/``sky`` come back all-zero at
    the target resolution, so every GT loss sees an empty mask and only the online teacher
    supervises these clips. See ``data/video_only_dataset.py`` for the full key contract."""

    def _load_clip(self, img_files, indices, video_id, crop_xy=None):
        """Resize+crop one clip's RGB. ``crop_xy=None`` picks the box from the first frame
        (random train / centre val); passing one reuses it (the paired-twin subclass)."""
        H, W = self.image_size
        assert H == W, "VDA-semantics mode requires a square target (e.g. 518x518)"

        frames = []
        raw_wh = None
        for i in indices:
            raw_img = Image.open(img_files[i]).convert("RGB")
            if raw_wh is None:
                raw_wh = raw_img.size
            else:
                assert raw_img.size == raw_wh, (
                    "every frame in a clip must share the same raw resolution -- the crop "
                    "box is chosen once (from the first frame) and reused for the whole "
                    f"clip, so a size change mid-clip would silently mis-crop later frames: "
                    f"{video_id} frame {i} is {raw_img.size}, clip started at {raw_wh}"
                )
            resized, crop_xy_i, _ = _vda_frame_and_crop(raw_img, H, self.split)
            if crop_xy is None:
                crop_xy = crop_xy_i
            x0, y0 = crop_xy
            frames.append(torch.from_numpy(_crop(resized, x0, y0, H).copy()).permute(2, 0, 1))
        return torch.stack(frames, 1).contiguous(), crop_xy

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_files, _, video_id, start = self.clips[idx]
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]
        stacked, _ = self._load_clip(img_files, indices, video_id)
        return {"frames": stacked, **self._empty_gt(stacked),
                "video_id": video_id, "start_frame": start}


class VDASemanticsDegradedVideoOnlyDataset(_VDADegradedMixin, VDASemanticsVideoOnlyDataset):
    pass


class VDASemanticsPairedVideoDataset(VDASemanticsVideoOnlyDataset):
    """DIST-01 arm (e), 2026-09-20: REAL paired bright/dark video (BVI-RLV), no GT.

    ``rgb_root/<seq>/<frame>`` is the bright twin and ``dark_root/<seq>/<frame>`` the
    registered dark twin (same sequence and file names; BVI-RLV pairs measured at <= 1 px
    offset). ``frames`` is the bright clip, ``degraded_frames`` the REAL dark clip under the
    SAME crop box -- no synthetic degradation is applied. ``degrade_params`` is all -1: the
    sentinel ``train.py`` reads (``online_teacher.input: degraded_or_bright_twin``) to show
    the teacher the bright twin for exactly these clips."""

    PAIRED_SENTINEL = -1.0

    def __init__(self, rgb_root: str, dark_root: str, **kwargs) -> None:
        super().__init__(rgb_root=rgb_root, **kwargs)
        self.dark_root = Path(dark_root)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_files, _, video_id, start = self.clips[idx]
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]
        bright, crop_xy = self._load_clip(img_files, indices, video_id)
        dark_files = [self.dark_root / video_id / f.name for f in img_files]
        dark, _ = self._load_clip(dark_files, indices, video_id, crop_xy=crop_xy)
        return {"frames": bright, **self._empty_gt(bright), "video_id": video_id, "start_frame": start,
                "degraded_frames": dark,
                "degrade_params": torch.full((5,), self.PAIRED_SENTINEL, dtype=torch.float32)}



class VDASemanticsClipListDataset(VDASemanticsVideoOnlyDataset):
    """DN-01 K46 WS2: the UNLABELED real degraded video stream (loss.rg_* arms), from a COMMITTED list
    (``provenance/dn01_ws2/unlabeled_v1.json``: ``{"sources": {name: {"sequences": [{"id", "dir", "frames"}]}}}``,
    ``dir`` relative to ``roots[name]``, a per-host path from the config). Frames are used as they are (real low light
    / night; no synthetic degradation, no clean target, no depth): train.py gives these clips ONLY L_rg. Every
    sequence's frame count must match the list (a host with a different copy fails loudly). Clips = ``num_frames``
    consecutive frames at every ``clip_stride``; VDA-semantics resize + one random 518^2 crop per clip. ``self.source``
    is each clip's source name (train.py samples the sources with equal mass)."""

    def __init__(self, list_path: str, roots: dict, num_frames: int = 32, clip_stride: int = 16,
                 image_size=(518, 518), split: str = "train") -> None:
        import json
        self.num_frames, self.frame_stride, self.image_size, self.split = num_frames, 1, tuple(image_size), split
        self.clips, self.source = [], []
        for name, src in sorted(json.loads(Path(list_path).read_text())["sources"].items()):
            for seq in src["sequences"]:
                files = sorted(p for p in (Path(roots[name]) / seq["dir"]).iterdir() if p.suffix == ".png")
                if len(files) != seq["frames"]:
                    raise FileNotFoundError(f"{roots[name]}/{seq['dir']}: {len(files)} frames, list {seq['frames']}")
                starts = range(0, len(files) - num_frames + 1, clip_stride)
                self.clips += [(files, None, seq["id"], s) for s in starts]
                self.source += [name] * len(starts)

"""ScanNet++ iPhone TRAIN source (protocol v3 WP14): the same iPhone streams as the eval loader (``data/scannetpp_dataset.py``: ``iphone/rgb.mkv`` 1920x1440 @ 60 fps + ``iphone/depth.bin`` LZ4 192x256 uint16 mm, ARKit
LiDAR, aligned 1:1) as a VDA-semantics TRAIN source, with the scene list taken from official split files and checked against the eval scenes.

``splits`` (default ``[nvs_sem_train]``) = the official split lists to train on; ``exclude_splits`` (default ``[nvs_sem_val, nvs_test_iphone]``, the ScanNet++ scenes our eval reads) must stay disjoint from the
training scenes, ASSERTED (a scene in both is an error, never silently dropped). A requested scene that is not on disk is skipped and recorded in ``self.missing`` (``allow_missing``) -- unlike the eval loader there is NO
fallback to "all local scenes". Clips stride the 60 fps video at ``frame_stride`` (default 6 = 10 Hz) over the whole video. Depth is LiDAR-dense-ish (192x256 upsampled by nearest): flag ``sparse_gt: false``; cap 10 m.
LOCAL STATE (2026-10-03): 0 of the 856 nvs_sem_train scenes are on disk (the 112 local scenes are nvs_sem_val 50 + sem_test 50 + nvs_test_iphone 12).
"""
from __future__ import annotations

import hashlib
import zlib
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from .scannetpp_dataset import _FRAME_BYTES, _H, _W, _index_depth_bin
from .v3_clip import V3ClipDataset

EVAL_SPLITS = ("nvs_sem_val", "nvs_test_iphone")
_REPO = Path(__file__).resolve().parents[1]


def read_scene_list(path: str, sha256: str | None = None) -> List[str]:
    """A committed scene list (one id per line; a repo-relative path resolves against the repo root). Its file sha256 must equal ``sha256`` when given (protocol v3 WP27 triage: the subset140 list is pinned)."""
    f = Path(path)
    f = f if f.is_absolute() else _REPO / f
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    if sha256 is not None and h != sha256:
        raise ValueError(f"ScanNet++ scene list {f}: sha256 {h} != pinned {sha256}")
    return sorted(set(f.read_text().split()))


def read_split(root: Path, name: str) -> List[str]:
    f = root / "splits" / f"{name}.txt"
    if not f.is_file():
        raise FileNotFoundError(f"ScanNet++ split list not found: {f}")
    return sorted(set(f.read_text().split()))


def decode_depth_frame(buf: bytes) -> np.ndarray:
    """One compressed depth.bin frame -> (192, 256) float32 metres, 0 = invalid. The decompressed payload size picks the dtype: 98,304 B = uint16 millimetres (the usual scenes, LZ4 or raw-deflate); 196,608 B = float32 metres (11 subset140 scenes, raw-deflate). Any other size raises."""
    try:
        import lz4.block
        raw = lz4.block.decompress(buf, uncompressed_size=_FRAME_BYTES)
    except Exception:
        raw = zlib.decompress(buf, wbits=-zlib.MAX_WBITS)
    if len(raw) == _FRAME_BYTES:
        return np.frombuffer(raw, np.uint16).reshape(_H, _W).astype(np.float32) / 1000.0
    if len(raw) == 2 * _FRAME_BYTES:
        d = np.frombuffer(raw, np.float32).reshape(_H, _W).copy()
        d[~(np.isfinite(d) & (d > 0))] = 0.0                       # same invalid convention as uint16 (0)
        return d
    raise ValueError(f"scannetpp depth frame decodes to {len(raw)} bytes; expected {_FRAME_BYTES} (uint16 mm) or {2 * _FRAME_BYTES} (float32 m)")


class ScanNetPPTrainDataset(V3ClipDataset):
    def __init__(self, root: str, splits: List[str] | None = None, exclude_splits: List[str] | None = None, allow_missing: bool = True,
                 extra_root: str | None = None, extra_scene_list: str | None = None, extra_scene_list_sha256: str | None = None, **kw) -> None:
        kw.setdefault("max_depth", 10.0)
        kw.setdefault("frame_stride", 6)
        super().__init__(**kw)
        import cv2
        root_p = Path(root)
        want = sorted({s for sp in (splits or ["nvs_sem_train"]) for s in read_split(root_p, sp)})
        excl = {s for sp in (exclude_splits if exclude_splits is not None else EVAL_SPLITS) for s in read_split(root_p, sp)}
        scene_root = {n: root_p for n in want}                       # scene -> the root it lives under
        if extra_root is not None:                                   # a SECOND root + a committed scene list (the nvs_sem_train subset140, WP27 triage); same eval-disjointness assert, same loader
            assert extra_scene_list, "scannetpp_train: extra_root needs extra_scene_list"
            extra = read_scene_list(extra_scene_list, extra_scene_list_sha256)
            dup = sorted(set(extra) & set(want))
            assert not dup, f"ScanNet++ extra scene list overlaps the primary scenes: {dup[:5]} ({len(dup)})"
            scene_root.update({n: Path(extra_root) for n in extra})
            want = sorted(set(want) | set(extra))
            excl |= {s for sp in EVAL_SPLITS for s in read_split(Path(extra_root), sp)}
        clash = sorted(set(want) & excl)
        assert not clash, f"ScanNet++ training scenes overlap the eval scenes: {clash[:5]} ({len(clash)})"
        self._scene_root = scene_root
        self.scenes, self.missing = [], []
        for name in want:
            sc = scene_root[name] / "data" / name
            if not (sc / "iphone" / "depth.bin").exists() or not (sc / "iphone" / "rgb.mkv").exists():
                self.missing.append(name)
                continue
            spans = _index_depth_bin(sc / "iphone" / "depth.bin")
            cap = cv2.VideoCapture(str(sc / "iphone" / "rgb.mkv"))
            n = min(len(spans), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
            self._spans = getattr(self, "_spans", {})
            self._spans[name] = spans
            self.scenes.append(name)
            self.add_video(name, list(range(n)), list(range(n)))     # frame indices stand in for files (load_clip below)
        self.root = root_p
        if self.missing and not allow_missing:
            raise FileNotFoundError(f"scannetpp_train: {len(self.missing)} of {len(want)} scene(s) not on disk, e.g. {self.missing[:3]}")
        if not self.clips:
            raise FileNotFoundError(f"scannetpp_train: no usable scene ({len(self.missing)} of {len(want)} requested scenes are not under {root}/data)")

    def _depth(self, scene: str, frame: int) -> np.ndarray:
        off, n = self._spans[scene][frame]
        with open(self._scene_root[scene] / "data" / scene / "iphone" / "depth.bin", "rb") as f:
            f.seek(off)
            buf = f.read(n)
        return decode_depth_frame(buf)

    def load_clip(self, idx: int, indices: List[int]):
        import cv2
        _a, _b, scene, _start = self.clips[idx]
        first, last = indices[0], indices[-1]
        cap = cv2.VideoCapture(str(self._scene_root[scene] / "data" / scene / "iphone" / "rgb.mkv"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        want, frames, depths = set(indices), [], []
        for j in range(first, last + 1):
            ok, fr = cap.read()
            if j in want:
                if not ok:
                    raise IOError(f"{scene}: cannot decode frame {j}")
                frames.append(Image.fromarray(fr[..., ::-1].copy()))
        cap.release()
        # bring RGB (INTER_AREA) and the 192x256 LiDAR depth (nearest) straight to VDA's short-edge target size (a 1440x1920 float depth per frame x 32 frames would be ~350 MB); the shared resize + crop is then an identity
        from .vda_resize import vda_resize_size
        w0, h0 = frames[0].size
        tw, th = vda_resize_size(w0, h0, self.image_size[0])
        frames = [Image.fromarray(cv2.resize(np.asarray(f), (tw, th), interpolation=cv2.INTER_AREA)) for f in frames]
        for i in indices:
            depths.append(cv2.resize(self._depth(scene, i), (tw, th), interpolation=cv2.INTER_NEAREST))
        return frames, depths

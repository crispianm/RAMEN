"""ScanNet++ iPhone test loader — real-sensor indoor video depth (zero-shot reporting).

Each scene ships `iphone/rgb.mkv` (1920x1440 @ 60 fps) and `iphone/depth.bin`: per-frame
LZ4-block streams with 4-byte little-endian size headers, each decompressing to a
192x256 uint16 depth map in millimetres from the ARKit LiDAR pipeline, aligned 1:1 with
the video frames (per the official toolkit's prepare_iphone_data.py; a few dataset
variants use whole-file raw-deflate zlib instead — handled as fallback).

Scenes are restricted to the official ``nvs_sem_val`` split when present (held-out).
Clips stride the 60 fps video at ``frame_stride`` (default 6 → 10 Hz, KITTI cadence),
spread evenly across each video. Depth is nearest-resized; GT capped at ``max_depth``
(10 m — LiDAR range). Degradation matches every other test set (_degrade_clip).
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .video_clip_dataset import _degrade_clip

_H, _W = 192, 256
_FRAME_BYTES = _H * _W * 2


def _index_depth_bin(path: Path) -> List[Tuple[int, int]]:
    """Walk the header chain once → [(byte_offset, compressed_size)] per frame."""
    spans, off, end = [], 0, path.stat().st_size
    with open(path, "rb") as f:
        while off < end:
            n = int.from_bytes(f.read(4), "little")
            off += 4
            spans.append((off, n))
            f.seek(n, 1)
            off += n
    return spans


class ScanNetPPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        num_frames: int = 8,
        image_size: Tuple[int, int] = (518, 518),
        split: str = "nvs_sem_val",
        max_depth: float = 10.0,
        frame_stride: int = 6,
        max_clips_per_video: int = 3,
        seed: int = 42,
        degrade: bool = True,
        degrade_mode="full",
        degrade_severity: Optional[float] = None,
        **_ignore,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.image_size = tuple(image_size)
        self.max_depth = max_depth
        self.frame_stride = frame_stride
        self.degrade = degrade
        self.degrade_mode = degrade_mode
        self.degrade_severity = degrade_severity
        self.severity_max = torch.ones(1).share_memory_()
        self.seed = int(seed)

        root_p = Path(root)
        scenes = sorted(p for p in (root_p / "data").iterdir()
                        if (p / "iphone" / "depth.bin").exists())
        split_f = root_p / "splits" / f"{split}.txt"
        if split_f.exists():
            keep = set(split_f.read_text().split())
            held = [p for p in scenes if p.name in keep]
            if held:
                scenes = held
        if not scenes:
            raise FileNotFoundError(f"No ScanNet++ iphone scenes under {root}")

        import cv2
        self.clips: List[Tuple[Path, int]] = []   # (scene_dir, start_frame)
        self._spans: Dict[str, List[Tuple[int, int]]] = {}
        span = (num_frames - 1) * frame_stride + 1
        for sc in scenes:
            spans = _index_depth_bin(sc / "iphone" / "depth.bin")
            cap = cv2.VideoCapture(str(sc / "iphone" / "rgb.mkv"))
            n = min(len(spans), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
            if n < span:
                continue
            self._spans[sc.name] = spans
            starts = np.linspace(0, n - span, max_clips_per_video).astype(int)
            self.clips += [(sc, int(s)) for s in sorted(set(starts.tolist()))]

    def __len__(self) -> int:
        return len(self.clips)

    def _read_depth(self, scene: Path, indices: List[int]) -> np.ndarray:
        spans = self._spans[scene.name]
        out = np.empty((len(indices), _H, _W), np.float32)
        with open(scene / "iphone" / "depth.bin", "rb") as f:
            for i, fi in enumerate(indices):
                off, n = spans[fi]
                f.seek(off)
                buf = f.read(n)
                try:
                    import lz4.block
                    raw = lz4.block.decompress(buf, uncompressed_size=_FRAME_BYTES)
                except Exception:
                    raw = zlib.decompress(buf, wbits=-zlib.MAX_WBITS)
                out[i] = np.frombuffer(raw, np.uint16).reshape(_H, _W).astype(np.float32) / 1000.0
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import cv2
        scene, start = self.clips[idx]
        T, stride = self.num_frames, self.frame_stride
        indices = [start + i * stride for i in range(T)]

        cap = cv2.VideoCapture(str(scene / "iphone" / "rgb.mkv"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        rgb = []
        for j in range(indices[-1] - start + 1):
            ok, fr = cap.read()
            if not ok:
                fr = rgb[-1] if rgb else np.zeros((1440, 1920, 3), np.uint8)
            if j % stride == 0:
                rgb.append(fr)
        cap.release()
        rgb = np.stack(rgb[:T])                                        # (T,H,W,3) BGR
        frames = torch.from_numpy(rgb[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0

        H, W = self.image_size
        frames = F.interpolate(frames, size=(H, W), mode="bilinear", align_corners=False)
        depth = torch.from_numpy(self._read_depth(scene, indices))[:, None]  # (T,1,h,w)
        depth = F.interpolate(depth, size=(H, W), mode="nearest")
        valid = (depth > 1e-3) & (depth < self.max_depth)

        result = {
            "frames": frames.permute(1, 0, 2, 3).contiguous(),         # (3,T,H,W)
            "depth": depth.permute(1, 0, 2, 3).contiguous(),           # (1,T,H,W)
            "mask": valid.float().permute(1, 0, 2, 3).contiguous(),
            "video_id": f"scannetpp/{scene.name}",
        }
        if self.degrade:
            rng = np.random.default_rng(self.seed * 100_003 + idx)
            sev = self.degrade_severity
            degraded, _ = _degrade_clip(frames, rng, self.degrade_mode, severity=sev)
            result["degraded_frames"] = degraded.permute(1, 0, 2, 3).contiguous()
        return result

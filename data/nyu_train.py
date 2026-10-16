"""NYUv2 TRAIN source (protocol v3 WP14): the raw-Eigen train shards (47,584 RGB-D frames, 291 scene dirs over ``train-0000NN/train/<scene>/%05d.h5``; ``rgb`` (3,480,640) uint8, ``depth`` (480,640) float32 metres, dense
filled depth). A scene's frames are ordered by filename and its shards merged (some scenes span two shards); clips are ``num_frames`` consecutive files (the files are every 5th Kinect frame, ~6 Hz). Indoor cap
``max_depth`` 10 m. The official 654-image test set (the ``val-*`` shards) is never read; scene / frame disjointness from it is audited by scripts/dn01_v3_nyu_overlap.py (WP14 RUNLOG) and pinned in tests.
Needs h5py (as the NYU eval loader).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from .v3_clip import V3ClipDataset

# VDA's NYUv2 VIDEO test scenes (the 8 raw scenes of data/TEST/nyuv2_raw_video; session_briefs/.../runs/WP_downloads/nyu.sh). Some of them (basement_0001a, cafe_0001a, kitchen_0003, ...) are ALSO scene dirs of the HF train
# shards, so they are excluded from training, together with the other captures of the same room (basement_0001b / _0001c: a trailing letter marks a re-capture of one room).
NYU_VIDEO_TEST_SCENES = ("basement_0001a", "bookstore_0001a", "cafe_0001a", "classroom_0001a", "kitchen_0003", "office_0004", "playroom_0002", "study_0002")


def _room(name: str) -> str:
    return name[:-1] if name[-1:].isalpha() and name[-2:-1].isdigit() else name


def video_test_exclusions(names, test_scenes=NYU_VIDEO_TEST_SCENES) -> set:
    rooms = {_room(s) for s in test_scenes}
    return {n for n in names if n in set(test_scenes) or _room(n) in rooms}


def nyu_train_scenes(root: str | Path) -> Dict[str, List[Path]]:
    """{scene: [h5 files sorted by filename]} over every ``train-*/train/<scene>/*.h5`` under ``root`` (shards merged)."""
    scenes: Dict[str, List[Path]] = {}
    for shard in sorted(Path(root).glob("train-*")):
        if not shard.is_dir():
            continue
        for sc in sorted((shard / "train").glob("*")):
            if sc.is_dir():
                scenes.setdefault(sc.name, []).extend(sc.glob("*.h5"))
    return {k: sorted(v, key=lambda p: p.name) for k, v in sorted(scenes.items())}


class NYUTrainDataset(V3ClipDataset):
    def __init__(self, root: str, val_fraction: float = 0.0, split_seed: int | None = None, exclude_video_test: bool = True, **kw) -> None:
        kw.setdefault("max_depth", 10.0)
        super().__init__(**kw)
        scenes = nyu_train_scenes(root)
        if not scenes:
            raise FileNotFoundError(f"nyu_train: no train-*/train/<scene>/*.h5 under {root} (untar train-000005.tar?)")
        names = sorted(scenes)
        self.excluded = sorted(video_test_exclusions(names)) if exclude_video_test else []
        names = [n for n in names if n not in set(self.excluded)]
        assert not set(names) & set(NYU_VIDEO_TEST_SCENES) or not exclude_video_test
        if val_fraction > 0:
            rng = random.Random(self.seed if split_seed is None else split_seed)
            shuf = list(names)
            rng.shuffle(shuf)
            n_val = max(1, int(len(shuf) * val_fraction))
            names = sorted(shuf[:n_val] if self.split == "val" else shuf[n_val:])
        elif self.split == "val":
            raise ValueError("nyu_train: split='val' needs val_fraction > 0 (the default trains on all scenes)")
        self.scenes = names
        for n in names:
            self.add_video(n, scenes[n], scenes[n])
        if not self.clips:
            raise FileNotFoundError(f"nyu_train: no scene under {root} has >= {self.num_frames} frames")

    def read_rgb(self, path) -> Image.Image:
        import h5py
        with h5py.File(path, "r") as h:
            return Image.fromarray(np.ascontiguousarray(np.asarray(h["rgb"]).transpose(1, 2, 0)))

    def read_depth(self, path) -> np.ndarray:
        import h5py
        with h5py.File(path, "r") as h:
            return np.asarray(h["depth"], np.float32)

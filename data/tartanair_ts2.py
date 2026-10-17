"""TS2 TartanAir sky / valid-mask recipe (opt-in; the released loader rule is untouched).

Logic of record: ``outdoor_debug/parts/TA_sky/ts_v2_draft.py::ts_v2_guarded`` (ported unchanged in semantics; the
test suite checks equality against the draft). Masks are computed at RAW resolution from the integer seg (uint8 class
ids as stored in the tartanairpy seg zips: ``<env>/<diff>/<traj>/seg_<cam>/<frame>_<cam>_seg.png``, single channel);
the caller then resizes the resulting FLOAT masks exactly as the released code does. Class ids are never interpolated.

Per-env flags come from ONE frozen table (``outdoor_debug/parts/TA_sky/ts2_env_table.json``, built by
``make_ts2_env_table.py``). A missing env or a missing seg file raises: there is no fallback to the released rule.
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("opencv-python is required to read the TartanAir seg PNGs.") from e

SAT_M = 1.0e4
FAR_M = 9.0e3     # far-dome threshold (median seg-sky stored depth per env >= 9000 m)
DEFAULT_TABLE = Path(__file__).resolve().parents[1] / "outdoor_debug" / "parts" / "TA_sky" / "ts2_env_table.json"


def ts2_sky_valid(depth, seg, sky_ids, far_dome, seg_sky_ok=True, sat_nonsky_is_sky=False, max_depth=1.0e4,
                  fallback_sat_is_sky=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (sky, mask, uncertain) bool arrays at the input resolution. See ts_v2_guarded for the rule."""
    d = np.asarray(depth, np.float32)
    seg = np.asarray(seg)
    assert seg.shape == d.shape, f"seg shape {seg.shape} != depth shape {d.shape}"
    ids = set(int(i) for i in sky_ids)
    sat = ~np.isfinite(d) | (d >= SAT_M)
    seg_sky = np.isin(seg, list(ids - {0})) if ids else np.zeros(d.shape, bool)
    seg_sky |= (seg == 0) & sat & (0 in ids)
    zero_finite = (seg == 0) & ~sat & (0 in ids)       # id 0 with finite depth: uncertain
    far_ok = sat | (np.nan_to_num(d, nan=0.0, posinf=1e9) >= FAR_M)
    if not seg_sky_ok:
        sky = np.zeros(d.shape, bool); unc = seg_sky.copy()
    elif far_dome:
        sky = seg_sky & far_ok; unc = seg_sky & ~far_ok
    else:
        sky = seg_sky.copy(); unc = np.zeros(d.shape, bool)
    nonsky_sat = sat & ~seg_sky
    if sat_nonsky_is_sky or (fallback_sat_is_sky and not ids):
        sky |= nonsky_sat
    else:
        unc |= nonsky_sat
    unc |= zero_finite
    mask = np.isfinite(d) & (d > 0) & (d < max_depth) & ~sky & ~unc
    return sky, mask, unc


def load_env_table(path=None) -> Dict[str, dict]:
    p = Path(path) if path else DEFAULT_TABLE
    if not p.is_file():
        raise FileNotFoundError(f"TS2 env table not found: {p}")
    t = json.load(open(p))
    if t.get("schema") != "ts2_env_table/1":
        raise ValueError(f"{p}: unexpected schema {t.get('schema')!r}")
    return t["envs"]


def table_sha256(path=None) -> str:
    return hashlib.sha256(open(Path(path) if path else DEFAULT_TABLE, "rb").read()).hexdigest()


class TS2Labeller:
    """Reads the seg for a depth file and returns TS2 (sky, mask, uncertain) at raw resolution.
    The open zip handles are cached per process (keyed by pid, so DataLoader workers never share a file offset)."""

    def __init__(self, seg_root, env_table=None, max_depth: float = 1.0e4):
        self.seg_root = Path(seg_root)
        if not self.seg_root.is_dir():
            raise FileNotFoundError(f"TS2 seg_root not found: {self.seg_root}")
        self.table = load_env_table(env_table)
        self.max_depth = float(max_depth)
        self._zips: Dict[Tuple[int, str], zipfile.ZipFile] = {}

    def __getstate__(self):
        s = dict(self.__dict__); s["_zips"] = {}
        return s

    @staticmethod
    def split_depth_path(dep_path) -> Tuple[str, str, str, str, str]:
        """root/<env>/<diff>/<traj>/depth_<cam>/<frame>_<cam>_depth.png -> (env, diff, traj, cam, frame stem)."""
        p = Path(dep_path)
        dep_dir = p.parent.name
        if not dep_dir.startswith("depth_") or not p.stem.endswith("_depth"):
            raise ValueError(f"not a TartanAir depth path: {dep_path}")
        cam = dep_dir[len("depth_"):]
        traj, diff, env = p.parent.parent.name, p.parent.parent.parent.name, p.parent.parent.parent.parent.name
        return env, diff, traj, cam, p.stem[:-len("_depth")]

    def _zip(self, path: Path) -> zipfile.ZipFile:
        key = (os.getpid(), str(path))
        z = self._zips.get(key)
        if z is None:
            if not path.is_file():
                raise FileNotFoundError(f"TS2 seg zip missing: {path}")
            z = self._zips[key] = zipfile.ZipFile(path)
        return z

    def read_seg(self, dep_path) -> np.ndarray:
        env, diff, traj, cam, stem = self.split_depth_path(dep_path)
        z = self._zip(self.seg_root / env / diff / f"seg_{cam}.zip")
        member = f"{env}/{diff}/{traj}/seg_{cam}/{stem}_seg.png"
        try:
            raw = z.read(member)
        except KeyError:
            raise FileNotFoundError(f"TS2 seg frame missing in zip: {member}") from None
        seg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if seg is None or seg.ndim != 2 or seg.dtype != np.uint8:
            raise ValueError(f"TS2 seg {member}: expected single-channel uint8, got "
                             f"{None if seg is None else (seg.shape, seg.dtype)}")
        return seg

    def flags(self, env: str) -> dict:
        if env not in self.table:
            raise KeyError(f"TS2: env {env!r} is not in the env table ({len(self.table)} envs); refusing to fall back to the released rule")
        return self.table[env]

    def label(self, dep_path, d_raw) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        env = self.split_depth_path(dep_path)[0]
        f = self.flags(env)
        seg = self.read_seg(dep_path)
        assert seg.shape == d_raw.shape, f"TS2: seg shape {seg.shape} != depth shape {d_raw.shape} ({dep_path})"
        return ts2_sky_valid(d_raw, seg, f["sky_ids"], f["far_dome"], f["seg_sky_ok"], f["sat_nonsky_is_sky"],
                             self.max_depth, f["fallback_sat_is_sky"])

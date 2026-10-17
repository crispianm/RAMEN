"""Argoverse 2 (AV2) Sensor VAL held-out loader for the TS2+H300 confirmatory run (see AV2_REGISTRATION.md).

STATUS: DRAFT. The pure functions (selection rule, pose chain, projection, z-buffer, mask store) are unit-tested on SYNTHETIC data only.
The REAL-DATA branch (Log class: feather/jpg reading) has NEVER been executed: no AV2 payload has been opened.  [UNVERIFIED until data exists]

Conventions (official docs, argoverse.github.io/user-guide/datasets/sensor.html and av2/datasets/sensor/av2_sensor_dataloader.py as quoted by a docs fetch):
  * <log>/sensors/cameras/ring_front_center/<timestamp_ns>.jpg ; <log>/sensors/lidar/<timestamp_ns>.feather (x,y,z,intensity,laser_number,offset_ns);
    <log>/city_SE3_egovehicle.feather ; <log>/calibration/{egovehicle_SE3_sensor,intrinsics}.feather.
  * Timestamps are integer nanoseconds (file stems). Cameras 20 Hz, LiDAR 10 Hz; the two 32-beam LiDARs are merged into one sweep, motion-compensated to the
    sweep's egovehicle reference timestamp, points in the EGOVEHICLE frame.
  * city_SE3_egovehicle (qw,qx,qy,qz,tx_m,ty_m,tz_m) maps egovehicle-frame points to the city frame; the API takes the pose at EXACTLY the lidar timestamp
    and EXACTLY the camera timestamp (RuntimeError if absent) and chains: p_ego_cam_t = inv(city_SE3_ego(t_cam)) * city_SE3_ego(t_lidar) * p_ego_lidar_t.
  * egovehicle_SE3_sensor gives ego_SE3_cam (camera -> egovehicle); camera frame x right, y down, z forward (z = depth). [UNVERIFIED: axis convention taken from the standard AV2 PinholeCamera]
  * ring_front_center is 2048 (H) x 1550 (W), PORTRAIT; all imagery is undistorted (docs) so distortion terms are NOT applied. [width/height are asserted against intrinsics.feather]
  * Depth = camera z, valid 0 < z <= 200 m, pixel index round(u), round(v), collisions keep the minimum depth (z-buffer): identical to ddad_lr_dataset.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "LR_DEV"))
import ddad_lr_dataset as D          # noqa: E402  registered DDAD loader: reuse quat_to_R / sparse_depth_from_cam_points (pure functions)

CAMERA = "ring_front_center"
NATIVE_HW = (2048, 1550)             # (H, W), portrait. Docs; asserted against intrinsics at load.
MAX_DEPTH_M = 200.0
BINS_M = [(0.0, 20.0), (20.0, 50.0), (50.0, 80.0), (80.0, 120.0), (120.0, 200.0)]   # [lo,hi) except the last, closed
# ---- selection rule (names only; AV2_REGISTRATION.md section "Frame rule")
LOG_STRIDE = 3                       # logs: sorted(all 150 val log ids)[::3] -> 50 logs
CLIP_CAM_STRIDE = 2                  # clip = every 2nd ring_front_center frame in ascending timestamp order (20 Hz -> 10 Hz)
SCORE_CLIP_STRIDE = 10               # scored = clip indices 0,10,20,... (within the first MAX_LEN clip frames)
MAX_LEN = 160                        # window limit as DDAD_LR_SPEC / bench_joint
MAX_PAIR_DT_NS = 25_000_000          # a scored frame needs a LiDAR sweep within 25 ms (Auditor O37); else dropped and counted
SHORT_SIDE = 378
AV2_SPEC = dict(max_depth=MAX_DEPTH_M, min_depth=0.0, max_len=MAX_LEN, short_side=SHORT_SIDE)
S3_VAL_PREFIX = "s3://argoverse/datasets/av2/sensor/val/"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- selection (pure; names only)
def select_logs(all_log_ids: list[str]) -> list[str]:
    ids = sorted(all_log_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate log ids")
    return ids[::LOG_STRIDE]


def nearest_sweep(cam_ts: int, lidar_ts_sorted: np.ndarray) -> tuple[int, int]:
    """(lidar_ts, |dt| ns) of the sweep nearest to cam_ts; ties go to the EARLIER sweep."""
    a = np.asarray(lidar_ts_sorted, np.int64)
    i = int(np.searchsorted(a, cam_ts))
    cands = [j for j in (i - 1, i) if 0 <= j < len(a)]
    best = min(cands, key=lambda j: (abs(int(a[j]) - cam_ts), int(a[j])))
    return int(a[best]), abs(int(a[best]) - cam_ts)


def select_frames(cam_ts: list[int], lidar_ts: list[int]) -> dict:
    """One log. Returns dict(clip=[cam_ts...] (<=MAX_LEN), scored=[(clip_idx, cam_ts, lidar_ts, dt_ns)], dropped=[(clip_idx, cam_ts, dt_ns|None)]).
    Rule: clip = sorted cam timestamps [::2][:MAX_LEN]; scored candidates = clip indices 0,10,20,...; each is paired with the nearest sweep; |dt| > 25 ms -> dropped."""
    cam = sorted(int(t) for t in cam_ts)
    lid = np.array(sorted(int(t) for t in lidar_ts), np.int64)
    clip = cam[::CLIP_CAM_STRIDE][:MAX_LEN]
    scored, dropped = [], []
    for i in range(0, len(clip), SCORE_CLIP_STRIDE):
        if len(lid) == 0:
            dropped.append((i, clip[i], None)); continue
        lt, dt = nearest_sweep(clip[i], lid)
        if dt <= MAX_PAIR_DT_NS:
            scored.append((i, clip[i], lt, dt))
        else:
            dropped.append((i, clip[i], dt))
    return dict(clip=clip, scored=scored, dropped=dropped)


def frame_id(log: str, cam_ts: int) -> str:
    return f"{log}:{int(cam_ts)}"


# ----------------------------------------------------------------------------- geometry (pure)
def se3(qw, qx, qy, qz, tx, ty, tz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = D.quat_to_R(qw, qx, qy, qz)
    T[:3, 3] = [tx, ty, tz]
    return T


def lidar_ego_to_camera(pts_ego_lidar_t: np.ndarray, city_T_ego_lidar_t: np.ndarray, city_T_ego_cam_t: np.ndarray, ego_T_cam: np.ndarray) -> np.ndarray:
    """Official chain: points (egovehicle frame at the lidar time) -> city -> egovehicle at the camera time -> camera. (N,3) in, (N,3) out."""
    T = np.linalg.inv(ego_T_cam) @ np.linalg.inv(city_T_ego_cam_t) @ city_T_ego_lidar_t
    return pts_ego_lidar_t @ T[:3, :3].T + T[:3, 3]


def K_from_row(fx, fy, cx, cy) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def sparse_gt(pts_cam: np.ndarray, K: np.ndarray, hw=NATIVE_HW) -> np.ndarray:
    """(H,W) float32 metres, NaN where no valid return; 0 < z <= 200, z-buffer."""
    return D.sparse_depth_from_cam_points(pts_cam, K, None, hw, MAX_DEPTH_M)


# ----------------------------------------------------------------------------- frozen sky-mask store
class MaskStore:
    """npz keyed by frame id ('<log>:<cam_ts>'): '<id>|certain' and '<id>|any' = packbits(bool native mask), '<id>|shape' = [H,W].
    certain = both segmenters sky with p>=0.9, eroded, no LiDAR return within 5 px (the PRIMARY sky set).
    any     = sky by EITHER segmenter at argmax (no threshold, no erosion; used ONLY to exclude pixels from the far-range guard)."""

    def __init__(self, path):
        self.path = Path(path)
        self._z = np.load(self.path)

    def get(self, fid: str) -> tuple[np.ndarray, np.ndarray]:
        h, w = (int(v) for v in self._z[f"{fid}|shape"])
        if (h, w) != NATIVE_HW:
            raise ValueError(f"mask {fid}: shape {(h, w)} != native {NATIVE_HW}")
        out = []
        for k in ("certain", "any"):
            out.append(np.unpackbits(self._z[f"{fid}|{k}"])[: h * w].reshape(h, w).astype(bool))
        if (out[0] & ~out[1]).any():
            raise ValueError(f"mask {fid}: certain-sky must be a subset of any-sky")
        return out[0], out[1]


def save_mask_store(path, masks: dict) -> None:
    """masks: {fid: (certain, any)} bool arrays. (Used by tests and by the future mask-generation script.)"""
    d = {}
    for fid, (c, a) in masks.items():
        d[f"{fid}|shape"] = np.array(c.shape, np.int64)
        d[f"{fid}|certain"] = np.packbits(c.ravel())
        d[f"{fid}|any"] = np.packbits(a.ravel())
    np.savez_compressed(path, **d)


# ----------------------------------------------------------------------------- REAL-DATA branch: UNTESTED [UNVERIFIED until data exists]
class Log:
    """One downloaded AV2 val log. NEVER executed so far (no data). Column names follow the AV2 feather files as documented; mismatches raise loudly."""

    def __init__(self, root, log_id: str):
        import pandas as pd                                   # pyarrow needed for read_feather (not verified installed)
        self.dir = Path(root) / log_id
        self.id = log_id
        self.poses = pd.read_feather(self.dir / "city_SE3_egovehicle.feather")
        ex = pd.read_feather(self.dir / "calibration" / "egovehicle_SE3_sensor.feather")
        it = pd.read_feather(self.dir / "calibration" / "intrinsics.feather")
        r = ex[ex["sensor_name"] == CAMERA]
        assert len(r) == 1, f"{log_id}: expected one {CAMERA} extrinsics row"
        r = r.iloc[0]
        self.ego_T_cam = se3(r.qw, r.qx, r.qy, r.qz, r.tx_m, r.ty_m, r.tz_m)
        i = it[it["sensor_name"] == CAMERA]
        assert len(i) == 1
        i = i.iloc[0]
        assert (int(i.height_px), int(i.width_px)) == NATIVE_HW, f"{log_id}: intrinsics image size {(i.height_px, i.width_px)} != {NATIVE_HW}"
        self.K = K_from_row(i.fx_px, i.fy_px, i.cx_px, i.cy_px)
        self._pose_by_ts = {int(t): k for k, t in enumerate(self.poses["timestamp_ns"].to_numpy())}

    def cam_timestamps(self) -> list[int]:
        return sorted(int(p.stem) for p in (self.dir / "sensors" / "cameras" / CAMERA).glob("*.jpg"))

    def lidar_timestamps(self) -> list[int]:
        return sorted(int(p.stem) for p in (self.dir / "sensors" / "lidar").glob("*.feather"))

    def city_T_ego(self, ts: int) -> np.ndarray:
        k = self._pose_by_ts.get(int(ts))
        if k is None:
            raise RuntimeError(f"{self.id}: no city_SE3_egovehicle pose at exactly {ts} (the official API raises too)")
        r = self.poses.iloc[k]
        return se3(r.qw, r.qx, r.qy, r.qz, r.tx_m, r.ty_m, r.tz_m)

    def sparse_gt(self, cam_ts: int, lidar_ts: int) -> np.ndarray:
        import pandas as pd
        sw = pd.read_feather(self.dir / "sensors" / "lidar" / f"{lidar_ts}.feather")
        pts = sw[["x", "y", "z"]].to_numpy(np.float64)
        pc = lidar_ego_to_camera(pts, self.city_T_ego(lidar_ts), self.city_T_ego(cam_ts), self.ego_T_cam)
        return sparse_gt(pc, self.K)

    def image_path(self, cam_ts: int) -> Path:
        return self.dir / "sensors" / "cameras" / CAMERA / f"{int(cam_ts)}.jpg"


def load_rgb(path) -> np.ndarray:
    return D.load_rgb(path)

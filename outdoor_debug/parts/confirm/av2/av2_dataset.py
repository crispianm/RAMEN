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


def _select_phase(cam, lid, phase):
    clip = cam[phase::CLIP_CAM_STRIDE][:MAX_LEN]
    scored, dropped = [], []
    for i in range(0, len(clip), SCORE_CLIP_STRIDE):
        if len(lid) == 0:
            dropped.append((i, clip[i], None)); continue
        lt, dt = nearest_sweep(clip[i], lid)
        if dt <= MAX_PAIR_DT_NS:
            scored.append((i, clip[i], lt, dt))
        else:
            dropped.append((i, clip[i], dt))
    return clip, scored, dropped


def select_frames(cam_ts: list[int], lidar_ts: list[int]) -> dict:
    """One log. Returns dict(clip=[cam_ts...] (<=MAX_LEN), scored=[(clip_idx, cam_ts, lidar_ts, dt_ns)], dropped=[(clip_idx, cam_ts, dt_ns|None)], phase=p).
    Rule (R1, Auditor O43): for phase p in {0,1}: clip_p = sorted cam timestamps [p::2][:MAX_LEN]; scored candidates = clip indices 0,10,20,...; each is paired with the
    nearest sweep; |dt| > 25 ms -> dropped.  p = the phase with MORE scored frames; ties -> 0."""
    cam = sorted(int(t) for t in cam_ts)
    lid = np.array(sorted(int(t) for t in lidar_ts), np.int64)
    r = [_select_phase(cam, lid, p) for p in (0, 1)]
    p = 1 if len(r[1][1]) > len(r[0][1]) else 0
    clip, scored, dropped = r[p]
    return dict(clip=clip, scored=scored, dropped=dropped, phase=p)


def frame_id(log: str, cam_ts: int) -> str:
    return f"{log}:{int(cam_ts)}"


# ----------------------------------------------------------------------------- registered inputs (frames tsv + pinned clip table + object list)
OBJ_PFX = "datasets/av2/sensor/val/"
# sha256 of the registered R1 tables (Auditor O45): the scorer and the mask builder refuse any other file.
REGISTERED_SHAS = dict(frames_sha="1207c93b73bcf2967643445e1a27d479e415a581b7aa08bfcce14f26db604e59",
                       clips_sha="3f57db92806fec2d7e559584af1234fbf290a51ea322750c007df334cb05c1ee",
                       objects_sha="9d577f318630b1056b08b109c079b54859971d2cbcaa4a0e21f484465e909f64")


def read_frames_tsv(path) -> list:
    out = []
    for ln in open(path):
        p = ln.rstrip("\n").split("\t")
        if len(p) == 5:
            out.append((p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4])))        # log, cam_ts, lidar_ts, dt_ns, clip_idx
    return out


def read_clips_tsv(path) -> dict:
    """{log: (phase, [cam_ts ...])} from the pinned clip table (log, phase, n_clip, comma-separated clip camera timestamps). Never from a directory glob."""
    out = {}
    for ln in open(path):
        if not ln.strip():
            continue
        log, phase, n, ts = ln.rstrip("\n").split("\t")
        clip = [int(t) for t in ts.split(",")]
        assert len(clip) == int(n) <= MAX_LEN, f"{log}: n_clip {n} != {len(clip)} or > {MAX_LEN}"
        assert clip == sorted(set(clip)), f"{log}: clip not strictly ascending"
        assert int(phase) in (0, 1)
        assert log not in out, f"duplicate log {log} in the clip table"
        out[log] = (int(phase), clip)
    return out


def load_registered_inputs(frames_path, clips_path, objects_path, root=None, registered=None, check_files=True) -> dict:
    """The ONLY source of the model clip and the scored frames. Checks, in order: sha256 of the three tables against `registered` (default REGISTERED_SHAS);
    every scored row: clip[clip_idx] == cam_ts (and the log has a clip); every clip jpg and every paired sweep appears in objects.tsv (key list of the R1 download);
    if check_files: every one of those files exists under root. Returns {log: dict(clip=[cam_ts], rows=[(log,cam,lidar,dt,idx)])}."""
    reg = REGISTERED_SHAS if registered is None else registered
    for name, p in (("frames_sha", frames_path), ("clips_sha", clips_path), ("objects_sha", objects_path)):
        got = sha256_file(p)
        if got != reg[name]:
            raise AssertionError(f"{name}: {got} != registered {reg[name]} ({p})")
    clips = read_clips_tsv(clips_path)
    keys = {ln.split("\t")[0] for ln in open(objects_path) if ln.strip()}
    by_log: dict = {}
    for r in read_frames_tsv(frames_path):
        by_log.setdefault(r[0], []).append(r)
    out = {}
    for log, rows in by_log.items():
        if log not in clips:
            raise AssertionError(f"{log}: scored frames but no row in the clip table")
        clip = clips[log][1]
        for (_, cam, lid, dt, idx) in rows:
            if not (0 <= idx < len(clip)) or clip[idx] != cam:
                raise AssertionError(f"{log}: clip[{idx}] != scored camera timestamp {cam}")
            if dt > MAX_PAIR_DT_NS:
                raise AssertionError(f"{log}:{cam}: dt {dt} ns > {MAX_PAIR_DT_NS}")
            k = f"{OBJ_PFX}{log}/sensors/lidar/{lid}.feather"
            if k not in keys:
                raise AssertionError(f"{k} not in objects.tsv")
        for t in clip:
            k = f"{OBJ_PFX}{log}/sensors/cameras/{CAMERA}/{t}.jpg"
            if k not in keys:
                raise AssertionError(f"{k} not in objects.tsv")
        if check_files:
            d = Path(root) / log
            miss = [str(d / "sensors" / "cameras" / CAMERA / f"{t}.jpg") for t in clip if not (d / "sensors" / "cameras" / CAMERA / f"{t}.jpg").is_file()]
            miss += [str(d / "sensors" / "lidar" / f"{r[2]}.feather") for r in rows if not (d / "sensors" / "lidar" / f"{r[2]}.feather").is_file()]
            if miss:
                raise FileNotFoundError(f"{log}: {len(miss)} registered files missing, e.g. {miss[0]}")
        out[log] = dict(clip=clip, rows=rows)
    return out


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


def project_all_returns(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N,2) float64 pixel coords (x=col, y=row) of ALL points with z > 0, NO range cap and NO image-bounds filter (the registered LiDAR exclusion of the
    sky mask applies to every return in front of the camera, not only the <=200 m GT returns)."""
    u, v, z, front = D.project_points(np.asarray(pts_cam, np.float64), K, None, NATIVE_HW)
    ok = front & np.isfinite(u) & np.isfinite(v)
    return np.stack([u[ok], v[ok]], 1)


# ----------------------------------------------------------------------------- frozen sky-mask store
class MaskStore:
    """npz keyed by frame id ('<log>:<cam_ts>'): '<id>|certain' and '<id>|any' = packbits(bool native mask), '<id>|shape' = [H,W].
    certain = both segmenters sky with p>=0.9, eroded, no projected LiDAR return within 9 px (round(5*1550/900); the PRIMARY sky set).
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
        self.xyz_dtypes: set = set()
        self._pose_by_ts = {int(t): k for k, t in enumerate(self.poses["timestamp_ns"].to_numpy())}

    def city_T_ego(self, ts: int) -> np.ndarray:
        k = self._pose_by_ts.get(int(ts))
        if k is None:
            raise RuntimeError(f"{self.id}: no city_SE3_egovehicle pose at exactly {ts} (the official API raises too)")
        r = self.poses.iloc[k]
        return se3(r.qw, r.qx, r.qy, r.qz, r.tx_m, r.ty_m, r.tz_m)

    def _pts_cam(self, cam_ts: int, lidar_ts: int) -> np.ndarray:
        """ALL returns of the paired sweep in the camera frame at cam_ts (no range cap, no image-bounds filter).
        The sweep's x,y,z columns may be stored as float16 (~0.125 m spacing at 128-256 m): they are cast to float64 here and the stored dtype is recorded in
        self.xyz_dtypes (disclosed in the mask meta and the scorer meta; no rule depends on it)."""
        import pandas as pd
        sw = pd.read_feather(self.dir / "sensors" / "lidar" / f"{lidar_ts}.feather")
        for c in ("x", "y", "z"):
            self.xyz_dtypes.add(str(sw[c].dtype))
        pts = sw[["x", "y", "z"]].to_numpy(np.float64)
        return lidar_ego_to_camera(pts, self.city_T_ego(lidar_ts), self.city_T_ego(cam_ts), self.ego_T_cam)

    def sparse_gt(self, cam_ts: int, lidar_ts: int) -> np.ndarray:
        return sparse_gt(self._pts_cam(cam_ts, lidar_ts), self.K)

    def all_returns_uv(self, cam_ts: int, lidar_ts: int) -> np.ndarray:
        """(N,2) float (x=col, y=row) of EVERY return in front of the camera (z > 0), whatever its range: used for the LiDAR exclusion of the sky mask (D12)."""
        return project_all_returns(self._pts_cam(cam_ts, lidar_ts), self.K)

    def image_path(self, cam_ts: int) -> Path:
        return self.image_path_of(self.dir.parent, self.id, cam_ts)

    @staticmethod
    def image_path_of(root, log_id: str, cam_ts: int) -> Path:
        return Path(root) / log_id / "sensors" / "cameras" / CAMERA / f"{int(cam_ts)}.jpg"


def load_rgb(path) -> np.ndarray:
    """AV2 jpg reader (own message; the DDAD loader's text says PNG and is not touched)."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"JPG decode failure: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

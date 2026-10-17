"""DDAD val long-range LiDAR DEV loader (report-only DEV, see REGISTRATION.md in this directory).

Interface mirrors bench_joint: ``DDAD_LR_SPEC`` is the analogue of a ``DATASETS`` entry, ``load_clips`` returns
``[(scene_key, [ {image, gt_depth, factor, ...}, ... ])]`` (one clip per scene, ALL CAMERA_01 frames in timestamp
order, windowed to ``max_len`` by the caller exactly as bench_joint does with ``frames[:max_len]``),
``load_frames`` returns the RGB clip ``(N,3,H,W)`` in [0,1], ``load_gt`` returns the per-frame GT stack.

Calibration conventions (dgp = TRI-ML Dataset Governance Policy, dgp/utils/camera.py ``Camera`` / ``Pose``, and
dgp/datasets/base_dataset.py; DDAD README):
  * ``datum.pose`` in scene_<sha>.json is the SENSOR-to-WORLD pose (dgp ``Pose`` = ``Twc`` / ``Twl``): p_world = R q + t,
    quaternion order (qw,qx,qy,qz), translation in metres. Both the LIDAR and the CAMERA_01 datum of the paired sample carry one.
  * The LIDAR npz (key ``data``, columns X,Y,Z,INTENSITY, float64, metres) is in the LIDAR sensor frame.
  * p_cam = inv(Twc) @ Twl @ p_lidar  (camera and LiDAR timestamps differ; the world poses absorb the ego motion).
  * calibration/<key>.json: per-sensor ``intrinsics`` (fx, fy, cx, cy, skew) and ``extrinsics`` (sensor-to-vehicle);
    the camera intrinsics come from here (index of 'CAMERA_01' in ``names``). The extrinsics are NOT needed for the
    world-pose chain; they are exposed for the consistency check (LIDAR extrinsic = identity, so vehicle = LIDAR frame).
  * Pinhole: u = fx*x/z + skew*y/z + cx, v = fy*y/z + cy; pixel coordinates have integer = pixel centre (OpenCV/dgp
    convention). DDAD calibrations carry no distortion terms (images are undistorted); if k1,k2,p1,p2,k3 keys are
    present they are applied with the OpenCV radial-tangential model, as dgp does through cv2.projectPoints.
  * Depth is the camera-frame z (z-depth), as in packnet-sfm DDAD evaluation.

Valid mask: projected, inside the native image, z > 0 and z <= 200 m. Sparse map at native 1936x1216; pixel collisions
keep the minimum depth (z-buffer); pixel index = round(u), round(v).

Resolution rule (the amendment to REGISTRATION.md; no GT resizing, ever): predictions (any resolution, depth in metres)
are bilinearly resized (align_corners=False) to the native 1936x1216 and sampled at the integer pixels of the valid sparse
mask. GT stays sparse at native resolution.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

NATIVE_HW = (1216, 1936)
MAX_DEPTH_M = 200.0
CAMERA = "CAMERA_01"
ROOT = Path(os.environ.get("DDAD_LR_ROOT", "/mnt/DATA/DegradedDepth/data/DEV_LR"))
SCENES_DIR = ROOT / "ddad_val" / "ddad_train_val"
PAIRS_TSV = ROOT / "ddad_val_pairs.tsv"
PAIRS_SHA256 = "962d499612c40bca7d314f8ff16ce1d6c7b03962ae78938f472a101766150245"
FRAMES_TSV = Path(__file__).resolve().parent / "ddad_val_frames.tsv"
FRAMES_SHA256 = "b8d9664fb0c28df0fab61ad71383d48660c5444c4538a8f471fd4bbc8f0e5098"
SANITY_FRAMES = [("000150", "15616458249936530"), ("000151", "15609732323309162"), ("000152", "15609722540809162"),
                 ("000153", "15616462546436512"), ("000154", "15609730485809162")]
# analogue of a bench_joint.DATASETS entry (short_side provisional, mirrors kitti; the scorer owns it). crop = native, no crop.
DDAD_LR_SPEC = dict(max_depth=MAX_DEPTH_M, min_depth=0.0, max_len=160, crop=(0, 1216, 0, 1936), short_side=378)
BINS_M = [(0, 20), (20, 50), (50, 80), (80, 120), (120, 200)]   # registered; [120,200] closed on the right


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def available() -> bool:
    return PAIRS_TSV.is_file() and SCENES_DIR.is_dir()


def read_pairs(check_sha: bool = True) -> dict[tuple[str, str], str]:
    """(scene, camera_ts) -> LIDAR path relative to the scene dir, for the 395 scored frames."""
    if check_sha and sha256_file(PAIRS_TSV) != PAIRS_SHA256:
        raise ValueError(f"{PAIRS_TSV}: sha256 differs from the registered {PAIRS_SHA256}")
    out = {}
    for ln in open(PAIRS_TSV):
        p = ln.rstrip("\n").split("\t")
        if len(p) == 3:
            out[(p[0], p[1])] = p[2]
    return out


def read_frame_list() -> list[tuple[str, str]]:
    if sha256_file(FRAMES_TSV) != FRAMES_SHA256:
        raise ValueError("ddad_val_frames.tsv sha256 differs from the registered one")
    return [tuple(ln.split("\t")) for ln in open(FRAMES_TSV).read().split("\n") if ln]


# ----------------------------------------------------------------------------- geometry

def quat_to_R(qw, qx, qy, qz) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_to_T(pose: dict) -> np.ndarray:
    """dgp pose dict {rotation:{qw,qx,qy,qz}, translation:{x,y,z}} -> 4x4 sensor-to-parent."""
    T = np.eye(4)
    r, t = pose["rotation"], pose["translation"]
    T[:3, :3] = quat_to_R(r["qw"], r["qx"], r["qy"], r["qz"])
    T[:3, 3] = [t["x"], t["y"], t["z"]]
    return T


def intrinsics_K(intr: dict) -> np.ndarray:
    return np.array([[intr["fx"], intr.get("skew", 0.0), intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]])


def project_points(pts_cam: np.ndarray, K: np.ndarray, dist: dict | None = None, hw=NATIVE_HW):
    """Pinhole (+ optional OpenCV radial-tangential) projection of camera-frame points (N,3).
    Returns (u, v, z, in_front) with integer-centred pixel coordinates (float)."""
    z = pts_cam[:, 2]
    front = z > 0
    zs = np.where(front, z, 1.0)
    x, y = pts_cam[:, 0] / zs, pts_cam[:, 1] / zs
    if dist:
        k1, k2, k3 = dist.get("k1", 0.0), dist.get("k2", 0.0), dist.get("k3", 0.0)
        p1, p2 = dist.get("p1", 0.0), dist.get("p2", 0.0)
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x, y = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = K[0, 0] * x + K[0, 1] * y + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return u, v, z, front


def sparse_depth_from_cam_points(pts_cam: np.ndarray, K: np.ndarray, dist: dict | None = None, hw=NATIVE_HW,
                                 max_depth: float = MAX_DEPTH_M) -> np.ndarray:
    """Sparse depth map (H,W) float32 in metres, NaN where no valid return. Valid = in front (z>0), projected inside the
    image (pixel index round(u), round(v) within bounds), 0 < z <= max_depth. Collisions keep the nearest point (z-buffer)."""
    H, W = hw
    u, v, z, front = project_points(pts_cam, K, dist, hw)
    with np.errstate(invalid="ignore"):
        iu, iv = np.rint(u), np.rint(v)
        ok = front & np.isfinite(u) & np.isfinite(v) & np.isfinite(z) & (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H) & (z > 0) & (z <= max_depth)
    flat = np.full(H * W, np.inf, np.float64)
    idx = iv[ok].astype(np.int64) * W + iu[ok].astype(np.int64)
    np.minimum.at(flat, idx, z[ok])
    out = np.where(np.isfinite(flat), flat, np.nan).astype(np.float32)
    return out.reshape(H, W)


def lidar_to_camera(pts_lidar: np.ndarray, T_w_lidar: np.ndarray, T_w_cam: np.ndarray) -> np.ndarray:
    T = np.linalg.inv(T_w_cam) @ T_w_lidar
    return pts_lidar @ T[:3, :3].T + T[:3, 3]


# ----------------------------------------------------------------------------- scene access

class Scene:
    """One DDAD val scene: scene json (pairing/poses) + calibration (CAMERA_01 intrinsics)."""

    def __init__(self, name: str, root: Path | None = None):
        self.name = name
        self.dir = Path(root or SCENES_DIR) / name
        sj = sorted(self.dir.glob("scene_*.json"))
        if len(sj) != 1:
            raise FileNotFoundError(f"{self.dir}: expected exactly one scene_*.json, found {len(sj)}")
        d = json.load(open(sj[0]))
        self._data = {x["key"]: x for x in d["data"]}
        self._samples = d["samples"]
        self._cam_to_sample: dict[str, dict] = {}
        for s in self._samples:
            for k in s["datum_keys"]:
                x = self._data[k]
                if x["id"]["name"] == CAMERA:
                    self._cam_to_sample[Path(x["datum"]["image"]["filename"]).stem] = s
        self.calib_key = self._samples[0]["calibration_key"]
        c = json.load(open(self.dir / "calibration" / f"{self.calib_key}.json"))
        i = c["names"].index(CAMERA)
        self.K = intrinsics_K(c["intrinsics"][i])
        self.dist = {k: c["intrinsics"][i][k] for k in ("k1", "k2", "k3", "p1", "p2") if k in c["intrinsics"][i]} or None
        self.T_vehicle_cam = pose_to_T(c["extrinsics"][i])
        self.T_vehicle_lidar = pose_to_T(c["extrinsics"][c["names"].index("LIDAR")])

    def cam_timestamps(self) -> list[str]:
        return sorted(p.stem for p in (self.dir / "rgb" / CAMERA).glob("*.png"))

    def sample_poses(self, cam_ts: str):
        """(T_world_cam, T_world_lidar, lidar_rel_path) of the sample containing camera frame ``cam_ts`` (scene json pairing)."""
        s = self._cam_to_sample[cam_ts]
        cam = lid = None
        for k in s["datum_keys"]:
            x = self._data[k]
            if x["id"]["name"] == CAMERA and Path(x["datum"]["image"]["filename"]).stem == cam_ts:
                cam = pose_to_T(x["datum"]["image"]["pose"])
            elif x["id"]["name"] == "LIDAR":
                lid = pose_to_T(x["datum"]["point_cloud"]["pose"])
                lpath = x["datum"]["point_cloud"]["filename"]
        return cam, lid, lpath

    def sparse_gt(self, cam_ts: str, lidar_rel: str | None = None) -> np.ndarray:
        T_wc, T_wl, lpath = self.sample_poses(cam_ts)
        pts = load_lidar_npz(self.dir / (lidar_rel or lpath))
        return sparse_depth_from_cam_points(lidar_to_camera(pts, T_wl, T_wc), self.K, self.dist)


def load_lidar_npz(path) -> np.ndarray:
    """(N,3) float64 metres in the LIDAR sensor frame (npz key 'data': X,Y,Z,INTENSITY)."""
    a = np.load(path)["data"]
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError(f"{path}: unexpected point cloud shape {a.shape}")
    return a[:, :3].astype(np.float64)


def load_rgb(path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"PNG decode failure: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------- bench_joint-style interface

def load_clips(limit: int | None = None) -> list[tuple[str, list[dict]]]:
    """One clip per val scene (split order = sorted scene names 000150..000199): ALL CAMERA_01 frames in timestamp order.
    Frame dict: image (path relative to SCENES_DIR), gt_depth (LIDAR npz path relative to SCENES_DIR or None), factor=1.0,
    scored (bool), ts. Only the 395 frames of ddad_val_frames.tsv are scored (GT); the rest are context frames."""
    pairs = read_pairs()
    scored = set(read_frame_list())
    clips = []
    for sc in sorted({s for s, _ in scored}):
        frames = []
        for ts in Scene(sc).cam_timestamps():
            is_sc = (sc, ts) in scored
            frames.append(dict(image=f"{sc}/rgb/{CAMERA}/{ts}.png", gt_depth=f"{sc}/{pairs[(sc, ts)]}" if is_sc else None,
                               factor=1.0, scored=is_sc, ts=ts, scene=sc))
        clips.append((sc, frames))
    return clips[:limit]


def load_frames(frames: list[dict], root: Path | None = None):
    """Clean clip (N,3,H,W) float in [0,1] at NATIVE resolution (the model-side resize belongs to the scorer), raw (H,W)."""
    import torch
    root = Path(root or SCENES_DIR)
    imgs = [load_rgb(root / f["image"]) for f in frames[:DDAD_LR_SPEC["max_len"]]]
    return torch.from_numpy(np.stack(imgs).astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous(), imgs[0].shape[:2]


_SCENE_CACHE: dict[str, Scene] = {}


def load_gt(frames: list[dict], root: Path | None = None) -> np.ndarray:
    """(N,H,W) float32 sparse metric GT at native resolution, NaN where invalid / not a scored frame (bench_joint's -1 is
    ``np.nan_to_num(g, nan=-1)``). Windowed to max_len like bench_joint."""
    out = np.full((len(frames[:DDAD_LR_SPEC["max_len"]]), *NATIVE_HW), np.nan, np.float32)
    for i, f in enumerate(frames[:DDAD_LR_SPEC["max_len"]]):
        if f["gt_depth"] is None:
            continue
        sc = _SCENE_CACHE.setdefault(f["scene"], Scene(f["scene"], root))
        out[i] = sc.sparse_gt(f["ts"], f["gt_depth"].split("/", 1)[1])
    return out


def sample_prediction(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resolution rule: bilinear-resize one prediction (h,w) (metres) to native 1936x1216, sample at the valid GT pixels.
    Returns (pred_at_valid, gt_at_valid)."""
    import torch
    import torch.nn.functional as F
    m = np.isfinite(gt) & (gt > 0)
    p = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32))[None, None]
    if tuple(p.shape[-2:]) != NATIVE_HW:
        p = F.interpolate(p, size=NATIVE_HW, mode="bilinear", align_corners=False)
    return p[0, 0].numpy()[m], gt[m]

"""DN-01 temporal metrics (user 2026-09-26: the paper's six metrics = AbsRel + delta1, PSNR + SSIM, TAE + tOF; Auditor
K38). Both temporal metrics are LOWER-is-better.

TAE -- Video Depth Anything's temporal alignment error, from the VERBATIM vendored vendor/vda_eval/eval_tae.py (VDA
  @4f5ae23, file e371c2b, sha256 8e481e94...). `tae_clip` reproduces `eval_TAE` in memory (tested equal on identical
  inputs): the model's disparity (1 / depth for metric models) gets ONE least-squares scale/shift per clip against
  1/(gt + 1e-8) on the GT-valid pixels (1e-3 < gt < cap); aligned disparity is clipped >= 1e-3, inverted to depth and
  clipped to [1e-3, cap]; for every consecutive pair (t, t+1) the PREDICTED depth of one frame is back-projected with K,
  moved by the GT relative pose inv(c2w_{t+1}) c2w_t (and the reverse), re-projected (rounded) and compared with the
  other frame's predicted depth: mean |d - d_proj| / d over the pixels both cover (no mask, as VDA's ScanNet setting);
  the clip value is the mean over the 2(N-1) directed pairs, x 100. Sparse LiDAR GT only enters the alignment fit.
  Note: upstream tae_torch writes colliding re-projected points with an index_put, which is thread-order dependent on a
  multi-threaded CPU (~1e-6 relative run to run); single-threaded it is deterministic (tested equal to eval_TAE).
  Poses are camera-to-world of the camera whose K is used, at the GT's resolution / crop (see the per-dataset helpers).
tOF -- TecoGAN's temporal optical-flow error on the RESTORED video vs the clean GT video:
  mean |OF(out_{t-1}, out_t) - OF(gt_{t-1}, gt_t)| over the pairs t = 1..N-1, all pixels and both flow channels (the clip
  value), with ONE fixed flow network for every model: torchvision raft_large, weights Raft_Large_Weights.C_T_SKHT_V2
  (sha256 ff5fadd5...), 12 flow updates, fp32, inputs in [0,1] mapped to [-1,1] (the weights' transform), at bench_joint's
  model resolution, replicate-padded to a multiple of 8 and cropped back.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RAFT_SHA256 = "ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322"


# ----------------------------------------------------------------------------- TAE

def tae_clip(pred: np.ndarray, metric: bool, gt: np.ndarray, K: np.ndarray, c2w: np.ndarray, max_depth: float,
             device: str = "cpu", gt_valid_only: bool = False) -> float:
    """``pred`` (N,H,W) model output at GT size (metres if ``metric``, else disparity); ``gt`` (N,H,W) metres (<= 0
    invalid); ``K`` (N,3,3) or (3,3) at GT size; ``c2w`` (N,4,4). Returns VDA's TAE x 100 (eval_TAE, in memory).
    ``gt_valid_only`` (sky-fix guard, DESCRIPTIVE; the reported TAE stays the all-pixel one): the same error evaluated only at pixels whose DESTINATION frame has valid GT (1e-3 < gt < cap), i.e. with no sky / no-return
    pixels (each directed pair: depth2's pixels under ``mask``); the default (False) is byte-identical to before."""
    from vendor.vda_eval.eval_tae import depth2disparity
    infs = (1.0 / np.clip(pred, 1e-3, None) if metric else pred).astype(np.float64)
    gts = gt.astype(np.float64)
    valid = (gts > 1e-3) & (gts < max_depth)
    gt_disp = 1.0 / (gts[valid].reshape(-1, 1) + 1e-8)
    infs = np.clip(infs, 1e-3, None)
    A = np.concatenate([infs[valid].reshape(-1, 1), np.ones((int(valid.sum()), 1))], axis=-1)
    scale, shift = np.linalg.lstsq(A, gt_disp, rcond=None)[0]
    aligned = np.clip(scale * infs + shift, 1e-3, None)
    pred_depth = np.clip(depth2disparity(aligned), 1e-3, max_depth)
    Ks = np.broadcast_to(K, (len(pred), 3, 3))
    err = 0.0
    threads = torch.get_num_threads()
    torch.set_num_threads(1)                 # deterministic index_put of colliding points (see the module docstring)
    try:
        err = _pairs(pred_depth, c2w, Ks, device, valid if gt_valid_only else None)
    finally:
        torch.set_num_threads(threads)
    return err / (2 * (len(pred) - 1)) * 100


def _pairs(pred_depth: np.ndarray, c2w: np.ndarray, Ks: np.ndarray, device: str, valid: np.ndarray | None = None) -> float:
    from vendor.vda_eval.eval_tae import tae_torch
    err = 0.0
    for i in range(len(pred_depth) - 1):
        T21 = np.linalg.inv(c2w[i + 1]) @ c2w[i]
        T12 = np.linalg.inv(T21)
        d1 = torch.from_numpy(np.ascontiguousarray(pred_depth[i])).to(device)
        d2 = torch.from_numpy(np.ascontiguousarray(pred_depth[i + 1])).to(device)
        ones = torch.ones_like(d1, dtype=torch.bool)
        m2 = ones if valid is None else torch.from_numpy(np.ascontiguousarray(valid[i + 1])).to(device)       # forward: depth2 = frame i+1
        m1 = ones if valid is None else torch.from_numpy(np.ascontiguousarray(valid[i])).to(device)           # reverse: depth2 = frame i
        err += float(tae_torch(d1, d2, torch.from_numpy(T21[:3, :3]).to(device), T21[:3, 3], Ks[i], m2))
        err += float(tae_torch(d2, d1, torch.from_numpy(T12[:3, :3]).to(device), T12[:3, 3], Ks[i], m1))
    return err


# ----------------------------------------------------------------------------- tOF

class TOF:
    """One fixed RAFT for every model (K38(b)); ``__call__(out, gt)`` -> tOF of one clip (lower is better)."""

    def __init__(self, device: str = "cuda", batch: int = 8, updates: int = 12):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        w = Raft_Large_Weights.C_T_SKHT_V2
        path = Path(torch.hub.get_dir()) / "checkpoints" / Path(w.url).name
        self.net = raft_large(weights=w, progress=False).to(device).eval()
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != RAFT_SHA256:
            raise RuntimeError(f"RAFT weights {path} sha256 {got} != pinned {RAFT_SHA256}")
        self.device, self.batch, self.updates = device, batch, updates

    @torch.no_grad()
    def flows(self, v: torch.Tensor) -> torch.Tensor:
        """(N,3,H,W) in [0,1] -> (N-1,2,H,W) flow of each consecutive pair (t-1 -> t)."""
        n, _, h, w = v.shape
        ph, pw = -h % 8, -w % 8
        x = F.pad(v.float().to(self.device), (0, pw, 0, ph), mode="replicate") * 2 - 1
        a, b = x[:-1], x[1:]                                               # the pairs (t-1, t)
        out = [self.net(a[s:s + self.batch], b[s:s + self.batch], num_flow_updates=self.updates)[-1]
               for s in range(0, n - 1, self.batch)]
        return torch.cat(out)[:, :, :h, :w]

    def __call__(self, out: torch.Tensor, gt: torch.Tensor) -> float:
        if out.shape[0] < 2:
            return float("nan")
        return float((self.flows(out) - self.flows(gt)).abs().mean())


# ----------------------------------------------------------------------------- poses (camera-to-world, K at GT size)

def _rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# Bonn RGB-D Dynamic (Palazzolo et al., IROS 2019), dataset page: groundtruth.txt is in the motion-capture file frame;
# the sensor pose is T_ROS^-1 T_file T_ROS T_m (T_m = sensor -> markers, as published, incl. its slight scale).
BONN_K = np.array([[542.822841, 0, 315.593520], [0, 542.576870, 237.756098], [0, 0, 1]])
BONN_T_ROS = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
BONN_T_M = np.array([[1.0157, 0.1828, -0.2389, 0.0113], [0.0009, -0.8431, -0.6413, -0.0098],
                     [-0.3009, 0.6147, -0.8085, 0.0111], [0, 0, 0, 1.0]])


def bonn_poses(scene_dir: Path, rgb_names: list[str]) -> np.ndarray:
    """c2w (N,4,4) of Bonn's RGB-D sensor for frames named rgb/<timestamp>.png: the nearest groundtruth.txt row."""
    rows = np.array([[float(v) for v in ln.split()] for ln in (scene_dir / "groundtruth.txt").read_text().splitlines()
                     if ln and not ln.startswith("#")])
    inv_ros = np.linalg.inv(BONN_T_ROS)
    out = []
    for name in rgb_names:
        ts = float(Path(name).stem)
        r = rows[np.abs(rows[:, 0] - ts).argmin()]
        T = _rt(quat_to_R(*r[4:8]), r[1:4])
        out.append(inv_ros @ T @ BONN_T_ROS @ BONN_T_M)
    return np.stack(out)


# KITTI raw (Geiger et al.): OXTS -> IMU-to-world (Mercator, first frame's scale), then camera 2 via the calibration
# chain (pykitti's convention): T_cam2_imu = T_cam2_cam0 . R_rect_00 . T_cam0_velo . T_velo_imu; c2w = T_w_imu . T_cam2_imu^-1.
def _read_calib(path: Path) -> dict:
    out = {}
    for ln in path.read_text().splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            try:
                out[k] = np.array([float(x) for x in v.split()])
            except ValueError:
                pass
    return out


def kitti_calib(calib_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(K of the rectified camera 2 (3,3), T_cam2_imu (4,4)) from calib_{cam_to_cam,velo_to_cam,imu_to_velo}.txt."""
    c2c = _read_calib(calib_dir / "calib_cam_to_cam.txt")
    v2c = _read_calib(calib_dir / "calib_velo_to_cam.txt")
    i2v = _read_calib(calib_dir / "calib_imu_to_velo.txt")
    P2 = c2c["P_rect_02"].reshape(3, 4)
    R0 = np.eye(4)
    R0[:3, :3] = c2c["R_rect_00"].reshape(3, 3)
    T_cam0_velo = _rt(v2c["R"].reshape(3, 3), v2c["T"])
    T_velo_imu = _rt(i2v["R"].reshape(3, 3), i2v["T"])
    T_cam2_cam0 = np.eye(4)
    T_cam2_cam0[0, 3] = P2[0, 3] / P2[0, 0]
    return P2[:, :3], T_cam2_cam0 @ R0 @ T_cam0_velo @ T_velo_imu


def kitti_oxts_poses(oxts_rows: list[np.ndarray]) -> np.ndarray:
    """T_w_imu (N,4,4) from OXTS rows (lat, lon, alt, roll, pitch, yaw, ...), pykitti's pose_from_oxts_packet."""
    er = 6378137.0
    scale = math.cos(oxts_rows[0][0] * math.pi / 180.0)
    out = []
    for r in oxts_rows:
        lat, lon, alt, roll, pitch, yaw = r[:6]
        tx = scale * lon * math.pi * er / 180.0
        ty = scale * er * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
        Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        out.append(_rt(Rz @ Ry @ Rx, np.array([tx, ty, alt])))
    return np.stack(out)


def kitti_poses(oxts_rows: list[np.ndarray], T_cam2_imu: np.ndarray) -> np.ndarray:
    """c2w (N,4,4) of rectified camera 2."""
    return kitti_oxts_poses(oxts_rows) @ np.linalg.inv(T_cam2_imu)

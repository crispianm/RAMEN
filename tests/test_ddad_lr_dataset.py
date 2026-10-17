"""Regression test for the DDAD val long-range LiDAR DEV loader (outdoor_debug/parts/LR_DEV/REGISTRATION.md).
Real-data checks touch ONLY the 5 registered sanity frames and assert units/mask/projection/calibration sanity, no other statistic."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[1] / "outdoor_debug" / "parts" / "LR_DEV" / "ddad_lr_dataset.py"
_spec = importlib.util.spec_from_file_location("ddad_lr_dataset", _P)
D = importlib.util.module_from_spec(_spec)
sys.modules["ddad_lr_dataset"] = D
_spec.loader.exec_module(D)

needs_data = pytest.mark.skipif(not D.available(), reason="DDAD val data / pairs file absent")


def _frame_ready(scene, ts):
    d = D.SCENES_DIR / scene
    return (D.available() and (d / "rgb" / D.CAMERA / f"{ts}.png").is_file() and any(d.glob("scene_*.json"))
            and (d / "calibration").is_dir() and (d / D.read_pairs(check_sha=False).get((scene, ts), "missing")).is_file())


# ---- synthetic (no data) ----
K = np.array([[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1.0]])
HW = (80, 100)


def test_synthetic_zbuffer_and_exclusions():
    pts = np.array([[0, 0, 10.0],      # centre pixel (50,40), z=10
                    [0, 0, 5.0],       # same pixel, nearer -> z-buffer keeps 5
                    [0, 0, 30.0],      # same pixel, farther
                    [0.04, 0, 20.0],    # u = 50.5 would be a tie; use 0.04 -> u=50.2 -> pixel 50 (collides, z=20 farther)
                    [1.0, 0, 20.0],    # u=55, v=40
                    [0, 0, 0.0],       # 0 m excluded
                    [0, 0, -3.0],      # behind camera excluded
                    [0, 0, 200.0],     # exactly 200 kept
                    [0.0, 0.1, 200.5], # >200 excluded (pixel (50,40.05)->40 but z>200)
                    [10.0, 0, 5.0],    # u=250 outside
                    [0, -10.0, 5.0]])  # v<0 outside
    d = D.sparse_depth_from_cam_points(pts, K, None, HW)
    assert d.shape == HW and d.dtype == np.float32
    assert d[40, 50] == 5.0                      # nearest wins over 10, 30, 20(0.1/20 -> same pixel) and the 200 m point
    assert d[40, 55] == 20.0
    valid = np.isfinite(d)
    assert valid.sum() == 2                       # only those two pixels
    assert np.nanmax(d) <= 200.0 and np.nanmin(d) > 0


def test_synthetic_200m_and_zero_only():
    pts = np.array([[0, 0, 0.0], [0, 0, 200.0], [1.0, 0, 200.0001], [-1.0, 0, 199.999]])
    d = D.sparse_depth_from_cam_points(pts, K, None, HW)
    assert np.isfinite(d).sum() == 2 and np.nanmax(d) == 200.0


def test_synthetic_pose_chain():
    # LIDAR 10 m ahead of the camera along camera z; camera and lidar share the world frame, camera pose = identity-with-axes
    T_wc = np.eye(4)
    T_wl = np.eye(4); T_wl[:3, 3] = [0, 0, 10.0]
    p = D.lidar_to_camera(np.array([[0.0, 0, 5.0]]), T_wl, T_wc)
    assert np.allclose(p, [[0, 0, 15.0]])
    # quaternion: 90 deg about z maps x->y
    R = D.quat_to_R(np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4))
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12) and np.allclose(R @ R.T, np.eye(3))


def test_synthetic_resolution_rule():
    gt = np.full(D.NATIVE_HW, np.nan, np.float32)
    gt[100, 200] = 50.0
    pred = np.full((608, 968), 25.0, np.float32)   # constant -> bilinear stays constant
    p, g = D.sample_prediction(pred, gt)
    assert p.shape == (1,) and g.shape == (1,) and np.isclose(p[0], 25.0) and g[0] == 50.0


# ---- real data: pairs + frame list integrity ----
@needs_data
def test_pairs_and_frames_sha256():
    assert D.sha256_file(D.PAIRS_TSV) == D.PAIRS_SHA256
    assert D.sha256_file(D.FRAMES_TSV) == D.FRAMES_SHA256
    pairs = D.read_pairs(); frames = D.read_frame_list()
    assert len(pairs) == 395 == len(frames) and set(pairs) == set(frames)
    assert D.SANITY_FRAMES == [(s, sorted(t for ss, t in frames if ss == s)[0]) for s, _ in D.SANITY_FRAMES]
    assert [s for s, _ in D.SANITY_FRAMES] == sorted({s for s, _ in frames})[:5]


# ---- real data: the 5 sanity frames ----
@pytest.mark.parametrize("scene,ts", D.SANITY_FRAMES)
def test_sanity_frame(scene, ts):
    if not _frame_ready(scene, ts):
        pytest.skip("sanity frame files not (yet) present")
    sc = D.Scene(scene)
    # decode
    img = D.load_rgb(sc.dir / "rgb" / D.CAMERA / f"{ts}.png")
    assert img.shape == (1216, 1936, 3) and img.dtype == np.uint8
    # intrinsics / extrinsics
    H, W = D.NATIVE_HW
    assert sc.K.shape == (3, 3) and sc.K[0, 0] > 0 and sc.K[1, 1] > 0
    assert 0 < sc.K[0, 2] < W and 0 < sc.K[1, 2] < H and sc.dist is None
    assert np.allclose(sc.T_vehicle_lidar, np.eye(4))
    R = sc.T_vehicle_cam[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
    # LIDAR decode + projection
    pairs = D.read_pairs()
    lrel = pairs[(scene, ts)]
    T_wc, T_wl, lpath = sc.sample_poses(ts)
    assert lpath == lrel                           # the pairs file agrees with the scene json
    pts = D.load_lidar_npz(sc.dir / lrel)
    assert pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0 and np.isfinite(pts).all()
    gt = sc.sparse_gt(ts, lrel)
    gt2 = sc.sparse_gt(ts, lrel)
    assert gt.shape == (H, W) and np.array_equal(gt, gt2, equal_nan=True)   # deterministic
    m = np.isfinite(gt)
    assert m.any() and (gt[m] > 0).all() and gt[m].max() <= D.MAX_DEPTH_M    # metres, positive, capped
    # projection lands inside the image by construction; the camera-frame points in front must exist
    pc = D.lidar_to_camera(pts, T_wl, T_wc)
    u, v, z, front = D.project_points(pc, sc.K)
    inside = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    assert inside.any()
    assert (np.rint(u[inside]) < W).sum() >= 1

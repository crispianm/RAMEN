"""DN-01 temporal metrics (evaluation/temporal_metrics.py; Auditor K38): our in-memory TAE equals VDA's vendored eval_TAE
on identical inputs for a real clip; zero cases; tOF == 0 when out == GT; the per-dataset pose conventions are sane."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation import temporal_metrics as tm

BONN = Path("/mnt/DATA/DegradedDepth/data/TEST/bonn")
KCAL = Path("/mnt/DATA/DegradedDepth/data/TEST/kitti_c/kitti_calib/2011_09_26")
KZIP = Path("/mnt/DATA/DegradedDepth/data/TEST/kitti_raw_val_drives/2011_09_26_drive_0002_sync.zip")


@pytest.mark.skipif(not BONN.is_dir(), reason="needs the Bonn TEST data")
def test_tae_equals_vda_eval_tae_on_a_real_bonn_clip(tmp_path, monkeypatch):
    import bench_joint as bj
    import vendor.vda_eval.eval_tae as vt
    from vendor.vda_eval.eval_tae import eval_TAE
    monkeypatch.setattr(vt, "device", torch.device("cpu"))   # upstream picks cuda if visible; GPU scatter is racy
    key, frames = bj.load_clips("bonn", "dev")[0]
    frames = frames[:6]
    gt = bj.load_gt("bonn", frames)                                          # (6,480,640) metres, invalid -1
    c2w = tm.bonn_poses(BONN / key, [f["image"] for f in frames])
    rng = np.random.RandomState(0)
    fill = np.where(gt > 0, gt, np.median(gt[gt > 0]))
    disp = (1.0 / fill) * (1 + 0.05 * rng.standard_normal(gt.shape))           # a noisy disparity "prediction"
    threads = torch.get_num_threads()
    torch.set_num_threads(1)      # VDA's tae_torch index_put with colliding pixels is thread-order dependent (~1e-6)
    ours = tm.tae_clip(disp.astype(np.float32), False, gt, tm.BONN_K, c2w, 10.0)
    inf_paths, gt_paths = [], []
    for i in range(len(frames)):
        inf_paths.append(str(tmp_path / f"inf{i}.npy"))
        gt_paths.append(str(tmp_path / f"gt{i}.npy"))
        np.save(inf_paths[-1], disp[i].astype(np.float32))
        np.save(gt_paths[-1], np.where(gt[i] > 0, gt[i], 0.0))
    args = Namespace(max_depth_eval=10.0, mask=False, hard_crop=False, a=0, b=None, c=0, d=None)
    ref = float(eval_TAE(inf_paths, gt_paths, [1.0] * len(frames), [], [tm.BONN_K] * len(frames), list(c2w), args))
    torch.set_num_threads(threads)
    assert ours == pytest.approx(ref, rel=1e-12) and ours > 0                  # single-threaded: identical


def test_tae_is_zero_for_a_static_pair_under_identity_motion():
    d = np.random.RandomState(0).uniform(1, 5, (2, 24, 32))
    d[1] = d[0]
    K = np.array([[20.0, 0, 16], [0, 20.0, 12], [0, 0, 1]])
    assert tm.tae_clip(d, True, d, K, np.stack([np.eye(4)] * 2), 10.0) == pytest.approx(0.0, abs=1e-9)


def test_tof_is_zero_when_out_equals_gt():
    tof = tm.TOF("cpu", updates=4)
    g = torch.Generator().manual_seed(0)
    v = torch.rand(3, 3, 128, 160, generator=g)                              # RAFT needs >= ~128 px
    assert tof(v, v) == 0.0
    assert tof((v + 0.2 * torch.rand(v.shape, generator=g)).clamp(0, 1), v) > 0


@pytest.mark.skipif(not BONN.is_dir(), reason="needs the Bonn TEST data")
def test_bonn_poses_are_smooth_rigid_camera_motion():
    import bench_joint as bj
    key, frames = bj.load_clips("bonn", "dev")[0]
    c2w = tm.bonn_poses(BONN / key, [f["image"] for f in frames[:10]])
    rel = [np.linalg.inv(c2w[i + 1]) @ c2w[i] for i in range(9)]
    assert all(np.linalg.norm(r[:3, 3]) < 0.1 for r in rel)                  # < 10 cm per frame at ~30 fps


@pytest.mark.skipif(not (BONN.is_dir() and KCAL.is_dir() and KZIP.is_file()), reason="needs Bonn + KITTI raw data")
def test_bench_joint_camera_gives_one_pose_per_frame_for_kitti_and_bonn_dev():
    import bench_joint as bj
    for ds in ("kitti", "bonn"):
        key, frames = bj.load_clips(ds, "dev")[0]
        frames = frames[:bj.DATASETS[ds]["max_len"]]
        K, c2w = bj.camera(ds, key, frames)
        assert K.shape == (3, 3) and c2w.shape == (len(frames), 4, 4) and np.isfinite(c2w).all()
        rel = np.linalg.inv(c2w[1]) @ c2w[0]
        assert np.linalg.norm(rel[:3, 3]) < 3.0


NUSC = Path("/mnt/DATA/DegradedDepth/RAMEN/results/nuscenes_manifest/out")


@pytest.mark.skipif(not (NUSC / "nuscenes_night_v1_poses.json").is_file(), reason="needs the nuScenes pose files")
@pytest.mark.parametrize("name", ["nuscenes_night_v1", "nuscenes_night_test_v1"])
def test_nuscenes_camera_poses_are_rigid_and_smooth(name):
    import json
    man = json.loads((NUSC / f"{name}.json").read_text())
    poses = json.loads((NUSC / f"{name}_poses.json").read_text())
    by_scene = {}
    for f in sorted(man["frames"], key=lambda r: r["timestamp_us"]):
        by_scene.setdefault(f["scene"], []).append(poses[f["camera_token"]])
    assert sum(len(v) for v in by_scene.values()) == len(man["frames"]) == len(poses)
    for recs in by_scene.values():
        c2w = np.array([r["c2w"] for r in recs])
        R = c2w[:, :3, :3]
        assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-6)            # rigid
        steps = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
        assert steps.max() < 15.0                                                      # 2 Hz keyframes, < 30 m/s
        K = np.array(recs[0]["K"])
        assert 1200 < K[0, 0] < 1300 and 780 < K[0, 2] < 850 and 450 < K[1, 2] < 520   # CAM_FRONT (~1266, 816, 492)


@pytest.mark.skipif(not (KCAL.is_dir() and KZIP.is_file()), reason="needs KITTI calib + a raw drive")
def test_kitti_poses_move_forward_along_camera_z():
    import zipfile
    K, T_cam2_imu = tm.kitti_calib(KCAL)
    assert 700 < K[0, 0] < 730 and 600 < K[0, 2] < 620
    with zipfile.ZipFile(KZIP) as z:
        names = sorted(n for n in z.namelist() if "/oxts/data/" in n and n.endswith(".txt"))[:6]
        rows = [np.array([float(v) for v in z.read(n).decode().split()]) for n in names]
    c2w = tm.kitti_poses(rows, T_cam2_imu)
    rel = [np.linalg.inv(c2w[i + 1]) @ c2w[i] for i in range(5)]             # frame t in frame t+1's camera
    moves = np.stack([r[:3, 3] for r in rel])
    assert (np.linalg.norm(moves, axis=1) < 3.0).all()                        # < 3 m per 0.1 s
    if np.linalg.norm(moves, axis=1).mean() > 0.2:                            # moving: mostly along the optical axis
        assert (np.abs(moves[:, 2]) > np.abs(moves[:, :2]).max(1)).all()

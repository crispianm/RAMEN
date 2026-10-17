"""Synthetic AV2-like world for the end-to-end tests (fake feathers, jpgs, S3 listing). NO AV2 data. Two logs, a 20 Hz camera, 10 Hz sweeps 10 ms after every even camera frame
(phase 0), 40 camera frames per log -> clip of 20, scored clip indices 0 and 10. Decoy (pre-R1-style) jpgs and sweeps are written to the data dir and are NOT in the listing/objects."""
import shutil, sys
from pathlib import Path
import numpy as np

AV2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AV2)); sys.path.insert(0, str(AV2.parent / "seg"))
import av2_dataset as A
import make_clip_tsv as MC
import select_av2_frames as SEL

LOGS = ["logA", "logB"]
BASE = 315969904000000000
STEP = 50_000_000
NCAM = 40
PFX = SEL.PFX
H, W = A.NATIVE_HW
FX = FY = 1000.0
CX, CY = W / 2.0, H / 2.0
SKY_ROWS = 700
FAR_SKY_UV = (175, 524)           # projected position of the single 300 m return (z=300, x=-180, y=-150)
NEAR_SKY_UV = (1075, 524)         # projected position of the <=300 m sky returns
GREY_SKY, GREY_TINY = 200, 100    # fake segmenter: mean grey > 150 -> 700 sky rows; else 20 rows (-> mask-dropped frame)
TINY_FRAME = ("logB", 10)         # (log, clip_idx) of the frame painted GREY_TINY


def ego_R_cam():
    # camera looks along ego +x: cam z = ego x, cam x = -ego y, cam y = -ego z
    return np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)


def cam_ts():
    return [BASE + k * STEP for k in range(NCAM)]


def lidar_ts():
    return [BASE + k * 100_000_000 + 10_000_000 for k in range(NCAM // 2)]


def points_cam(seed):
    rng = np.random.default_rng(seed)
    z = rng.uniform(5, 150, 12000); x = rng.uniform(-0.2, 0.2, z.size) * z; y = np.full(z.size, 1.5)       # ground below the camera
    g = np.stack([x, y, z], 1)
    zs = rng.uniform(50, 199, 30)                                                                         # sky returns at 50-199 m (inside any-sky), all at NEAR_SKY_UV
    near = np.stack([0.3 * zs, -0.5 * zs, zs], 1)
    far = np.array([[-180.0, -150.0, 300.0]] + [[0.3 * zz, -0.5 * zz, zz] for zz in (250.0, 320.0)])        # > 200 m returns
    return np.concatenate([g, near, far])


def build_world(tmp: Path, lidar_dtypes=("float32", "float16"), decoys=True):
    import pandas as pd, cv2
    from scipy.spatial.transform import Rotation as Rot
    root = tmp / "data"; listing = tmp / "listing"; root.mkdir(); listing.mkdir()
    cams, lids = cam_ts(), lidar_ts()
    R = ego_R_cam(); q = Rot.from_matrix(R).as_quat(); qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    assert np.allclose(A.D.quat_to_R(qw, qx, qy, qz), R, atol=1e-9), "quaternion convention mismatch"
    t_cam_in_ego = np.array([1.5, 0.0, 1.5])
    for li, log in enumerate(LOGS):
        d = root / log
        (d / "calibration").mkdir(parents=True); (d / "sensors" / "cameras" / A.CAMERA).mkdir(parents=True); (d / "sensors" / "lidar").mkdir(parents=True)
        pd.DataFrame(dict(sensor_name=[A.CAMERA, "other"], qw=[qw, 1.0], qx=[qx, 0.0], qy=[qy, 0.0], qz=[qz, 0.0], tx_m=[t_cam_in_ego[0], 0.0], ty_m=[0.0, 0.0], tz_m=[t_cam_in_ego[2], 0.0])
                     ).to_feather(d / "calibration" / "egovehicle_SE3_sensor.feather")
        pd.DataFrame(dict(sensor_name=[A.CAMERA, "other"], fx_px=[FX, 1.0], fy_px=[FY, 1.0], cx_px=[CX, 1.0], cy_px=[CY, 1.0], width_px=[W, 1], height_px=[H, 1])
                     ).to_feather(d / "calibration" / "intrinsics.feather")
        allts = sorted(set(cams) | set(lids))
        pd.DataFrame(dict(timestamp_ns=allts, qw=1.0, qx=0.0, qy=0.0, qz=0.0, tx_m=[(t - BASE) * 1e-9 * 10.0 for t in allts], ty_m=0.0, tz_m=0.0)).to_feather(d / "city_SE3_egovehicle.feather")
        keys = []
        base = PFX + log + "/"
        for rel in ("city_SE3_egovehicle.feather", "calibration/egovehicle_SE3_sensor.feather", "calibration/intrinsics.feather"):
            keys.append((base + rel, (d / rel).stat().st_size))
        # camera frames: the clip frames are written as jpgs; the rest of the 20 Hz stream exists only in the LISTING (not downloaded)
        sel = SEL.select(cams, lids)
        clip = sel[0]
        for k, t in enumerate(clip):
            grey = GREY_TINY if (log, k) == TINY_FRAME else GREY_SKY
            im = np.full((H, W, 3), grey, np.uint8); im[::2, ::2] += 5          # non-constant (decode sanity), mean stays on its side of 150
            cv2.imwrite(str(d / "sensors" / "cameras" / A.CAMERA / f"{t}.jpg"), im)
        for t in cams:
            keys.append((base + f"sensors/cameras/{A.CAMERA}/{t}.jpg", 1000))
        for t in lids:
            keys.append((base + f"sensors/lidar/{t}.feather", 1000))
        pts_cam = points_cam(li)
        pts_ego = pts_cam @ R.T + t_cam_in_ego                       # static world: ego == city up to the motion offset (ignored by the chain's identity rotation)
        for (_, _, lt, _) in sel[1]:
            tx = (lt - BASE) * 1e-9 * 10.0
            pe = pts_ego.copy(); pe[:, 0] += 0.0                      # points are given in the ego frame at the lidar time
            cols = {c: pe[:, i].astype(lidar_dtypes[li]) for i, c in enumerate("xyz")}
            pd.DataFrame(dict(**cols, intensity=np.zeros(len(pe), np.uint8))).to_feather(d / "sensors" / "lidar" / f"{lt}.feather")
        if decoys:                                                    # extra unused jpgs (between clip frames) and sweeps: a directory glob would change the selection
            src = d / "sensors" / "cameras" / A.CAMERA / f"{clip[0]}.jpg"
            for off in (25_000_000, 125_000_000, 225_000_000):
                shutil.copy(src, d / "sensors" / "cameras" / A.CAMERA / f"{cams[0] + off}.jpg")
            for off in (60_000_000, 160_000_000):
                shutil.copy(d / "sensors" / "lidar" / f"{sel[1][0][2]}.feather", d / "sensors" / "lidar" / f"{cams[0] + off}.feather")
        (listing / f"{log}.keys").write_text("".join(f"{k}\t{s}\n" for k, s in keys))
    logs_txt = tmp / "logs.txt"; logs_txt.write_text("\n".join(LOGS) + "\n")
    frames, objects, clips = tmp / "av2_frames.tsv", tmp / "objects.tsv", tmp / "av2_clips.tsv"
    SEL.main(str(listing), str(logs_txt), str(frames), str(objects))
    MC.main(str(listing), str(logs_txt), str(clips))
    (tmp / "SHA256SUMS").write_text("synthetic\n")
    return dict(root=root, listing=listing, logs=logs_txt, frames=frames, objects=objects, clips=clips, sums=tmp / "SHA256SUMS", tmp=tmp)

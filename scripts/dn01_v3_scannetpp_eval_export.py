"""Protocol v3 WP8: export the ScanNet++ EVAL scenes (the ScanNet* column when ScanNet v2 access does not land) to the VDA frame-file format bench_joint reads: for each of the 62 eval scenes (nvs_sem_val + nvs_test_iphone)
the first ``--frames`` (90) frames at ``--stride`` (6 = 10 Hz) of the iPhone video -> <out>/<scene>/rgb/%05d.jpg (1440x1920 -> 480x640, INTER_AREA, q95) and <out>/<scene>/depth/%05d.png (the 192x256 uint16-mm ARKit LiDAR
depth nearest-upsampled to 480x640, factor 1000; 0 = no return), plus ``scannetpp_video.json`` = {"scannetpp": [{scene: [{image, gt_depth, factor, K, pose}]}]} with the per-frame intrinsics K (scaled to 480x640) and the
ARKit camera-to-world ``pose`` (metric) for TAE, like VDA's scannet TAE json. Training scenes are NEVER exported here: the eval list excludes nvs_sem_train by construction (asserted).
``python scripts/dn01_v3_scannetpp_eval_export.py [--out DIR] [--workers 4]``"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SPP = Path("/mnt/DATA/DegradedDepth/data/TEST/ScanNet++")
OUT = Path("/mnt/DATA/DegradedDepth/data/TEST/scannetpp_eval")
H, W = 480, 640


def eval_scenes() -> list[str]:
    rd = lambda n: set((SPP / "splits" / f"{n}.txt").read_text().split())
    ev = sorted(rd("nvs_sem_val") | rd("nvs_test_iphone"))
    assert not set(ev) & rd("nvs_sem_train"), "an nvs_sem_train scene is in the eval list"
    return [s for s in ev if (SPP / "data" / s / "iphone" / "depth.bin").exists()]


def export_scene(args) -> tuple[str, list]:
    scene, out, frames, stride = args
    from data.scannetpp_dataset import _FRAME_BYTES, _H, _W, _index_depth_bin
    import zlib
    sd = SPP / "data" / scene / "iphone"
    spans = _index_depth_bin(sd / "depth.bin")
    pose = json.loads((sd / "pose_intrinsic_imu.json").read_text())
    want = [i * stride for i in range(frames)]
    assert want[-1] < len(spans), f"{scene}: video too short"
    cap = cv2.VideoCapture(str(sd / "rgb.mkv"))
    rows = []
    wset, k = set(want), 0
    with open(sd / "depth.bin", "rb") as fb:
        for j in range(want[-1] + 1):
            ok, fr = cap.read()
            if j not in wset:
                continue
            assert ok, f"{scene}: cannot decode frame {j}"
            off, n = spans[j]
            fb.seek(off)
            buf = fb.read(n)
            try:
                import lz4.block
                raw = lz4.block.decompress(buf, uncompressed_size=_FRAME_BYTES)
            except Exception:
                raw = zlib.decompress(buf, wbits=-zlib.MAX_WBITS)
            d = np.frombuffer(raw, np.uint16).reshape(_H, _W)
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
            (out / scene / "rgb").mkdir(parents=True, exist_ok=True)
            (out / scene / "depth").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / scene / "rgb" / f"{k:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out / scene / "depth" / f"{k:05d}.png"), d)
            p = pose[f"frame_{j:06d}"]
            K = np.array(p["intrinsic"], np.float64) * np.array([[W / 1920.0] * 3, [H / 1440.0] * 3, [1.0] * 3])      # fx, cx scale by W/1920; fy, cy by H/1440
            rows.append({"image": f"{scene}/rgb/{k:05d}.jpg", "gt_depth": f"{scene}/depth/{k:05d}.png", "factor": 1000.0, "K": K.tolist(), "pose": p["pose"]})
            k += 1
    cap.release()
    return scene, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    scenes = eval_scenes()[: a.limit]
    with Pool(a.workers) as p:
        res = p.map(export_scene, [(s, out, a.frames, a.stride) for s in scenes])
    man = {"scannetpp": [{s: rows} for s, rows in sorted(res)]}
    (out / "scannetpp_video.json").write_text(json.dumps(man, indent=1) + "\n")
    print("exported", len(res), "scenes,", sum(len(r) for _, r in res), "frames ->", out)


if __name__ == "__main__":
    main()

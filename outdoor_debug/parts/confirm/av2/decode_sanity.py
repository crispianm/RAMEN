#!/usr/bin/env python3
"""Decode-sanity check (registration frame rule 6, O9a-style, fixed rule): clip_idx 0 of the first 5 logs (sorted) of the registered frames tsv. GT side ONLY, no model output,
no depth statistic used for any choice; pass/fail only. The exact checks per frame (each raises on failure):
  1. jpg decodes to uint8 (2048, 1550, 3), finite by construction, not constant (max > min);
  2. intrinsics image size == (2048, 1550) (asserted in A.Log), K finite with fx, fy > 0 and 0 < cx < 1550, 0 < cy < 2048;
  3. the paired sweep reads; its x,y,z are finite for every row used; dtype recorded;
  4. sparse GT: shape (2048, 1550) float32, at least one valid return, every valid value in (0, 200] m, no inf;
  5. the pose chain finds city_SE3_egovehicle at the EXACT camera and lidar timestamps (A.Log.city_T_ego raises otherwise).
The 5 logs must be logs 1-5 of the registered stage-1 list (av2_val_logs_selected50.txt == av2_val_logs_stage1_50.txt, sorted), asserted against LOGS_TXT.
usage: decode_sanity.py ROOT FRAMES_TSV [N=5] LOGS_TXT"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import av2_dataset as A


def check_frame(root, log, cam, lid) -> dict:
    L = A.Log(root, log)
    img = A.load_rgb(L.image_path(cam))
    assert img.dtype == np.uint8 and img.shape == (*A.NATIVE_HW, 3), f"{log}:{cam}: image {img.dtype} {img.shape}"
    assert int(img.max()) > int(img.min()), f"{log}:{cam}: constant image"
    K = L.K
    assert np.isfinite(K).all() and K[0, 0] > 0 and K[1, 1] > 0 and 0 < K[0, 2] < A.NATIVE_HW[1] and 0 < K[1, 2] < A.NATIVE_HW[0], f"{log}: bad K {K}"
    pts = L._pts_cam(cam, lid)
    assert np.isfinite(pts).all(), f"{log}:{cam}: non-finite sweep/pose values"
    gt = L.sparse_gt(cam, lid)
    assert gt.dtype == np.float32 and gt.shape == A.NATIVE_HW, f"{log}:{cam}: GT {gt.dtype} {gt.shape}"
    v = gt[np.isfinite(gt)]
    assert v.size > 0 and bool((v > 0).all()) and bool((v <= A.MAX_DEPTH_M).all()), f"{log}:{cam}: GT valid-range check failed"
    assert not np.isinf(gt).any()
    return dict(log=log, cam=cam, ok=True, n_valid=int(v.size), xyz_dtypes=sorted(L.xyz_dtypes))


def main(root, frames, n=5, logs_txt=None):
    if not logs_txt:
        raise SystemExit("decode_sanity: LOGS_TXT (the registered log list) is required")
    rows = A.read_frames_tsv(frames)
    logs = sorted({r[0] for r in rows})[: int(n)]
    registered = [l.strip() for l in open(logs_txt) if l.strip()][: int(n)]
    assert logs == registered, f"decode-sanity logs {logs} != the first {n} of the registered list {registered}"
    out = []
    for log in logs:
        r0 = [r for r in rows if r[0] == log and r[4] == 0]
        assert r0, f"{log}: no clip_idx 0 scored frame"
        out.append(check_frame(root, log, r0[0][1], r0[0][2]))
    for o in out:
        print(o)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]), a[3])

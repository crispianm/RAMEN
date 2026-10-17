#!/usr/bin/env python3
"""DESCRIPTIVE-ONLY addendum (not a decision; the pinned readout.py is unchanged): per-arm sky medians from scores.json.
kitti_sky and vkitti_sky: pooled median, frame-balanced median, median of frame medians, frac_gt80, frac_gt200, n_px, frames. Reference = GH200 start-checkpoint row."""
import json
R = "/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/"
ROWS = [("start(GH200 ref)", R + "GH200_REF/scores.json")] + [(a, R + f"ARMS/{a}/scores.json") for a in ["A0", "A0b", "TS2", "TS2_H80", "TS2_H300", "TS2_VK655"]]
S = {n: json.load(open(p)) for n, p in ROWS}
for blk in ("kitti_sky", "vkitti_sky"):
    print(f"\n# {blk} (metres; GH200; exploratory, descriptive only)")
    print(f"{'arm':18s} {'pooled':>8s} {'framebal':>9s} {'med_of_frame_med':>17s} {'frac>80':>8s} {'frac>200':>9s} {'n_px':>10s} {'frames':>7s}")
    for n, _ in ROWS:
        b = S[n]["secondary"][blk]
        print(f"{n:18s} {b['pooled_median']:8.1f} {b['framebalanced_median']:9.1f} {b['median_of_frame_medians']:17.1f} {b['frac_gt80']:8.3f} {b['frac_gt200']:9.3f} {b['n_px']:10d} {b['frames']:7d}")

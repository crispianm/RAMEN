#!/usr/bin/env python3
"""Ifig8: build the 500-frame Bonn manifest for the long-video figure (CPU only, reads file NAMES, no image).

Same construction as Video-Depth-Anything benchmark/dataset_extract/dataset_extract_bonn.py + eval_utils.gen_json: the sorted rgb file list and the sorted depth file list of a scene are
paired BY INDEX (rgb[i] <-> depth[i]); frames start_id .. end_id-1. VDA's own ``bonn_video.json`` uses 30..139 (the 110-frame benchmark manifest) and ``bonn_video_500.json`` 0..499;
this figure uses 30..529 so that its first 110 frames ARE the scored 110-frame clip (same frames, same llvar noise prefix). Format = bench_joint / VDA manifest:
{"bonn": [{"<scene>": [{"image": "<scene>/rgb/<ts>.png", "gt_depth": "<scene>/depth/<ts>.png", "factor": 5000.0}, ...]}]}.
Usage: build_bonn_long_manifest.py --root <bonn root> --scene rgbd_bonn_placing_obstructing_box --out bonn_video_long500.json [--start 30 --n 500]
"""
import argparse, json, os
from pathlib import Path


def natkey(s):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def build(root, scenes, start=30, n=500):
    out = []
    for sc in scenes:
        rgb = sorted(os.listdir(Path(root) / sc / "rgb"), key=natkey)
        dep = sorted(os.listdir(Path(root) / sc / "depth"), key=natkey)
        assert len(rgb) >= start + n and len(dep) >= start + n, f"{sc}: needs {start + n} rgb+depth frames, has {len(rgb)}/{len(dep)}"
        out.append({sc: [{"image": f"{sc}/rgb/{rgb[i]}", "gt_depth": f"{sc}/depth/{dep[i]}", "factor": 5000.0} for i in range(start, start + n)]})
    return {"bonn": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scene", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=30)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    json.dump(build(a.root, a.scene, a.start, a.n), open(a.out, "w"), indent=0)

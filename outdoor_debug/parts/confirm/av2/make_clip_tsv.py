#!/usr/bin/env python3
"""Stdlib-only. Names-only: writes the pinned CLIP table from the S3 key listings with the R1 rule (select_av2_frames.select): one row per log: log, phase, n_clip, then the clip camera timestamps (comma-separated).
usage: make_clip_tsv.py LISTING_DIR LOGS_TXT OUT_CLIPS_TSV   (phase = 0 if clip[0] is cam_sorted[0] else 1; derived from the same select(), see below)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_av2_frames as Sel

PFX = Sel.PFX


def main(listing_dir, logs_txt, out):
    rows = []
    for log in [l.strip() for l in open(logs_txt) if l.strip()]:
        keys = [ln.split("\t")[0] for ln in open(Path(listing_dir) / (log + ".keys"))]
        base = PFX + log + "/"
        cam = sorted(int(k.split("/")[-1][:-4]) for k in keys if k.startswith(base + "sensors/cameras/ring_front_center/") and k.endswith(".jpg"))
        lid = sorted(int(k.split("/")[-1][:-8]) for k in keys if k.startswith(base + "sensors/lidar/") and k.endswith(".feather"))
        clip, scored, dropped = Sel.select(cam, lid)
        phase = cam.index(clip[0]) % 2 if clip else 0
        assert clip == cam[phase::2][:Sel.MAX_LEN]
        rows.append("%s\t%d\t%d\t%s" % (log, phase, len(clip), ",".join(str(t) for t in clip)))
    open(out, "w").write("\n".join(rows) + "\n")
    print("logs", len(rows))


if __name__ == "__main__":
    main(*sys.argv[1:4])

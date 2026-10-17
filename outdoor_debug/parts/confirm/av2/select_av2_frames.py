#!/usr/bin/env python3
"""Stdlib-only (python >= 3.6 OK) names-only frame selection for AV2 val. Reads S3 key listings (one 'key<TAB>size' file per log), applies the registered rule
(identical to av2_dataset.select_frames; a unit test checks the equivalence), writes the frames tsv and the object download list. Opens NO payload.
usage: select_av2_frames.py LISTING_DIR LOGS_TXT OUT_FRAMES_TSV OUT_OBJECTS_TSV
LISTING_DIR/<log>.keys : 'datasets/av2/sensor/val/<log>/...<TAB>size' lines.   OUT_FRAMES_TSV: log, cam_ts, lidar_ts, dt_ns, clip_idx.   OUT_OBJECTS_TSV: key, size."""
import bisect, sys
from pathlib import Path

CLIP_CAM_STRIDE, SCORE_CLIP_STRIDE, MAX_LEN, MAX_PAIR_DT_NS = 2, 10, 160, 25000000
PFX = "datasets/av2/sensor/val/"


def nearest(lid, t):
    i = bisect.bisect_left(lid, t)
    c = [j for j in (i - 1, i) if 0 <= j < len(lid)]
    j = min(c, key=lambda j: (abs(lid[j] - t), lid[j]))
    return lid[j], abs(lid[j] - t)


def select(cam, lid):
    cam, lid = sorted(cam), sorted(lid)
    clip = cam[::CLIP_CAM_STRIDE][:MAX_LEN]
    scored, dropped = [], []
    for i in range(0, len(clip), SCORE_CLIP_STRIDE):
        if not lid:
            dropped.append((i, clip[i], None)); continue
        lt, dt = nearest(lid, clip[i])
        if dt <= MAX_PAIR_DT_NS:
            scored.append((i, clip[i], lt, dt))
        else:
            dropped.append((i, clip[i], dt))
    return clip, scored, dropped


def main(listing_dir, logs_txt, out_frames, out_objs):
    logs = [l.strip() for l in open(logs_txt) if l.strip()]
    frames, objs, ndrop = [], [], 0
    for log in logs:
        keys = {}
        for ln in open(Path(listing_dir) / (log + ".keys")):
            k, s = ln.rstrip("\n").split("\t"); keys[k] = int(s)
        base = PFX + log + "/"
        cam = [int(k.split("/")[-1][:-4]) for k in keys if k.startswith(base + "sensors/cameras/ring_front_center/") and k.endswith(".jpg")]
        lid = [int(k.split("/")[-1][:-8]) for k in keys if k.startswith(base + "sensors/lidar/") and k.endswith(".feather")]
        clip, scored, dropped = select(cam, lid)
        ndrop += len(dropped)
        need = [base + "city_SE3_egovehicle.feather", base + "calibration/egovehicle_SE3_sensor.feather", base + "calibration/intrinsics.feather"]
        need += [base + "sensors/cameras/ring_front_center/%d.jpg" % t for t in clip]
        need += [base + "sensors/lidar/%d.feather" % lt for _, _, lt, _ in scored]
        for k in need:
            if k not in keys:
                raise SystemExit("MISSING KEY in listing: " + k)
            objs.append((k, keys[k]))
        frames += [(log, c, lt, dt, i) for i, c, lt, dt in scored]
    with open(out_frames, "w") as f:
        for r in frames:
            f.write("\t".join(str(x) for x in r) + "\n")
    with open(out_objs, "w") as f:
        for k, s in objs:
            f.write("%s\t%d\n" % (k, s))
    print("logs=%d frames=%d dropped(no sweep within 25 ms)=%d objects=%d total_bytes=%d" % (len(logs), len(frames), ndrop, len(objs), sum(s for _, s in objs)))


if __name__ == "__main__":
    main(*sys.argv[1:5])

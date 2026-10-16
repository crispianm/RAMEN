"""Protocol v3 WP14: NYUv2 train-vs-test overlap audit. The official 654 test images (data/TEST/nyuv2/data/val-*/val/official/*.h5) carry no scene name, so scene / frame disjointness from the 47,584 train frames is
checked on CONTENT: a 16x16 difference hash (256 bits) of the grayscale RGB of every file; for each test image the minimum Hamming distance to any train frame. A duplicate or a near-duplicate frame (same scene, same
view) would sit at a small distance. Writes provenance/dn01_v3/nyu_overlap_report.json (counts, the distance distribution, the closest pairs). ``python scripts/dn01_v3_nyu_overlap.py [--root DIR] [--workers N]``."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT_DEFAULT = "/mnt/DATA/DegradedDepth/data/TEST/nyuv2/data"
OUT = Path(__file__).resolve().parents[1] / "provenance" / "dn01_v3" / "nyu_overlap_report.json"
NEAR_DUP_BITS = 25            # <= 25 / 256 bits (10 %) would be called a near-duplicate view


def dhash(path: str) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as h:
        g = cv2.cvtColor(np.ascontiguousarray(np.asarray(h["rgb"]).transpose(1, 2, 0)), cv2.COLOR_RGB2GRAY)
    s = cv2.resize(g, (17, 16), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (s[:, 1:] > s[:, :-1]).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    root = Path(a.root)
    test = sorted(root.glob("val-*/val/official/*.h5"))
    train = sorted(root.glob("train-*/train/*/*.h5"))
    print(len(test), "test,", len(train), "train", flush=True)
    with Pool(a.workers) as p:
        ht = np.stack(p.map(dhash, map(str, test), chunksize=16))
        htr = np.stack(p.map(dhash, map(str, train), chunksize=64))
    import torch
    t = torch.from_numpy(ht.astype(np.float32) * 2 - 1)
    r = torch.from_numpy(htr.astype(np.float32) * 2 - 1)
    dist = ((256 - t @ r.T) / 2).round().int()            # Hamming
    mn, arg = dist.min(1)
    mn = mn.numpy()
    order = np.argsort(mn)[:10]
    rec = {"method": "dhash 16x16 (256 bit) of grayscale RGB; Hamming distance, test -> nearest train frame", "n_test": len(test), "n_train": len(train),
           "n_train_scene_dirs": len({p.parent.name for p in train}), "near_dup_threshold_bits": NEAR_DUP_BITS,
           "min_distance_bits": int(mn.min()), "p5_distance_bits": float(np.percentile(mn, 5)), "median_distance_bits": float(np.median(mn)),
           "n_test_with_near_dup_train": int((mn <= NEAR_DUP_BITS).sum()),
           "closest_pairs": [{"test": str(test[i].relative_to(root)), "train": str(train[int(arg[i])].relative_to(root)), "bits": int(mn[i])} for i in order]}
    rec["train_list_sha256"] = hashlib.sha256("\n".join(str(p.relative_to(root)) for p in train).encode()).hexdigest()
    rec["test_list_sha256"] = hashlib.sha256("\n".join(str(p.relative_to(root)) for p in test).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "closest_pairs"}, indent=1))
    print("closest:", rec["closest_pairs"][:3])


if __name__ == "__main__":
    sys.exit(main())

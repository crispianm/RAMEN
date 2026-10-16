"""DN-01 KITTI B/R channel-swap fix (audit/SWAP_impact_predeclared_20261001.md, K78).

VDA's extractor wrote PIL-RGB arrays with cv2.imwrite, so the stored KITTI PNGs hold the TRUE RGB frame in the order cv2.imread returns it, and every loader's cv2.imread + BGR2RGB then feeds B/R-SWAPPED
frames. The fix is DATA, not a loader hack: a NEW root whose PNGs are the channel-reversed stored arrays (cv2.imread + BGR2RGB then yields the true RGB), a NEW manifest (image paths under ``rgb_fixed/`` so
``manifest_sha256`` in every result json tells fixed rows from swapped ones), GT depth linked from the old root. The old root is never touched and is not in PATHS any more (the swapped repro uses the
test-only switch ``BENCH_KITTI_SWAP_TESTONLY`` of bench_joint on the fixed frames, which a unit test proves bit-identical to the old root).

  hashlist : (crispwork, from the RAW KITTI zips, independent of the PNGs) per-frame sha256 of the decoded uint8 true-RGB array -> provenance/dn01_kittifix/kitti_rgb_sha256.json
  build    : old root + manifest -> new root + manifest; every written frame is checked against the hash list (decoded array, NOT the PNG bytes: PNG bytes can differ across libpng / zlib builds)
  verify   : a built root against the hash list (every frame of every clip, or --clips k1 k2 ...)
Usage:  python scripts/dn01_kitti_rgbfix.py hashlist --zips <kitti_raw_val_drives> --manifest <old kitti_video.json> --out provenance/dn01_kittifix/kitti_rgb_sha256.json
        python scripts/dn01_kitti_rgbfix.py build --src <old root> --dst <new root> --hashlist provenance/dn01_kittifix/kitti_rgb_sha256.json
        python scripts/dn01_kitti_rgbfix.py verify --root <new root> --hashlist provenance/dn01_kittifix/kitti_rgb_sha256.json"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

FIXED_DIR = "rgb_fixed"
MANIFEST = "kitti_video.json"


def arr_sha(a: np.ndarray) -> str:
    """sha256 over shape + the decoded uint8 RGB bytes (independent of any PNG encoder)."""
    a = np.ascontiguousarray(a, dtype=np.uint8)
    return hashlib.sha256(repr(a.shape).encode() + a.tobytes()).hexdigest()


def read_manifest(path: str | Path) -> list[tuple[str, list[dict]]]:
    return [(k, v) for entry in json.load(open(path))["kitti"] for k, v in entry.items()]


def raw_frame(z: zipfile.ZipFile, clip: str, image: str) -> np.ndarray:
    """The true RGB frame of ``image`` (``<clip>/rgb/<n>.png``) from the raw KITTI zip image_02 (PIL RGB), cropped as VDA copy_crop_files does."""
    from PIL import Image
    n = Path(image).name
    hits = [m for m in z.namelist() if m.endswith(f"{clip}/image_02/data/{n}")]
    if len(hits) != 1:
        raise FileNotFoundError(f"{clip}: image_02/data/{n}: {len(hits)} matches in the zip")
    img = np.array(Image.open(io.BytesIO(z.read(hits[0]))).convert("RGB"))
    h, w = img.shape[:2]
    return img[:h - h % 2, :w - w % 2]            # VDA benchmark/dataset_extract/eval_utils.py copy_crop_files (kitti / bonn): top-left crop to EVEN height and width (even_or_odd)


def fixed_array(stored_png: str | Path) -> np.ndarray:
    """True RGB from a STORED (swapped) PNG: what cv2.imread returns IS the true RGB array, so the fixed loader input is that array."""
    return cv2.imread(str(stored_png))


def cmd_hashlist(a) -> None:
    out: dict = {"schema": 1, "what": "sha256(repr(shape) + decoded uint8 true-RGB bytes) of KITTI raw image_02 frames, per manifest frame", "clips": {}}
    for clip, frames in read_manifest(a.manifest):
        z = zipfile.ZipFile(Path(a.zips) / f"{clip}.zip")
        out["clips"][clip] = {Path(f["image"]).name: arr_sha(raw_frame(z, clip, f["image"])) for f in frames}
        print(clip, len(frames), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("wrote", a.out, hashlib.sha256(Path(a.out).read_bytes()).hexdigest())


def cmd_build(a) -> None:
    src, dst = Path(a.src), Path(a.dst)
    hl = json.load(open(a.hashlist))["clips"]
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"{dst} exists and is not empty -- refusing (a new root is only ever created, never overwritten)")
    man = json.load(open(src / MANIFEST))
    new_entries = []
    for entry in man["kitti"]:
        ne = {}
        for clip, frames in entry.items():
            (dst / clip / FIXED_DIR).mkdir(parents=True, exist_ok=True)
            link = dst / clip / "depth"
            link.symlink_to(os.path.relpath(src / clip / "depth", link.parent))      # GT is untouched: the new root links the old depth dir
            nf = []
            for f in frames:
                n = Path(f["image"]).name
                A = fixed_array(src / f["image"])                                    # = the true RGB array (cv2.imread of the stored PNG)
                if arr_sha(A) != hl[clip][n]:
                    raise SystemExit(f"{clip}/{n}: decoded array != the raw-zip hash")
                p = dst / clip / FIXED_DIR / n
                if not cv2.imwrite(str(p), np.ascontiguousarray(A[..., ::-1])):    # cv2 writes BGR: reversing makes the file standard RGB; cv2.imread + BGR2RGB then returns the true RGB
                    raise SystemExit(f"cannot write {p}")
                nf.append({**f, "image": f"{clip}/{FIXED_DIR}/{n}"})
            ne[clip] = nf
        new_entries.append(ne)
    (dst / MANIFEST).write_text(json.dumps({"kitti": new_entries}))
    print("built", dst, "manifest sha256", hashlib.sha256((dst / MANIFEST).read_bytes()).hexdigest())
    cmd_verify(argparse.Namespace(root=str(dst), hashlist=a.hashlist, clips=None))


def loader_array(path: str | Path) -> np.ndarray:
    """EXACTLY what bench_joint.load_frames / kitti_sky_masks do to one KITTI file before resizing."""
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def cmd_verify(a) -> None:
    root = Path(a.root)
    hl = json.load(open(a.hashlist))["clips"]
    bad, n = [], 0
    for clip, frames in read_manifest(root / MANIFEST):
        if a.clips and clip not in a.clips:
            continue
        for f in frames:
            nm = Path(f["image"]).name
            n += 1
            if arr_sha(loader_array(root / f["image"])) != hl[clip][nm]:
                bad.append(f"{clip}/{nm}")
    if bad:
        raise SystemExit(f"VERIFY FAILED: {len(bad)} of {n} frames differ from the raw-zip hashes: {bad[:5]}")
    print(f"VERIFIED {n} frames: the loader's decoded input == the raw KITTI image_02 RGB (sha256 of the decoded arrays)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hashlist")
    h.add_argument("--zips", required=True)
    h.add_argument("--manifest", required=True)
    h.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", required=True)
    b.add_argument("--dst", required=True)
    b.add_argument("--hashlist", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--root", required=True)
    v.add_argument("--hashlist", required=True)
    v.add_argument("--clips", nargs="*")
    a = ap.parse_args()
    {"hashlist": cmd_hashlist, "build": cmd_build, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

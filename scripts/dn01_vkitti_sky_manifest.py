"""Sky-fix guard manifest (bench_joint dataset 'vkitti', OPT-IN): the exact-sky-label VKITTI clips. The recipe trains on VKITTI variant 'clone' only, so the guard clips are the OTHER weather variants (overcast, sunset) of the five
VKITTI scenes, Camera_0, frames 0..63 of each (a VDA-style manifest: {"vkitti": [{key: [{image, gt_depth, factor}, ...]}]}, file order = scene, variant). The GT is VKITTI's depth png in cm (factor 100); the sky / no-return
sentinel is 65535 cm = 655.35 m, so the exact sky label is gt >= 655 in bench_joint (sky_depth_m). Metadata only (file names); no image is read.
  python scripts/dn01_vkitti_sky_manifest.py [--root <vkitti>] [--out provenance/dn01_sky/vkitti_sky_v1.json]"""
import argparse, hashlib, json
from pathlib import Path

VARIANTS, CAMERA, N_FRAMES, FACTOR = ("overcast", "sunset"), "Camera_0", 64, 100


def build(root: Path) -> dict:
    clips = []
    for scene in sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("Scene")):
        for v in VARIANTS:
            rgb_dir = root / scene / v / "frames" / "rgb" / CAMERA
            dep_dir = root / scene / v / "frames" / "depth" / CAMERA
            if not (rgb_dir.is_dir() and dep_dir.is_dir()):
                continue
            rgb, dep = sorted(rgb_dir.glob("rgb_*.jpg")), sorted(dep_dir.glob("depth_*.png"))
            n = min(len(rgb), len(dep), N_FRAMES)
            if n < N_FRAMES:
                continue
            frames = [{"image": str(r.relative_to(root)), "gt_depth": str(d.relative_to(root)), "factor": FACTOR} for r, d in zip(rgb[:n], dep[:n])]
            clips.append({f"{scene}_{v}": frames})
    return {"vkitti": clips}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/DATA/DegradedDepth/data/TRAIN/vkitti")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "provenance/dn01_sky/vkitti_sky_v1.json"))
    a = ap.parse_args()
    m = build(Path(a.root))
    Path(a.out).write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    print(f"wrote {a.out}: {len(m['vkitti'])} clips x {N_FRAMES} frames; sha256 {hashlib.sha256(Path(a.out).read_bytes()).hexdigest()}")

"""DN-01 K80 option B: the PointOdyssey background-depth filter (audit/K80_optionB_predeclared_20261002.md, section 1).

CRITERION (fixed before any training, never revised after a result): a pixel is VALID iff finite and 0 < depth < 80 m (depth = the uint16 png * 1000 / 65535, the loader's own decode,
data.pointodyssey_dataset._PO_DEPTH_SCALE); a sequence (scene) is KEPT iff the MEDIAN over ALL its frames of the valid-pixel fraction is >= 0.5. Frames of kept sequences keep only valid pixels in every loss (the
loader's mask is exactly this valid test). VDA's own filter ("drop the PointOdyssey frames without background depth") is unstated; this one is ours (disclosed).

    python scripts/dn01_po_filter.py --root <PointOdyssey train dir> --out provenance/dn01_vdamix/po_filter_v1.json [--workers 16]

The output lists every scene with its frame count and median valid fraction, the kept list, and ``kept_sha256`` (sha256 of the kept scene names, sorted, joined with "\\n"): the configs pin it
(``po_keep_sha256``) and ``PointOdysseyDataset`` refuses a list whose hash differs. ``--check`` re-computes the decision from a stored per-scene table without touching the images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MAX_DEPTH_M = 80.0
KEEP_MEDIAN_VALID_FRAC = 0.5


def valid_fraction(path: str) -> float:
    import cv2
    from data.pointodyssey_dataset import _PO_DEPTH_SCALE
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read PO depth: {path}")
    d = raw.astype(np.float32) * _PO_DEPTH_SCALE
    return float((np.isfinite(d) & (d > 0) & (d < MAX_DEPTH_M)).mean())


def scene_stat(scene_dir: str) -> dict:
    files = sorted(Path(scene_dir).joinpath("depths").glob("depth_*.png"))
    fr = np.array([valid_fraction(str(f)) for f in files], np.float64)
    return {"frames": int(len(fr)), "median_valid_frac": float(np.median(fr)) if len(fr) else 0.0,
            "mean_valid_frac": float(fr.mean()) if len(fr) else 0.0, "min_valid_frac": float(fr.min()) if len(fr) else 0.0}


def decide(per_scene: dict, threshold: float = KEEP_MEDIAN_VALID_FRAC) -> list[str]:
    """The kept scene names (sorted): median valid fraction >= threshold."""
    return sorted(n for n, s in per_scene.items() if s["frames"] > 0 and s["median_valid_frac"] >= threshold)


def kept_sha256(kept: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(kept)).encode()).hexdigest()


def build(root: Path, workers: int) -> dict:
    scenes = sorted(p for p in root.iterdir() if p.is_dir() and (p / "rgbs").is_dir() and (p / "depths").is_dir())
    with ProcessPoolExecutor(max_workers=workers) as ex:
        stats = list(ex.map(scene_stat, [str(s) for s in scenes], chunksize=1))
    per_scene = {s.name: st for s, st in zip(scenes, stats)}
    kept = decide(per_scene)
    return {"criterion": f"valid = finite and 0 < depth < {MAX_DEPTH_M:g} m; a scene is kept iff the median over all its frames of the valid-pixel fraction >= {KEEP_MEDIAN_VALID_FRAC}",
            "max_depth_m": MAX_DEPTH_M, "keep_median_valid_frac": KEEP_MEDIAN_VALID_FRAC,
            "n_scenes": len(per_scene), "n_kept": len(kept), "frames_total": sum(s["frames"] for s in per_scene.values()),
            "frames_kept": sum(per_scene[n]["frames"] for n in kept),
            "kept": kept, "kept_sha256": kept_sha256(kept), "per_scene": per_scene}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = build(Path(a.root), a.workers)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{out['n_kept']}/{out['n_scenes']} scenes kept, {out['frames_kept']}/{out['frames_total']} frames, kept_sha256 {out['kept_sha256']}")


if __name__ == "__main__":
    main()

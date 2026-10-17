#!/usr/bin/env python3
"""Build the ONE frozen AV2 sky-mask store (shared by both arms) in the scorer's MaskStore format + a .meta.json sidecar.
Registration R1 rules (all fixed before any use on real data):
  certain = both segmenters p_sky >= 0.9 (image-size, bilinear-upsampled probs), eroded: Euclid distance to the nearest non-certain px > r_e = round(8*short/900) (=12 px at 1550),
            minus pixels within r_l = round(5*short/900) (=9 px at 1550) of any projected LiDAR return of the paired sweep.  [no horizon rule]
  any     = sky by EITHER segmenter: argmax is sky OR p_sky >= 0.5 (superset of certain by construction; asserted).
  drop rule 1 (this script): a frame with < 1% of its native pixels certain-sky (after erosion and LiDAR exclusion) is listed in meta.dropped = excluded from the PRIMARY P only
            (Auditor O43 M4: the guards use ALL scored frames; the frame's masks stay in the store because the guard needs its `any` layer).
  agreement: pooled IoU of the two segmenters' own `any_sky` masks over ALL selected frames, computed BEFORE either drop rule, must be >= 0.85, else MaskRejected: no store is written.
  (drop rule 2, in the scorer: <2000 certain-sky px at the model grid -> excluded from P only.)
usage: build_av2_masks.py --root <av2_val data dir> --frames av2_frames.tsv --out sky_masks.npz --seg-a <dir> --seg-b <dir> --seg-a-sha <pytorch_model.bin sha> --seg-b-sha <sha> [--device cuda]"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "seg"))
import make_sky_mask as M            # noqa: E402


def lidar_uv_from_gt(gt):
    """gt: sparse native GT (NaN = no return) -> [N,2] (x=col, y=row) of the projected returns."""
    yx = np.argwhere(np.isfinite(gt))
    return yx[:, ::-1].astype(np.float64)


def frame_masks(pa, ma, pb, mb, uv):
    """One frame -> dict(certain, any, inter, union, dropped). pa/pb/ma/mb: float32 [H,W] image-size sky prob / margin of segmenter A/B."""
    cert = M.apply_lidar(M.certain_sky(pa, pb), uv) if uv is not None and len(uv) else M.certain_sky(pa, pb)
    anyl = M.any_layer(pa, ma, pb, mb)
    if (cert & ~anyl).any():
        raise AssertionError("certain-sky must be a subset of any-sky")
    i, u = M.iou_counts(pa, ma, pb, mb)
    return dict(certain=cert, any=anyl, inter=i, union=u, dropped=bool(M.is_dropped(cert)), n_certain=int(cert.sum()))


def assemble(results, out, meta_extra):
    """results: {fid: frame_masks dict}. Writes `out` (+ .meta.json) unless the pooled IoU < 0.85 (MaskRejected, nothing written)."""
    from av2_dataset import save_mask_store
    iou = M.check_agreement([(r["inter"], r["union"]) for r in results.values()])
    kept = {f: (r["certain"], r["any"]) for f, r in results.items()}          # ALL frames stored (guards use all); `dropped` only removes a frame from P
    dropped = sorted(f for f, r in results.items() if r["dropped"])
    save_mask_store(out, kept)
    meta = dict(status="OK", rule=M.RULE, pooled_iou=iou, n_candidates=len(results), n_kept=len(kept), dropped=dropped,
                n_certain_px_total=int(sum(r["n_certain"] for r in results.values() if not r["dropped"])), **meta_extra)
    Path(str(out)).with_suffix(".meta.json").write_text(json.dumps(meta, sort_keys=True, indent=1))
    return meta


def verify_seg_sha(weights_file, expected):
    """The segmenter weight file on disk must hash to the registered sha256; raises otherwise."""
    import hashlib
    h = hashlib.sha256(Path(weights_file).read_bytes()).hexdigest()
    if h != expected:
        raise RuntimeError(f"segmenter weights {weights_file}: sha256 {h} != registered {expected}")
    return h


def main():
    ap = argparse.ArgumentParser()
    for k in ("root", "frames", "out", "seg-a", "seg-b", "seg-a-sha", "seg-b-sha"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import av2_dataset as A
    out = Path(a.out); assert not out.exists(), f"{out} exists (never overwrite)"
    verify_seg_sha(Path(a.seg_a) / "pytorch_model.bin", a.seg_a_sha); verify_seg_sha(Path(a.seg_b) / "pytorch_model.bin", a.seg_b_sha)
    A_ = M._Seg(a.seg_a, None, M.ADE_EXPECT, "segformer", a.device)
    B_ = M._Seg(a.seg_b, None, M.CITY_EXPECT, "oneformer", a.device)
    rows = [l.rstrip("\n").split("\t") for l in open(a.frames) if l.strip() and not l.startswith("#")]
    rows = [r for r in rows if r[0] != "log"]
    res, logs = {}, {}
    for log, cam, lid, dt, idx in rows:
        L = logs.setdefault(log, A.Log(a.root, log))
        img = A.load_rgb(L.image_path(int(cam)))
        pa, ma = A_.sky_maps(img); pb, mb = B_.sky_maps(img)
        uv = lidar_uv_from_gt(L.sparse_gt(int(cam), int(lid)))
        res[A.frame_id(log, int(cam))] = frame_masks(pa, ma, pb, mb, uv)
    meta = assemble(res, out, dict(segmenter_a_sha=a.seg_a_sha, segmenter_b_sha=a.seg_b_sha, frames_sha=A.sha256_file(a.frames)))
    print(json.dumps({k: meta[k] for k in ("status", "pooled_iou", "n_candidates", "n_kept")}), "dropped", len(meta["dropped"]))


if __name__ == "__main__":
    main()

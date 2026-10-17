#!/usr/bin/env python3
"""Build the ONE frozen AV2 sky-mask store (shared by both arms) in the scorer's MaskStore format + a .meta.json sidecar.
Registration R1 rules (all fixed before any use on real data):
  certain = both segmenters p_sky >= 0.9 (image-size, bilinear-upsampled probs), eroded: Euclid distance to the nearest non-certain px > r_e = round(8*short/900) (=14 px at 1550),
            minus pixels within r_l = round(5*short/900) (=9 px at 1550) of any projected LiDAR return of the paired sweep.  [no horizon rule]
  any     = sky by EITHER segmenter: argmax is sky OR p_sky >= 0.5 (superset of certain by construction; asserted).
  drop rule 1 (this script): a frame with < 1% of its native pixels certain-sky (after erosion and LiDAR exclusion) is listed in meta.dropped = excluded from the PRIMARY P only
            (Auditor O43 M4: the guards use ALL scored frames; the frame's masks stay in the store because the guard needs its `any` layer).
  agreement: pooled IoU of the two segmenters' own `any_sky` masks over ALL selected frames, computed BEFORE either drop rule, must be >= 0.85, else MaskRejected: no store is written.
  (drop rule 2, in the scorer: <2000 certain-sky px at the model grid -> excluded from P only.)
  LiDAR exclusion: EVERY return of the paired sweep in front of the camera (z > 0, no 200 m cap, A.project_all_returns), not the capped GT map.
  inputs: ONLY av2_frames.tsv + the pinned clip table av2_clips.tsv + objects.tsv (A.load_registered_inputs: shas, clip[clip_idx]==cam_ts, files in objects.tsv, files exist). No directory glob.
  segmenter pins are the hard-coded constants SEG_A / SEG_B below (SEG_PIN.md); no CLI sha.
usage: build_av2_masks.py --root <av2_val data dir> --frames av2_frames.tsv --clips av2_clips.tsv --objects objects.tsv --out sky_masks.npz --seg-a <dir> --seg-b <dir> [--device cuda]"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "seg"))
import make_sky_mask as M            # noqa: E402


# ---- hard-coded segmenter pins (SEG_PIN.md; AV2_REGISTRATION.md 'Segmenter pins'). Never taken from the command line.
SEG_A = dict(repo="nvidia/segformer-b5-finetuned-ade-640-640", revision="739f5d4692954e4a185eac280dec1ba5a7d52f1d",
             bin_sha="a3dbf2d7a9912203078f304d88df8cfee9fd752ca2d2413f82036bdb02c5e7df")
SEG_B = dict(repo="shi-labs/oneformer_cityscapes_swin_large", revision="8ed649db2cc1756b8bc36f0cad00b0812e333027",
             bin_sha="876855252aa674c7ad6e920c268d6722add811a927806372811faa724f19856c")


def frame_masks(pa, ma, pb, mb, uv):
    """One frame -> dict(certain, any, inter, union, dropped). pa/pb/ma/mb: float32 [H,W] image-size sky prob / margin of segmenter A/B."""
    cert = M.apply_lidar(M.certain_sky(pa, pb), uv) if uv is not None and len(uv) else M.certain_sky(pa, pb)    # uv = ALL returns in front of the camera (D12)
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


def verify_segmenters(dir_a, dir_b, pin_a=None, pin_b=None):
    """Hard-coded pins only: each weight dir must contain pytorch_model.bin with the pinned sha256 and NO *.safetensors (from_pretrained would prefer it).
    Returns the meta fragment (repo, revision, sha per segmenter)."""
    pa, pb = pin_a or SEG_A, pin_b or SEG_B
    out = {}
    for tag, d, pin in (("a", dir_a, pa), ("b", dir_b, pb)):
        st = sorted(str(q) for q in Path(d).rglob("*.safetensors"))
        if st:
            raise RuntimeError(f"segmenter {tag}: {st} present; only the verified pytorch_model.bin may exist in the weight dir")
        verify_seg_sha(Path(d) / "pytorch_model.bin", pin["bin_sha"])
        out[f"segmenter_{tag}_sha"] = pin["bin_sha"]
        out[f"segmenter_{tag}"] = dict(repo=pin["repo"], revision=pin["revision"], pytorch_model_bin_sha256=pin["bin_sha"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("root", "frames", "clips", "objects", "out", "seg-a", "seg-b"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    import av2_dataset as A
    out = Path(a.out); assert not out.exists(), f"{out} exists (never overwrite)"
    segmeta = verify_segmenters(a.seg_a, a.seg_b)
    reg = A.load_registered_inputs(a.frames, a.clips, a.objects, a.root)
    A_ = M._Seg(a.seg_a, None, M.ADE_EXPECT, "segformer", a.device, use_safetensors=False)
    B_ = M._Seg(a.seg_b, None, M.CITY_EXPECT, "oneformer", a.device, use_safetensors=False)
    res, dtypes = {}, set()
    for log, d in reg.items():
        L = A.Log(a.root, log)
        for (_, cam, lid, dt, idx) in d["rows"]:
            img = A.load_rgb(L.image_path(cam))
            assert img.shape[:2] == A.NATIVE_HW, f"{log}:{cam}: image {img.shape} != native {A.NATIVE_HW}"
            pa, ma = A_.sky_maps(img); pb, mb = B_.sky_maps(img)
            res[A.frame_id(log, cam)] = frame_masks(pa, ma, pb, mb, L.all_returns_uv(cam, lid))
        dtypes |= L.xyz_dtypes
    meta = assemble(res, out, dict(**segmeta, frames_sha=A.sha256_file(a.frames), clips_sha=A.sha256_file(a.clips), objects_sha=A.sha256_file(a.objects),
                                   lidar_xyz_dtypes=sorted(dtypes), lidar_exclusion="all returns with z>0, no range cap"))
    print(json.dumps({k: meta[k] for k in ("status", "pooled_iou", "n_candidates", "n_kept")}), "dropped", len(meta["dropped"]))


if __name__ == "__main__":
    main()

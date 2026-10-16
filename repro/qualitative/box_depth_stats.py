#!/usr/bin/env python3
"""Read-only: predicted vs GT depth inside a region of a render's <stem>_depth.npz (CPU, numpy only).

  box_depth_stats.py STEM_depth.npz --rect x0 y0 x1 y1 [--space panel|native] [--panel-w W]
  box_depth_stats.py STEM_depth.npz --mask mask.png        (non-zero = inside; resized nearest to the array shape)

--rect is in NATIVE array pixels (x0,y0 inclusive, x1,y1 exclusive). Prints, per panel (ft, c3_raw, c3_tag when present) and GT:
median / p10 / p90 over the region (GT: valid pixels only), the fraction of region pixels at/below the colour floor lo and at/above the ceiling hi
(lo, hi = color_range_m), and the median pred-GT and median |pred-GT|/GT over valid GT pixels."""
import argparse, sys
import numpy as np


def region_mask(shape, rect=None, mask_png=None):
    m = np.zeros(shape, bool)
    if rect is not None:
        x0, y0, x1, y1 = rect
        m[max(0, y0):y1, max(0, x0):x1] = True
    else:
        from PIL import Image
        im = np.array(Image.open(mask_png).convert("L").resize((shape[1], shape[0]), Image.NEAREST))
        m = im > 0
    return m


def stats(d, region, lo, hi, valid=None):
    sel = region & (valid if valid is not None else np.isfinite(d))
    v = d[sel].astype(np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "frac_at_floor": float((v <= lo).mean()), "frac_at_ceiling": float((v >= hi).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rect", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    g.add_argument("--mask")
    a = ap.parse_args(argv)
    z = np.load(a.npz, allow_pickle=False)
    lo, hi = (float(x) for x in z["color_range_m"])
    gt, valid = z["gt"], z["gt_valid"]
    region = region_mask(gt.shape, a.rect, a.mask)
    print(f"frame {int(z['frame_index'])}  colour range [{lo:.4g}, {hi:.4g}] m  region px {int(region.sum())} (valid GT {int((region & valid).sum())})")
    print(f"{'panel':8s} {'n':>7s} {'median':>8s} {'p10':>8s} {'p90':>8s} {'@floor':>7s} {'@ceil':>7s}  {'med(pred-GT)':>12s} {'med|rel err|':>12s}")
    for k in ["gt"] + [k for k in ("ft", "c3_raw", "c3_tag") if k in z.files]:
        d = z[k]
        if d.shape != gt.shape:
            print(f"{k}: shape {d.shape} != GT {gt.shape}; skipped (resize first)")
            continue
        s = stats(d, region, lo, hi, valid if k == "gt" else None)
        s_pred_region = stats(d, region, lo, hi)
        s = s if k == "gt" else s_pred_region
        if s["n"] == 0:
            print(f"{k:8s} empty")
            continue
        both = region & valid
        extra = ("", "") if k == "gt" else (f"{np.median(d[both] - gt[both]):12.4f}", f"{np.median(np.abs(d[both] - gt[both]) / gt[both]):12.4f}")
        print(f"{k:8s} {s['n']:7d} {s['median']:8.4f} {s['p10']:8.4f} {s['p90']:8.4f} {s['frac_at_floor']:7.3f} {s['frac_at_ceiling']:7.3f}  {extra[0]:>12s} {extra[1]:>12s}")


if __name__ == "__main__":
    main()

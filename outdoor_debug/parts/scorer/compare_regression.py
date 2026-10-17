#!/usr/bin/env python3
"""Compare a scorer JSON of c3_n46 step12000 with the earlier numbers (source files in README). Prints value / earlier / abs diff / rel diff / verdict."""
import json, sys
j = json.load(open(sys.argv[1]))
g, p, s = j["guards"], j["primary"], j["secondary"]
rows = [  # (name, new, earlier, rel tolerance same-host-type, source)
    ("VKITTI far 80-655 pred:gt, legacy-subsample median", p["vkitti_far_80_655"].get("legacy_subsample_median"), 0.8417303562164307, 1e-4, "T_trajectory/trajectory.csv"),
    ("VKITTI far 80-655 pred:gt, EXACT pooled median", p["vkitti_far_80_655"]["pooled_median"], 0.8417303562164307, 3e-3, "T_trajectory (sampled; exact differs by sampling error)"),
    ("VKITTI near<30 pred:gt legacy median", s["vkitti_near_lt30_pred_over_gt"].get("legacy_subsample_median"), 1.1001617908477783, 1e-4, "T_trajectory"),
    ("VKITTI absrel_raw >80", j["by_range"]["vkitti"][">80"]["absrel_raw"], 0.21134385865700342, 1e-4, "T_trajectory/baseline_metrics.csv"),
    ("VKITTI absrel_raw 0-10", j["by_range"]["vkitti"]["0-10"]["absrel_raw"], 0.14398058202849973, 1e-4, "T_trajectory"),
    ("VKITTI absrel_raw 10-30", j["by_range"]["vkitti"]["10-30"]["absrel_raw"], 0.16972464281237404, 1e-4, "T_trajectory"),
    ("VKITTI absrel_raw 30-80", j["by_range"]["vkitti"]["30-80"]["absrel_raw"], 0.2018785600382735, 1e-4, "T_trajectory"),
    ("KITTI pooled absrel_raw", j["pooled_absrel_raw"]["kitti"]["value"], 0.1477352948568263, 1e-4, "T_trajectory kitti_dev absrel_raw (base=gt>0.1&~sky)"),
    ("KITTI sky median of frame medians", s["kitti_sky"]["median_of_frame_medians"], 160.49712371826172, 1e-3, "T_trajectory"),
    ("KITTI sky pooled median", s["kitti_sky"]["pooled_median"], 322.2507629394531, 1e-3, "T_trajectory"),
    ("VKITTI sky median (legacy 200k subsample)", s["vkitti_sky"].get("legacy_subsample_median"), 323.7591247558594, 1e-3, "T_trajectory"),
    ("ScanNet++ pooled absrel_raw", j["pooled_absrel_raw"]["scannetpp"]["value"], 0.12083008882329532, 1e-4, "T_trajectory scannetpp_dev"),
    ("NYUv2 pooled absrel_raw (160 fr)", j["pooled_absrel_raw"]["nyuv2"]["value"], 0.09485469182551873, 1e-4, "D_baseline baseline_metrics.csv (all)"),
    ("KITTI per-frame-macro AbsRel unaligned (bj video_macro, clip-macro)", g["kitti"]["video_macro"]["abs_rel_unaligned"], None, None, "no earlier clip-macro; D_baseline cap_perframe_macro (frame-macro) = 0.14867"),
    ("ScanNet++ video_macro abs_rel_unaligned", g["scannetpp"]["video_macro"]["abs_rel_unaligned"], None, None, "D_baseline cap_perframe_macro 0.12066 (frame-macro)"),
    ("NYUv2 video_macro abs_rel_unaligned (160 fr)", g["nyuv2"]["video_macro"]["abs_rel_unaligned"], None, None, "D_baseline cap_perframe_macro 0.09295 (frame-macro)"),
    ("horse sky_top15 median", s["horse"]["sky_top15_median_m"], 199.59031677246094, 1e-3, "T_trajectory"),
    ("horse treeline median", s["horse"]["treeline_38_46_median_m"], 81.55253601074219, 1e-3, "T_trajectory"),
]
for name, new, old, tol, src in rows:
    if old is None:
        print(f"INFO  {name}: new {new:.6f}   ({src})"); continue
    if new is None:
        print(f"MISS  {name}"); continue
    rel = abs(new - old) / abs(old)
    print(f"{'OK  ' if rel <= tol else 'DIFF'}  {name}: new {new:.6f} earlier {old:.6f} rel {rel:.2e} tol {tol:.0e}  [{src}]")
print("TAE guards (no earlier number): " + ", ".join(f"{d}: tae_gtvalid {g[d]['video_macro'].get('tae_gtvalid')}" for d in g))

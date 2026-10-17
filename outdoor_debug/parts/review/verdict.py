import json,sys
def verdict(path):
    d=json.load(open(path)); fails=[]
    for k,r in d["c2"].items():
        for s in ("seed0","seed1"):
            x=r[s]
            if not x["masks_differ"]: fails.append(f"c2 {k} {s}: TS2 masks == A0 masks (no change)")
            if x["max_abs_diff_mask_vs_ts_v2_guarded"]!=0 or x["max_abs_diff_sky_vs_ts_v2_guarded"]!=0: fails.append(f"c2 {k} {s}: differs from ts_v2_guarded ref (mask {x['max_abs_diff_mask_vs_ts_v2_guarded']}, sky {x['max_abs_diff_sky_vs_ts_v2_guarded']}); TS2 sky%={x['TS2_sky_pct']:.3f} A0 sky%={x['A0_sky_pct']}")
    for k,r in d["c3"].items():
        v=r["VK655"]
        if not (v["sentinel_px_pct"]>1 and v["mask_pct"]==100.0 and v["sky_pct"]==0): fails.append(f"c3 {k}: VK655 sentinel_px%={v['sentinel_px_pct']:.2f} mask%={v['mask_pct']:.2f} sky%={v['sky_pct']:.2f} depth_max={v['valid_depth_max']}")
    return fails
for p in sys.argv[1:]:
    f=verdict(p); print(p, "->", "PASS (no fails)" if not f else "FAIL"); [print("   ",x) for x in f]

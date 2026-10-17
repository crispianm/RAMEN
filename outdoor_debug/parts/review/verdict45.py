import json,sys,math
def verdict(p):
    d=json.load(open(p)); f=[]
    c4=d["c4"]
    if c4["TS2_VK655"]["silog_mask_px"]!=8586368 or c4["TS2_VK655"]["sentinel_px_in_silog_mask"]<=0: f.append(f"c4 VK655: silog_mask_px={c4['TS2_VK655']['silog_mask_px']}, sentinel px in SiLog mask={c4['TS2_VK655']['sentinel_px_in_silog_mask']} (claimed 8,586,368 / 3,667,968)")
    if c4["TS2_VK655"]["min(GT,80)"]["grad_abs_sum_on_sentinel_px"]<=0: f.append(f"c4 VK655: grad on sentinel px = {c4['TS2_VK655']['min(GT,80)']['grad_abs_sum_on_sentinel_px']} (claimed 9.05e-3)")
    if c4["TS2"]["silog_mask_px"]!=4918400: f.append(f"c4 TS2 mask px {c4['TS2']['silog_mask_px']} != 4918400")
    for n in ("TS2_H80","TS2_H300"):
        r=d["c5"][n]; exp=r["expected_pred10"]; got=r["pred=10m"]["sky_hinge_weighted"]
        if got is None or abs(got-exp)>1e-6: f.append(f"c5 {n}: hinge at pred=10m = {got} expected {exp:.5f} (w={r['cfg_sky_hinge_weight']}, floor={r['floor_used']})")
        if r["pred=1000m"]["sky_hinge_weighted"] not in (0.0,): f.append(f"c5 {n}: hinge nonzero above floor: {r['pred=1000m']['sky_hinge_weighted']}")
    return f
for p in sys.argv[1:]:
    f=verdict(p); print(p,"->","PASS" if not f else "FAIL"); [print("   ",x) for x in f]

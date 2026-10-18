import json, sys, math
new = json.load(open(sys.argv[1])); ref = json.load(open(sys.argv[2]))
ok = True; mx = 0.0
def cmp(k, a, b):
    global ok, mx
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = abs(a - b); mx = max(mx, d)
        return d
    return 0.0 if a == b else float("inf")
# counts
for k in ref["bins"]:
    for f in ("n_px", "n_frames", "n_scenes"):
        if new["bins"][k][f] != ref["bins"][k][f]: ok = False; print("COUNT DIFF", k, f, new["bins"][k][f], ref["bins"][k][f])
for f in ("n_scored_frames", "n_scenes", "overall_n_px", "n_nonfinite_pred_px"):
    if new.get(f) != ref.get(f): ok = False; print("COUNT DIFF", f, new.get(f), ref.get(f))
rows = []
for k in ref["bins"]:
    for f, v in ref["bins"][k].items():
        if f in ("n_px", "n_frames", "n_scenes"): continue
        a = new["bins"][k].get(f)
        if isinstance(v, (int, float)):
            d = cmp(k + "." + f, a, v); rows.append((d, k, f))
for f in ("overall_absrel_0_200",):
    rows.append((cmp(f, new[f], ref[f]), "overall", f))
rows.sort(reverse=True)
print("fields compared:", len(rows), "| max abs diff over all bin metrics + overall AbsRel: %.3e" % rows[0][0], "at", rows[0][1], rows[0][2])
for k in ref["bins"]:
    print(k, "median new %.6f ref %.6f | AbsRel new %.6f ref %.6f" % (new["bins"][k]["median_pred_over_gt"], ref["bins"][k]["median_pred_over_gt"], new["bins"][k]["absrel_unaligned_pooled"], ref["bins"][k]["absrel_unaligned_pooled"]))
okm = rows[0][0] <= 1e-4
print("COUNTS IDENTICAL:", ok, "| MAX DIFF <= 1e-4:", okm, "| IDENTITY CONTROL:", "PASS" if (ok and okm) else "FAIL")

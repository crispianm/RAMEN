"""Gate evaluation for AUD-FASTTAG-B-20261013 from per_clip.csv + per_pixel_stats.csv (no scorer code; arithmetic only). Writes gates.json + RESULT.md body pieces."""
import csv, json, math, sys
from collections import defaultdict
from pathlib import Path
out = Path(sys.argv[1])
rows = list(csv.DictReader(open(out / "per_clip.csv")))
pix = list(csv.DictReader(open(out / "per_pixel_stats.csv")))
META = {"dataset","cell","clip","arm","setting","backend","agf_backend","raw_depth_sha256","input_sha256","out_sha256","raw_shape","raw_dtype"}
keys = [k for k in rows[0] if k not in META]
LISTED = ["abs_rel_unaligned","abs_rel_aligned","delta1_unaligned","delta1_aligned","tae","tae_gtvalid","flicker","abs_rel_static","eps_d","tof"]
def f(x):
    try: return float(x)
    except: return float("nan")
def rel(a, b):
    if math.isnan(a) and math.isnan(b): return 0.0
    if math.isnan(a) != math.isnan(b): return float("inf")
    return abs(a-b)/max(abs(b), 1e-12)
CL = defaultdict(dict)
for r in rows: CL[(r["dataset"], r["cell"], r["clip"])][(r["arm"], r["setting"])] = r
res = {"expected_clips": {"kitti":7,"bonn":13,"scannetpp":31}, "fail": [], "flag": []}
cnt = defaultdict(set)
for (ds,c,cl) in CL: cnt[(ds,c)].add(cl)
for ds,n in res["expected_clips"].items():
    for c in ("llvar","st_llvar"):
        if len(cnt[(ds,c)]) != n: res["fail"].append(f"count {ds}/{c}: {len(cnt[(ds,c)])} != {n}")
for k,v in CL.items():
    exp = {("none","none")} | {(a,s) for a in ("REF","V1","V3") for s in ("s02","s05")}
    if set(v) != exp: res["fail"].append(f"missing/extra arms {k}: {sorted(set(v)^exp)}")
# raw depth identical across arms
for k,v in CL.items():
    if len({r["raw_depth_sha256"] for r in v.values()}) != 1: res["fail"].append(f"raw sha differs across arms {k}")
# per-clip metric gate
pcw = {}
for arm in ("V1","V3"):
    for s in ("s02","s05"):
        worst = (0.0, None); nviol = 0; lw = (0.0, None); lviol = 0
        for k,v in CL.items():
            a, b = v[(arm,s)], v[("REF",s)]
            for m in keys:
                d = rel(f(a[m]), f(b[m]))
                if d > worst[0]: worst = (d, f"{k}|{m}")
                if d > 1e-3: nviol += 1
        pcw[f"{arm}_{s}"] = {"max_rel_all_keys": worst[0], "at": worst[1], "n_violations_1e-3": nviol}
        if nviol: res["fail"].append(f"per-clip gate {arm} {s}: {nviol} violations")
res["per_clip"] = pcw
# macro (per dataset x cell, nanmean as scorer) + overall macro over all clips of dataset-cells
def nanmean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs)/len(xs) if xs else float("nan")
MAC = defaultdict(dict)
for (ds,c,cl),v in CL.items():
    for (arm,s),r in v.items():
        for m in keys: MAC[(ds,c,arm,s)].setdefault(m, []).append(f(r[m]))
DSP = sorted({k[0] for k in MAC})
macro = {k: {m: nanmean(xs) for m,xs in d.items()} for k,d in MAC.items()}
mw = {}
for arm in ("V1","V3"):
    for s in ("s02","s05"):
        worst_l = (0.0, None); worst_a = (0.0, None); nv = 0
        for ds in DSP:
            for c in ("llvar","st_llvar"):
                for m in keys:
                    d = rel(macro[(ds,c,arm,s)][m], macro[(ds,c,"REF",s)][m])
                    if m in LISTED and d > worst_l[0]: worst_l = (d, f"{ds}|{c}|{m}")
                    if d > worst_a[0]: worst_a = (d, f"{ds}|{c}|{m}")
                    if d > 1e-4: nv += 1
        mw[f"{arm}_{s}"] = {"max_rel_listed_keys": worst_l[0], "at_listed": worst_l[1], "max_rel_all_keys": worst_a[0], "at_all": worst_a[1], "n_violations_1e-4_all_keys": nv}
        if worst_l[0] > 1e-4 or nv: res["fail"].append(f"macro gate {arm} {s}: listed max {worst_l[0]:.3g}, all-key violations {nv}")
res["macro"] = mw
# flips: sign of (TAG - none) per macro metric/dataset/cell/setting; setting ordering; rank of {none,s02,s05}
flips = []
for ds in DSP:
    for c in ("llvar","st_llvar"):
        for m in keys:
            def order(arm):
                vals = {"none": macro[(ds,c,"REF","none")][m] if False else macro[(ds,c,"none","none")][m], "s02": macro[(ds,c,arm,"s02")][m], "s05": macro[(ds,c,arm,"s05")][m]}
                sg = {s: (vals[s] > vals["none"]) - (vals[s] < vals["none"]) for s in ("s02","s05")}
                rk = tuple(sorted(vals, key=lambda z: (vals[z] if not math.isnan(vals[z]) else 9e99)))
                return sg, rk
            r0 = order("REF")
            for arm in ("V1","V3"):
                if order(arm) != r0: flips.append(f"{arm} {ds}|{c}|{m}: {order(arm)} vs REF {r0}")
# per-clip sign flips of TAG-vs-none
cf = 0
for k,v in CL.items():
    for m in LISTED:
        if m not in keys: continue
        for s in ("s02","s05"):
            sg = lambda arm: (f(v[(arm,s)][m]) > f(v[("none","none")][m])) - (f(v[(arm,s)][m]) < f(v[("none","none")][m]))
            if sg("V1") != sg("REF") or sg("V3") != sg("REF"): cf += 1
res["flips_macro"] = flips; res["per_clip_sign_flips_listed_keys"] = cf
if flips or cf: res["flag"].append(f"{len(flips)} macro rank/sign flips, {cf} per-clip sign flips (listed keys)")
# per-pixel
ps = {}
for arm in ("V1","V3"):
    for s in ("s02","s05"):
        P = [p for p in pix if p["arm"] == arm and p["setting"] == s]
        R = [p for p in pix if p["arm"] == "REF" and p["setting"] == s]
        mx = max(f(p["max_rel"]) for p in P); p999 = max(f(p["p999_rel"]) for p in P); med = max(f(p["median_rel"]) for p in P)
        nan = sum(int(p["nan_y"]) for p in P); inf = sum(int(p["inf_y"]) for p in P); fm = sum(int(p["finite_mask_changes"]) for p in P)
        bi = sum(p["byte_identical"] == "True" for p in P)
        ps[f"{arm}_{s}"] = {"n_clipcells": len(P), "max_rel": mx, "worst_p999_rel": p999, "worst_median_rel": med, "nan_px": nan, "inf_px": inf, "finite_mask_changes": fm, "byte_identical_clipcells": bi,
                            "ref_nan_px": sum(int(p["nan_y"]) for p in R), "ref_inf_px": sum(int(p["inf_y"]) for p in R)}
        if mx > 1e-4: res["fail"].append(f"per-pixel gate {arm} {s}: max {mx:.3g}")
        if nan != ps[f"{arm}_{s}"]["ref_nan_px"] or inf != ps[f"{arm}_{s}"]["ref_inf_px"] or fm: res["fail"].append(f"NaN/inf/mask {arm} {s}: nan {nan} inf {inf} mask {fm}")
res["per_pixel"] = ps
res["raw_depth_nonfinite"] = sum(1 for p in pix if p["arm"]=="REF" and (int(p["nan_ref"]) or int(p["inf_ref"])))
res["n_rows"] = len(rows); res["n_pix_rows"] = len(pix)
res["verdict"] = "FAIL" if res["fail"] else ("FLAG" if res["flag"] else "PASS")
(out / "gates.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=2, default=str))

"""Far/near-dome class per env (74). args: <depth_root> <seg_root> <sky_ids_74.json> <out.csv> [N]. Same 20-frame pick rule as sky_audit_74.py.
Class = median stored depth of seg-sky px (non-finite counted as +inf): >= 9000 m far-dome, else near-dome; no seg-sky px in sample -> no_seg_sky."""
import sys, os, json, zipfile, csv, collections
import numpy as np, cv2
DEPTH_ROOT, SEG_ROOT, IDS, OUT = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 20
ids_all = json.load(open(IDS))["envs"]
rows = []
for env in sorted(e for e in os.listdir(DEPTH_ROOT) if e in ids_all):
    info = ids_all[env]; sky_ids = set(info["sky_ids"])
    z = zipfile.ZipFile(f"{SEG_ROOT}/{env}/Data_easy/seg_lcam_front.zip")
    names = sorted(n for n in z.namelist() if n.endswith(".png")); bt = collections.defaultdict(list)
    for n in names: bt[n.split("/")[-3]].append(n)
    trajs = [t for t in sorted(bt) if os.path.isdir(f"{DEPTH_ROOT}/{env}/Data_easy/{t}/depth_lcam_front")]
    vals = []; used = 0
    for k in range(N):
        t = trajs[k % len(trajs)]; lst = bt[t]; j = k // len(trajs); per = -(-N // len(trajs))
        p = lst[int((j + 0.5) * len(lst) / per) % len(lst)]
        dp = f"{DEPTH_ROOT}/{env}/Data_easy/{t}/depth_lcam_front/" + os.path.basename(p).replace("_seg", "_depth")
        if not os.path.exists(dp): continue
        seg = cv2.imdecode(np.frombuffer(z.read(p), np.uint8), cv2.IMREAD_UNCHANGED)
        d = np.squeeze(np.ascontiguousarray(cv2.imdecode(np.fromfile(dp, np.uint8), cv2.IMREAD_UNCHANGED)).view("<f4"), -1)
        used += 1
        sat = ~np.isfinite(d) | (d >= 1e4)
        sk = np.isin(seg, list(sky_ids - {0})) if sky_ids else np.zeros(d.shape, bool)
        if 0 in sky_ids: sk |= (seg == 0) & sat
        if sk.any(): vals.append(np.where(np.isfinite(d[sk]), d[sk], np.inf).astype(np.float64))
    v = np.concatenate(vals) if vals else np.array([])
    if v.size:
        med = float(np.median(v)); q10, q90 = (float(x) for x in np.percentile(v, [10, 90])); fr = float((v >= 9000).mean())
        cls = "far_dome" if med >= 9000 else "near_dome"
    else:
        med = q10 = q90 = fr = float("nan"); cls = "no_seg_sky"
    rows.append(dict(env=env, sky_ids=";".join(map(str, sorted(sky_ids))), frames=used, n_seg_sky_px=int(v.size),
                     median_m=med, p10_m=q10, p90_m=q90, frac_ge_9000=round(fr, 4), cls=cls)); print(rows[-1], flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

"""OldTownNight far-plane stats. args: <depth_root> <out_csv>. All trajectories, q = 0.2/0.5/0.8 plus a 20-frame spread; per frame max, #px == max, #px >= 0.98*max, #px >= 0.9*max, top-5 distinct values."""
import sys, os, csv
import numpy as np, cv2
ROOT, OUT = sys.argv[1:3]; base = f"{ROOT}/OldTownNight/Data_easy"; rows = []
for t in sorted(x for x in os.listdir(base) if x.startswith("P") and os.path.isdir(f"{base}/{x}/depth_lcam_front")):
    fs = sorted(os.listdir(f"{base}/{t}/depth_lcam_front"))
    for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        f = fs[int(q * (len(fs) - 1))]
        d = np.squeeze(np.ascontiguousarray(cv2.imdecode(np.fromfile(f"{base}/{t}/depth_lcam_front/{f}", np.uint8), cv2.IMREAD_UNCHANGED)).view("<f4"), -1)
        m = float(d[np.isfinite(d)].max()); u, c = np.unique(d, return_counts=True); top = np.argsort(-u)[:5]
        rows.append(dict(traj=t, q=q, frame=f, n_nonfinite=int((~np.isfinite(d)).sum()), max_m=m, n_eq_max=int((d == m).sum()), n_ge_098max=int((d >= .98 * m).sum()),
                         n_ge_09max=int((d >= .9 * m).sum()), n_ge_150=int((d >= 150).sum()), median_m=float(np.median(d)), top5="; ".join(f"{u[i]:.2f}x{c[i]}" for i in top)))
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

"""Run in the isolated tartanair==1.4.0 venv: official reader outputs (sha256 of bytes) for sampled TRAIN frames."""
import sys, os, json, hashlib, glob
import numpy as np
from tartanair.reader import TartanAirImageReader
ROOT = "/mnt/DATA/DegradedDepth/data/TRAIN/tartanair"
out = sys.argv[1]
r = TartanAirImageReader()
h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
ent = []
for env in sorted(os.listdir(ROOT)):
    trajs = sorted(t for t in os.listdir(f"{ROOT}/{env}/Data_easy") if t.startswith("P") and os.path.isdir(f"{ROOT}/{env}/Data_easy/{t}/depth_lcam_front"))
    if not trajs: continue
    fixed = []; scan_hit = None
    for k, t in enumerate(trajs[:3]):
        fs = sorted(os.listdir(f"{ROOT}/{env}/Data_easy/{t}/depth_lcam_front"))
        for q in (0.1, 0.5, 0.9):
            fixed.append((t, fs[int(q * (len(fs) - 1))]))
        if scan_hit is None:
            for f in fs[:: max(1, len(fs) // 25)]:
                d = r.read_depth(f"{ROOT}/{env}/Data_easy/{t}/depth_lcam_front/{f}")
                if (d >= 65000).any() or not np.isfinite(d).all():
                    scan_hit = (t, f); break
    sel = fixed[:3] + fixed[4:5] + ([scan_hit] if scan_hit else [])
    for t, f in sel:
        dp = f"{ROOT}/{env}/Data_easy/{t}/depth_lcam_front/{f}"
        ip = f"{ROOT}/{env}/Data_easy/{t}/image_lcam_front/" + f.replace("_depth", "")
        d = r.read_depth(dp); im = r.read_rgb(ip)
        ent.append(dict(env=env, traj=t, depth=f, image=os.path.basename(ip), depth_dtype=str(d.dtype), depth_shape=list(d.shape),
            depth_sha256=h(d), rgb_shape=list(im.shape), rgb_dtype=str(im.dtype), rgb_sha256=h(im),
            n_nonfinite=int((~np.isfinite(d)).sum()), n_65504=int((d == 65504).sum()), dmax_finite=float(d[np.isfinite(d)].max()) if np.isfinite(d).any() else None))
json.dump(dict(tartanair_version="1.4.0", reader="tartanair.reader.TartanAirImageReader", entries=ent), open(out, "w"), indent=1)
print(len(ent), "entries")

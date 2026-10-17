"""Parity of our depth/RGB decoding with the OFFICIAL tartanairpy 1.4.0 reader on the cluster tree.
args: <depth_root> <out.csv> [frames_per_env=3].  Frames: first trajectory (sorted) at q=0.1/0.5/0.9 of its index.
Official reader imported unmodified via a sys.modules stub (package __init__ does not import on aarch64)."""
import sys, os, types, importlib, site, csv, io
import numpy as np, cv2
from PIL import Image
pkg = next(os.path.join(p, "tartanair") for p in site.getsitepackages() if os.path.isdir(os.path.join(p, "tartanair")))
stub = types.ModuleType("tartanair"); stub.__path__ = [pkg]; sys.modules["tartanair"] = stub
off = importlib.import_module("tartanair.reader").TartanAirImageReader()
ROOT, out = sys.argv[1], sys.argv[2]
def ours_depth(b):  # copy of data.video_clip_dataset._decode_depth
    a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    return np.squeeze(np.ascontiguousarray(a).view("<f4"), axis=-1)
rows = []
for env in sorted(e for e in os.listdir(ROOT) if os.path.isdir(f"{ROOT}/{e}/Data_easy")):
    base = f"{ROOT}/{env}/Data_easy"
    trajs = sorted(t for t in os.listdir(base) if t.startswith("P") and os.path.isdir(f"{base}/{t}/depth_lcam_front"))
    if not trajs: rows.append([env, "", "", "NO_TRAJ", "", ""]); continue
    t = trajs[0]; fs = sorted(os.listdir(f"{base}/{t}/depth_lcam_front"))
    for q in (0.1, 0.5, 0.9):
        f = fs[int(q * (len(fs) - 1))]
        dp = f"{base}/{t}/depth_lcam_front/{f}"; ip = f"{base}/{t}/image_lcam_front/" + f.replace("_depth", "")
        d0 = ours_depth(open(dp, "rb").read()); d1 = off.read_depth(dp)
        r0 = np.asarray(Image.open(ip).convert("RGB")); r1 = off.read_rgb(ip)
        deq = d0.dtype == d1.dtype and d0.shape == d1.shape and np.array_equal(d0, d1, equal_nan=True)
        req = r0.dtype == r1.dtype and r0.shape == r1.shape and np.array_equal(r0, r1)
        rows.append([env, t, f, "OK" if deq and req else "DIFF", int(deq), int(req)])
        print(rows[-1], flush=True)
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["env", "traj", "depth_file", "result", "depth_equal", "rgb_equal"]); w.writerows(rows)
print("n", len(rows), "DIFF", sum(r[3] == "DIFF" for r in rows), "NO_TRAJ", sum(r[3] == "NO_TRAJ" for r in rows))

import sys; sys.path.insert(0, "/home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts/B1_datasets")
from common import *
from data.video_clip_dataset import _decode_depth
import pandas as pd
from pathlib import Path
root = Path(f"{T}/tartanair"); rng = random.Random(3); rows = []
mism = 0; ntraj = 0
for env in sorted(p for p in root.iterdir() if p.is_dir()):
    diff = env / "Data_easy"
    if not diff.is_dir(): continue
    trajs = sorted(diff.iterdir())
    for tr in trajs:
        di, dd = tr / "image_lcam_front", tr / "depth_lcam_front"
        if not (di.is_dir() and dd.is_dir()): continue
        ntraj += 1
        a = sorted(p.name for p in di.glob("*.png")); b = sorted(p.name for p in dd.glob("*.png"))
        if len(a) != len(b) or any(x.replace("_lcam_front.png", "") != y.replace("_lcam_front_depth.png", "") for x, y in zip(a, b)): mism += 1
    tr = rng.choice(trajs); dd = tr / "depth_lcam_front"; fl = sorted(dd.glob("*.png"))
    for f in rng.sample(fl, min(4, len(fl))):
        d = _decode_depth(str(f)); s = raw_stats(d); fin = d[np.isfinite(d)]
        s.update(env=env.name, traj=tr.name, frame=f.name, hw=f"{d.shape[0]}x{d.shape[1]}", f_1e3_1e4=float(((d > 1e3) & (d <= 1e4)).mean()), f_80_655=float(((d > 80) & (d < 655)).mean()),
                 max_finite=float(fin.max()) if fin.size else np.nan, f_eq_1e4=float((d == 1e4).mean()), f_nonfinite_or_gt1e4=float((~np.isfinite(d) | (d > 1e4)).mean()))
        rows.append(s)
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/tartanair_env_survey.csv", index=False)
pd.set_option("display.width", 250)
g = df.groupby("env")[["f_inf", "f_nan", "f_gt80", "f_80_655", "f_ge655", "f_1e3_1e4", "f_gt1e4", "f_nonfinite_or_gt1e4", "max_finite", "med"]].mean().round(4)
print(g.to_string()); print("mean over all frames:", df[["f_inf", "f_nan", "f_gt80", "f_ge655", "f_gt1e4", "f_nonfinite_or_gt1e4", "f_zero", "f_neg"]].mean().round(4).to_dict())
print("global max finite", df.max_finite.max(), "trajs", ntraj, "image/depth name mismatching trajs", mism)

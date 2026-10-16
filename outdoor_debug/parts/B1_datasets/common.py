import sys, os, glob, random, json
REPO = "/home/wg19671/.claude/jobs/a0d391cb/tmp/od"
sys.path.insert(0, REPO)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np, cv2
OUT = "/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/B1_datasets"
T = "/mnt/DATA/DegradedDepth/data/TRAIN"
PARTS = REPO + "/outdoor_debug/parts/B1_datasets"

def pct(x):
    if x.size == 0: return dict(min=np.nan, p1=np.nan, med=np.nan, p99=np.nan, max=np.nan)
    q = np.percentile(x, [1, 50, 99])
    return dict(min=float(x.min()), p1=float(q[0]), med=float(q[1]), p99=float(q[2]), max=float(x.max()))

def raw_stats(d, sentinel_fn=None):
    """d: float32 metres, ORIGINAL decoded. returns dict of percentile (finite & >0) + fractions."""
    n = d.size
    fin = np.isfinite(d)
    pos = fin & (d > 0)
    s = dict(n=n, f_zero=float((d == 0).mean()), f_neg=float((d < 0).mean()), f_nan=float(np.isnan(d).mean()),
             f_inf=float(np.isinf(d).mean()), f_gt80=float((pos & (d > 80)).mean()), f_gt200=float((pos & (d > 200)).mean()),
             f_ge655=float((pos & (d >= 655)).mean()), f_gt1e4=float((d > 1e4).mean()))
    s.update(pct(d[pos]))
    return s

def color_depth(d, valid, lo=1.0, hi=1000.0):
    import matplotlib
    from matplotlib import cm
    x = np.log(np.clip(d, lo, hi) / lo) / np.log(hi / lo)
    c = (cm.turbo(x)[..., :3] * 255).astype(np.uint8)
    c[~valid] = (30, 30, 30)
    return c

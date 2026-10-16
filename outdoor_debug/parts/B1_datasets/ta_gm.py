import sys; sys.path.insert(0, "/home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts/B1_datasets")
from common import *
from data.vda_semantics_dataset import VDASemanticsTartanAirDataset
from data.video_clip_dataset import _decode_depth
from PIL import Image
import collections, pandas as pd
ds = VDASemanticsTartanAirDataset(root=f"{T}/tartanair", num_frames=32, image_size=(518,518), split="train", clip_stride=32, frame_stride=1, max_depth=10000.0, seed=45, val_fraction=0.1, degrade=False)
cnt = collections.Counter(c[2].split("/")[0] for c in ds.clips); tot = len(ds.clips)
print("clips", tot); print({k: round(v/tot,3) for k, v in cnt.most_common(8)})
rng = random.Random(5); gm = [i for i, c in enumerate(ds.clips) if c[2].startswith("GreatMarsh")]
tiles = []; rows = []
for i in rng.sample(gm, 6):
    c = ds.clips[i]; f = c[1][c[3]+16]; d = _decode_depth(str(f)); im = np.asarray(Image.open(c[0][c[3]+16]).convert("RGB"))
    cat = np.zeros(d.shape, np.uint8); cat[d >= 80] = 1; cat[d >= 655] = 2; cat[d >= 1e4] = 3
    pal = np.array([[40,160,60],[240,220,40],[240,120,0],[220,0,0]], np.uint8)
    tiles.append(np.concatenate([im, (0.5*im+0.5*pal[cat]).astype(np.uint8), color_depth(d, np.ones(d.shape, bool), 1, 1e5)], 1))
    q = np.percentile(d[d > 0], [1, 50, 99]); rows.append((c[2], [round(float((cat==k).mean()),3) for k in range(4)], q.round(1).tolist(), float(d.max())))
for r in rows: print(r)
s = np.concatenate(tiles, 0); Image.fromarray(s).resize((s.shape[1]//2, s.shape[0]//2)).save(f"{OUT}/tartanair_greatmarsh.png")
# overall far-valid share over all train clips (middle frame of 400 random clips)
tot655 = []; tot1e4 = []
for i in rng.sample(range(len(ds)), 400):
    c = ds.clips[i]; d = _decode_depth(str(c[1][c[3]+16])); tot655.append(((d>=655)&(d<1e4)).mean()); tot1e4.append((d>=1e4).mean())
print("clip-weighted mean frac 655-1e4 (valid far, NOT sky):", np.mean(tot655), " >=1e4 (sky):", np.mean(tot1e4))

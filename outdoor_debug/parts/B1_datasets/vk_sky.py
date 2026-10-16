import sys; sys.path.insert(0, "/home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts/B1_datasets")
from common import *
from PIL import Image
import collections, json
root = Path = None
from pathlib import Path
root = Path(f"{T}/vkitti")
allf = []
for sc in sorted(p for p in root.iterdir() if p.is_dir()):
    for v in sorted(p for p in sc.iterdir() if p.is_dir()):
        for cam in ("Camera_0", "Camera_1"):
            dd = v / "frames/depth" / cam
            if dd.is_dir():
                allf += [(sc.name, v.name, cam, p.name) for p in sorted(dd.glob("depth_*.png"))]
print("total VKITTI depth frames", len(allf))
rng = random.Random(7)
sel = rng.sample(allf, 300)
vals = collections.Counter(); maxnon = 0; nsky = 0; ntot = 0; sky_bottom = []; holes = []; thin = []
tiles = []
for i, (sc, v, cam, fn) in enumerate(sel):
    raw = cv2.imread(str(root / sc / v / "frames/depth" / cam / fn), cv2.IMREAD_UNCHANGED)
    assert raw.dtype == np.uint16, raw.dtype
    big = raw[raw >= 60000]
    vals.update(big.tolist())
    non = raw[raw < 65535]
    maxnon = max(maxnon, int(non.max()))
    sky = raw >= 65535
    nsky += sky.sum(); ntot += raw.size
    H = raw.shape[0]
    sky_bottom.append(float(sky[H // 2:].mean() / max(sky.mean(), 1e-9)) if sky.any() else 0.0)
    # sky "holes": sky pixel with non-sky (valid) pixel ABOVE it in the same column (excluding thin things) -> geometry above sky
    cs = np.cumsum(~sky, axis=0) > 0
    holes.append(float((sky & cs).sum() / max(sky.sum(), 1)))
    if i < 120:
        rgb = np.asarray(Image.open(root / sc / v / "frames/rgb" / cam / fn.replace("depth_", "rgb_").replace(".png", ".jpg")).convert("RGB")).astype(np.float32)
        assert rgb.shape[:2] == raw.shape, (rgb.shape, raw.shape)
        ov = rgb.copy(); ov[sky] = 0.45 * ov[sky] + 0.55 * np.array([255, 0, 0])
        im = Image.fromarray(ov.astype(np.uint8)).resize((310, 94))
        tiles.append(np.asarray(im))
cols = 6
rows = (len(tiles) + cols - 1) // cols
sheet = np.zeros((rows * 94, cols * 310, 3), np.uint8)
for k, t in enumerate(tiles):
    r, c = divmod(k, cols); sheet[r*94:(r+1)*94, c*310:(c+1)*310] = t
Image.fromarray(sheet).save(f"{OUT}/vkitti_sky_overlay_contact_sheet_120frames.png")
res = dict(frames_sampled=len(sel), sky_pixel_frac=float(nsky / ntot), max_non_sentinel_raw=maxnon,
           top_values_ge60000=vals.most_common(8), n_distinct_values_ge60000=len(vals),
           values_in_65000_65534=int(sum(c for v, c in vals.items() if 65000 <= v < 65535)),
           sky_in_bottom_half_ratio_mean=float(np.mean(sky_bottom)), sky_with_valid_above_in_column_mean=float(np.mean(holes)))
# cross-variant sky identity: same frame, clone vs other variants
same = []; 
for sc in ("Scene01", "Scene18", "Scene20"):
    for cam in ("Camera_0",):
        for fn in ("depth_00100.png",):
            ref = cv2.imread(str(root / sc / "clone/frames/depth" / cam / fn), cv2.IMREAD_UNCHANGED)
            for v in ("fog", "rain", "overcast", "sunset", "15-deg-left"):
                o = cv2.imread(str(root / sc / v / "frames/depth" / cam / fn), cv2.IMREAD_UNCHANGED)
                if o is None: continue
                same.append((sc, v, float(((o >= 65535) == (ref >= 65535)).mean()), float((o == ref).mean())))
res["variant_sky_agreement(scene,variant,sky_agree,depth_equal)"] = same
json.dump(res, open(f"{PARTS}/vk_sky_result.json", "w"), indent=1)
print(json.dumps(res, indent=1))

"""TartanAir raw depth: how is sky encoded? Render RGB | log-depth | pixels classified (sky>1e4 / far valid 80..1e4 / <80) for several outdoor envs."""
import sys, glob, random, pathlib, json
sys.path.insert(0, '.')
import numpy as np, cv2
from PIL import Image
from data.video_clip_dataset import _decode_depth
R = pathlib.Path('/mnt/DATA/DegradedDepth/data/TRAIN/tartanair'); OUT = pathlib.Path('/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/B2_transforms_masks')
random.seed(7)
envs = sys.argv[1:] or ['Downtown', 'AmusementPark', 'GreatMarsh', 'ModernCityDowntown', 'AmericanDiner', 'ForestEnv', 'Hospital', 'HongKong', 'Rome', 'NordicHarbor']
rows = []; table = {}
for e in envs:
    ps = sorted(glob.glob(f'{R}/{e}/Data_easy/P00*/depth_lcam_front/*.png'))
    p = random.choice(ps); d = _decode_depth(p)
    rgb = np.asarray(Image.open(p.replace('depth_lcam_front', 'image_lcam_front').replace('_depth', '').replace('depth', 'left') if False else p.replace('depth_lcam_front', 'image_lcam_front').replace('_lcam_front_depth', '_lcam_front')).convert('RGB'))
    fin = d[np.isfinite(d)]
    cls = np.zeros(d.shape + (3,), np.uint8)
    cls[d < 80] = (40, 160, 60); cls[(d >= 80) & (d < 655)] = (230, 200, 40); cls[(d >= 655) & (d <= 1e4)] = (230, 60, 60); cls[d > 1e4] = (0, 120, 255)
    x = (np.log(np.clip(d, 0.5, 1e5)) - np.log(0.5)) / (np.log(1e5) - np.log(0.5))
    dep = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
    row = np.concatenate([cv2.resize(a, (400, 300)) for a in (rgb, dep, cls)], 1)
    q = lambda v: [round(float(t), 1) for t in np.percentile(fin, v)]
    table[e] = dict(file=p.split('tartanair/')[1], pct_lt80=round(100 * (d < 80).mean(), 2), pct_80_655=round(100 * ((d >= 80) & (d < 655)).mean(), 2), pct_655_1e4=round(100 * ((d >= 655) & (d <= 1e4)).mean(), 2),
                    pct_gt1e4=round(100 * (d > 1e4).mean(), 2), p50_p99_max=q([50, 99, 100]))
    cv2.putText(row, f"{e}  <80:green 80-655:yellow 655-1e4:red >1e4(sky label):blue", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    rows.append(row)
Image.fromarray(np.concatenate(rows, 0)).save(OUT / 'ta_sky_encoding_probe.jpg', quality=80)
json.dump(table, open('outdoor_debug/parts/B2_transforms_masks/ta_sky_probe.json', 'w'), indent=1)
for k, v in table.items(): print(k, v)

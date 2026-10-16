"""Before/after-transform panels per outdoor source + sky-overlay contact sheet (real dataset classes, CPU)."""
import sys, json, random, pathlib
sys.path.insert(0, '.')
import numpy as np, torch, cv2
from PIL import Image
import train
from data.vda_resize import vda_resize_size, vda_crop_box
T = pathlib.Path('/mnt/DATA/DegradedDepth/data/TRAIN')
OUT = pathlib.Path('/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/B2_transforms_masks'); OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open('outdoor_debug/parts/B2_transforms_masks/effective_C3.json')); d = cfg['data']
common = dict(num_frames=1, image_size=(518, 518), clip_stride=16, frame_stride=1, max_depth=80.0, seed=45, degrade=False, degrade_mode=['awgn_blur'],
              degrade_severity=0.0, max_clips_per_video=None, resize_mode='vda')
roots = {'tartanair': T/'tartanair', 'vkitti': T/'vkitti', 'pointodyssey': T/'point_odyssey/train', 'irs': T/'irs', 'kitti_raw': T/'kitti_raw', 'spring': T/'spring'}
def mk(spec):
    spec = dict(spec); spec['root'] = str(roots[spec['type']])
    spec.pop('degrade', None)
    if 'gt_root' in spec: spec['gt_root'] = str(T/'kitti_raw_gt')
    if 'po_keep_list' in spec:
        kl = json.load(open(spec['po_keep_list'])); kl['kept'] = [k for k in kl['kept'] if (roots['pointodyssey']/k).is_dir()]
        p = OUT/'po_keep_local_pointodyssey_train.json'; json.dump(kl, open(p, 'w')); spec['po_keep_list'] = str(p); spec.pop('po_keep_sha256', None)
    spec.pop('metric_supervision', None); spec.pop('sparse_gt', None); spec.pop('silog_max_depth', None)
    # clean (non-degraded) real class: degrade False so the non-degraded VDASemantics*/v3 classes (same getitem body) are used
    c = dict(common); c['degrade'] = False
    return train._make_source(spec, 'train', c)
specs = {s['type']: s for s in d['sources'] if not (s['type'] == 'pointodyssey' and s['root'].endswith('/val'))}
def colz(dep, lo=0.5, hi=700, valid=None):
    x = np.log(np.clip(dep, lo, hi)); x = (x - np.log(lo)) / (np.log(hi) - np.log(lo))
    im = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    if valid is not None: im[~valid] = 0
    return im
def raw_of(ds, i):
    """raw rgb + raw depth (metres, as the loader sees it, before resize/crop)"""
    t = type(ds).__name__
    c = ds.clips[i]
    rgbp = c[0][c[3]]
    img = np.asarray(Image.open(rgbp).convert('RGB'))
    return img, c
def panel(name, ds, idxs, ncol_note=''):
    rows = []
    for k, i in enumerate(idxs):
        random.seed(1000 + k); it = ds[i]
        random.seed(1000 + k)
        raw = np.asarray(Image.open(ds.clips[i][0][ds.clips[i][3]]).convert('RGB'))
        h0, w0 = raw.shape[:2]; nw, nh = vda_resize_size(w0, h0, 518); x0, y0 = vda_crop_box(nw, nh, 518, 'train')
        sx, sy = w0 / nw, h0 / nh
        r = raw.copy(); cv2.rectangle(r, (int(x0 * sx), int(y0 * sy)), (int((x0 + 518) * sx), int((y0 + 518) * sy)), (255, 0, 0), max(2, w0 // 300))
        r = cv2.resize(r, (int(w0 * 420 / h0), 420))
        f = (it['frames'][:, 0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        D = it['depth'][0, 0].numpy(); M = it['mask'][0, 0].numpy() > .5; S = it['sky'][0, 0].numpy() > .5
        vis = f.copy(); vis[S] = (0.5 * vis[S] + 0.5 * np.array([0, 120, 255])).astype(np.uint8)
        inv = (~M & ~S); vis[inv] = (0.5 * vis[inv] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        cat = np.full((518, 518, 3), 40, np.uint8)
        dimg = colz(D, valid=M); dimg[S] = (255, 255, 255)
        tiles = [cv2.resize(f, (420, 420)), cv2.resize(vis, (420, 420), interpolation=cv2.INTER_NEAREST), cv2.resize(dimg, (420, 420), interpolation=cv2.INTER_NEAREST)]
        row = np.concatenate([np.pad(r, ((0, 0), (0, 0), (0, 0)))] + tiles, 1)
        txt = f"{name} clip{i} raw {w0}x{h0} -> {nw}x{nh} crop@({x0},{y0}) | valid {100*M.mean():.1f}% sky {100*S.mean():.1f}% invalid {100*inv.mean():.1f}% far>80 {100*(M&(D>80)).mean():.1f}% maxD {D[M].max() if M.any() else 0:.0f}"
        cv2.putText(row, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        rows.append(row)
    w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    big = np.concatenate(rows, 0)
    Image.fromarray(big).save(OUT/f'panel_{name}.jpg', quality=82)
    small = cv2.resize(big, (big.shape[1] // 2, big.shape[0] // 2), interpolation=cv2.INTER_AREA)
    Image.fromarray(small).save(OUT/f'panel_{name}_small.jpg', quality=70)
random.seed(0)
dss = {}
for t in ('tartanair', 'vkitti', 'pointodyssey', 'irs', 'kitti_raw', 'spring'):
    try: dss[t] = mk(specs[t])
    except Exception as e: print('FAIL', t, repr(e)[:200]); continue
    n = len(dss[t]); print(t, n, flush=True)
    panel(t, dss[t], random.sample(range(n), 3))
# contact sheet: >=100 random outdoor frames with sky masks (TartanAir + VKITTI); label sky %
tiles = []
for t, k in (('vkitti', 60), ('tartanair', 80)):
    ds = dss[t]; n = len(ds)
    for j, i in enumerate(random.sample(range(n), k)):
        random.seed(5000 + j); it = ds[i]
        f = (it['frames'][:, 0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        S = it['sky'][0, 0].numpy() > .5; M = it['mask'][0, 0].numpy() > .5
        v = f.copy(); v[S] = (0.45 * v[S] + 0.55 * np.array([0, 140, 255])).astype(np.uint8)
        v = cv2.resize(v, (200, 200), interpolation=cv2.INTER_AREA)
        cv2.putText(v, f"{t[:3]} s{100*S.mean():.0f}% v{100*M.mean():.0f}%", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append((t, 100 * S.mean(), 100 * M.mean(), v))
json.dump([(a, b, c) for a, b, c, _ in tiles], open(OUT/'contact_sheet_stats.json', 'w'))
cols = 14; rows_ = [np.concatenate([x[3] for x in tiles[i:i + cols]] + [np.zeros((200, 200, 3), np.uint8)] * (cols - len(tiles[i:i + cols])), 1) for i in range(0, len(tiles), cols)]
sheet = np.concatenate(rows_, 0); Image.fromarray(sheet).save(OUT/'contact_sheet_sky_overlays.jpg', quality=80)
print('contact sheet', len(tiles), 'frames', sheet.shape)
